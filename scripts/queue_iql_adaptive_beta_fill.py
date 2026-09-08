#!/usr/bin/env python3
"""Fill GPUs up to --jobs-per-gpu for a profile without killing existing trainers.

Respects already-running iql_adaptive_beta children (any suite). Assigns each new
job to the least-loaded physical GPU among --gpus. Does not silently restart
failures; records them and continues with the rest of the queue.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from algorithms.offline.iql_adaptive_beta_utils import (  # noqa: E402
    adaptive_beta_source_files,
    code_hash,
)
from scripts.launch_iql_adaptive_beta import (  # noqa: E402
    ENV_MAP,
    PROFILES,
    gpu_uuid,
    is_live_output,
    load_gate,
    verify_gate,
)


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def count_live_iql_by_gpu(gpu_ids: List[int]) -> Dict[int, int]:
    """Count live `iql_adaptive_beta` trainers per physical GPU via nvidia-smi + cmdline."""
    counts = {g: 0 for g in gpu_ids}
    # Map pid -> gpu index via nvidia-smi
    pid_to_gpu: Dict[int, int] = {}
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,gpu_uuid",
                "--format=csv,noheader",
            ],
            text=True,
        )
        uuid_to_idx = {}
        for g in gpu_ids:
            u = gpu_uuid(g)
            if u:
                uuid_to_idx[u] = g
        for line in out.strip().splitlines():
            if not line.strip():
                continue
            parts = [x.strip() for x in line.split(",")]
            if len(parts) < 2:
                continue
            pid = int(parts[0])
            uuid = parts[1]
            if uuid in uuid_to_idx:
                pid_to_gpu[pid] = uuid_to_idx[uuid]
    except Exception:
        pass

    # Confirm cmdline is our trainer
    try:
        ps = subprocess.check_output(["ps", "-eo", "pid,cmd"], text=True)
    except Exception:
        return counts
    for line in ps.splitlines():
        if "iql_adaptive_beta" not in line or "queue_iql" in line:
            continue
        parts = line.strip().split(None, 1)
        if not parts:
            continue
        try:
            pid = int(parts[0])
        except ValueError:
            continue
        if not _pid_alive(pid):
            continue
        gpu = pid_to_gpu.get(pid)
        if gpu is None:
            # Parse CUDA_VISIBLE_DEVICES from /proc/<pid>/environ
            try:
                env_raw = Path(f"/proc/{pid}/environ").read_bytes()
                for item in env_raw.split(b"\0"):
                    if item.startswith(b"CUDA_VISIBLE_DEVICES="):
                        val = item.split(b"=", 1)[1].decode()
                        if val.strip() != "":
                            gpu = int(val.split(",")[0])
                        break
            except Exception:
                continue
        if gpu in counts:
            counts[gpu] += 1
    return counts


def pick_gpu(counts: Dict[int, int], jobs_per_gpu: int) -> Optional[int]:
    free = [(c, g) for g, c in counts.items() if c < jobs_per_gpu]
    if not free:
        return None
    free.sort()
    return free[0][1]


def start_one(
    *,
    env_id: str,
    physical_gpu: int,
    meta_config: Path,
    dataset_dir: Path,
    suite_dir: Path,
    seed: int,
) -> Dict[str, Any]:
    info = ENV_MAP[env_id]
    dataset_path = (dataset_dir / info["hdf5"]).resolve()
    if not dataset_path.is_file():
        raise FileNotFoundError(dataset_path)
    yaml_rel = info["yaml"]
    with open(REPO_ROOT / yaml_rel) as f:
        cfg = yaml.safe_load(f)
    if cfg.get("env") != env_id:
        raise ValueError(f"config/env mismatch for {env_id}")

    run_dir = suite_dir / env_id
    if is_live_output(run_dir):
        raise RuntimeError(f"live/completed output exists: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "-m",
        "algorithms.offline.iql_adaptive_beta",
        "--config",
        str(REPO_ROOT / yaml_rel),
        "--meta-config",
        str(meta_config),
        "--env",
        env_id,
        "--seed",
        str(seed),
        "--device",
        "cuda:0",
        "--dataset-path",
        str(dataset_path),
        "--output-dir",
        str(run_dir),
        "--mode",
        "train",
    ]
    child_env = os.environ.copy()
    child_env["CUDA_VISIBLE_DEVICES"] = str(physical_gpu)
    child_env.setdefault("D4RL_SUPPRESS_IMPORT_ERROR", "1")
    child_env.setdefault("WANDB_MODE", "disabled")
    child_env.setdefault("OMP_NUM_THREADS", "1")
    child_env.setdefault("MKL_NUM_THREADS", "1")
    child_env.setdefault("PYTHONUNBUFFERED", "1")
    # Avoid empty CVD from parent shells
    if child_env.get("CUDA_VISIBLE_DEVICES", None) == "":
        child_env["CUDA_VISIBLE_DEVICES"] = str(physical_gpu)

    stdout = open(run_dir / "stdout.log", "a")
    proc = subprocess.Popen(
        cmd,
        cwd=str(REPO_ROOT),
        env=child_env,
        stdout=stdout,
        stderr=subprocess.STDOUT,
        shell=False,
        start_new_session=True,
    )
    entry = {
        "env": env_id,
        "pid": proc.pid,
        "argv": cmd,
        "output_dir": str(run_dir),
        "gpu_physical": physical_gpu,
        "cuda_visible_devices": str(physical_gpu),
        "gpu_uuid": gpu_uuid(physical_gpu),
        "device": "cuda:0",
        "dataset_path": str(dataset_path),
        "started_at": time.time(),
    }
    with open(run_dir / "launcher_meta.json", "w") as f:
        json.dump(entry, f, indent=2)
    (run_dir / "launcher.pid").write_text(str(proc.pid) + "\n")
    return entry


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--profile", required=True, choices=list(PROFILES))
    p.add_argument("--meta-config", required=True)
    p.add_argument("--gpus", default="0,1")
    p.add_argument("--jobs-per-gpu", type=int, default=4)
    p.add_argument("--dataset-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--validation-gate", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--poll-seconds", type=float, default=30.0)
    p.add_argument("--once", action="store_true", help="Fill current free slots then exit")
    p.add_argument("--detach", action="store_true", help="Re-exec self detached as supervisor")
    args = p.parse_args()

    meta_config = Path(args.meta_config)
    if not meta_config.is_absolute():
        meta_config = (REPO_ROOT / meta_config).resolve()
    gate_path = Path(args.validation_gate)
    if not gate_path.is_absolute():
        gate_path = (REPO_ROOT / gate_path).resolve()
    gate = load_gate(gate_path)
    verify_gate(gate, meta_config)

    dataset_dir = Path(args.dataset_dir).resolve()
    suite_dir = Path(args.output_dir).resolve()
    suite_dir.mkdir(parents=True, exist_ok=True)
    gpu_list = [int(x) for x in args.gpus.split(",") if x.strip() != ""]

    queue = list(PROFILES[args.profile])
    # Skip already completed or currently live
    pending: List[str] = []
    for env_id in queue:
        run_dir = suite_dir / env_id
        if (run_dir / "completion.json").exists():
            print(f"[queue] skip completed {env_id}", flush=True)
            continue
        if is_live_output(run_dir):
            print(f"[queue] skip live {env_id}", flush=True)
            continue
        pending.append(env_id)

    state_path = suite_dir / "queue_state.json"
    launched: List[Dict[str, Any]] = []
    failed: List[Dict[str, Any]] = []

    if args.detach and not os.environ.get("IQL_QUEUE_SUPERVISOR"):
        log = open(suite_dir / "queue_supervisor.log", "a")
        env = os.environ.copy()
        env["IQL_QUEUE_SUPERVISOR"] = "1"
        cmd = [sys.executable, str(Path(__file__).resolve()), *[
            a for a in sys.argv[1:] if a != "--detach"
        ]]
        proc = subprocess.Popen(
            cmd,
            cwd=str(REPO_ROOT),
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            shell=False,
        )
        (suite_dir / "queue_supervisor.pid").write_text(str(proc.pid) + "\n")
        print(f"[queue] detached supervisor pid={proc.pid}", flush=True)
        return 0

    print(
        f"[queue] pending={pending} jobs_per_gpu={args.jobs_per_gpu} gpus={gpu_list}",
        flush=True,
    )

    while True:
        counts = count_live_iql_by_gpu(gpu_list)
        print(f"[queue] live_by_gpu={counts} pending={len(pending)}", flush=True)

        # Detect failures among launched: dead without completion
        still = []
        for entry in launched:
            pid = entry["pid"]
            out = Path(entry["output_dir"])
            if (out / "completion.json").exists():
                continue
            if _pid_alive(pid):
                still.append(entry)
                continue
            failed.append({**entry, "failed_at": time.time(), "reason": "process_exited"})
            print(f"[queue] FAILED {entry['env']} pid={pid}", flush=True)
        launched = still

        progressed = False
        while pending:
            gpu = pick_gpu(counts, args.jobs_per_gpu)
            if gpu is None:
                break
            env_id = pending.pop(0)
            try:
                entry = start_one(
                    env_id=env_id,
                    physical_gpu=gpu,
                    meta_config=meta_config,
                    dataset_dir=dataset_dir,
                    suite_dir=suite_dir,
                    seed=args.seed,
                )
                launched.append(entry)
                counts[gpu] = counts.get(gpu, 0) + 1
                progressed = True
                print(json.dumps({"launched": entry}), flush=True)
            except Exception as exc:  # noqa: BLE001
                failed.append({"env": env_id, "error": repr(exc), "failed_at": time.time()})
                print(f"[queue] launch error {env_id}: {exc}", flush=True)

        state = {
            "profile": args.profile,
            "pending": pending,
            "launched": launched,
            "failed": failed,
            "live_by_gpu": counts,
            "jobs_per_gpu": args.jobs_per_gpu,
            "validation_gate": str(gate_path),
            "code_hash": gate.get("code_hash"),
            "updated_at": time.time(),
        }
        with open(state_path, "w") as f:
            json.dump(state, f, indent=2)

        if not pending:
            print("[queue] queue drained (no more pending launches)", flush=True)
            break
        if args.once and not progressed:
            print("[queue] --once and no free slots; exit", flush=True)
            break
        if args.once and progressed and pick_gpu(counts, args.jobs_per_gpu) is None:
            print("[queue] --once filled free slots; exit", flush=True)
            break
        time.sleep(float(args.poll_seconds))

    return 0 if not failed else 1


if __name__ == "__main__":
    # Allow `from scripts.launch_...` when run as script
    sys.path.insert(0, str(REPO_ROOT))
    raise SystemExit(main())
