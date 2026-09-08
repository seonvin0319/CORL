#!/usr/bin/env python3
"""Launch production runs for corl_iql_adaptive_beta_v1 after validation gate."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from algorithms.offline.iql_adaptive_beta_utils import (  # noqa: E402
    adaptive_beta_source_files,
    code_hash,
)

ENV_MAP = {
    "halfcheetah-medium-v2": {
        "yaml": "configs/offline/iql/halfcheetah/medium_v2.yaml",
        "hdf5": "halfcheetah_medium-v2.hdf5",
        "gpu_index_in_loco": 0,
    },
    "halfcheetah-medium-expert-v2": {
        "yaml": "configs/offline/iql/halfcheetah/medium_expert_v2.yaml",
        "hdf5": "halfcheetah_medium_expert-v2.hdf5",
        "gpu_index_in_loco": 0,
    },
    "halfcheetah-medium-replay-v2": {
        "yaml": "configs/offline/iql/halfcheetah/medium_replay_v2.yaml",
        "hdf5": "halfcheetah_medium_replay-v2.hdf5",
        "gpu_index_in_loco": 0,
    },
    "hopper-medium-v2": {
        "yaml": "configs/offline/iql/hopper/medium_v2.yaml",
        "hdf5": "hopper_medium_v2.hdf5",
        "gpu_index_in_loco": 1,
    },
    "hopper-medium-expert-v2": {
        "yaml": "configs/offline/iql/hopper/medium_expert_v2.yaml",
        "hdf5": "hopper_medium_expert-v2.hdf5",
        "gpu_index_in_loco": 1,
    },
    "hopper-medium-replay-v2": {
        "yaml": "configs/offline/iql/hopper/medium_replay_v2.yaml",
        "hdf5": "hopper_medium_replay-v2.hdf5",
        "gpu_index_in_loco": 1,
    },
    "walker2d-medium-v2": {
        "yaml": "configs/offline/iql/walker2d/medium_v2.yaml",
        "hdf5": "walker2d_medium-v2.hdf5",
        "gpu_index_in_loco": 0,
    },
    "walker2d-medium-expert-v2": {
        "yaml": "configs/offline/iql/walker2d/medium_expert_v2.yaml",
        "hdf5": "walker2d_medium_expert-v2.hdf5",
        "gpu_index_in_loco": 1,
    },
    "walker2d-medium-replay-v2": {
        "yaml": "configs/offline/iql/walker2d/medium_replay_v2.yaml",
        "hdf5": "walker2d_medium_replay-v2.hdf5",
        "gpu_index_in_loco": 0,
    },
    "antmaze-umaze-diverse-v2": {
        "yaml": "configs/offline/iql/antmaze/umaze_diverse_v2.yaml",
        "hdf5": "Ant_maze_u-maze_noisy_multistart_True_multigoal_True_sparse_fixed.hdf5",
        "gpu_index_in_loco": 1,
    },
    "antmaze-medium-play-v2": {
        "yaml": "configs/offline/iql/antmaze/medium_play_v2.yaml",
        "hdf5": "Ant_maze_big-maze_noisy_multistart_True_multigoal_False_sparse_fixed.hdf5",
        "gpu_index_in_loco": 1,
    },
}

PROFILES = {
    "locomotion4": [
        "halfcheetah-medium-v2",
        "halfcheetah-medium-expert-v2",
        "hopper-medium-v2",
        "hopper-medium-expert-v2",
    ],
    "locomotion_rest5": [
        "halfcheetah-medium-replay-v2",
        "hopper-medium-replay-v2",
        "walker2d-medium-v2",
        "walker2d-medium-expert-v2",
        "walker2d-medium-replay-v2",
    ],
    "antmaze2": [
        "antmaze-umaze-diverse-v2",
        "antmaze-medium-play-v2",
    ],
    "remaining7": [
        "halfcheetah-medium-replay-v2",
        "hopper-medium-replay-v2",
        "walker2d-medium-v2",
        "walker2d-medium-expert-v2",
        "walker2d-medium-replay-v2",
        "antmaze-umaze-diverse-v2",
        "antmaze-medium-play-v2",
    ],
}


def load_gate(path: Path) -> Dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def verify_gate(gate: Dict[str, Any], meta_config: Path) -> None:
    if gate.get("status") != "passed":
        raise RuntimeError(f"validation gate status is not passed: {gate.get('status')}")
    sources = [p for p in adaptive_beta_source_files(REPO_ROOT) if p.exists()]
    chash = code_hash(sources)
    if gate.get("code_hash") != chash:
        raise RuntimeError(
            f"code hash mismatch: gate={gate.get('code_hash')} current={chash}. "
            "Re-run validation."
        )
    with open(meta_config) as f:
        meta_raw = yaml.safe_load(f)
    import hashlib

    prod_hash = hashlib.sha256(json.dumps(meta_raw, sort_keys=True).encode()).hexdigest()
    if gate.get("production_semantic_config_hash") != prod_hash:
        raise RuntimeError("production semantic config hash mismatch; re-run validation")


def gpu_uuid(index: int) -> Optional[str]:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid",
                "--format=csv,noheader",
            ],
            text=True,
        )
        for line in out.strip().splitlines():
            idx, uuid = [x.strip() for x in line.split(",")]
            if int(idx) == index:
                return uuid
    except Exception:
        return None
    return None


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def is_live_output(output_dir: Path) -> bool:
    """Refuse only completed runs or currently-alive trainers in this output dir."""
    if (output_dir / "completion.json").exists():
        return True
    manifest = output_dir / "runtime_manifest.json"
    if manifest.exists():
        try:
            data = json.loads(manifest.read_text())
            pid = int(data.get("pid") or 0)
            if pid and _pid_alive(pid):
                return True
        except Exception:
            pass
    # Fallback: launcher.pid from older layout
    pid_path = output_dir / "launcher.pid"
    if pid_path.exists():
        try:
            pid = int(pid_path.read_text().strip())
            if _pid_alive(pid):
                return True
        except Exception:
            pass
    return False


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--profile", required=True, choices=list(PROFILES))
    p.add_argument("--meta-config", required=True)
    p.add_argument("--gpus", default="0,1")
    p.add_argument("--jobs-per-gpu", type=int, default=2)
    p.add_argument("--dataset-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--validation-gate", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--detach", action="store_true")
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
    envs = PROFILES[args.profile]

    # Respect existing CUDA_VISIBLE_DEVICES if set: logical ids map into that list.
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd:
        visible = [int(x) for x in cvd.split(",") if x.strip() != ""]
    else:
        visible = None

    children: List[Dict[str, Any]] = []
    for env_id in envs:
        info = ENV_MAP[env_id]
        yaml_rel = info["yaml"]
        hdf5 = info["hdf5"]
        dataset_path = (dataset_dir / hdf5).resolve()
        if not dataset_path.is_file():
            raise FileNotFoundError(dataset_path)

        with open(REPO_ROOT / yaml_rel) as f:
            cfg = yaml.safe_load(f)
        if cfg.get("env") != env_id:
            raise ValueError(f"config/env mismatch for {env_id}")

        if args.profile == "locomotion4":
            logical = info["gpu_index_in_loco"]
        else:
            # antmaze2: both on first listed GPU by default (jobs-per-gpu=2)
            logical = 0 if len(gpu_list) == 1 else info["gpu_index_in_loco"]

        if logical >= len(gpu_list):
            raise ValueError(f"GPU index {logical} out of range for --gpus {args.gpus}")
        physical = gpu_list[logical]
        device = f"cuda:{logical}" if visible is None else "cuda:0"
        # When we set CUDA_VISIBLE_DEVICES per child, always use cuda:0 inside.
        child_cvd = str(physical) if visible is None else str(visible[physical] if physical < len(visible) else physical)
        # Simpler: always isolate one GPU per process via CUDA_VISIBLE_DEVICES=physical
        child_cvd = str(physical)
        device = "cuda:0"

        run_dir = suite_dir / env_id
        if is_live_output(run_dir):
            raise RuntimeError(
                f"Refusing duplicate live/completed output: {run_dir}. "
                "Choose a new --output-dir or clean completed runs deliberately."
            )
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
            str(args.seed),
            "--device",
            device,
            "--dataset-path",
            str(dataset_path),
            "--output-dir",
            str(run_dir),
            "--mode",
            "train",
        ]
        child_env = os.environ.copy()
        child_env["CUDA_VISIBLE_DEVICES"] = child_cvd
        child_env.setdefault("D4RL_SUPPRESS_IMPORT_ERROR", "1")
        child_env.setdefault("WANDB_MODE", "disabled")
        child_env.setdefault("OMP_NUM_THREADS", "1")
        child_env.setdefault("MKL_NUM_THREADS", "1")
        child_env.setdefault("PYTHONUNBUFFERED", "1")

        stdout = open(run_dir / "launcher_stdout.log", "a")
        stderr = open(run_dir / "launcher_stderr.log", "a")
        popen_kwargs: Dict[str, Any] = {
            "cwd": str(REPO_ROOT),
            "env": child_env,
            "stdout": stdout,
            "stderr": stderr,
            "shell": False,
        }
        if args.detach:
            popen_kwargs["start_new_session"] = True

        proc = subprocess.Popen(cmd, **popen_kwargs)
        entry = {
            "env": env_id,
            "pid": proc.pid,
            "argv": cmd,
            "output_dir": str(run_dir),
            "gpu_logical": logical,
            "gpu_physical": physical,
            "cuda_visible_devices": child_cvd,
            "gpu_uuid": gpu_uuid(physical),
            "device": device,
            "dataset_path": str(dataset_path),
            "started_at": time.time(),
        }
        children.append(entry)
        print(json.dumps(entry), flush=True)

    manifest = {
        "profile": args.profile,
        "suite_dir": str(suite_dir),
        "validation_gate": str(gate_path),
        "code_hash": gate.get("code_hash"),
        "children": children,
        "python": sys.executable,
        "detach": bool(args.detach),
        "no_silent_restart": True,
    }
    with open(suite_dir / "suite_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote {suite_dir / 'suite_manifest.json'}", flush=True)

    if not args.detach:
        # Foreground: wait for all
        # Re-open wait via pid poll
        failures = []
        for entry in children:
            pid = entry["pid"]
            while True:
                try:
                    os.waitpid(pid, 0)
                    break
                except ChildProcessError:
                    break
                except Exception:
                    time.sleep(1)
                    if not Path(f"/proc/{pid}").exists():
                        break
        print("All children exited", flush=True)


if __name__ == "__main__":
    main()
