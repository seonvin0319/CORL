#!/usr/bin/env python3
"""Validation gate for corl_iql_adaptive_beta_v1."""

from __future__ import annotations

import argparse
import hashlib
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
    is_meta_step,
)

ENV_MAP = {
    "halfcheetah-medium-v2": {
        "yaml": "configs/offline/iql/halfcheetah/medium_v2.yaml",
        "hdf5": "halfcheetah_medium-v2.hdf5",
        "beta": 3.0,
        "iql_tau": 0.7,
        "tau": 0.005,
        "deterministic": False,
        "normalize_reward": False,
    },
    "halfcheetah-medium-expert-v2": {
        "yaml": "configs/offline/iql/halfcheetah/medium_expert_v2.yaml",
        "hdf5": "halfcheetah_medium_expert-v2.hdf5",
        "beta": 3.0,
        "iql_tau": 0.7,
        "tau": 0.005,
        "deterministic": False,
        "normalize_reward": False,
    },
    "halfcheetah-medium-replay-v2": {
        "yaml": "configs/offline/iql/halfcheetah/medium_replay_v2.yaml",
        "hdf5": "halfcheetah_medium_replay-v2.hdf5",
        "beta": 3.0,
        "iql_tau": 0.7,
        "tau": 0.005,
        "deterministic": False,
        "normalize_reward": False,
    },
    "hopper-medium-v2": {
        "yaml": "configs/offline/iql/hopper/medium_v2.yaml",
        "hdf5": "hopper_medium-v2.hdf5",
        "beta": 3.0,
        "iql_tau": 0.7,
        "tau": 0.001,
        "deterministic": True,
        "normalize_reward": True,
    },
    "hopper-medium-expert-v2": {
        "yaml": "configs/offline/iql/hopper/medium_expert_v2.yaml",
        "hdf5": "hopper_medium_expert-v2.hdf5",
        "beta": 6.0,
        "iql_tau": 0.5,
        "tau": 0.005,
        "deterministic": False,
        "normalize_reward": False,
    },
    "hopper-medium-replay-v2": {
        "yaml": "configs/offline/iql/hopper/medium_replay_v2.yaml",
        "hdf5": "hopper_medium_replay-v2.hdf5",
        "beta": 3.0,
        "iql_tau": 0.7,
        "tau": 0.001,
        "deterministic": True,
        "normalize_reward": True,
    },
    "walker2d-medium-v2": {
        "yaml": "configs/offline/iql/walker2d/medium_v2.yaml",
        "hdf5": "walker2d_medium-v2.hdf5",
        "beta": 3.0,
        "iql_tau": 0.7,
        "tau": 0.005,
        "deterministic": False,
        "normalize_reward": False,
    },
    "walker2d-medium-expert-v2": {
        "yaml": "configs/offline/iql/walker2d/medium_expert_v2.yaml",
        "hdf5": "walker2d_medium_expert-v2.hdf5",
        "beta": 3.0,
        "iql_tau": 0.7,
        "tau": 0.005,
        "deterministic": False,
        "normalize_reward": False,
    },
    "walker2d-medium-replay-v2": {
        "yaml": "configs/offline/iql/walker2d/medium_replay_v2.yaml",
        "hdf5": "walker2d_medium_replay-v2.hdf5",
        "beta": 3.0,
        "iql_tau": 0.7,
        "tau": 0.005,
        "deterministic": False,
        "normalize_reward": False,
    },
    # D4RL stores antmaze under Ant_maze_* filenames (not antmaze-*.hdf5).
    "antmaze-umaze-diverse-v2": {
        "yaml": "configs/offline/iql/antmaze/umaze_diverse_v2.yaml",
        "hdf5": "Ant_maze_u-maze_noisy_multistart_True_multigoal_True_sparse_fixed.hdf5",
        "beta": 10.0,
        "iql_tau": 0.9,
        "tau": 0.005,
        "deterministic": False,
        "normalize_reward": True,
    },
    "antmaze-medium-play-v2": {
        "yaml": "configs/offline/iql/antmaze/medium_play_v2.yaml",
        "hdf5": "Ant_maze_big-maze_noisy_multistart_True_multigoal_False_sparse_fixed.hdf5",
        "beta": 10.0,
        "iql_tau": 0.9,
        "tau": 0.005,
        "deterministic": False,
        "normalize_reward": True,
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


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def preflight_env(env_id: str, dataset_dir: Path) -> Dict[str, Any]:
    info = ENV_MAP[env_id]
    yaml_path = REPO_ROOT / info["yaml"]
    hdf5_path = dataset_dir / info["hdf5"]
    if not yaml_path.is_file():
        raise FileNotFoundError(yaml_path)
    if not hdf5_path.is_file():
        raise FileNotFoundError(hdf5_path)
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)
    checks = {
        "env_match": cfg["env"] == env_id,
        "beta": float(cfg["beta"]) == float(info["beta"]),
        "iql_tau": float(cfg["iql_tau"]) == float(info["iql_tau"]),
        "tau": float(cfg["tau"]) == float(info["tau"]),
        "iql_deterministic": bool(cfg["iql_deterministic"]) == bool(info["deterministic"]),
        "normalize_reward": bool(cfg["normalize_reward"]) == bool(info["normalize_reward"]),
        "normalize": bool(cfg.get("normalize", True)) is True,
    }
    if not all(checks.values()):
        raise AssertionError(f"preflight failed for {env_id}: {checks}")
    return {
        "env": env_id,
        "yaml": str(yaml_path),
        "dataset_path": str(hdf5_path.resolve()),
        "dataset_sha256": sha256_file(hdf5_path),
        "checks": checks,
        "expected": info,
    }


def run_pytest() -> Dict[str, Any]:
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        str(REPO_ROOT / "tests"),
        "--ignore-glob=*not_adaptive*",
    ]
    # Restrict to adaptive beta tests
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        str(REPO_ROOT / "tests/test_iql_adaptive_beta_parity.py"),
        str(REPO_ROOT / "tests/test_iql_adaptive_beta_adam.py"),
        str(REPO_ROOT / "tests/test_iql_adaptive_beta_gradients.py"),
        str(REPO_ROOT / "tests/test_iql_adaptive_beta_schedule.py"),
        str(REPO_ROOT / "tests/test_iql_adaptive_beta_resume.py"),
    ]
    env = os.environ.copy()
    env.setdefault("D4RL_SUPPRESS_IMPORT_ERROR", "1")
    env.setdefault("WANDB_MODE", "disabled")
    proc = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
    )
    return {
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "passed": proc.returncode == 0,
        "cmd": cmd,
    }


def run_smoke(
    env_id: str,
    *,
    meta_config: Path,
    device: str,
    dataset_dir: Path,
    output_dir: Path,
) -> Dict[str, Any]:
    info = ENV_MAP[env_id]
    smoke_dir = output_dir / f"smoke_{env_id}"
    if smoke_dir.exists():
        # Use unique subdir to avoid refuse_output_dir
        smoke_dir = output_dir / f"smoke_{env_id}_{int(time.time())}"
    cmd = [
        sys.executable,
        "-m",
        "algorithms.offline.iql_adaptive_beta",
        "--config",
        str(REPO_ROOT / info["yaml"]),
        "--meta-config",
        str(meta_config),
        "--env",
        env_id,
        "--seed",
        "0",
        "--device",
        device,
        "--dataset-path",
        str((dataset_dir / info["hdf5"]).resolve()),
        "--output-dir",
        str(smoke_dir),
        "--mode",
        "smoke",
        "--max-timesteps",
        "240",
        "--meta-warmup-steps",
        "80",
        "--meta-interval",
        "20",
        "--eval-episodes",
        "1",
        "--eval-freq",
        "240",
        "--checkpoint-freq",
        "240",
        "--metrics-freq",
        "40",
    ]
    env = os.environ.copy()
    env.setdefault("D4RL_SUPPRESS_IMPORT_ERROR", "1")
    env.setdefault("WANDB_MODE", "disabled")
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("PYTHONUNBUFFERED", "1")
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), env=env, capture_output=True, text=True)
    result: Dict[str, Any] = {
        "env": env_id,
        "returncode": proc.returncode,
        "stdout_tail": proc.stdout[-4000:],
        "stderr_tail": proc.stderr[-4000:],
        "output_dir": str(smoke_dir),
        "cmd": cmd,
    }
    if proc.returncode != 0:
        result["passed"] = False
        return result

    meta_path = smoke_dir / "meta.jsonl"
    steps = []
    if meta_path.exists():
        with open(meta_path) as f:
            for line in f:
                row = json.loads(line)
                steps.append(int(row["step"]))
    expected = [s for s in range(1, 241) if is_meta_step(s, 80, 20)]
    result["meta_steps"] = steps
    result["expected_meta_steps"] = expected
    result["meta_steps_ok"] = steps == expected
    result["completion_exists"] = (smoke_dir / "completion.json").exists()
    # Warmup: rho fixed at beta_initial through step 80 — check metrics
    metrics_path = smoke_dir / "metrics.jsonl"
    beta_ok = True
    if metrics_path.exists():
        with open(metrics_path) as f:
            for line in f:
                row = json.loads(line)
                if int(row["step"]) <= 80:
                    if abs(float(row["beta_used"]) - float(info["beta"])) > 1e-6:
                        beta_ok = False
    result["warmup_beta_fixed"] = beta_ok
    result["passed"] = (
        result["meta_steps_ok"] and result["completion_exists"] and beta_ok
    )
    return result


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--profile", required=True, choices=list(PROFILES))
    p.add_argument("--meta-config", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dataset-dir", required=True)
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir = Path(args.dataset_dir).resolve()
    meta_config = Path(args.meta_config)
    if not meta_config.is_absolute():
        meta_config = (REPO_ROOT / meta_config).resolve()

    envs = PROFILES[args.profile]
    report: Dict[str, Any] = {
        "status": "running",
        "profile": args.profile,
        "python": sys.executable,
        "device": args.device,
        "checks": {},
    }

    # Pytest once
    print("=== pytest ===", flush=True)
    pytest_res = run_pytest()
    report["checks"]["pytest"] = {
        "passed": pytest_res["passed"],
        "returncode": pytest_res["returncode"],
        "stdout": pytest_res["stdout"],
        "stderr": pytest_res["stderr"][-2000:],
    }
    print(pytest_res["stdout"], flush=True)
    if not pytest_res["passed"]:
        print(pytest_res["stderr"], flush=True)
        report["status"] = "failed"
        with open(output_dir / "validation_failed.json", "w") as f:
            json.dump(report, f, indent=2)
        sys.exit(1)

    # Preflight + smoke per env
    preflights = []
    smokes = []
    for env_id in envs:
        print(f"=== preflight {env_id} ===", flush=True)
        pf = preflight_env(env_id, dataset_dir)
        preflights.append(pf)
        print(f"=== smoke {env_id} ===", flush=True)
        sm = run_smoke(
            env_id,
            meta_config=meta_config,
            device=args.device,
            dataset_dir=dataset_dir,
            output_dir=output_dir,
        )
        smokes.append(sm)
        print(f"smoke {env_id} passed={sm['passed']} meta={sm.get('meta_steps')}", flush=True)
        if not sm["passed"]:
            report["status"] = "failed"
            report["checks"]["preflight"] = preflights
            report["checks"]["smoke"] = smokes
            with open(output_dir / "validation_failed.json", "w") as f:
                json.dump(report, f, indent=2)
            sys.exit(1)

    sources = [p for p in adaptive_beta_source_files(REPO_ROOT) if p.exists()]
    chash = code_hash(sources)
    try:
        base = subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        base = ""

    with open(meta_config) as f:
        meta_raw = yaml.safe_load(f)
    prod_hash = hashlib.sha256(
        json.dumps(meta_raw, sort_keys=True).encode()
    ).hexdigest()

    gate = {
        "status": "passed",
        "profile": args.profile,
        "base_commit": base,
        "code_hash": chash,
        "production_semantic_config_hash": prod_hash,
        "python": sys.executable,
        "device": args.device,
        "smoke_override": {
            "max_steps": 240,
            "meta_warmup_steps": 80,
            "meta_interval": 20,
            "eval_episodes": 1,
        },
        "production": {
            "meta_config": str(meta_config),
            "meta": meta_raw,
            "max_timesteps": 1000000,
            "seed": 0,
        },
        "checks": {
            "pytest": report["checks"]["pytest"],
            "preflight": preflights,
            "smoke": [
                {
                    "env": s["env"],
                    "passed": s["passed"],
                    "meta_steps": s.get("meta_steps"),
                    "expected_meta_steps": s.get("expected_meta_steps"),
                    "output_dir": s.get("output_dir"),
                }
                for s in smokes
            ],
        },
        "datasets": {pf["env"]: pf["dataset_sha256"] for pf in preflights},
        "created_at": time.time(),
    }
    gate_path = output_dir / "validation_gate.json"
    with open(gate_path, "w") as f:
        json.dump(gate, f, indent=2)
    print(f"Wrote {gate_path}", flush=True)


if __name__ == "__main__":
    main()
