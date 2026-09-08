"""Utilities for CORL IQL adaptive inverse-temperature (beta) training."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import h5py
import numpy as np
import torch
import torch.nn as nn
import yaml
try:
    from torch.func import functional_call
except ImportError:  # pragma: no cover
    from torch.nn.utils.stateless import functional_call

from algorithms.offline.iql import (
    EXP_ADV_MAX,
    DeterministicPolicy,
    GaussianPolicy,
    ReplayBuffer,
    compute_mean_std,
    modify_reward,
    normalize_states,
)

TensorBatch = List[torch.Tensor]


# ---------------------------------------------------------------------------
# Safe exponential weights
# ---------------------------------------------------------------------------


def safe_exp_weights(
    beta: torch.Tensor,
    adv: torch.Tensor,
    cap: float = EXP_ADV_MAX,
) -> torch.Tensor:
    """w(beta, A) = min(exp(beta * A), cap) with overflow-safe forward.

    Caps the exponent at log(cap) before exp. In the non-clipped regime this
    matches ``torch.exp(beta * adv).clamp(max=cap)`` in value and gradient.
    Clipped samples have zero gradient w.r.t. beta / adv through the weight.
    """
    log_cap = math.log(cap)
    x = beta * adv
    # Cap exponent first (overflow-safe), then clamp to guarantee <= cap
    # even when exp(log(cap)) is slightly above due to float rounding.
    return torch.clamp(torch.exp(torch.clamp(x, max=log_cap)), max=cap)


# ---------------------------------------------------------------------------
# Functional Adam (matches torch.optim.Adam; eps outside sqrt)
# ---------------------------------------------------------------------------


class _SafeSqrt(torch.autograd.Function):
    """Forward exact sqrt; backward avoids 0/0 NaNs for higher-order grads."""

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        ctx.save_for_backward(x)
        return torch.sqrt(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # type: ignore[override]
        (x,) = ctx.saved_tensors
        denom = torch.sqrt(x.clamp(min=1e-12))
        return grad_output * (0.5 / denom)


def safe_sqrt(x: torch.Tensor) -> torch.Tensor:
    return _SafeSqrt.apply(x)


def functional_adam_step(
    named_params: Dict[str, torch.Tensor],
    named_grads: Dict[str, Optional[torch.Tensor]],
    adam_state: Dict[str, Dict[str, Any]],
    *,
    lr: float,
    betas: Tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
    weight_decay: float = 0.0,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Dict[str, Any]]]:
    """One Adam update matching ``torch.optim.Adam`` (decoupled WD not used).

    Returns ``(theta_plus, new_state)``. ``new_state`` is detached/copied for
    inspection; ``theta_plus`` keeps the autograd graph through ``grads``.
    """
    b1, b2 = betas
    theta_plus: Dict[str, torch.Tensor] = {}
    new_state: Dict[str, Dict[str, Any]] = {}

    for name, p in named_params.items():
        g = named_grads.get(name, None)
        st = adam_state.get(name)

        if g is None:
            # Match PyTorch: parameters with grad=None are left unchanged and
            # their optimizer state is not advanced.
            theta_plus[name] = p
            if st is not None:
                new_state[name] = {
                    "step": int(st["step"]),
                    "exp_avg": st["exp_avg"].clone(),
                    "exp_avg_sq": st["exp_avg_sq"].clone(),
                }
            continue

        if weight_decay != 0.0:
            g = g + weight_decay * p

        if st is None:
            step = 0
            exp_avg = torch.zeros_like(p)
            exp_avg_sq = torch.zeros_like(p)
        else:
            step = int(st["step"])
            exp_avg = st["exp_avg"]
            exp_avg_sq = st["exp_avg_sq"]

        step_plus = step + 1
        # Keep graph on moment updates that depend on g.
        exp_avg_p = exp_avg * b1 + (1.0 - b1) * g
        exp_avg_sq_p = exp_avg_sq * b2 + (1.0 - b2) * (g * g)

        bias_c1 = 1.0 - b1**step_plus
        bias_c2 = 1.0 - b2**step_plus
        m_hat = exp_avg_p / bias_c1
        v_hat = exp_avg_sq_p / bias_c2
        denom = safe_sqrt(v_hat) + eps
        theta_plus[name] = p - lr * m_hat / denom

        new_state[name] = {
            "step": step_plus,
            "exp_avg": exp_avg_p.detach().clone(),
            "exp_avg_sq": exp_avg_sq_p.detach().clone(),
            "v_hat": v_hat.detach().clone(),
            "m_hat": m_hat.detach().clone(),
        }

    return theta_plus, new_state


def adam_state_from_optimizer(
    optimizer: torch.optim.Optimizer,
    named_params: Dict[str, nn.Parameter],
) -> Dict[str, Dict[str, Any]]:
    """Extract per-parameter Adam state keyed by parameter name."""
    id_to_name = {id(p): n for n, p in named_params.items()}
    out: Dict[str, Dict[str, Any]] = {}
    for p in optimizer.param_groups[0]["params"]:
        name = id_to_name[id(p)]
        if p not in optimizer.state or len(optimizer.state[p]) == 0:
            continue
        st = optimizer.state[p]
        out[name] = {
            "step": int(st["step"]),
            "exp_avg": st["exp_avg"].detach().clone(),
            "exp_avg_sq": st["exp_avg_sq"].detach().clone(),
        }
    return out


def named_parameters_dict(module: nn.Module) -> Dict[str, nn.Parameter]:
    return dict(module.named_parameters())


def tensors_from_params(params: Dict[str, nn.Parameter]) -> Dict[str, torch.Tensor]:
    return {k: v for k, v in params.items()}


# ---------------------------------------------------------------------------
# Actor losses
# ---------------------------------------------------------------------------


def actor_ell(actor: nn.Module, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    """Per-sample imitation loss ell; shape [B]."""
    policy_out = actor(states)
    if isinstance(policy_out, torch.distributions.Distribution):
        ell = -policy_out.log_prob(actions).sum(-1)
    elif torch.is_tensor(policy_out):
        if policy_out.shape != actions.shape:
            raise RuntimeError(
                f"Actions shape mismatch: policy {tuple(policy_out.shape)} "
                f"vs actions {tuple(actions.shape)}"
            )
        ell = torch.sum((policy_out - actions) ** 2, dim=1)
    else:
        raise NotImplementedError(type(policy_out))
    assert_vec1d(ell, "ell")
    return ell


def actor_ell_functional(
    actor: nn.Module,
    params: Dict[str, torch.Tensor],
    states: torch.Tensor,
    actions: torch.Tensor,
) -> torch.Tensor:
    """ell(theta; s, a) via functional_call (supports Parameter log_std)."""
    policy_out = functional_call(actor, params, (states,))
    if isinstance(policy_out, torch.distributions.Distribution):
        ell = -policy_out.log_prob(actions).sum(-1)
    elif torch.is_tensor(policy_out):
        if policy_out.shape != actions.shape:
            raise RuntimeError(
                f"Actions shape mismatch: policy {tuple(policy_out.shape)} "
                f"vs actions {tuple(actions.shape)}"
            )
        ell = torch.sum((policy_out - actions) ** 2, dim=1)
    else:
        raise NotImplementedError(type(policy_out))
    assert_vec1d(ell, "ell_functional")
    return ell


def assert_vec1d(t: torch.Tensor, name: str) -> None:
    if t.ndim != 1:
        raise AssertionError(f"{name} must be rank-1 [B], got shape {tuple(t.shape)}")


def squeeze_to_1d(t: torch.Tensor, name: str) -> torch.Tensor:
    if t.ndim == 2 and t.shape[-1] == 1:
        t = t.squeeze(-1)
    assert_vec1d(t, name)
    return t


def is_meta_step(step: int, meta_warmup_steps: int, meta_interval: int) -> bool:
    return step > meta_warmup_steps and (step - meta_warmup_steps) % meta_interval == 0


# ---------------------------------------------------------------------------
# CPU replay buffer (dataset-sized; sample then .to(device))
# ---------------------------------------------------------------------------


class CPUReplayBuffer(ReplayBuffer):
    """ReplayBuffer forced onto CPU and sized to the dataset."""

    def __init__(self, state_dim: int, action_dim: int, buffer_size: int):
        super().__init__(state_dim, action_dim, buffer_size, device="cpu")

    def sample_indices(self, batch_size: int, rng: Optional[np.random.RandomState] = None) -> np.ndarray:
        high = min(self._size, self._pointer)
        if rng is None:
            return np.random.randint(0, high, size=batch_size)
        return rng.randint(0, high, size=batch_size)

    def sample_with_indices(self, indices: np.ndarray) -> TensorBatch:
        states = self._states[indices]
        actions = self._actions[indices]
        rewards = self._rewards[indices]
        next_states = self._next_states[indices]
        dones = self._dones[indices]
        return [states, actions, rewards, next_states, dones]

    def sample(self, batch_size: int) -> TensorBatch:
        # Preserve original np.random.randint flow for inner sampling.
        return super().sample(batch_size)

    def sample_outer(self, batch_size: int, rng: np.random.RandomState) -> TensorBatch:
        indices = self.sample_indices(batch_size, rng=rng)
        return self.sample_with_indices(indices)


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------


def sha256_file(path: Union[str, Path], chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def load_raw_hdf5(path: Union[str, Path]) -> Dict[str, np.ndarray]:
    path = Path(path)
    required = ("observations", "actions", "rewards", "terminals", "timeouts")
    with h5py.File(path, "r") as f:
        missing = [k for k in required if k not in f]
        if missing:
            raise KeyError(f"HDF5 {path} missing keys: {missing}")
        raw = {k: np.array(f[k]) for k in required}
    return raw


def load_qlearning_dataset_from_hdf5(env, dataset_path: Union[str, Path]) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """Load local HDF5 and convert via d4rl.qlearning_dataset(env, dataset=raw)."""
    import d4rl  # noqa: F401  # env registration side effects

    dataset_path = Path(dataset_path).resolve()
    if not dataset_path.is_file():
        raise FileNotFoundError(f"dataset-path not found: {dataset_path}")

    raw = load_raw_hdf5(dataset_path)
    file_sha = sha256_file(dataset_path)
    converted = d4rl.qlearning_dataset(env, dataset=raw)

    manifest: Dict[str, Any] = {
        "dataset_path": str(dataset_path),
        "raw_hdf5_sha256": file_sha,
        "n_transitions": int(converted["observations"].shape[0]),
        "observation_dim": int(converted["observations"].shape[1]),
        "action_dim": int(converted["actions"].shape[1]),
        "action_min": float(np.min(converted["actions"])),
        "action_max": float(np.max(converted["actions"])),
        "terminal_timeout_handling": (
            "d4rl.qlearning_dataset with raw timeouts field; "
            "timeout transitions discarded (terminate_on_end=False)"
        ),
        "raw_keys": sorted(raw.keys()),
        "raw_n_rows": int(raw["rewards"].shape[0]),
    }
    return converted, manifest


def apply_normalization_and_reward(
    dataset: Dict[str, np.ndarray],
    env_name: str,
    *,
    normalize: bool,
    normalize_reward: bool,
    max_episode_steps: int = 1000,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    reward_info: Dict[str, Any] = {
        "normalize_reward": bool(normalize_reward),
        "transform": "none",
        "coefficients": {},
    }
    if normalize_reward:
        if any(s in env_name for s in ("halfcheetah", "hopper", "walker2d")):
            from algorithms.offline.iql import return_reward_range

            min_ret, max_ret = return_reward_range(dataset, max_episode_steps)
            denom = max_ret - min_ret
            dataset["rewards"] = dataset["rewards"] / denom * max_episode_steps
            reward_info["transform"] = "locomotion_return_range_scale"
            reward_info["coefficients"] = {
                "min_return": float(min_ret),
                "max_return": float(max_ret),
                "scale_to": float(max_episode_steps),
            }
        elif "antmaze" in env_name:
            dataset["rewards"] = dataset["rewards"] - 1.0
            reward_info["transform"] = "antmaze_minus_one"
            reward_info["coefficients"] = {"subtract": 1.0}
        else:
            modify_reward(dataset, env_name, max_episode_steps=max_episode_steps)
            reward_info["transform"] = "modify_reward_fallback"

    if normalize:
        state_mean, state_std = compute_mean_std(dataset["observations"], eps=1e-3)
    else:
        state_mean, state_std = 0.0, 1.0

    dataset["observations"] = normalize_states(dataset["observations"], state_mean, state_std)
    dataset["next_observations"] = normalize_states(
        dataset["next_observations"], state_mean, state_std
    )

    norm_info = {
        "normalize": bool(normalize),
        "state_mean": np.asarray(state_mean, dtype=np.float64).tolist()
        if normalize
        else 0.0,
        "state_std": np.asarray(state_std, dtype=np.float64).tolist() if normalize else 1.0,
        "reward": reward_info,
    }
    return dataset, norm_info


# ---------------------------------------------------------------------------
# RNG snapshot / restore
# ---------------------------------------------------------------------------


def get_rng_state(outer_rng: Optional[np.random.RandomState] = None) -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "outer_rng": outer_rng.get_state() if outer_rng is not None else None,
    }
    return state


def set_rng_state(state: Dict[str, Any], outer_rng: Optional[np.random.RandomState] = None) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if state.get("torch_cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])
    if outer_rng is not None and state.get("outer_rng") is not None:
        outer_rng.set_state(state["outer_rng"])


# ---------------------------------------------------------------------------
# Meta / ESS diagnostics
# ---------------------------------------------------------------------------


def ess_stats(w: torch.Tensor) -> Dict[str, float]:
    w = w.detach().float().reshape(-1)
    s = float(w.sum().item())
    s2 = float((w * w).sum().item())
    if s2 <= 0.0 or not math.isfinite(s2):
        return {
            "ess": float("nan"),
            "ess_fraction": float("nan"),
            "ess_undefined": True,
            "weight_sum": s,
            "weight_sumsq": s2,
        }
    ess = (s * s) / s2
    return {
        "ess": float(ess),
        "ess_fraction": float(ess / max(w.numel(), 1)),
        "ess_undefined": False,
        "weight_sum": s,
        "weight_sumsq": s2,
    }


def weight_diagnostics(w: torch.Tensor, cap: float = EXP_ADV_MAX) -> Dict[str, float]:
    w = w.detach().float().reshape(-1)
    qs = torch.quantile(w, torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0], device=w.device)).cpu().numpy()
    clipped = (w >= cap - 1e-6).float().mean().item()
    near_zero = (w <= 1e-12).float().mean().item()
    out = {
        "weight_mean": float(w.mean().item()),
        "weight_max": float(w.max().item()),
        "weight_q0": float(qs[0]),
        "weight_q25": float(qs[1]),
        "weight_q50": float(qs[2]),
        "weight_q75": float(qs[3]),
        "weight_q100": float(qs[4]),
        "weight_clipped_fraction": float(clipped),
        "weight_numerical_zero_fraction": float(near_zero),
    }
    out.update(ess_stats(w))
    return out


def advantage_diagnostics(adv: torch.Tensor) -> Dict[str, float]:
    a = adv.detach().float().reshape(-1)
    qs = torch.quantile(
        a, torch.tensor([0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0], device=a.device)
    ).cpu().numpy()
    return {
        "adv_mean": float(a.mean().item()),
        "adv_std": float(a.std(unbiased=False).item()),
        "adv_q0": float(qs[0]),
        "adv_q10": float(qs[1]),
        "adv_q25": float(qs[2]),
        "adv_q50": float(qs[3]),
        "adv_q75": float(qs[4]),
        "adv_q90": float(qs[5]),
        "adv_q100": float(qs[6]),
    }


# ---------------------------------------------------------------------------
# Checkpoint / logging helpers
# ---------------------------------------------------------------------------


def code_hash(paths: Sequence[Union[str, Path]]) -> str:
    h = hashlib.sha256()
    for p in sorted(Path(x).resolve() for x in paths):
        h.update(str(p).encode())
        h.update(b"\0")
        with open(p, "rb") as f:
            h.update(f.read())
    return h.hexdigest()


def adaptive_beta_source_files(repo_root: Union[str, Path]) -> List[Path]:
    root = Path(repo_root)
    return [
        root / "algorithms/offline/iql_adaptive_beta.py",
        root / "algorithms/offline/iql_adaptive_beta_utils.py",
        root / "configs/offline/iql_adaptive_beta/meta_v1.yaml",
    ]


def dump_yaml(path: Union[str, Path], obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(obj, f, sort_keys=False)


def append_jsonl(path: Union[str, Path], row: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(row, default=_json_default) + "\n")
        f.flush()
        os.fsync(f.fileno())


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    if torch.is_tensor(o):
        return o.detach().cpu().tolist()
    raise TypeError(f"Object of type {type(o)} is not JSON serializable")


def refuse_output_dir(output_dir: Path, *, resume: bool) -> None:
    output_dir = Path(output_dir)
    if not output_dir.exists():
        return
    completion = output_dir / "completion.json"
    if completion.exists() and not resume:
        raise RuntimeError(
            f"Refusing to overwrite completed output dir: {output_dir} "
            f"(found completion.json). Use a new --output-dir or --resume."
        )
    live_marker = output_dir / "runtime_manifest.json"
    metrics = output_dir / "metrics.jsonl"
    if live_marker.exists() and metrics.exists() and not resume:
        # Treat existing live/partial runs as protected.
        raise RuntimeError(
            f"Refusing to overwrite existing live/partial output dir: {output_dir}. "
            f"Use a new --output-dir or --resume with a compatible checkpoint."
        )


def optimizer_state_to_cpu(state: Dict[str, Any]) -> Dict[str, Any]:
    return torch_nested_to(state, "cpu")


def torch_nested_to(obj: Any, device: Union[str, torch.device]) -> Any:
    if torch.is_tensor(obj):
        return obj.detach().to(device)
    if isinstance(obj, dict):
        return {k: torch_nested_to(v, device) for k, v in obj.items()}
    if isinstance(obj, list):
        return [torch_nested_to(v, device) for v in obj]
    if isinstance(obj, tuple):
        return tuple(torch_nested_to(v, device) for v in obj)
    return obj


@dataclass
class MetaConfig:
    adaptive_enabled: bool = True
    beta_initial: float = 3.0
    beta_fixed: float = 3.0
    beta_min: float = 0.05
    beta_max: float = 100.0
    rho_parameterization: str = "log_beta"
    rho_dtype: str = "float64"
    rho_lr: float = 1e-4
    rho_adam_betas: List[float] = field(default_factory=lambda: [0.0, 0.999])
    rho_adam_eps: float = 1e-8
    rho_weight_decay: float = 0.0
    meta_warmup_steps: int = 100000
    meta_interval: int = 20
    outer_batch_size: int = 256
    outer_reference: str = "current_beta_stopgrad"
    weight_cap: float = 100.0

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "MetaConfig":
        with open(path) as f:
            raw = yaml.safe_load(f)
        return cls(**{k: raw[k] for k in cls.__dataclass_fields__ if k in raw})

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def project_rho(rho: torch.Tensor, beta_min: float, beta_max: float) -> Tuple[torch.Tensor, bool]:
    lo = math.log(beta_min)
    hi = math.log(beta_max)
    projected = False
    with torch.no_grad():
        if float(rho.item()) < lo:
            rho.fill_(lo)
            projected = True
        elif float(rho.item()) > hi:
            rho.fill_(hi)
            projected = True
    return rho, projected


def beta_from_rho(rho: torch.Tensor) -> torch.Tensor:
    """Preserve graph: do not use .item() / float() / new tensor from Python float."""
    return torch.exp(rho)


def make_actor(
    *,
    deterministic: bool,
    state_dim: int,
    action_dim: int,
    max_action: float,
    actor_dropout: Optional[float],
    device: str,
) -> nn.Module:
    if deterministic:
        actor: nn.Module = DeterministicPolicy(
            state_dim, action_dim, max_action, dropout=actor_dropout
        )
    else:
        actor = GaussianPolicy(
            state_dim, action_dim, max_action, dropout=actor_dropout
        )
    return actor.to(device)


def gaussian_log_std_stats(actor: nn.Module) -> Dict[str, float]:
    if not isinstance(actor, GaussianPolicy):
        return {}
    from algorithms.offline.iql import LOG_STD_MAX, LOG_STD_MIN

    ls = actor.log_std.detach()
    clamped = ls.clamp(LOG_STD_MIN, LOG_STD_MAX)
    hit_lo = (ls <= LOG_STD_MIN).float().mean().item()
    hit_hi = (ls >= LOG_STD_MAX).float().mean().item()
    return {
        "log_std_mean": float(ls.mean().item()),
        "log_std_min": float(ls.min().item()),
        "log_std_max": float(ls.max().item()),
        "log_std_clamp_lo_fraction": float(hit_lo),
        "log_std_clamp_hi_fraction": float(hit_hi),
        "log_std_clamped_mean": float(clamped.mean().item()),
    }


def unwrap_env(env):
    e = env
    while hasattr(e, "env"):
        e = e.env
    return e


def snapshot_env(env) -> Dict[str, Any]:
    """Best-effort env snapshot for paired evaluation (loco + antmaze)."""
    base = unwrap_env(env)
    snap: Dict[str, Any] = {"obs_space_seed": None}
    if hasattr(base, "sim") and hasattr(base.sim, "get_state"):
        snap["qpos"] = base.sim.data.qpos.copy()
        snap["qvel"] = base.sim.data.qvel.copy()
    if hasattr(base, "target_goal"):
        snap["target_goal"] = np.array(base.target_goal, copy=True)
    if hasattr(base, "_elapsed_steps"):
        snap["_elapsed_steps"] = int(base._elapsed_steps)
    # Gym TimeLimit wrapper
    e = env
    while e is not None:
        if hasattr(e, "_elapsed_steps"):
            snap["wrapper_elapsed_steps"] = int(e._elapsed_steps)
            snap["wrapper_id"] = id(e)
            break
        e = getattr(e, "env", None)
    return snap


def restore_env(env, snap: Dict[str, Any]) -> None:
    base = unwrap_env(env)
    if "qpos" in snap and hasattr(base, "set_state"):
        base.set_state(snap["qpos"], snap["qvel"])
    elif "qpos" in snap and hasattr(base, "sim"):
        base.sim.data.qpos[:] = snap["qpos"]
        base.sim.data.qvel[:] = snap["qvel"]
        base.sim.forward()
    if "target_goal" in snap and hasattr(base, "target_goal"):
        base.target_goal = np.array(snap["target_goal"], copy=True)
        if hasattr(base, "set_target"):
            try:
                base.set_target(base.target_goal)
            except Exception:
                pass
    if "wrapper_elapsed_steps" in snap:
        e = env
        while e is not None:
            if hasattr(e, "_elapsed_steps"):
                e._elapsed_steps = snap["wrapper_elapsed_steps"]
                break
            e = getattr(e, "env", None)
    if "_elapsed_steps" in snap and hasattr(base, "_elapsed_steps"):
        base._elapsed_steps = snap["_elapsed_steps"]


def current_observation(env) -> np.ndarray:
    base = unwrap_env(env)
    if hasattr(base, "_get_obs"):
        return np.asarray(base._get_obs(), dtype=np.float32)
    if hasattr(base, "state_vector"):
        return np.asarray(base.state_vector(), dtype=np.float32)
    raise RuntimeError("Cannot read current observation from env")
