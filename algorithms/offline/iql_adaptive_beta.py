"""CORL IQL with fixed-beta and adaptive-beta actors (corl_iql_adaptive_beta_v1)."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.optim.lr_scheduler import CosineAnnealingLR

from algorithms.offline.iql import (
    TwinQ,
    ValueFunction,
    asymmetric_l2_loss,
    eval_actor,
    set_seed,
    soft_update,
    wrap_env,
)
from algorithms.offline.iql_adaptive_beta_utils import (
    CPUReplayBuffer,
    MetaConfig,
    adam_state_from_optimizer,
    advantage_diagnostics,
    append_jsonl,
    apply_normalization_and_reward,
    assert_vec1d,
    beta_from_rho,
    code_hash,
    current_observation,
    dump_yaml,
    functional_adam_step,
    gaussian_log_std_stats,
    get_rng_state,
    is_meta_step,
    load_qlearning_dataset_from_hdf5,
    make_actor,
    named_parameters_dict,
    project_rho,
    refuse_output_dir,
    restore_env,
    safe_exp_weights,
    set_rng_state,
    sha256_file,
    snapshot_env,
    squeeze_to_1d,
    actor_ell,
    actor_ell_functional,
    adaptive_beta_source_files,
    weight_diagnostics,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="algorithms.offline.iql_adaptive_beta",
        description="CORL IQL adaptive inverse-temperature experiment",
    )
    p.add_argument("--config", type=str, required=True, help="CORL env YAML")
    p.add_argument("--meta-config", type=str, required=True, help="meta_v1.yaml")
    p.add_argument("--env", type=str, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--dataset-path", type=str, required=True)
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("--resume", type=str, default="", help="checkpoint path to resume")
    p.add_argument("--mode", type=str, default="train", choices=["train", "smoke", "eval"])
    p.add_argument("--checkpoint", type=str, default="", help="checkpoint for eval mode")
    # Smoke / override knobs (validator uses these; unknown flags still error)
    p.add_argument("--max-timesteps", type=int, default=None)
    p.add_argument("--meta-warmup-steps", type=int, default=None)
    p.add_argument("--meta-interval", type=int, default=None)
    p.add_argument("--eval-episodes", type=int, default=None)
    p.add_argument("--eval-freq", type=int, default=None)
    p.add_argument("--checkpoint-freq", type=int, default=None)
    p.add_argument("--metrics-freq", type=int, default=None)
    p.add_argument("--adaptive-enabled", type=str, default=None)
    args = p.parse_args(argv)
    return args


def load_corl_yaml(path: str) -> Dict[str, Any]:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return cfg


def validate_env_config(env: str, corl: Dict[str, Any]) -> None:
    if corl.get("env") != env:
        raise ValueError(
            f"--env {env} does not match config env {corl.get('env')} in YAML"
        )


class IQLAdaptiveBetaTrainer:
    def __init__(
        self,
        *,
        corl: Dict[str, Any],
        meta: MetaConfig,
        device: str,
        seed: int,
        max_action: float,
        state_dim: int,
        action_dim: int,
    ):
        self.device = device
        self.seed = seed
        self.corl = corl
        self.meta = meta
        self.discount = float(corl["discount"])
        self.tau = float(corl["tau"])
        self.iql_tau = float(corl["iql_tau"])
        self.batch_size = int(corl["batch_size"])
        self.max_timesteps = int(corl["max_timesteps"])
        self.iql_deterministic = bool(corl["iql_deterministic"])
        actor_dropout = corl.get("actor_dropout", None)

        self.qf = TwinQ(state_dim, action_dim).to(device)
        self.q_target = copy.deepcopy(self.qf).requires_grad_(False).to(device)
        self.vf = ValueFunction(state_dim).to(device)

        # Preserve CORL init order: Q, V, then actor; second actor = deepcopy.
        actor0 = make_actor(
            deterministic=self.iql_deterministic,
            state_dim=state_dim,
            action_dim=action_dim,
            max_action=max_action,
            actor_dropout=actor_dropout,
            device=device,
        )
        self.actor_fixed = actor0
        self.actor_adaptive = copy.deepcopy(actor0)

        lr_v = float(corl["vf_lr"])
        lr_q = float(corl["qf_lr"])
        lr_a = float(corl["actor_lr"])
        self.v_optimizer = torch.optim.Adam(self.vf.parameters(), lr=lr_v, betas=(0.9, 0.999), eps=1e-8)
        self.q_optimizer = torch.optim.Adam(self.qf.parameters(), lr=lr_q, betas=(0.9, 0.999), eps=1e-8)
        self.actor_fixed_optimizer = torch.optim.Adam(
            self.actor_fixed.parameters(), lr=lr_a, betas=(0.9, 0.999), eps=1e-8
        )
        self.actor_adaptive_optimizer = torch.optim.Adam(
            self.actor_adaptive.parameters(), lr=lr_a, betas=(0.9, 0.999), eps=1e-8
        )
        self.actor_fixed_lr_schedule = CosineAnnealingLR(
            self.actor_fixed_optimizer, T_max=self.max_timesteps
        )
        self.actor_adaptive_lr_schedule = CosineAnnealingLR(
            self.actor_adaptive_optimizer, T_max=self.max_timesteps
        )

        beta0 = float(meta.beta_initial)
        self.beta_fixed = float(meta.beta_fixed)
        self.weight_cap = float(meta.weight_cap)
        dtype = torch.float64 if meta.rho_dtype == "float64" else torch.float32
        self.rho = torch.nn.Parameter(
            torch.tensor(math.log(beta0), dtype=dtype, device=device)
        )
        betas_rho = tuple(meta.rho_adam_betas)
        self.rho_optimizer = torch.optim.Adam(
            [self.rho],
            lr=float(meta.rho_lr),
            betas=betas_rho,
            eps=float(meta.rho_adam_eps),
            weight_decay=float(meta.rho_weight_decay),
        )

        self.total_it = 0
        self.completed_meta_events = 0
        self.outer_rng = np.random.RandomState(seed + 100003)
        self.max_action = max_action

    # ---- beta helpers ----
    def current_beta_tensor(self) -> torch.Tensor:
        if not self.meta.adaptive_enabled:
            # Keep as tensor for API uniformity; no graph needed.
            return torch.tensor(
                self.beta_fixed, dtype=torch.float32, device=self.device
            )
        # Cast for model math while preserving graph from float64 rho.
        return beta_from_rho(self.rho).to(dtype=torch.float32)

    def current_beta_value(self) -> float:
        with torch.no_grad():
            return float(self.current_beta_tensor().detach().float().cpu().item())

    # ---- one training step ----
    def train_step(
        self,
        batch_b: List[torch.Tensor],
        batch_c: Optional[List[torch.Tensor]],
        *,
        do_meta: bool,
    ) -> Tuple[Dict[str, float], Optional[Dict[str, Any]]]:
        self.total_it += 1
        step = self.total_it

        states, actions, rewards, next_states, dones = batch_b
        rewards_b = squeeze_to_1d(rewards, "rewards_B")
        dones_b = squeeze_to_1d(dones, "dones_B")

        # Cache advantages BEFORE Q/V updates.
        with torch.no_grad():
            next_v_b = self.vf(next_states)
            next_v_b = squeeze_to_1d(next_v_b, "next_v_B")
            target_q_b = self.q_target(states, actions)
            target_q_b = squeeze_to_1d(target_q_b, "target_q_B")
            v_b = self.vf(states)
            v_b = squeeze_to_1d(v_b, "v_B")
            adv_b = target_q_b - v_b
            assert_vec1d(adv_b, "adv_B")

            adv_c = None
            if do_meta:
                assert batch_c is not None
                s_c, a_c, _, _, _ = batch_c
                target_q_c = squeeze_to_1d(self.q_target(s_c, a_c), "target_q_C")
                v_c = squeeze_to_1d(self.vf(s_c), "v_C")
                adv_c = target_q_c - v_c
                assert_vec1d(adv_c, "adv_C")

        # --- V update (once) ---
        # Recompute graph for V: adv = target_q - V(states) with target_q stopped.
        with torch.no_grad():
            target_q_for_v = self.q_target(states, actions)
            target_q_for_v = squeeze_to_1d(target_q_for_v, "target_q_for_v")
        v = squeeze_to_1d(self.vf(states), "v_online")
        adv_for_v = target_q_for_v - v
        v_loss = asymmetric_l2_loss(adv_for_v, self.iql_tau)
        self.v_optimizer.zero_grad()
        v_loss.backward()
        self.v_optimizer.step()

        # --- Q update (once) ---
        y_b = rewards_b + self.discount * (1.0 - dones_b) * next_v_b
        assert_vec1d(y_b, "y_B")
        qs = self.qf.both(states, actions)
        q1 = squeeze_to_1d(qs[0], "q1")
        q2 = squeeze_to_1d(qs[1], "q2")
        # Match original: sum(mse)/len(qs) == 0.5*(mse1+mse2)
        q_loss = sum(F.mse_loss(q, y_b) for q in (q1, q2)) / 2
        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()
        soft_update(self.q_target, self.qf, self.tau)

        adv_detached = adv_b.detach()

        # --- Fixed actor (always normal Adam) ---
        beta_fixed_t = torch.tensor(
            self.beta_fixed, dtype=torch.float32, device=self.device
        )
        w_fixed = safe_exp_weights(beta_fixed_t, adv_detached, cap=self.weight_cap)
        ell_fixed = actor_ell(self.actor_fixed, states, actions)
        loss_fixed = torch.mean(w_fixed * ell_fixed)
        self.actor_fixed_optimizer.zero_grad()
        loss_fixed.backward()
        self.actor_fixed_optimizer.step()
        self.actor_fixed_lr_schedule.step()

        # --- Adaptive actor ---
        meta_row: Optional[Dict[str, Any]] = None
        beta_used_t = self.current_beta_tensor()
        beta_used_val = float(beta_used_t.detach().float().cpu().item())

        if do_meta and self.meta.adaptive_enabled:
            meta_row = self._meta_adaptive_update(
                states=states,
                actions=actions,
                adv_b=adv_detached,
                batch_c=batch_c,
                adv_c=adv_c,
                beta_used_t=beta_used_t,
            )
            self.completed_meta_events += 1
        else:
            # Normal CORL Adam update with beta_used (no graph through rho).
            w_ad = safe_exp_weights(
                beta_used_t.detach(), adv_detached, cap=self.weight_cap
            )
            ell_ad = actor_ell(self.actor_adaptive, states, actions)
            loss_ad = torch.mean(w_ad * ell_ad)
            self.actor_adaptive_optimizer.zero_grad()
            loss_ad.backward()
            self.actor_adaptive_optimizer.step()
            self.actor_adaptive_lr_schedule.step()

        beta_next_val = self.current_beta_value()
        actor_lr = float(self.actor_adaptive_optimizer.param_groups[0]["lr"])

        with torch.no_grad():
            q_data = self.qf(states, actions)
            q_data = squeeze_to_1d(q_data, "q_data")
            v_data = squeeze_to_1d(self.vf(states), "v_data")

        metrics: Dict[str, float] = {
            "step": float(step),
            "q_loss": float(q_loss.item()),
            "value_loss": float(v_loss.item()),
            "actor_fixed_loss": float(loss_fixed.item()),
            "actor_adaptive_loss": float(
                meta_row["L_inner"] if meta_row is not None else loss_ad.item()
            ),
            "beta_fixed": float(self.beta_fixed),
            "beta_used": beta_used_val,
            "beta_next": beta_next_val,
            "actor_lr": actor_lr,
            "Q_data_mean": float(q_data.mean().item()),
            "Q_data_std": float(q_data.std(unbiased=False).item()),
            "V_data_mean": float(v_data.mean().item()),
            "V_data_std": float(v_data.std(unbiased=False).item()),
        }
        metrics.update({f"{k}": v for k, v in advantage_diagnostics(adv_detached).items()})
        metrics.update(gaussian_log_std_stats(self.actor_adaptive))
        return metrics, meta_row

    def _meta_adaptive_update(
        self,
        *,
        states: torch.Tensor,
        actions: torch.Tensor,
        adv_b: torch.Tensor,
        batch_c: List[torch.Tensor],
        adv_c: torch.Tensor,
        beta_used_t: torch.Tensor,
    ) -> Dict[str, Any]:
        s_c, a_c, _, _, _ = batch_c
        named = named_parameters_dict(self.actor_adaptive)
        params = {k: v for k, v in named.items()}

        # Inner loss with beta graph preserved.
        w_inner = safe_exp_weights(beta_used_t, adv_b, cap=self.weight_cap)
        ell_inner = actor_ell(self.actor_adaptive, states, actions)
        L_inner = torch.mean(w_inner * ell_inner)

        grads = torch.autograd.grad(
            L_inner,
            list(params.values()),
            create_graph=True,
            retain_graph=True,
            allow_unused=True,
        )
        named_grads = {n: g for n, g in zip(params.keys(), grads)}

        adam_state = adam_state_from_optimizer(self.actor_adaptive_optimizer, named)
        lr = float(self.actor_adaptive_optimizer.param_groups[0]["lr"])
        betas = self.actor_adaptive_optimizer.param_groups[0]["betas"]
        eps = self.actor_adaptive_optimizer.param_groups[0]["eps"]

        theta_plus, _new_st = functional_adam_step(
            params,
            named_grads,
            adam_state,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=0.0,
        )

        # Outer reference weights: stopgrad on beta and weights.
        w_ref_c = safe_exp_weights(
            beta_used_t.detach(), adv_c.detach(), cap=self.weight_cap
        ).detach()
        ell_outer = actor_ell_functional(self.actor_adaptive, theta_plus, s_c, a_c)
        L_meta = torch.mean(w_ref_c * ell_outer)

        g_rho = torch.autograd.grad(L_meta, self.rho, retain_graph=False)[0]

        # Real adaptive actor update: ONE Adam.step with detached inner grads.
        self.actor_adaptive_optimizer.zero_grad(set_to_none=True)
        for p, g in zip(params.values(), grads):
            if g is None:
                p.grad = None
            else:
                p.grad = g.detach()
        self.actor_adaptive_optimizer.step()
        self.actor_adaptive_lr_schedule.step()

        # Rho update + projection.
        rho_before = float(self.rho.detach().cpu().item())
        self.rho_optimizer.zero_grad(set_to_none=True)
        self.rho.grad = g_rho.detach()
        self.rho_optimizer.step()
        _, projected = project_rho(self.rho, self.meta.beta_min, self.meta.beta_max)
        rho_after = float(self.rho.detach().cpu().item())

        # Rho Adam diagnostics.
        rho_st = self.rho_optimizer.state.get(self.rho, {})
        exp_avg = float(rho_st["exp_avg"].detach().cpu().item()) if rho_st else 0.0
        exp_avg_sq = float(rho_st["exp_avg_sq"].detach().cpu().item()) if rho_st else 0.0
        step_rho = int(rho_st["step"]) if rho_st else 0
        b1, b2 = self.rho_optimizer.param_groups[0]["betas"]
        if step_rho > 0:
            v_hat = exp_avg_sq / (1.0 - b2**step_rho)
            sqrt_v_hat = math.sqrt(max(v_hat, 0.0))
        else:
            sqrt_v_hat = 0.0

        row: Dict[str, Any] = {
            "step": self.total_it,
            "completed_meta_events": self.completed_meta_events + 1,
            "rho_used": rho_before,
            "rho_next": rho_after,
            "beta_used": float(math.exp(rho_before)),
            "beta_next": float(math.exp(rho_after)),
            "L_inner": float(L_inner.detach().cpu().item()),
            "L_meta": float(L_meta.detach().cpu().item()),
            "g_rho": float(g_rho.detach().cpu().item()),
            "delta_rho": rho_after - rho_before,
            "rho_exp_avg": exp_avg,
            "rho_sqrt_v_hat": sqrt_v_hat,
            "rho_optimizer_step": step_rho,
            "actor_optimizer_step": int(
                next(iter(self.actor_adaptive_optimizer.state.values()))["step"]
            )
            if self.actor_adaptive_optimizer.state
            else 0,
            "actor_learning_rate": lr,
            "projection_applied": bool(projected),
        }
        row.update({f"inner_{k}": v for k, v in weight_diagnostics(w_inner, self.weight_cap).items()})
        row.update({f"outer_{k}": v for k, v in weight_diagnostics(w_ref_c, self.weight_cap).items()})
        row.update({f"inner_{k}": v for k, v in advantage_diagnostics(adv_b).items()})
        row.update({f"outer_{k}": v for k, v in advantage_diagnostics(adv_c).items()})
        return row

    # ---- checkpoint ----
    def state_dict(self) -> Dict[str, Any]:
        return {
            "qf": self.qf.state_dict(),
            "q_target": self.q_target.state_dict(),
            "vf": self.vf.state_dict(),
            "actor_fixed": self.actor_fixed.state_dict(),
            "actor_adaptive": self.actor_adaptive.state_dict(),
            "q_optimizer": self.q_optimizer.state_dict(),
            "v_optimizer": self.v_optimizer.state_dict(),
            "actor_fixed_optimizer": self.actor_fixed_optimizer.state_dict(),
            "actor_adaptive_optimizer": self.actor_adaptive_optimizer.state_dict(),
            "actor_fixed_lr_schedule": self.actor_fixed_lr_schedule.state_dict(),
            "actor_adaptive_lr_schedule": self.actor_adaptive_lr_schedule.state_dict(),
            "rho": self.rho.detach().cpu(),
            "rho_optimizer": self.rho_optimizer.state_dict(),
            "total_it": self.total_it,
            "completed_meta_events": self.completed_meta_events,
            "rng": get_rng_state(self.outer_rng),
            "meta": self.meta.to_dict(),
            "corl": self.corl,
            "format": "corl_iql_adaptive_beta_v1",
        }

    def load_state_dict(self, ckpt: Dict[str, Any], *, strict_target: bool = True) -> None:
        if ckpt.get("format") != "corl_iql_adaptive_beta_v1":
            raise ValueError(
                f"Incompatible checkpoint format: {ckpt.get('format')!r}; "
                "smoke checkpoints must not be resumed as production."
            )
        self.qf.load_state_dict(ckpt["qf"])
        if "q_target" not in ckpt:
            raise ValueError("Checkpoint missing target Q; refuse online-Q overwrite")
        self.q_target.load_state_dict(ckpt["q_target"])
        self.vf.load_state_dict(ckpt["vf"])
        self.actor_fixed.load_state_dict(ckpt["actor_fixed"])
        self.actor_adaptive.load_state_dict(ckpt["actor_adaptive"])
        self.q_optimizer.load_state_dict(ckpt["q_optimizer"])
        self.v_optimizer.load_state_dict(ckpt["v_optimizer"])
        self.actor_fixed_optimizer.load_state_dict(ckpt["actor_fixed_optimizer"])
        self.actor_adaptive_optimizer.load_state_dict(ckpt["actor_adaptive_optimizer"])
        self.actor_fixed_lr_schedule.load_state_dict(ckpt["actor_fixed_lr_schedule"])
        self.actor_adaptive_lr_schedule.load_state_dict(ckpt["actor_adaptive_lr_schedule"])
        with torch.no_grad():
            self.rho.copy_(ckpt["rho"].to(device=self.rho.device, dtype=self.rho.dtype))
        self.rho_optimizer.load_state_dict(ckpt["rho_optimizer"])
        self.total_it = int(ckpt["total_it"])
        self.completed_meta_events = int(ckpt["completed_meta_events"])
        set_rng_state(ckpt["rng"], self.outer_rng)


def evaluate_policies(
    env: gym.Env,
    actors: Dict[str, nn.Module],
    *,
    device: str,
    n_episodes: int,
    seed: int,
    step: int,
    paired: bool,
    protocol: str,
) -> List[Dict[str, Any]]:
    """Evaluate actors; optionally pair start states (AntMaze)."""
    rows: List[Dict[str, Any]] = []
    names = list(actors.keys())

    if not paired or len(names) < 2:
        for pid, actor in actors.items():
            returns = eval_actor(env, actor, device=device, n_episodes=n_episodes, seed=seed)
            lengths = [None] * n_episodes
            norms = []
            for r in returns:
                try:
                    norms.append(float(env.get_normalized_score(float(r)) * 100.0))
                except Exception:
                    norms.append(float("nan"))
            rows.append(
                {
                    "step": step,
                    "policy_id": pid,
                    "episode_count": n_episodes,
                    "returns": returns.tolist(),
                    "normalized_scores": norms,
                    "episode_lengths": lengths,
                    "mean_return": float(np.mean(returns)),
                    "std_return": float(np.std(returns, ddof=1)) if n_episodes > 1 else 0.0,
                    "mean_normalized": float(np.nanmean(norms)),
                    "std_normalized": float(np.nanstd(norms, ddof=1)) if n_episodes > 1 else 0.0,
                    "episode_ids": list(range(n_episodes)),
                    "protocol": protocol,
                    "paired": False,
                }
            )
        return rows

    # Paired evaluation: same start for both policies per episode.
    per_policy: Dict[str, Dict[str, List[Any]]] = {
        pid: {"returns": [], "norms": [], "lengths": [], "ids": []} for pid in names
    }
    for ep in range(n_episodes):
        env.seed(seed + ep)
        obs = env.reset()
        snap = snapshot_env(env)
        start_obs = np.array(obs, copy=True)
        ep_id = ep
        for pid in names:
            restore_env(env, snap)
            # Re-apply observation transform by reading through wrappers if needed.
            actor = actors[pid]
            actor.eval()
            state = start_obs
            done = False
            ep_ret = 0.0
            ep_len = 0
            while not done:
                action = actor.act(state, device)
                state, reward, done, _ = env.step(action)
                ep_ret += float(reward)
                ep_len += 1
            actor.train()
            per_policy[pid]["returns"].append(ep_ret)
            try:
                per_policy[pid]["norms"].append(
                    float(env.get_normalized_score(ep_ret) * 100.0)
                )
            except Exception:
                per_policy[pid]["norms"].append(float("nan"))
            per_policy[pid]["lengths"].append(ep_len)
            per_policy[pid]["ids"].append(ep_id)

    # Self-replay check on first episode of first policy (optional soft check).
    for pid in names:
        rets = np.asarray(per_policy[pid]["returns"], dtype=np.float64)
        norms = np.asarray(per_policy[pid]["norms"], dtype=np.float64)
        rows.append(
            {
                "step": step,
                "policy_id": pid,
                "episode_count": n_episodes,
                "returns": rets.tolist(),
                "normalized_scores": norms.tolist(),
                "episode_lengths": per_policy[pid]["lengths"],
                "mean_return": float(np.mean(rets)),
                "std_return": float(np.std(rets, ddof=1)) if n_episodes > 1 else 0.0,
                "mean_normalized": float(np.nanmean(norms)),
                "std_normalized": float(np.nanstd(norms, ddof=1)) if n_episodes > 1 else 0.0,
                "episode_ids": per_policy[pid]["ids"],
                "protocol": protocol,
                "paired": True,
            }
        )
    if "fixed_beta" in per_policy and "adaptive_beta" in per_policy:
        diff = (
            np.asarray(per_policy["adaptive_beta"]["norms"])
            - np.asarray(per_policy["fixed_beta"]["norms"])
        )
        rows.append(
            {
                "step": step,
                "policy_id": "adaptive_minus_fixed",
                "episode_count": n_episodes,
                "normalized_score_diffs": diff.tolist(),
                "mean_normalized_diff": float(np.nanmean(diff)),
                "episode_ids": per_policy["fixed_beta"]["ids"],
                "protocol": protocol,
                "paired": True,
            }
        )
    return rows


def eval_episode_count(env_name: str, step: int, max_timesteps: int, default_n: int) -> int:
    if "antmaze" in env_name:
        if step % 50000 == 0 or step == max_timesteps:
            return 100
        return 10
    return default_n


def run(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir).resolve()
    resume_path = args.resume.strip()
    refuse_output_dir(output_dir, resume=bool(resume_path))
    output_dir.mkdir(parents=True, exist_ok=True)

    stdout_path = output_dir / "stdout.log"
    # Tee-ish: keep printing and also append.
    class _Tee:
        def __init__(self, stream, path):
            self.stream = stream
            self.f = open(path, "a", buffering=1)

        def write(self, data):
            self.stream.write(data)
            self.f.write(data)
            self.f.flush()

        def flush(self):
            self.stream.flush()
            self.f.flush()

    sys.stdout = _Tee(sys.stdout, stdout_path)  # type: ignore
    sys.stderr = _Tee(sys.stderr, stdout_path)  # type: ignore

    corl = load_corl_yaml(args.config)
    validate_env_config(args.env, corl)
    meta = MetaConfig.from_yaml(args.meta_config)

    # CLI overrides (smoke / explicit)
    if args.max_timesteps is not None:
        corl["max_timesteps"] = int(args.max_timesteps)
    if args.meta_warmup_steps is not None:
        meta.meta_warmup_steps = int(args.meta_warmup_steps)
    if args.meta_interval is not None:
        meta.meta_interval = int(args.meta_interval)
    if args.eval_episodes is not None:
        corl["n_episodes"] = int(args.eval_episodes)
    if args.eval_freq is not None:
        corl["eval_freq"] = int(args.eval_freq)
    if args.adaptive_enabled is not None:
        meta.adaptive_enabled = str(args.adaptive_enabled).lower() in ("1", "true", "yes")

    # Sync beta from CORL yaml into meta (env-specific).
    meta.beta_initial = float(corl["beta"])
    meta.beta_fixed = float(corl["beta"])

    corl["seed"] = int(args.seed)
    corl["device"] = args.device
    corl["env"] = args.env

    metrics_freq = int(args.metrics_freq or 1000)
    checkpoint_freq = int(args.checkpoint_freq or 50000)
    eval_freq = int(corl.get("eval_freq", 5000))
    max_timesteps = int(corl["max_timesteps"])

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA requested ({device}) but not available")

    import d4rl  # noqa: F401

    env = gym.make(args.env)
    eval_env = gym.make(args.env)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    dataset, data_manifest = load_qlearning_dataset_from_hdf5(env, args.dataset_path)
    dataset, norm_info = apply_normalization_and_reward(
        dataset,
        args.env,
        normalize=bool(corl.get("normalize", True)),
        normalize_reward=bool(corl.get("normalize_reward", False)),
        max_episode_steps=getattr(env, "_max_episode_steps", 1000),
    )
    data_manifest.update(norm_info)

    state_mean = norm_info["state_mean"]
    state_std = norm_info["state_std"]
    env = wrap_env(env, state_mean=state_mean, state_std=state_std)
    eval_env = wrap_env(eval_env, state_mean=state_mean, state_std=state_std)

    n_transitions = int(dataset["observations"].shape[0])
    replay = CPUReplayBuffer(state_dim, action_dim, n_transitions)
    replay.load_d4rl_dataset(dataset)

    set_seed(int(args.seed), env)

    trainer = IQLAdaptiveBetaTrainer(
        corl=corl,
        meta=meta,
        device=device,
        seed=int(args.seed),
        max_action=max_action,
        state_dim=state_dim,
        action_dim=action_dim,
    )

    sources = adaptive_beta_source_files(REPO_ROOT)
    sources = [p for p in sources if p.exists()]
    chash = code_hash(sources) if sources else ""
    try:
        import subprocess

        base = subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        base = ""

    resolved = {
        "corl": corl,
        "meta": meta.to_dict(),
        "cli": vars(args),
        "code_hash": chash,
        "base_commit": base,
        "mode": args.mode,
    }
    dump_yaml(output_dir / "resolved_config.yaml", resolved)
    with open(output_dir / "dataset_manifest.json", "w") as f:
        json.dump(data_manifest, f, indent=2)
    runtime = {
        "python": sys.executable,
        "device": device,
        "cuda_available": torch.cuda.is_available(),
        "torch_version": torch.__version__,
        "output_dir": str(output_dir),
        "pid": os.getpid(),
        "start_time": time.time(),
        "code_hash": chash,
        "base_commit": base,
        "mode": args.mode,
    }
    if torch.cuda.is_available():
        runtime["cuda_device_name"] = torch.cuda.get_device_name(
            torch.device(device).index or 0
        )
    with open(output_dir / "runtime_manifest.json", "w") as f:
        json.dump(runtime, f, indent=2)

    if resume_path:
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        if args.mode == "smoke" or ckpt.get("mode") == "smoke":
            if args.mode != "smoke":
                raise RuntimeError("Refusing to resume smoke checkpoint as production")
        trainer.load_state_dict(ckpt)

    if args.mode == "eval":
        ckpt_path = args.checkpoint or resume_path
        if not ckpt_path:
            raise ValueError("--checkpoint required for eval mode")
        trainer.load_state_dict(
            torch.load(ckpt_path, map_location=device, weights_only=False)
        )
        n_ep = int(corl.get("n_episodes", 10))
        rng_before = get_rng_state(trainer.outer_rng)
        rows = evaluate_policies(
            eval_env,
            {
                "fixed_beta": trainer.actor_fixed,
                "adaptive_beta": trainer.actor_adaptive,
            },
            device=device,
            n_episodes=n_ep,
            seed=int(args.seed),
            step=trainer.total_it,
            paired="antmaze" in args.env,
            protocol="eval_mode",
        )
        set_rng_state(rng_before, trainer.outer_rng)
        for row in rows:
            append_jsonl(output_dir / "eval.jsonl", row)
        return

    t0 = time.time()
    peak_alloc = 0.0
    start_it = trainer.total_it

    try:
        for _ in range(start_it, max_timesteps):
            batch_b = replay.sample(trainer.batch_size)
            batch_b = [b.to(device) for b in batch_b]

            next_step = trainer.total_it + 1
            do_meta = (
                meta.adaptive_enabled
                and is_meta_step(next_step, meta.meta_warmup_steps, meta.meta_interval)
            )
            batch_c = None
            if do_meta:
                batch_c = replay.sample_outer(meta.outer_batch_size, trainer.outer_rng)
                batch_c = [b.to(device) for b in batch_c]

            metrics, meta_row = trainer.train_step(batch_b, batch_c, do_meta=do_meta)
            step = trainer.total_it

            if torch.cuda.is_available():
                peak_alloc = max(
                    peak_alloc,
                    float(torch.cuda.max_memory_allocated()) / (1024**2),
                )
            metrics["elapsed_seconds"] = time.time() - t0
            metrics["gpu_peak_alloc_mb"] = peak_alloc

            if step % metrics_freq == 0 or step == max_timesteps:
                append_jsonl(output_dir / "metrics.jsonl", metrics)
                print(
                    f"[step {step}] q={metrics['q_loss']:.4f} v={metrics['value_loss']:.4f} "
                    f"af={metrics['actor_fixed_loss']:.4f} aa={metrics['actor_adaptive_loss']:.4f} "
                    f"beta={metrics['beta_used']:.4f}->{metrics['beta_next']:.4f}",
                    flush=True,
                )

            if meta_row is not None:
                append_jsonl(output_dir / "meta.jsonl", meta_row)

            # Evaluation
            if step % eval_freq == 0 or step == max_timesteps:
                n_ep = eval_episode_count(
                    args.env, step, max_timesteps, int(corl.get("n_episodes", 10))
                )
                # AntMaze: skip duplicate 10-ep when 100-ep is scheduled.
                if "antmaze" in args.env and n_ep == 100:
                    protocol = "antmaze_100"
                elif "antmaze" in args.env:
                    protocol = "antmaze_10"
                else:
                    protocol = "locomotion_10"

                rng_before = get_rng_state(trainer.outer_rng)
                rows = evaluate_policies(
                    eval_env,
                    {
                        "fixed_beta": trainer.actor_fixed,
                        "adaptive_beta": trainer.actor_adaptive,
                    },
                    device=device,
                    n_episodes=n_ep,
                    seed=int(args.seed),
                    step=step,
                    paired="antmaze" in args.env,
                    protocol=protocol,
                )
                set_rng_state(rng_before, trainer.outer_rng)
                for row in rows:
                    append_jsonl(output_dir / "eval.jsonl", row)
                print(
                    f"[eval {step}] "
                    + ", ".join(
                        f"{r['policy_id']}={r.get('mean_normalized', float('nan')):.2f}"
                        for r in rows
                        if "mean_normalized" in r
                    ),
                    flush=True,
                )

            if step % checkpoint_freq == 0 or step == max_timesteps:
                ckpt = trainer.state_dict()
                ckpt["mode"] = args.mode
                ckpt["code_hash"] = chash
                ckpt["base_commit"] = base
                ckpt["dataset_manifest"] = data_manifest
                ckpt["norm_info"] = norm_info
                ckpt["resolved_config"] = resolved
                path = output_dir / f"checkpoint_{step}.pt"
                torch.save(ckpt, path)
                print(f"[ckpt] wrote {path}", flush=True)

        completion = {
            "status": "completed",
            "total_it": trainer.total_it,
            "completed_meta_events": trainer.completed_meta_events,
            "elapsed_seconds": time.time() - t0,
            "code_hash": chash,
            "mode": args.mode,
        }
        with open(output_dir / "completion.json", "w") as f:
            json.dump(completion, f, indent=2)
        print("[done] training complete", flush=True)
    except Exception as e:
        err = {
            "status": "failed",
            "error": str(e),
            "traceback": traceback.format_exc(),
            "total_it": trainer.total_it,
        }
        with open(output_dir / "failure.json", "w") as f:
            json.dump(err, f, indent=2)
        raise


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
