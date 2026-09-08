"""Schedule / meta-step / beta-timing unit checks."""

from __future__ import annotations

import math

import torch

from algorithms.offline.iql_adaptive_beta import IQLAdaptiveBetaTrainer
from algorithms.offline.iql_adaptive_beta_utils import MetaConfig, is_meta_step, safe_exp_weights


def test_meta_schedule_production():
    warmup, interval = 100000, 20
    assert not is_meta_step(100000, warmup, interval)
    assert not is_meta_step(100001, warmup, interval)
    assert is_meta_step(100020, warmup, interval)
    assert is_meta_step(100040, warmup, interval)
    assert not is_meta_step(100030, warmup, interval)


def test_meta_schedule_smoke():
    warmup, interval = 80, 20
    expected = list(range(100, 241, 20))
    got = [s for s in range(1, 241) if is_meta_step(s, warmup, interval)]
    assert got == expected


def test_safe_weights_match_corl_in_normal_range():
    beta = torch.tensor(3.0)
    adv = torch.linspace(-2.0, 2.0, 50)
    w_safe = safe_exp_weights(beta, adv, cap=100.0)
    w_ref = torch.exp(beta * adv).clamp(max=100.0)
    assert torch.allclose(w_safe, w_ref, atol=1e-6)


def test_beta_timing_meta_updates_next_iter():
    device = "cpu"
    corl = {
        "discount": 0.99,
        "tau": 0.005,
        "iql_tau": 0.7,
        "batch_size": 8,
        "max_timesteps": 1_000_000,
        "iql_deterministic": False,
        "vf_lr": 3e-4,
        "qf_lr": 3e-4,
        "actor_lr": 3e-4,
        "actor_dropout": None,
        "beta": 3.0,
    }
    meta = MetaConfig(
        adaptive_enabled=True,
        beta_initial=3.0,
        beta_fixed=3.0,
        meta_warmup_steps=5,
        meta_interval=5,
        outer_batch_size=8,
    )
    trainer = IQLAdaptiveBetaTrainer(
        corl=corl,
        meta=meta,
        device=device,
        seed=0,
        max_action=1.0,
        state_dim=3,
        action_dim=2,
    )
    # Force total_it so next step is a meta step: want step==10
    trainer.total_it = 9
    beta_before = trainer.current_beta_value()
    assert abs(beta_before - 3.0) < 1e-8

    def batch():
        return [
            torch.randn(8, 3),
            torch.randn(8, 2),
            torch.randn(8, 1),
            torch.randn(8, 3),
            torch.zeros(8, 1),
        ]

    b = batch()
    c = batch()
    metrics, meta_row = trainer.train_step(b, c, do_meta=True)
    assert meta_row is not None
    # This step used beta_before
    assert abs(metrics["beta_used"] - beta_before) < 1e-8
    # beta_next may differ after rho step
    assert "beta_next" in metrics
    # Rho changed (or projected); at least optimizer stepped
    assert trainer.rho_optimizer.state[trainer.rho]["step"] >= 1
