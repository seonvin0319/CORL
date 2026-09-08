"""Parity: adaptive-beta trainer vs original ImplicitQLearning."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from algorithms.offline.iql import (
    DeterministicPolicy,
    GaussianPolicy,
    ImplicitQLearning,
    TwinQ,
    ValueFunction,
)
from algorithms.offline.iql_adaptive_beta import IQLAdaptiveBetaTrainer
from algorithms.offline.iql_adaptive_beta_utils import MetaConfig


def _make_batch(n: int, state_dim: int, action_dim: int, device: str, seed: int):
    g = torch.Generator().manual_seed(seed)
    states = torch.randn(n, state_dim, generator=g, device=device)
    actions = torch.randn(n, action_dim, generator=g, device=device)
    rewards = torch.randn(n, 1, generator=g, device=device)
    next_states = torch.randn(n, state_dim, generator=g, device=device)
    dones = (torch.rand(n, 1, generator=g, device=device) > 0.9).float()
    return [states, actions, rewards, next_states, dones]


def _assert_close_state(a: nn.Module, b: nn.Module, atol=1e-6, rtol=1e-5):
    for (na, pa), (nb, pb) in zip(a.named_parameters(), b.named_parameters()):
        assert na == nb
        assert torch.allclose(pa, pb, atol=atol, rtol=rtol), f"mismatch at {na}"


def _assert_opt_close(opt_a, mod_a, opt_b, mod_b, atol=1e-6):
    for pa, pb in zip(mod_a.parameters(), mod_b.parameters()):
        if pa not in opt_a.state:
            assert pb not in opt_b.state
            continue
        sa, sb = opt_a.state[pa], opt_b.state[pb]
        assert int(sa["step"]) == int(sb["step"])
        assert torch.allclose(sa["exp_avg"], sb["exp_avg"], atol=atol)
        assert torch.allclose(sa["exp_avg_sq"], sb["exp_avg_sq"], atol=atol)


def _run_parity(deterministic: bool, n_iters: int = 5):
    device = "cpu"
    state_dim, action_dim = 4, 2
    torch.manual_seed(0)
    np.random.seed(0)

    corl = {
        "discount": 0.99,
        "tau": 0.005,
        "iql_tau": 0.7,
        "batch_size": 32,
        "max_timesteps": 1_000_000,
        "iql_deterministic": deterministic,
        "vf_lr": 3e-4,
        "qf_lr": 3e-4,
        "actor_lr": 3e-4,
        "actor_dropout": None,
        "beta": 3.0,
    }
    meta = MetaConfig(
        adaptive_enabled=False,
        beta_initial=3.0,
        beta_fixed=3.0,
        meta_warmup_steps=100000,
        meta_interval=20,
    )
    trainer = IQLAdaptiveBetaTrainer(
        corl=corl,
        meta=meta,
        device=device,
        seed=0,
        max_action=1.0,
        state_dim=state_dim,
        action_dim=action_dim,
    )
    if deterministic:
        actor = DeterministicPolicy(state_dim, action_dim, 1.0).to(device)
    else:
        actor = GaussianPolicy(state_dim, action_dim, 1.0).to(device)
    actor.load_state_dict(trainer.actor_fixed.state_dict())
    q = TwinQ(state_dim, action_dim).to(device)
    q.load_state_dict(trainer.qf.state_dict())
    v = ValueFunction(state_dim).to(device)
    v.load_state_dict(trainer.vf.state_dict())
    actor_opt = torch.optim.Adam(actor.parameters(), lr=3e-4)
    q_opt = torch.optim.Adam(q.parameters(), lr=3e-4)
    v_opt = torch.optim.Adam(v.parameters(), lr=3e-4)
    iql = ImplicitQLearning(
        max_action=1.0,
        actor=actor,
        actor_optimizer=actor_opt,
        q_network=q,
        q_optimizer=q_opt,
        v_network=v,
        v_optimizer=v_opt,
        iql_tau=0.7,
        beta=3.0,
        max_steps=1_000_000,
        discount=0.99,
        tau=0.005,
        device=device,
    )
    iql.q_target.load_state_dict(trainer.q_target.state_dict())

    batches = [
        _make_batch(32, state_dim, action_dim, device, seed=100 + i) for i in range(n_iters)
    ]

    for batch in batches:
        log = iql.train([t.clone() for t in batch])
        metrics, _ = trainer.train_step([t.clone() for t in batch], None, do_meta=False)
        assert abs(log["q_loss"] - metrics["q_loss"]) < 1e-5
        assert abs(log["value_loss"] - metrics["value_loss"]) < 1e-5
        assert abs(log["actor_loss"] - metrics["actor_fixed_loss"]) < 1e-5
        _assert_close_state(iql.qf, trainer.qf)
        _assert_close_state(iql.q_target, trainer.q_target)
        _assert_close_state(iql.vf, trainer.vf)
        _assert_close_state(iql.actor, trainer.actor_fixed)
        _assert_opt_close(iql.q_optimizer, iql.qf, trainer.q_optimizer, trainer.qf)
        _assert_opt_close(iql.v_optimizer, iql.vf, trainer.v_optimizer, trainer.vf)
        _assert_opt_close(
            iql.actor_optimizer, iql.actor, trainer.actor_fixed_optimizer, trainer.actor_fixed
        )
        assert (
            iql.actor_lr_schedule.state_dict()["last_epoch"]
            == trainer.actor_fixed_lr_schedule.state_dict()["last_epoch"]
        )


def test_parity_gaussian():
    _run_parity(deterministic=False, n_iters=5)


def test_parity_deterministic():
    _run_parity(deterministic=True, n_iters=5)


def test_adaptive_flag_does_not_change_qv_fixed():
    device = "cpu"
    state_dim, action_dim = 3, 2
    torch.manual_seed(1)
    corl = {
        "discount": 0.99,
        "tau": 0.005,
        "iql_tau": 0.7,
        "batch_size": 16,
        "max_timesteps": 1_000_000,
        "iql_deterministic": False,
        "vf_lr": 3e-4,
        "qf_lr": 3e-4,
        "actor_lr": 3e-4,
        "actor_dropout": None,
        "beta": 3.0,
    }
    meta_off = MetaConfig(adaptive_enabled=False, beta_initial=3.0, beta_fixed=3.0)
    meta_on = MetaConfig(
        adaptive_enabled=True,
        beta_initial=3.0,
        beta_fixed=3.0,
        meta_warmup_steps=100000,
        meta_interval=20,
    )
    t_off = IQLAdaptiveBetaTrainer(
        corl=corl,
        meta=meta_off,
        device=device,
        seed=0,
        max_action=1.0,
        state_dim=state_dim,
        action_dim=action_dim,
    )
    t_on = IQLAdaptiveBetaTrainer(
        corl=corl,
        meta=meta_on,
        device=device,
        seed=0,
        max_action=1.0,
        state_dim=state_dim,
        action_dim=action_dim,
    )
    t_on.qf.load_state_dict(t_off.qf.state_dict())
    t_on.q_target.load_state_dict(t_off.q_target.state_dict())
    t_on.vf.load_state_dict(t_off.vf.state_dict())
    t_on.actor_fixed.load_state_dict(t_off.actor_fixed.state_dict())
    t_on.actor_adaptive.load_state_dict(t_off.actor_adaptive.state_dict())

    batches = [
        _make_batch(16, state_dim, action_dim, device, seed=50 + i) for i in range(4)
    ]
    for batch in batches:
        t_off.train_step([t.clone() for t in batch], None, do_meta=False)
        t_on.train_step([t.clone() for t in batch], None, do_meta=False)
        _assert_close_state(t_off.qf, t_on.qf)
        _assert_close_state(t_off.q_target, t_on.q_target)
        _assert_close_state(t_off.vf, t_on.vf)
        _assert_close_state(t_off.actor_fixed, t_on.actor_fixed)
