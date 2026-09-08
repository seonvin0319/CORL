"""Functional Adam vs torch.optim.Adam."""

from __future__ import annotations

import copy

import torch
import torch.nn as nn

from algorithms.offline.iql import GaussianPolicy
from algorithms.offline.iql_adaptive_beta_utils import (
    adam_state_from_optimizer,
    functional_adam_step,
    named_parameters_dict,
    safe_sqrt,
)


def _compare_one_step(actor: nn.Module, *, lr: float, seed: int, nonzero_moments: bool):
    torch.manual_seed(seed)
    actor = copy.deepcopy(actor)
    opt = torch.optim.Adam(actor.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8)

    # Optionally warm up moments
    if nonzero_moments:
        x = torch.randn(8, actor.net.net[0].in_features)
        a = torch.randn(8, actor.log_std.numel())
        for _ in range(3):
            dist = actor(x)
            loss = (-dist.log_prob(a).sum(-1)).mean()
            opt.zero_grad()
            loss.backward()
            # zero out one coordinate to test None/zero-grad handling later
            opt.step()

    named = named_parameters_dict(actor)
    # Build loss / grads
    x = torch.randn(8, actor.net.net[0].in_features)
    a = torch.randn(8, actor.log_std.numel())
    dist = actor(x)
    loss = (-dist.log_prob(a).sum(-1)).mean()
    # Also include a param with forced zero grad
    params = list(named.values())
    grads = torch.autograd.grad(loss, params, allow_unused=True)
    named_grads = {n: g for n, g in zip(named.keys(), grads)}
    # Force one weight grad to exactly zero (not None)
    first_name = next(iter(named_grads))
    if named_grads[first_name] is not None:
        g0 = named_grads[first_name].clone()
        g0.view(-1)[0] = 0.0
        named_grads[first_name] = g0

    adam_state = adam_state_from_optimizer(opt, named)
    theta_plus, new_state = functional_adam_step(
        {k: v for k, v in named.items()},
        named_grads,
        adam_state,
        lr=lr,
        betas=(0.9, 0.999),
        eps=1e-8,
    )

    # Real Adam step with same grads
    opt.zero_grad(set_to_none=True)
    for p, g in zip(params, [named_grads[n] for n in named.keys()]):
        p.grad = None if g is None else g.detach().clone()
    before = {n: p.detach().clone() for n, p in named.items()}
    opt.step()
    after = {n: p.detach().clone() for n, p in named.items()}

    for n in named:
        assert torch.allclose(theta_plus[n].detach(), after[n], atol=1e-7, rtol=1e-5), n
        # State transition
        st = opt.state[named[n]]
        assert int(new_state[n]["step"]) == int(st["step"])
        assert torch.allclose(new_state[n]["exp_avg"], st["exp_avg"], atol=1e-7)
        assert torch.allclose(new_state[n]["exp_avg_sq"], st["exp_avg_sq"], atol=1e-7)
        # Changed (unless zero update)
        _ = before


def test_functional_adam_zero_moments():
    actor = GaussianPolicy(3, 2, 1.0)
    _compare_one_step(actor, lr=3e-4, seed=0, nonzero_moments=False)


def test_functional_adam_nonzero_moments_and_cosine_lr():
    actor = GaussianPolicy(3, 2, 1.0)
    # Use a reduced LR as if cosine schedule advanced
    _compare_one_step(actor, lr=1.5e-4, seed=1, nonzero_moments=True)


def test_safe_sqrt_eps_outside():
    v = torch.tensor([0.0, 1e-8, 4.0], dtype=torch.float64, requires_grad=True)
    y = safe_sqrt(v) + 1e-8  # Adam denom style
    loss = y.sum()
    g = torch.autograd.grad(loss, v, create_graph=True)[0]
    assert torch.isfinite(g).all()
    # Forward matches torch.sqrt for positive; 0 -> 0
    assert float(safe_sqrt(torch.tensor(0.0))) == 0.0
    assert abs(float(safe_sqrt(torch.tensor(4.0))) - 2.0) < 1e-12


def test_functional_adam_none_grad_leaves_param():
    actor = GaussianPolicy(2, 1, 1.0)
    opt = torch.optim.Adam(actor.parameters(), lr=1e-3)
    named = named_parameters_dict(actor)
    # Warm one step so state exists
    x = torch.randn(4, 2)
    a = torch.randn(4, 1)
    loss = (-actor(x).log_prob(a).sum(-1)).mean()
    opt.zero_grad()
    loss.backward()
    opt.step()

    adam_state = adam_state_from_optimizer(opt, named)
    named_grads = {n: None for n in named}  # all None
    theta_plus, new_state = functional_adam_step(
        {k: v for k, v in named.items()},
        named_grads,
        adam_state,
        lr=1e-3,
    )
    for n, p in named.items():
        assert torch.equal(theta_plus[n], p)
        assert int(new_state[n]["step"]) == int(adam_state[n]["step"])
