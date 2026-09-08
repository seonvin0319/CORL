"""Finite-difference hypergradient checks for adaptive beta."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from algorithms.offline.iql import DeterministicPolicy, GaussianPolicy
from algorithms.offline.iql_adaptive_beta_utils import (
    adam_state_from_optimizer,
    actor_ell,
    actor_ell_functional,
    beta_from_rho,
    functional_adam_step,
    named_parameters_dict,
    safe_exp_weights,
)


def _fd_hypergrad(
    *,
    deterministic: bool,
    eps: float = 1e-4,
    seed: int = 0,
):
    torch.manual_seed(seed)
    device = "cpu"
    dtype = torch.float64
    state_dim, action_dim, B, C = 3, 2, 16, 16
    lr = 2e-4

    if deterministic:
        actor = DeterministicPolicy(state_dim, action_dim, 1.0).to(device=device)
    else:
        actor = GaussianPolicy(state_dim, action_dim, 1.0).to(device=device)
    # Promote to float64 for FD stability
    actor = actor.double()

    opt = torch.optim.Adam(actor.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8)
    # Warm moments away from zero, avoid ReLU / clamp boundaries
    for i in range(5):
        s = torch.randn(B, state_dim, dtype=dtype) * 0.1
        a = torch.tanh(torch.randn(B, action_dim, dtype=dtype) * 0.1)
        loss = actor_ell(actor, s, a).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()

    # Fixed batches / advantages away from weight-clip boundary
    s_b = torch.randn(B, state_dim, dtype=dtype) * 0.05
    a_b = torch.tanh(torch.randn(B, action_dim, dtype=dtype) * 0.05)
    s_c = torch.randn(C, state_dim, dtype=dtype) * 0.05
    a_c = torch.tanh(torch.randn(C, action_dim, dtype=dtype) * 0.05)
    adv_b = torch.randn(B, dtype=dtype) * 0.2  # beta*A << log(100)
    adv_c = torch.randn(C, dtype=dtype) * 0.2

    rho0 = math.log(3.0)
    rho = torch.nn.Parameter(torch.tensor(rho0, dtype=dtype))

    def F_and_analytic(rho_param: torch.Tensor):
        named = named_parameters_dict(actor)
        params = {k: v for k, v in named.items()}
        beta = beta_from_rho(rho_param)
        w_inner = safe_exp_weights(beta, adv_b, cap=100.0)
        L_inner = torch.mean(w_inner * actor_ell(actor, s_b, a_b))
        grads = torch.autograd.grad(
            L_inner, list(params.values()), create_graph=True, allow_unused=True
        )
        named_grads = {n: g for n, g in zip(params.keys(), grads)}
        adam_state = adam_state_from_optimizer(opt, named)
        theta_plus, _ = functional_adam_step(
            params, named_grads, adam_state, lr=lr, betas=(0.9, 0.999), eps=1e-8
        )
        w_ref = safe_exp_weights(beta.detach(), adv_c, cap=100.0).detach()
        L_meta = torch.mean(w_ref * actor_ell_functional(actor, theta_plus, s_c, a_c))
        g = torch.autograd.grad(L_meta, rho_param)[0]
        return L_meta.detach(), g.detach()

    _, g_analytic = F_and_analytic(rho)

    def F_value(rho_val: float) -> float:
        # Rebuild graph from a fresh rho leaf; keep w_ref fixed at rho0's beta.
        named = named_parameters_dict(actor)
        params = {k: v for k, v in named.items()}
        rho_t = torch.tensor(rho_val, dtype=dtype, requires_grad=True)
        beta = beta_from_rho(rho_t)
        w_inner = safe_exp_weights(beta, adv_b, cap=100.0)
        L_inner = torch.mean(w_inner * actor_ell(actor, s_b, a_b))
        grads = torch.autograd.grad(
            L_inner, list(params.values()), create_graph=True, allow_unused=True
        )
        named_grads = {n: g for n, g in zip(params.keys(), grads)}
        adam_state = adam_state_from_optimizer(opt, named)
        theta_plus, _ = functional_adam_step(
            params, named_grads, adam_state, lr=lr, betas=(0.9, 0.999), eps=1e-8
        )
        beta_ref = torch.tensor(math.exp(rho0), dtype=dtype)
        w_ref = safe_exp_weights(beta_ref, adv_c, cap=100.0).detach()
        L_meta = torch.mean(w_ref * actor_ell_functional(actor, theta_plus, s_c, a_c))
        return float(L_meta.detach().cpu().item())

    f_pos = F_value(rho0 + eps)
    f_neg = F_value(rho0 - eps)
    g_fd = (f_pos - f_neg) / (2.0 * eps)
    g_an = float(g_analytic.cpu().item())
    abs_err = abs(g_fd - g_an)
    rel_err = abs_err / max(abs(g_an), abs(g_fd), 1e-12)
    return {
        "g_analytic": g_an,
        "g_fd": g_fd,
        "abs_err": abs_err,
        "rel_err": rel_err,
        "eps": eps,
    }


def test_fd_hypergrad_gaussian():
    r4 = _fd_hypergrad(deterministic=False, eps=1e-4, seed=0)
    r5 = _fd_hypergrad(deterministic=False, eps=1e-5, seed=0)
    assert r4["abs_err"] < 1e-7 or r4["rel_err"] < 1e-3, r4
    assert r5["abs_err"] < 1e-7 or r5["rel_err"] < 1e-3, r5
    # Consistency across eps
    assert abs(r4["g_fd"] - r5["g_fd"]) / max(abs(r4["g_fd"]), 1e-12) < 0.05 or abs(
        r4["g_fd"] - r5["g_fd"]
    ) < 1e-6


def test_fd_hypergrad_deterministic():
    r4 = _fd_hypergrad(deterministic=True, eps=1e-4, seed=1)
    r5 = _fd_hypergrad(deterministic=True, eps=1e-5, seed=1)
    assert r4["abs_err"] < 1e-7 or r4["rel_err"] < 1e-3, r4
    assert r5["abs_err"] < 1e-7 or r5["rel_err"] < 1e-3, r5


def test_clipped_weights_zero_beta_grad():
    dtype = torch.float64
    beta = torch.tensor(10.0, dtype=dtype, requires_grad=True)
    # Large positive advantages => clipped
    adv = torch.tensor([5.0, 8.0, 0.01], dtype=dtype)
    w = safe_exp_weights(beta, adv, cap=100.0)
    assert abs(float(w[0]) - 100.0) < 1e-12
    assert abs(float(w[1]) - 100.0) < 1e-12
    loss = w.sum()
    g = torch.autograd.grad(loss, beta)[0]
    # Only the unclipped sample contributes
    # w2 = exp(beta * 0.01), dw/dbeta = 0.01 * exp(beta*0.01)
    expected = 0.01 * math.exp(10.0 * 0.01)
    assert abs(float(g.item()) - expected) < 1e-10
