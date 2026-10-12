"""Differentiable Black-Scholes PDE and condition losses."""
from __future__ import annotations
import torch

def derivatives(model, x):
    if not x.requires_grad: x = x.clone().requires_grad_(True)
    value = model(x)
    if value.ndim == 1: value = value[:, None]
    grad = torch.autograd.grad(value.sum(), x, create_graph=True)[0]
    second = [torch.autograd.grad(grad[:, i].sum(), x, create_graph=True)[0][:, i]
              for i in range(x.shape[1] - 1)]
    return value, grad, torch.stack(second, dim=1)

def black_scholes_residual(model, x, rate, volatility):
    value, grad, second = derivatives(model, x)
    s = x[:, :-1]
    residual = grad[:, -1:] + rate * (s * grad[:, :-1]).sum(1, keepdim=True)
    residual += .5 * volatility**2 * (s.square() * second).sum(1, keepdim=True)
    return residual - rate * value

def pde_loss(model, x, rate, volatility):
    return black_scholes_residual(model, x, rate, volatility).square().mean()

def terminal_loss(model, s, payoff):
    t = torch.zeros((s.shape[0], 1), dtype=s.dtype, device=s.device)
    return (model(torch.cat((s, t), 1)) - payoff(s)).square().mean()

def total_loss(model, interior, terminal_s, payoff, rate, volatility, terminal_weight=1.0):
    return pde_loss(model, interior, rate, volatility) + terminal_weight * terminal_loss(model, terminal_s, payoff)


def heston_residual(model, x, rate, kappa, theta, sigma, rho):
    """Optional two-factor Heston residual for ``x = (S, v, t)``.

    The mixed derivative is included through torch autograd. This helper is
    intentionally separate from the basket Black-Scholes loss because Heston
    has a different state dimension and variance boundary behavior.
    """
    if x.shape[1] != 3:
        raise ValueError("Heston inputs must have columns (spot, variance, time)")
    if not x.requires_grad:
        x = x.clone().requires_grad_(True)
    value = model(x)
    if value.ndim == 1:
        value = value[:, None]
    grad = torch.autograd.grad(value.sum(), x, create_graph=True)[0]
    second_s = torch.autograd.grad(grad[:, 0].sum(), x, create_graph=True)[0][:, 0]
    second_v = torch.autograd.grad(grad[:, 1].sum(), x, create_graph=True)[0][:, 1]
    mixed = torch.autograd.grad(grad[:, 0].sum(), x, create_graph=True)[0][:, 1]
    spot, variance = x[:, 0:1], x[:, 1:2]
    diffusion = 0.5 * variance[:, 0] * spot[:, 0].square() * second_s
    variance_diffusion = 0.5 * sigma**2 * variance[:, 0] * second_v
    correlation = rho * sigma * variance[:, 0] * spot[:, 0] * mixed
    drift = rate * spot[:, 0] * grad[:, 0] + kappa * (theta - variance[:, 0]) * grad[:, 1]
    return (grad[:, 2] + drift + diffusion + variance_diffusion + correlation
            - rate * value[:, 0])
