"""Autograd Greeks for a trained pricing model."""
from __future__ import annotations
import torch

def greeks(model, x, create_graph=False):
    if not x.requires_grad: x = x.clone().requires_grad_(True)
    value = model(x)
    if value.ndim == 1: value = value[:, None]
    grad = torch.autograd.grad(value.sum(), x, create_graph=True)[0]
    delta = grad[:, :-1]
    gamma = torch.stack([torch.autograd.grad(delta[:, i].sum(), x, create_graph=True)[0][:, i]
                         for i in range(delta.shape[1])], 1)
    result = {"price": value, "delta": delta, "gamma": gamma}
    return result if create_graph else {k: v.detach() for k, v in result.items()}

def delta(model, x): return greeks(model, x)["delta"]
def gamma(model, x): return greeks(model, x)["gamma"]
