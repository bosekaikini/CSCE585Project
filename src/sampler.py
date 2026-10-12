"""Uniform and payoff-aware collocation point samplers."""
from __future__ import annotations
import torch

def uniform_sampler(n, dimension, spot_bounds=(0., 2.), maturity=1., device=None, dtype=torch.float32):
    if n <= 0 or dimension <= 0: raise ValueError("n and dimension must be positive")
    low, high = spot_bounds
    if not high > low or maturity <= 0: raise ValueError("invalid domain bounds")
    s = low + (high-low) * torch.rand(n, dimension, device=device, dtype=dtype)
    t = maturity * torch.rand(n, 1, device=device, dtype=dtype)
    return torch.cat((s, t), 1).requires_grad_(True)

def adaptive_sampler(n, dimension, strike=1., spot_bounds=(0., 2.), maturity=1.,
                     maturity_fraction=.5, kink_fraction=.5, device=None, dtype=torch.float32):
    if maturity_fraction < 0 or kink_fraction < 0 or maturity_fraction + kink_fraction > 1:
        raise ValueError("fractions must be nonnegative and sum to at most one")
    nm, nk = int(n*maturity_fraction), int(n*kink_fraction)
    parts = []
    if nm:
        s = spot_bounds[0] + (spot_bounds[1]-spot_bounds[0])*torch.rand(nm, dimension, device=device, dtype=dtype)
        t = maturity*(1-.1*torch.rand(nm, 1, device=device, dtype=dtype)); parts.append(torch.cat((s,t), 1))
    if nk:
        s = (strike+.1*torch.randn(nk, dimension, device=device, dtype=dtype)).clamp(*spot_bounds)
        t = maturity*torch.rand(nk, 1, device=device, dtype=dtype); parts.append(torch.cat((s,t), 1))
    nu = n-nm-nk
    if nu: parts.append(uniform_sampler(nu, dimension, spot_bounds, maturity, device, dtype).detach())
    return torch.cat(parts, 0).requires_grad_(True)
