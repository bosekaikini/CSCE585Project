"""Small, dependency-light PINN training loop."""
from __future__ import annotations
import torch
from .pde_loss import total_loss

def train(model, sampler, terminal_sampler, payoff, epochs=1000, lr=1e-3, rate=.05,
          volatility=.2, terminal_weight=1., optimizer=None):
    opt=optimizer or torch.optim.Adam(model.parameters(), lr=lr); history=[]
    for _ in range(epochs):
        opt.zero_grad()
        loss=total_loss(model,sampler(),terminal_sampler(),payoff,rate,volatility,terminal_weight)
        loss.backward(); opt.step(); history.append(float(loss.detach()))
    return {"model":model,"optimizer":opt,"loss":history}
