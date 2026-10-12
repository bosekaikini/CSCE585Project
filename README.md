# CSCE585Project
## Options Pricing Near Maturity and Greek Variance Reduction via Physics Informed Neural Networks

## Group Info
- Bose Kaikini (Computer Science), individual project
  - Email: bkaikini@email.sc.edu

## TODO
1. Fine tune project hyperparameters
2. Parallelize abd vectorize to reduce runtime beyond autograd
3. Make project demo, reproducbility guide

## Overview
PINN-Options is an open-source physics-informed neural network (PINN) solver for Black-Scholes and Heston-style option-pricing PDEs. It uses mesh-free adaptive sampling, tanh activations, and automatic differentiation so that the price and its Greeks come from a single pass (faster and more accurate pricing). We compare it against Monte Carlo and finite-difference baselines on medium-size baskets (d in {2, 5, 10, 20}), measuring relative L2 pricing error, Greek variance, runtime, and GPU memory. The systems question is whether a PINN's one-time training cost buys fast, low-variance inference as dimension grows. Ongoing.

## Problem Description
- Problem description: Classical PDE solvers (finite differences) suffer from the curse of dimensionality, and Monte Carlo is slow and noisy for the Greeks. We train a PINN that minimizes the PDE residual plus terminal-payoff and boundary penalties together, then evaluate prices and Greeks (delta, gamma, vega, etc.) through autograd. We test whether it matches Monte Carlo pricing accuracy at lower Greek variance and inference latency.
- Motivation
  - Teams can lose alpha when pricing and hedging because traditional methods are slow at scale.
  - Parabolic PDE solvers (for example for Heston) are slow, and grid methods become infeasible as the number of assets grows.
  - Monte Carlo Greeks are noisy; a trained PINN gives smooth derivatives from one global surface.
- Challenges
  - PINN training is hard (loss balancing, convergence), especially near maturity where the payoff kink is non-smooth.
  - Collocation points must concentrate near maturity and kinks, which uniform sampling does not do.
  - Memory and runtime growth with dimension must be measured, not assumed.

## Research Questions and Hypotheses
- **RQ1:** How accurately does PINN-Options price options and estimate Greeks compared with Monte Carlo and finite-difference baselines?
- **RQ2:** How do memory (VRAM) and runtime scale as asset dimension grows from 2 to 20?
- **RQ3:** Does adaptive mesh sampling beat uniform sampling at equal training budget?
- **H1:** The PINN reaches low relative L2 error against Monte Carlo reference prices on medium-size baskets.
- **H2:** PINN autograd Greeks have significantly lower variance than Monte Carlo Greeks.
- **H3:** Adaptive sampling lowers error near maturity relative to uniform sampling.

## Contribution
### [`Extension of existing work`], [`Novel contribution`]
We build on physics-informed neural networks (Raissi et al.), the Black-Scholes and Heston models, and Longstaff-Schwartz Monte Carlo as the baseline. We add:
- A multi-asset PDE loss module with a soft-penalty optimizer for terminal and boundary conditions.
- An adaptive collocation sampler that concentrates points near maturity and payoff kinks.
- A benchmark harness (PINN-Options package) that measures residuals, relative error, Greek variance, runtime, and VRAM against Monte Carlo and finite-difference baselines.

## Evaluation Design
| Factor | Values |
|---|---|
| Asset dimension d | 2, 5, 10, 20 |
| Sampling | uniform (baseline) vs adaptive mesh |
| Baselines | Monte Carlo (incl. Longstaff-Schwartz), finite difference (low d only) |
| Seeds | at least 5 per condition; report mean, std, and significance tests (p < 0.05) |
| Controls | volatility bounds, risk-free rate, hardware (USC Theia or Hyperion), software versions |

Metrics: relative L2 pricing error, Greek variance across domain steps, runtime per epoch, inference runtime, peak VRAM.

Planned figures (saved in `data/results/figures/`): dimension vs runtime and VRAM; price-surface comparison across d; Greek estimates with variance intervals; epoch vs loss.

## References
See [`references.bib`](references.bib).

---

## Dependencies
- Python 3.11
- Ubuntu 22.04 (record the actual OS/driver/CUDA on the cluster)
- [uv](https://docs.astral.sh/uv/) as package manager; `pyproject.toml` and `uv.lock` are included
- PyTorch, NumPy, SciPy, matplotlib; vectorbt (optional, only if used for backtesting)
- Hardware: USC Theia or Hyperion GPU node [record exact GPU]

## Directory Structure
```
|- data
|   |- market/       Yahoo Finance adjusted-close data and estimated parameters
|   |- results/        raw CSV logs and figures (*.csv, *.png)
|   |- configs/        experiment configs (seeds, d, rates, vol bounds)
|- src
|   |- pde_loss.py     multi-asset Black-Scholes / Heston residual loss
|   |- model.py        tanh MLP
|   |- sampler.py      uniform and adaptive collocation sampling
|   |- optimizer.py    soft-penalty training loop
|   |- greeks.py       autograd Greeks
|   |- baselines.py    Monte Carlo, Longstaff-Schwartz, finite difference
|   |- benchmark.py    error, variance, runtime, VRAM measurement
|- train.py
|- run.py
|- result.py
```