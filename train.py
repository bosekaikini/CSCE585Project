"""Train one PINN configuration and append measured metrics to the results CSV."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from src.baselines import black_scholes_call, monte_carlo_call, finite_difference_call
from src.benchmark import relative_l2
from src.greeks import greeks
from src.market_data import estimate_market_parameters
from src.model import PINN
from src.optimizer import train as train_pinn
from src.sampler import adaptive_sampler, uniform_sampler


ROOT = Path(__file__).resolve().parent


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _payoff(strike: float, smooth: bool = False, width: float = 0.01):
    def payoff(s: torch.Tensor) -> torch.Tensor:
        basket = s.mean(dim=1, keepdim=True)
        if smooth:
            return width * torch.nn.functional.softplus((basket - strike) / width)
        return torch.relu(basket - strike)
    return payoff


def _measure_inference_latency(model, x, repeats: int = 20, warmup: int = 5):
    if not torch.is_tensor(x):
        raise TypeError("x must be a torch tensor")
    model.eval()
    if x.device.type == "cuda":
        torch.cuda.synchronize(x.device)
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        samples = []
        for _ in range(repeats):
            start = time.perf_counter()
            model(x)
            if x.device.type == "cuda":
                torch.cuda.synchronize(x.device)
            samples.append(time.perf_counter() - start)
    return float(np.median(samples))


def run_training(config: dict, seed: int, epochs: int | None = None) -> tuple[dict, list[dict]]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = int(config["dimension"])
    market = _market_parameters(config)
    strike = float(config["strike"])
    maturity = float(config["maturity"])
    spot = float(market["spot"].mean())
    volatility = float(market["volatility"].mean())
    sampler_fn = adaptive_sampler if config["sampling"] == "adaptive" else uniform_sampler
    payoff = _payoff(strike, bool(config.get("smooth_payoff", False)))
    model = PINN(d + 1, int(config["width"]), int(config["depth"])).to(device)
    n_epochs = int(config.get("epochs", 1000) if epochs is None else epochs)
    n_points = int(config.get("collocation_points", 2048))

    def interior():
        return sampler_fn(
            n_points, d, strike=strike, spot_bounds=(0.01 * spot, 2.0 * spot),
            maturity=maturity, device=device
        ) if config["sampling"] == "adaptive" else sampler_fn(
            n_points, d, spot_bounds=(0.01 * spot, 2.0 * spot),
            maturity=maturity, device=device
        )

    def terminal():
        return (0.01 * spot + (1.99 * spot) * torch.rand(
            n_points, d, device=device
        )).requires_grad_(True)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    fit = train_pinn(
        model, interior, terminal, payoff, epochs=n_epochs,
        lr=float(config["learning_rate"]), rate=float(config["rate"]),
        volatility=volatility,
    )
    elapsed = time.perf_counter() - start

    eval_spots = torch.full((64, d), spot, device=device)
    eval_x = torch.cat(
        [eval_spots, torch.full((64, 1), maturity * 0.5, device=device)], dim=1
    ).requires_grad_(True)
    with torch.no_grad():
        prediction = model(eval_x).detach().cpu().numpy().reshape(-1)
    inference_seconds = _measure_inference_latency(model, eval_x)
    mc_start = time.perf_counter()
    monte_carlo_call(spot, strike, maturity * 0.5, float(config["rate"]),
                     volatility, paths=10000, seed=seed)
    mc_runtime = time.perf_counter() - mc_start
    fd_runtime = 0.0
    if d <= 2:
        fd_start = time.perf_counter()
        finite_difference_call(spot, strike, maturity * 0.5, float(config["rate"]),
                               volatility)
        fd_runtime = time.perf_counter() - fd_start
    reference = np.asarray(black_scholes_call(
        spot, strike, maturity * 0.5, float(config["rate"]), volatility
    )).reshape(-1)
    greek_values = greeks(model, eval_x)
    row = {
        "config": config["name"], "dimension": d, "sampling": config["sampling"],
        "seed": seed, "config_hash": hashlib.sha256(
            json.dumps(config, sort_keys=True).encode()
        ).hexdigest()[:12],
        "relative_l2": relative_l2(prediction, np.full_like(prediction, reference.item())),
        "runtime_seconds": elapsed, "runtime_per_epoch_seconds": elapsed / max(n_epochs, 1),
        "inference_seconds": inference_seconds,
        "mc_runtime_seconds": mc_runtime, "fd_runtime_seconds": fd_runtime,
        "peak_vram_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
        "greek_delta_mean": float(greek_values["delta"].mean().cpu()),
        "greek_gamma_mean": float(greek_values["gamma"].mean().cpu()),
        "final_loss": float(fit["loss"][-1]),
        "python_version": platform.python_version(), "torch_version": torch.__version__,
        "numpy_version": np.__version__, "scipy_version": _scipy_version(),
        "cuda_version": str(torch.version.cuda or "none"),
        "cuda_available": torch.cuda.is_available(), "platform": platform.platform(),
        "git_commit": _git_commit(),
    }
    history = [
        {"config": config["name"], "dimension": d, "sampling": config["sampling"],
         "seed": seed, "epoch": i + 1, "loss": value}
        for i, value in enumerate(fit["loss"])
    ]
    return row, history


def _market_parameters(config: dict):
    """Load configured Yahoo-derived parameters, failing if inputs are stale/missing."""
    path = ROOT / config["market_data"]
    if not path.exists():
        raise FileNotFoundError(
            f"Market parameters not found at {path}; run download_data.py first."
        )
    import pandas as pd
    parameters = pd.read_csv(path).set_index("ticker")
    missing = [ticker for ticker in config["tickers"] if ticker not in parameters.index]
    if missing:
        raise ValueError(f"Market parameters are missing tickers: {', '.join(missing)}")
    return parameters.loc[config["tickers"]]


def _scipy_version() -> str:
    try:
        import scipy
        return scipy.__version__
    except ImportError:
        return "unavailable"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--epochs", type=int)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    seeds = [args.seed] if args.seed is not None else config["seeds"]
    result_dir = ROOT / "data" / "results"
    result_dir.mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        row, history = run_training(config, seed, args.epochs)
        result_path = result_dir / "metrics.csv"
        with result_path.open("a", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            if handle.tell() == 0:
                writer.writeheader()
            writer.writerow(row)
        history_path = result_dir / "loss_history.csv"
        with history_path.open("a", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(history[0]))
            if handle.tell() == 0:
                writer.writeheader()
            writer.writerows(history)
        print(json.dumps(row, indent=2))


if __name__ == "__main__":
    main()
