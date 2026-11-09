"""Benchmark helpers that report measured, rather than fabricated, metrics."""
from __future__ import annotations

import time
from typing import Callable

import numpy as np


def relative_l2(prediction, reference, eps=1e-12):
    p, r = np.asarray(prediction), np.asarray(reference)
    return float(np.linalg.norm(p - r) / max(np.linalg.norm(r), eps))


def benchmark_model(model, inputs, reference=None):
    start = time.perf_counter()
    result = model(inputs)
    elapsed = time.perf_counter() - start
    out = {"runtime_seconds": elapsed, "prediction": result.detach().cpu().numpy()}
    if reference is not None:
        out["relative_l2"] = relative_l2(out["prediction"], reference)
    return out


def summarize(values):
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        raise ValueError("values must not be empty")
    return {
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
        "min": float(values.min()),
        "max": float(values.max()),
    }


def bootstrap_confidence_interval(
    values,
    *,
    confidence: float = 0.95,
    iterations: int = 2000,
    seed: int = 0,
    statistic: Callable[[np.ndarray], float] | None = None,
):
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        raise ValueError("values must not be empty")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1")
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    stat = statistic or np.mean
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(iterations, values.size), replace=True)
    replicated = np.array([float(stat(draw)) for draw in draws], dtype=float)
    alpha = 1.0 - confidence
    lower, upper = np.quantile(replicated, [alpha / 2.0, 1.0 - alpha / 2.0])
    return {
        "statistic": float(stat(values)),
        "lower": float(lower),
        "upper": float(upper),
        "confidence": confidence,
        "iterations": int(iterations),
    }


def paired_difference_summary(
    left,
    right,
    *,
    confidence: float = 0.95,
    iterations: int = 2000,
    seed: int = 0,
):
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    if left.shape != right.shape:
        raise ValueError("left and right arrays must have identical shapes")
    diffs = right - left
    ci = bootstrap_confidence_interval(
        diffs,
        confidence=confidence,
        iterations=iterations,
        seed=seed,
        statistic=np.mean,
    )
    ci["mean_difference"] = float(np.mean(diffs))
    ci["baseline_mean"] = float(np.mean(left))
    ci["candidate_mean"] = float(np.mean(right))
    return ci
