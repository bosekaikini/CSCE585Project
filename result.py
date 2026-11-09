"""Create result figures from measured CSV files; never invents missing data."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.benchmark import bootstrap_confidence_interval, paired_difference_summary

ROOT = Path(__file__).resolve().parent


def _require(path: Path):
    if not path.exists():
        raise SystemExit(
            f"Results are missing: {path}. Run `python run.py` after the real experiments."
        )
    frame = pd.read_csv(path)
    if frame.empty:
        raise SystemExit(f"Results are empty: {path}. No figures were generated.")
    return frame.drop_duplicates(subset=["config", "dimension", "sampling", "seed"], keep="last")


def _summarize_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    records = []
    for (dimension, sampling), group in metrics.groupby(["dimension", "sampling"], sort=True):
        relative = bootstrap_confidence_interval(group["relative_l2"].to_numpy(dtype=float), seed=0)
        runtime = bootstrap_confidence_interval(group["runtime_seconds"].to_numpy(dtype=float), seed=1)
        inference = bootstrap_confidence_interval(group["inference_seconds"].to_numpy(dtype=float), seed=2)
        pair = group.groupby("config", sort=True).agg(
            relative_l2_mean=("relative_l2", "mean"),
            runtime_mean=("runtime_seconds", "mean"),
        )
        records.append({
            "dimension": int(dimension),
            "sampling": sampling,
            "relative_l2_mean": float(group["relative_l2"].mean()),
            "relative_l2_ci_low": float(relative["lower"]),
            "relative_l2_ci_high": float(relative["upper"]),
            "runtime_mean": float(group["runtime_seconds"].mean()),
            "runtime_ci_low": float(runtime["lower"]),
            "runtime_ci_high": float(runtime["upper"]),
            "inference_mean": float(group["inference_seconds"].mean()),
            "inference_ci_low": float(inference["lower"]),
            "inference_ci_high": float(inference["upper"]),
            "seeds": int(group["seed"].nunique()),
            "configs": sorted(pair.index.tolist()),
        })
    return pd.DataFrame.from_records(records)


def print_summary(metrics: pd.DataFrame) -> None:
    summary = _summarize_metrics(metrics)
    print("Dimension-by-sampling summary with bootstrap 95% CI")
    for row in summary.itertuples(index=False):
        print(
            f"d={row.dimension} {row.sampling:>8}: "
            f"relative_l2={row.relative_l2_mean:.4f} [{row.relative_l2_ci_low:.4f}, {row.relative_l2_ci_high:.4f}], "
            f"runtime={row.runtime_mean:.3f}s [{row.runtime_ci_low:.3f}, {row.runtime_ci_high:.3f}], "
            f"inference={row.inference_mean:.6f}s [{row.inference_ci_low:.6f}, {row.inference_ci_high:.6f}]"
        )


def make_figures() -> None:
    metrics = _require(ROOT / "data" / "results" / "metrics.csv")
    history = _require(ROOT / "data" / "results" / "loss_history.csv")
    import matplotlib.pyplot as plt

    out = ROOT / "data" / "results" / "figures"
    out.mkdir(parents=True, exist_ok=True)

    grouped = metrics.groupby(["dimension", "sampling"], as_index=False).mean(numeric_only=True)
    fig, ax1 = plt.subplots()
    for label, group in grouped.groupby("sampling"):
        ax1.plot(group["dimension"], group["runtime_seconds"], marker="o", label=f"PINN {label}")
    if "mc_runtime_seconds" in grouped:
        ax1.plot(grouped["dimension"], grouped["mc_runtime_seconds"], "k:", marker="s",
                 label="MC baseline")
    if "fd_runtime_seconds" in grouped and grouped["fd_runtime_seconds"].any():
        ax1.plot(grouped["dimension"], grouped["fd_runtime_seconds"], "k--", marker="^",
                 label="finite-difference baseline")
    ax1.set_xlabel("Asset dimension d (count)")
    ax1.set_ylabel("Runtime (seconds)")
    ax2 = ax1.twinx()
    for label, group in grouped.groupby("sampling"):
        ax2.plot(group["dimension"], group["peak_vram_bytes"] / 2**20, "--", marker="x",
                 label=f"VRAM {label}")
    ax2.set_ylabel("Peak VRAM (MiB)")
    ax1.set_title("Measured runtime and VRAM versus dimension")
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2)
    fig.tight_layout()
    fig.savefig(out / "dimension_runtime_vram.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots()
    for label, group in metrics.groupby("sampling"):
        ax.plot(group["dimension"], group["relative_l2"], marker="o", label=f"PINN {label}")
    ax.set_xlabel("Asset dimension d (count)")
    ax.set_ylabel("Relative L2 pricing error (dimensionless)")
    ax.set_title("Measured PINN pricing error across dimensions")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "price_surface_comparison.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots()
    summary = metrics.groupby("dimension")["greek_delta_mean"]
    ax.errorbar(summary.mean().index, summary.mean().values,
                yerr=summary.std().fillna(0).values, fmt="o-", capsize=4,
                label="PINN delta across seeds")
    ax.set_xlabel("Asset dimension d (count)")
    ax.set_ylabel("Delta estimate (dimensionless)")
    ax.set_title("Measured Greek estimates with across-seed variance intervals")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "greek_variance.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots()
    for label, group in history.groupby("sampling"):
        ax.plot(group["epoch"], group["loss"], label=f"PINN {label}")
    ax.set_xlabel("Epoch (count)")
    ax.set_ylabel("Total training loss (dimensionless)")
    ax.set_title("Measured training loss for uniform and adaptive sampling")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "loss_curves.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", action="store_true", help="print dimension-level statistical summaries")
    args = parser.parse_args()
    metrics = _require(ROOT / "data" / "results" / "metrics.csv")
    if args.summary:
        print_summary(metrics)
    else:
        make_figures()
