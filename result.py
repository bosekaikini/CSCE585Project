"""Create result figures from measured CSV files; never invents missing data."""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent


def _require(path: Path):
    if not path.exists():
        raise SystemExit(
            f"Results are missing: {path}. Run `python run.py` after the real experiments."
        )
    frame = pd.read_csv(path)
    if frame.empty:
        raise SystemExit(f"Results are empty: {path}. No figures were generated.")
    return frame


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
    parser.parse_args()
    make_figures()
