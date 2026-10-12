"""Run or preview the complete dimension x sampler x seed experiment sweep."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from train import run_training


ROOT = Path(__file__).resolve().parent


def configs() -> list[Path]:
    return sorted((ROOT / "data" / "configs").glob("*.json"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--epochs", type=int, help="override epochs for a smoke run")
    args = parser.parse_args()
    paths = configs()
    plan = [
        (path, seed)
        for path in paths
        for seed in json.loads(path.read_text())["seeds"]
    ]
    print(f"Experiment plan: {len(plan)} runs")
    for path, seed in plan:
        print(f"- {path.relative_to(ROOT)} seed={seed}")
    if args.dry_run:
        return
    for path, seed in plan:
        config = json.loads(path.read_text())
        row, history = run_training(config, seed, args.epochs)
        result_dir = ROOT / "data" / "results"
        result_dir.mkdir(parents=True, exist_ok=True)
        import csv
        for filename, records in [("metrics.csv", [row]), ("loss_history.csv", history)]:
            target = result_dir / filename
            with target.open("a", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(records[0]))
                if handle.tell() == 0:
                    writer.writeheader()
                writer.writerows(records)


if __name__ == "__main__":
    main()
