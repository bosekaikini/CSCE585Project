"""Run or preview the complete dimension x sampler x seed experiment sweep."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from train import run_training


ROOT = Path(__file__).resolve().parent


def configs() -> list[Path]:
    return sorted((ROOT / "data" / "configs").glob("*.json"))


def _reset_result_files(result_dir: Path):
    for filename in ("metrics.csv", "loss_history.csv"):
        target = result_dir / filename
        if target.exists():
            target.unlink()


def _seen_keys(path: Path, key_fields):
    if not path.exists():
        return set()
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        return {
            tuple(row[field] for field in key_fields)
            for row in reader
            if all(field in row for field in key_fields)
        }


def _write_unique_rows(path: Path, rows: list[dict], key_fields: tuple[str, ...]):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    seen = _seen_keys(path, key_fields)
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        if handle.tell() == 0:
            writer.writeheader()
        for row in rows:
            key = tuple(str(row[field]) for field in key_fields)
            if key in seen:
                continue
            writer.writerow(row)
            seen.add(key)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--epochs", type=int, help="override epochs for a smoke run")
    parser.add_argument("--reset", action="store_true", help="clear existing CSV outputs before the run")
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
    result_dir = ROOT / "data" / "results"
    result_dir.mkdir(parents=True, exist_ok=True)
    if args.reset:
        _reset_result_files(result_dir)
    for path, seed in plan:
        config = json.loads(path.read_text())
        row, history = run_training(config, seed, args.epochs)
        _write_unique_rows(result_dir / "metrics.csv", [row], ("config", "dimension", "sampling", "seed"))
        _write_unique_rows(
            result_dir / "loss_history.csv",
            history,
            ("config", "dimension", "sampling", "seed", "epoch"),
        )


if __name__ == "__main__":
    main()
