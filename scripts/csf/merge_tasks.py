#!/usr/bin/env python3
"""Merge all task CSVs from job array into one file and print summary.
Usage:
    python merge_tasks.py --tasks-dir results/tasks --output results/sensitivity_merged.csv
"""
from __future__ import annotations
import argparse
from datetime import datetime
from pathlib import Path
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tasks-dir", type=Path, default=Path("results/tasks"))
    p.add_argument("--output",    type=Path, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    csvs = sorted(args.tasks_dir.glob("task_*.csv"))
    if not csvs:
        print(f"No task CSVs found in {args.tasks_dir}")
        return

    print(f"Found {len(csvs)} task files", flush=True)
    merged = pd.concat([pd.read_csv(f) for f in csvs], ignore_index=True)

    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = args.output or Path(f"results/sensitivity_merged_{ts}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out, index=False)

    valid = merged.dropna(subset=["cost"])
    valid = valid[valid["cost"] > 0]

    print(f"\nTotal rows: {len(merged)}, valid: {len(valid)}, "
          f"failed: {len(merged) - len(valid)}")
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()