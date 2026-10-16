"""Summarize a benchmark results.csv per oracle (and optionally per factor).

Usage: py multi_payload/benchmark/analyze.py <results.csv> [--by M layout nu]
"""
import argparse

import numpy as np
import pandas as pd


def summarize(df, by=("oracle",)):
    def agg(g):
        ok = g[g["status"].isin(["ok", "slack"])]
        return pd.Series({
            "tasks": len(g),
            "success": f"{int(g['success'].sum())}/{len(g)}",
            "status": ", ".join(f"{k}:{v}" for k, v in g["status"].value_counts().items()),
            "repair_converged": _rate(g, "info_repair_converged"),
            "repair_iters_med": _med(g, "info_repair_iterations"),
            "repair_iters_max": _max(g, "info_repair_iterations"),
            "recurring_conflicts": _sum(g, "info_repair_recurring_conflicts"),
            "oscillations": _sum(g, "info_repair_oscillation"),
            "corridor_expansions": _sum(g, "info_corridor_expansions"),
            "re_repairs": _sum(g, "info_re_repairs"),
            "min_sep/delta_min": _min_ratio(ok),
            "max_V": _max(ok, "max_V"),
            "gap_ratio_med": _med(ok, "gap_ratio"),
            "J_final_le_J_hat": _rate(ok, "certificate_ok"),
            "oracle_time_med_s": _med(g, "oracle_time"),
        })
    return df.groupby(list(by)).apply(agg, include_groups=False)


def _col(g, c):
    return pd.to_numeric(g[c], errors="coerce").dropna() if c in g else pd.Series(dtype=float)


def _rate(g, c):
    s = g[c].dropna() if c in g else []
    return f"{int(np.sum(s.astype(bool)))}/{len(s)}" if len(s) else "-"


def _med(g, c):
    s = _col(g, c)
    return round(float(s.median()), 3) if len(s) else np.nan


def _max(g, c):
    s = _col(g, c)
    return float(s.max()) if len(s) else np.nan


def _sum(g, c):
    s = _col(g, c)
    return int(s.sum()) if len(s) else 0


def _min_ratio(ok):
    if not len(ok) or "min_separation" not in ok:
        return np.nan
    return round(float((ok["min_separation"] / ok["delta"]).min()), 3)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("csv")
    p.add_argument("--by", nargs="+", default=["oracle"])
    a = p.parse_args()
    df = pd.read_csv(a.csv)
    pd.set_option("display.width", 250, "display.max_columns", 40)
    print(summarize(df, a.by).T.to_string())
    bad = df[~df["success"].astype(bool)]
    if len(bad):
        cols = [c for c in ("oracle", "layout", "M", "nu", "seed", "status", "error") if c in bad]
        print("\nunsuccessful tasks:\n" + bad[cols].to_string(index=False))


if __name__ == "__main__":
    main()
