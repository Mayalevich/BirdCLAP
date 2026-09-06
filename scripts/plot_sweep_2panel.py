"""
Two-panel dose-response figure for the paper: seen species (left) and unseen
holdout species (right), each with name-following (↓, left axis) and acoustic-only
mAP (↑, right axis) against the name-dropout probability p.

Reads the aggregated CSVs written by plot_sweep.py:
    results/sweep_summary.csv          (seen)
    results/sweep_summary_holdout.csv  (unseen)

Both must have columns: p, name_following_diff, name_following_ci_lo,
name_following_ci_hi, acoustic_only_map.

Usage:
    python scripts/plot_sweep_2panel.py
    python scripts/plot_sweep_2panel.py --seen results/sweep_summary.csv \
        --holdout results/sweep_summary_holdout.csv --out results/figures/dose_response_2panel.png
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

NF, MAP = "#c0392b", "#2471a3"   # name-following red, acoustic-mAP blue


def read_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k, v in list(r.items()):
            r[k] = float(v) if v not in ("", None) else None
    return sorted(rows, key=lambda r: r["p"])


def _pct(rows, key):
    """Return values as percentages; mAP stays as-is."""
    return [r[key] for r in rows]


def draw_panel(ax, rows, title, show_ylabel_left, show_ylabel_right):
    xs = [r["p"] for r in rows]
    nf = _pct(rows, "name_following_diff")
    lo = [r.get("name_following_ci_lo") for r in rows]
    hi = [r.get("name_following_ci_hi") for r in rows]
    amap = [r["acoustic_only_map"] for r in rows]

    ln1, = ax.plot(xs, nf, "o-", color=NF, label="name-following (diff-family)")
    if all(v is not None for v in lo + hi):
        ax.fill_between(xs, lo, hi, color=NF, alpha=0.15)
    ax.set_xlabel("name-dropout probability $p$")
    if show_ylabel_left:
        ax.set_ylabel("name-following %  (↓ better)", color=NF)
    ax.tick_params(axis="y", labelcolor=NF)
    ax.set_title(title)

    ax2 = ax.twinx()
    ln2, = ax2.plot(xs, amap, "s--", color=MAP, label="acoustic-only mAP")
    if show_ylabel_right:
        ax2.set_ylabel("acoustic-only mAP  (↑ better)", color=MAP)
    ax2.tick_params(axis="y", labelcolor=MAP)

    # mark the acoustic-mAP peak
    pk = max(range(len(amap)), key=lambda i: amap[i])
    ax2.annotate(f"peak $p$={xs[pk]:g}", (xs[pk], amap[pk]),
                 textcoords="offset points", xytext=(0, 8),
                 fontsize=8, color=MAP, ha="center")
    return ln1, ln2, ax2


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seen", default="results/sweep_summary.csv")
    ap.add_argument("--holdout", default="results/sweep_summary_holdout.csv")
    ap.add_argument("--out", default="results/figures/dose_response_2panel.png")
    args = ap.parse_args()

    seen = read_csv(Path(args.seen))
    hold = read_csv(Path(args.holdout))

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(9.5, 3.8), sharex=True)
    ln1, ln2, _ = draw_panel(axL, seen, "Seen species (val, 316 sp.)",
                             show_ylabel_left=True, show_ylabel_right=False)
    draw_panel(axR, hold, "Unseen species (holdout, 80 sp.)",
               show_ylabel_left=False, show_ylabel_right=True)

    fig.legend([ln1, ln2], [ln1.get_label(), ln2.get_label()],
               loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.06))
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
