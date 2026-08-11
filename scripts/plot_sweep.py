"""
Aggregate + plot the name-dropout dose-response curve.

Reads the per-checkpoint probe JSONs produced for each point in the name-dropout
sweep (see docs/name_dropout_sweep.md), builds a `p x metric` table, and plots the
headline dose-response figure: name-following (should fall with p) and acoustic-only
mAP (should rise with p) against the dropout probability p.

Expected layout (produced by sweep_evaluate.py):
    reports/sweep/p0.0/probe1_name_swap.json
    reports/sweep/p0.0/probe1_heldout_expanded.json   (optional)
    reports/sweep/p0.0/probe10_acoustic_and_stats.json (optional)
    reports/sweep/p0.25/...
    ...

The reader is tolerant: it pulls each metric from whichever probe JSON is present,
warns on anything missing, and plots what it has.

Usage:
    conda activate birdclap
    python scripts/plot_sweep.py
    python scripts/plot_sweep.py --ps 0.0,0.25,0.5,0.75,1.0 --sweep-dir reports/sweep
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def _get(d: dict, *path, default=None):
    """Safe nested lookup: _get(d, 'a', 'b') -> d['a']['b'] or default."""
    cur = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _load(p: Path) -> dict:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def collect_point(pdir: Path) -> dict:
    """Pull the headline metrics for one sweep point from whatever probes exist."""
    heldout = _load(pdir / "probe1_heldout_expanded.json")
    swap    = _load(pdir / "probe1_name_swap.json")
    p10     = _load(pdir / "probe10_acoustic_and_stats.json")

    m: dict[str, float | None] = {}

    # name-following, diff-family (fraction the model follows the NAME; lower better)
    nf = _get(p10, "diff_family_name_following", "point_species_mean")
    if nf is None:
        aw = _get(heldout, "ft_diff_family", "acoustics_win_rate_mean")
        nf = (1.0 - aw) if aw is not None else None
    m["name_following_diff"] = nf

    ci = _get(p10, "diff_family_name_following", "ci95_species_clustered")
    m["name_following_ci_lo"] = ci[0] if isinstance(ci, list) and len(ci) == 2 else None
    m["name_following_ci_hi"] = ci[1] if isinstance(ci, list) and len(ci) == 2 else None

    # acoustics-win, diff-family (higher better)
    m["acoustics_win_diff"] = _get(heldout, "ft_diff_family", "acoustics_win_rate_mean")

    # acoustic-only mAP (the 0.116 -> 0.129 headline; higher better)
    amap = _get(swap, "finetuned", "summary", "acoustic", "mAP_A")
    if amap is None:
        amap = _get(heldout, "ft_diff_family", "acoustic_mAP_A")
    m["acoustic_only_map"] = amap

    return m


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sweep-dir", default="reports/sweep",
                    help="Directory holding p<val>/ subdirs of probe JSONs")
    ap.add_argument("--ps", default="0.0,0.25,0.5,0.75,1.0",
                    help="Comma-separated dropout probabilities to plot")
    ap.add_argument("--out", default="reports/figures/sweep_dose_response.png")
    ap.add_argument("--csv", default="reports/sweep/sweep_summary.csv")
    args = ap.parse_args()

    ps = [float(x) for x in args.ps.split(",")]
    sweep_dir = Path(args.sweep_dir)

    rows: list[dict] = []
    for p in ps:
        pdir = sweep_dir / f"p{p}"
        if not pdir.exists():
            print(f"  [warn] {pdir} missing — skipping p={p}")
            continue
        m = collect_point(pdir)
        m["p"] = p
        rows.append(m)
        nf = m["name_following_diff"]; am = m["acoustic_only_map"]
        print(f"  p={p:<4}  name_following={nf if nf is None else round(nf,4)}  "
              f"acoustic_mAP={am if am is None else round(am,4)}")

    if not rows:
        print("No sweep points found — run sweep_evaluate.py first.")
        return 1

    # write CSV
    cols = ["p", "name_following_diff", "name_following_ci_lo", "name_following_ci_hi",
            "acoustics_win_diff", "acoustic_only_map"]
    Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
    with open(args.csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c) for c in cols})
    print(f"-> {args.csv}")

    # plot dose-response (twin axes)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # noqa: BLE001
        print(f"  [warn] matplotlib unavailable ({e}); wrote CSV only.")
        return 0

    xs   = [r["p"] for r in rows]
    nf   = [r["name_following_diff"] for r in rows]
    amap = [r["acoustic_only_map"] for r in rows]

    fig, ax1 = plt.subplots(figsize=(6, 4))
    c1, c2 = "#c0392b", "#2471a3"
    ax1.plot(xs, nf, "o-", color=c1, label="name-following (diff-family)")
    ax1.set_xlabel("name-dropout probability  p")
    ax1.set_ylabel("name-following  (↓ better)", color=c1)
    ax1.tick_params(axis="y", labelcolor=c1)
    lo = [r.get("name_following_ci_lo") for r in rows]
    hi = [r.get("name_following_ci_hi") for r in rows]
    if all(v is not None for v in lo + hi):
        ax1.fill_between(xs, lo, hi, color=c1, alpha=0.15)

    ax2 = ax1.twinx()
    ax2.plot(xs, amap, "s--", color=c2, label="acoustic-only mAP")
    ax2.set_ylabel("acoustic-only mAP  (↑ better)", color=c2)
    ax2.tick_params(axis="y", labelcolor=c2)

    ax1.set_title("Name-dropout dose-response\n(p=0 baseline, p=1 masked)")
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200)
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
