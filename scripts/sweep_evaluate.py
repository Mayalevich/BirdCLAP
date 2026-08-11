"""
Run the metric probes on every checkpoint of the name-dropout sweep and collect
their JSON outputs into per-p folders, ready for plot_sweep.py.

The probes (probe1_name_swap.py, probe10_acoustic_and_stats.py, ...) each take a
`--checkpoint` and write a fixed JSON under reports/. This orchestrator runs them
per checkpoint and moves each output into reports/sweep/p<val>/ so nothing is
overwritten across sweep points.

Adjust PROBES and --ckpt-template to match your checkpoint filenames and the exact
probe set you want reported. Whoever holds the checkpoints runs this.

Usage:
    conda activate birdclap
    python scripts/sweep_evaluate.py \
        --ps 0.0,0.25,0.5,0.75,1.0 \
        --ckpt-template "checkpoints/sweep_p{p}/best.pt"
    python scripts/plot_sweep.py     # then plot the dose-response curve
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

# script name -> the reports/<file> it writes. Confirm/extend for your eval set.
PROBES: dict[str, str] = {
    "probe1_name_swap.py":          "probe1_name_swap.json",
    "probe10_acoustic_and_stats.py": "probe10_acoustic_and_stats.json",
    "probe8_audio_to_text.py":       "probe8_audio_to_text.json",
    # "probe1_name_swap.py --heldout-expanded": "probe1_heldout_expanded.json",
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ps", default="0.0,0.25,0.5,0.75,1.0")
    ap.add_argument("--ckpt-template", default="checkpoints/sweep_p{p}/best.pt",
                    help="Checkpoint path per p; {p} is substituted")
    ap.add_argument("--reports-dir", default="reports")
    ap.add_argument("--out-dir", default="reports/sweep")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print the commands without running the probes")
    args = ap.parse_args()

    ps = [x.strip() for x in args.ps.split(",")]
    reports = Path(args.reports_dir)

    for p in ps:
        ckpt = Path(args.ckpt_template.format(p=p))
        dest = Path(args.out_dir) / f"p{p}"
        dest.mkdir(parents=True, exist_ok=True)
        if not ckpt.exists() and not args.dry_run:
            print(f"  [warn] checkpoint {ckpt} not found — skipping p={p}")
            continue
        print(f"== p={p}  checkpoint={ckpt} ==")
        for script, out_json in PROBES.items():
            parts = script.split()
            cmd = [sys.executable, f"scripts/{parts[0]}", *parts[1:],
                   "--checkpoint", str(ckpt)]
            print("   $", " ".join(cmd))
            if args.dry_run:
                continue
            r = subprocess.run(cmd)
            if r.returncode != 0:
                print(f"   [warn] {script} exited {r.returncode}; continuing")
                continue
            produced = reports / out_json
            if produced.exists():
                shutil.move(str(produced), str(dest / out_json))
                print(f"   -> {dest / out_json}")
            else:
                print(f"   [warn] expected {produced} not written by {script}")
    print(f"\nDone. Now: python scripts/plot_sweep.py --ps {args.ps} --sweep-dir {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
