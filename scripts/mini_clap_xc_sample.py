#!/usr/bin/env python3
"""
Mini end-to-end test: sample rows from xc_metadata_unified.csv, download MP3s from
Xeno-Canto, build text labels, run LAION CLAP (Hugging Face) and print audio–text
similarity scores.

Prerequisites:
  pip install -r requirements.txt
  pip install -r requirements-ml.txt
  ffmpeg on PATH (recommended for librosa + MP3)

Usage (from repo root):
  python scripts/mini_clap_xc_sample.py --sample 6
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import numpy as np
import requests
import torch
from tqdm import tqdm
from transformers import ClapModel, ClapProcessor

try:
    import librosa
except ImportError:
    print("Missing librosa. Run: pip install -r requirements-ml.txt", file=sys.stderr)
    raise

XC_ID_RE = re.compile(r"audio/xc/(\d+)\.", re.I)
DEFAULT_MODEL = "laion/clap-htsat-fused"
HEADERS = {
    "User-Agent": "lets-solve-it-mini-pipeline/1.0 (research; contact: team)",
}


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def resolve_csv(path: str | None) -> Path:
    if path:
        p = Path(path)
        if p.is_file():
            return p
        raise FileNotFoundError(path)
    root = repo_root()
    for candidate in (root / "scripts" / "xc_metadata_unified.csv", root / "xc_metadata_unified.csv"):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("xc_metadata_unified.csv (pass --csv)")


def xc_id_from_row(filepath: str) -> int | None:
    m = XC_ID_RE.search(str(filepath))
    return int(m.group(1)) if m else None


def download_xc_mp3(xc_id: int, dest: Path, sleep_s: float = 0.35) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://xeno-canto.org/{xc_id}/download"
    r = requests.get(url, headers=HEADERS, timeout=60)
    r.raise_for_status()
    dest.write_bytes(r.content)
    time.sleep(sleep_s)


def build_label(common_name: str, vocalization: str) -> str:
    v = "" if not vocalization or str(vocalization) == "nan" else str(vocalization).strip()
    if v:
        return f"Recording of {common_name}: {v}"
    return f"Recording of {common_name}"


def main() -> int:
    ap = argparse.ArgumentParser(description="Mini Xeno-Canto + CLAP smoke test.")
    ap.add_argument("--sample", type=int, default=6, help="Number of rows to use (default 6)")
    ap.add_argument("--csv", type=str, default=None, help="Path to xc_metadata_unified.csv")
    ap.add_argument("--out-dir", type=str, default=None, help="Download directory (default: scripts/data/xc_mini)")
    ap.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Hugging Face CLAP model id")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    args = ap.parse_args()

    import pandas as pd

    csv_path = resolve_csv(args.csv)
    out_dir = Path(args.out_dir) if args.out_dir else repo_root() / "scripts" / "data" / "xc_mini"

    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["filepath", "species_code", "common_name"])
    df["_xc_id"] = df["filepath"].map(xc_id_from_row)
    df = df.dropna(subset=["_xc_id"])
    df["_xc_id"] = df["_xc_id"].astype(int)

    if len(df) < args.sample:
        print(f"Only {len(df)} usable rows; taking all.", file=sys.stderr)
        sample = df
    else:
        sample = df.sample(n=args.sample, random_state=args.seed)

    print(f"CSV: {csv_path}")
    print(f"Downloads: {out_dir}")
    print(f"Model: {args.model}")

    paths: list[Path] = []
    texts: list[str] = []
    for _, row in tqdm(sample.iterrows(), total=len(sample), desc="download"):
        xc_id = int(row["_xc_id"])
        ext = Path(str(row["filepath"])).suffix or ".mp3"
        local = out_dir / f"{xc_id}{ext}"
        if not local.is_file():
            download_xc_mp3(xc_id, local)
        paths.append(local)
        vt = row.get("vocalization_type", "")
        texts.append(build_label(str(row["common_name"]), vt if pd.notna(vt) else ""))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    processor = ClapProcessor.from_pretrained(args.model)
    model = ClapModel.from_pretrained(args.model).to(device)
    model.eval()

    target_sr = 48000
    waveforms: list[np.ndarray] = []
    for p in paths:
        y, _ = librosa.load(str(p), sr=target_sr, mono=True)
        waveforms.append(np.asarray(y, dtype=np.float32))

    audio_embeds = []
    with torch.no_grad():
        for y in waveforms:
            inp = processor(audio=y, sampling_rate=target_sr, return_tensors="pt")
            inp = {k: v.to(device) for k, v in inp.items()}
            a = model.get_audio_features(**inp)
            audio_embeds.append(a.pooler_output)
        audio_embeds = torch.cat(audio_embeds, dim=0)

        text_inp = processor(text=texts, return_tensors="pt", padding=True)
        text_inp = {k: v.to(device) for k, v in text_inp.items()}
        text_embeds = model.get_text_features(**text_inp).pooler_output

    # pooler_output is already L2-normalized inside get_* for this model
    sims = (audio_embeds * text_embeds).sum(dim=-1)

    print("\n--- Per-file cosine similarity (audio vs its caption) ---")
    for p, t, s in zip(paths, texts, sims.cpu().tolist()):
        print(f"\n{s:+.4f}  {p.name}\n  text: {t[:120]}{'...' if len(t) > 120 else ''}")

    # Contrast with wrong pairing: shift text by 1
    wrong_text = texts[1:] + texts[:1]
    with torch.no_grad():
        w_inp = processor(text=wrong_text, return_tensors="pt", padding=True)
        w_inp = {k: v.to(device) for k, v in w_inp.items()}
        w_embeds = model.get_text_features(**w_inp).pooler_output
    wrong_sims = (audio_embeds * w_embeds).sum(dim=-1)
    print("\n--- Same audio vs *shifted* wrong caption (with more files, matched pairs usually score higher on average) ---")
    for p, s_right, s_wrong in zip(paths, sims.cpu().tolist(), wrong_sims.cpu().tolist()):
        print(f"{p.name}: matched={s_right:+.4f}  mismatched={s_wrong:+.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
