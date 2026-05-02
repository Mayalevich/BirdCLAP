#!/usr/bin/env python3
"""
Pre-compute ClapFeatureExtractor (mel-spectrogram) features for every audio
file referenced in the training and validation pair JSONs.

WHY
---
ClapProcessor takes ~14-15 s to process a batch of 8 samples on CPU.  With
4 DataLoader workers each running the processor, effective throughput is still
~3.5 s/batch — the dominant bottleneck regardless of disk speed.

Pre-computing stores the mel tensor once as a small sidecar file:
    scripts/data/xc_audio/audio/xc/1234567.clap.pt  (~256 KB each)

train_clap.py auto-detects these files and uses ClapPrecomputedDataset, which
loads the cached tensor and only tokenises the text string (cheap), reducing
wall time per batch from ~3.5 s to ~0.1 s.

Usage (from repo root):
    python scripts/precompute_clap_features.py
    python scripts/precompute_clap_features.py --workers 6
    python scripts/precompute_clap_features.py --force      # re-compute all
    python scripts/precompute_clap_features.py --dry-run    # report only
    python scripts/precompute_clap_features.py --audio-root D:/custom/path
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import ClapProcessor

try:
    import soundfile as sf
except ImportError:
    print("Missing soundfile.  Run: pip install -r requirements-ml.txt", file=sys.stderr)
    raise

try:
    import librosa
except ImportError:
    print("Missing librosa.  Run: pip install -r requirements-ml.txt", file=sys.stderr)
    raise

# ── constants (must match train_clap.py exactly) ───────────────────────────────
DEFAULT_MODEL    = "laion/clap-htsat-fused"
TARGET_SR        = 48_000
CLIP_DURATION_S  = 10.0
MIN_DURATION_S   = 0.5
DEFAULT_K_CROPS  = 4


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def load_audio(path: Path, sr: int = TARGET_SR, clip_s: float = CLIP_DURATION_S) -> np.ndarray | None:
    """Identical fast-path logic as train_clap.py."""
    target_len = int(clip_s * sr)
    min_len    = int(MIN_DURATION_S * sr)

    wav_path = path.with_suffix(".wav")
    if wav_path.is_file():
        try:
            y, file_sr = sf.read(str(wav_path), dtype="float32", always_2d=False)
            if file_sr == sr and len(y) == target_len:
                return y
            if len(y) < min_len:
                return None
            if len(y) >= target_len:
                start = (len(y) - target_len) // 2
                y = y[start : start + target_len]
            else:
                y = np.pad(y, (0, target_len - len(y)))
            return y.astype(np.float32)
        except Exception:
            pass

    try:
        y, _ = librosa.load(str(path), sr=sr, mono=True)
    except Exception:
        return None

    if len(y) < min_len:
        return None
    if len(y) >= target_len:
        start = (len(y) - target_len) // 2
        y = y[start : start + target_len]
    else:
        y = np.pad(y, (0, target_len - len(y)))
    return y.astype(np.float32)


def random_crop_starts(n_total: int, target_len: int, k: int) -> list[int]:
    """Evenly spaced crop starts (beginning, thirds, end) — max K positions."""
    if n_total <= target_len:
        return [0] * k
    max_start = n_total - target_len
    if k == 1:
        anchors = [max_start // 2]
    elif k >= 4:
        anchors = [0, max_start // 3, (2 * max_start) // 3, max_start]
    else:
        anchors = [int(round(i * max_start / (k - 1))) for i in range(k)]
    return anchors[:k]


def load_audio_full(path: Path, sr: int = TARGET_SR) -> np.ndarray | None:
    """Load entire waveform at target SR (mono), preferring WAV if longer than clipped."""
    target_len = int(CLIP_DURATION_S * sr)
    min_len = int(MIN_DURATION_S * sr)
    wav_path = path.with_suffix(".wav")

    def _finalize(y_arr: np.ndarray) -> np.ndarray | None:
        if len(y_arr) < min_len:
            return None
        return y_arr.astype(np.float32)

    if wav_path.is_file():
        try:
            y_w, file_sr = sf.read(str(wav_path), dtype="float32", always_2d=False)
            if file_sr != sr:
                try:
                    y_w = librosa.resample(y_w, orig_sr=file_sr, target_sr=sr)
                except Exception:
                    pass
            y_w = np.asarray(y_w, dtype=np.float32).reshape(-1)
            if len(y_w) >= target_len:
                return _finalize(y_w)
        except Exception:
            pass

    try:
        y, _ = librosa.load(str(path), sr=sr, mono=True)
    except Exception:
        return None
    return _finalize(np.asarray(y, dtype=np.float32).reshape(-1))


def crop_and_extract(
    feature_extractor,
    y_full: np.ndarray,
    target_len: int,
    starts: list[int],
    k_use: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    crops = []
    longer_flags = []
    for si in starts[:k_use]:
        seg = (
            np.pad(y_full, (0, target_len - len(y_full)), mode="constant")
            if len(y_full) < target_len
            else y_full[si : si + target_len]
        )
        if len(seg) < target_len:
            seg = np.pad(seg, (0, target_len - len(seg)))
        feats = feature_extractor(
            raw_speech=[seg],
            sampling_rate=TARGET_SR,
            return_tensors="pt",
        )
        crops.append(feats["input_features"].cpu())
        longer_flags.append(feats["is_longer"].cpu())
    stacked = torch.cat(crops, dim=0)
    iso = torch.cat(longer_flags, dim=0)
    return stacked, iso


def compute_one(
    mp3_path: Path,
    feature_extractor,
    force: bool,
    k_crops: int,
) -> tuple[str, str]:
    """
    Compute and save .clap.pt for a single audio file.
    Returns (path_str, status) where status is 'ok', 'skip', or 'error:<msg>'.
    """
    out_path = mp3_path.with_suffix(".clap.pt")
    if out_path.is_file() and not force:
        return str(mp3_path), "skip"

    target_len = int(CLIP_DURATION_S * TARGET_SR)
    y_full = load_audio_full(mp3_path)
    if y_full is None:
        return str(mp3_path), "error:load_failed"

    starts = random_crop_starts(len(y_full), target_len, k_crops)

    try:
        stacked, iso = crop_and_extract(
            feature_extractor, y_full, target_len, starts, k_crops
        )
        payload = {
            "input_features": stacked,
            "is_longer":      iso,
            "n_crops":        k_crops,
        }
        torch.save(payload, str(out_path))
    except Exception as exc:
        return str(mp3_path), f"error:{exc}"

    return str(mp3_path), "ok"


def main() -> int:
    root = repo_root()

    ap = argparse.ArgumentParser(
        description="Pre-compute CLAP mel features for all training audio files."
    )
    ap.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"HuggingFace ClapModel id (default: {DEFAULT_MODEL})",
    )
    ap.add_argument(
        "--audio-root",
        default=str(root / "scripts" / "data" / "xc_audio"),
        help="Root containing audio/xc/*.mp3 (default: scripts/data/xc_audio)",
    )
    ap.add_argument(
        "--train-pairs",
        default=str(root / "data" / "clap_train_pairs.json"),
    )
    ap.add_argument(
        "--val-pairs",
        default=str(root / "data" / "clap_val_pairs.json"),
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=6,
        help="Parallel threads (I/O + numpy, GIL released; default 6)",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="Re-compute even if .clap.pt already exists",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would be done without writing files",
    )
    ap.add_argument(
        "--k-crops",
        type=int,
        default=DEFAULT_K_CROPS,
        metavar="K",
        help=(
            "Number of 10 s mel crops stacked per recording (default 4: start, thirds, end). "
            "Training randomly picks one per epoch."
        ),
    )
    args = ap.parse_args()

    audio_root = Path(args.audio_root)

    # Collect unique audio paths from both pair files
    unique_paths: set[Path] = set()
    for json_path in (args.train_pairs, args.val_pairs):
        pairs = json.loads(Path(json_path).read_text(encoding="utf-8"))
        for p in pairs:
            fp = audio_root / p["audio"]
            if fp.is_file():
                unique_paths.add(fp)

    paths = sorted(unique_paths)
    total   = len(paths)
    already = sum(1 for p in paths if p.with_suffix(".clap.pt").is_file())
    to_do   = total if args.force else (total - already)
    est_gb = to_do * max(1, args.k_crops) * 280 / (1024 ** 3)  # ~280 KB tensor per crop

    print(f"Audio root      : {audio_root}")
    print(f"Unique audio    : {total:,}")
    print(f".clap.pt exists : {already:,}")
    print(f"To compute      : {to_do:,}")
    print(f"Est. disk usage : ~{est_gb:.1f} GB of new .clap.pt files")
    print(f"Workers         : {args.workers}")
    print(f"Est. time       : ~{to_do * 1.85 / args.workers / 60:.0f} min "
          f"(based on ~1.85 s/file/thread)\n")

    if args.dry_run:
        print("Dry-run — no files written.")
        return 0

    print("Loading ClapProcessor feature extractor ...")
    processor        = ClapProcessor.from_pretrained(args.model)
    feature_extractor = processor.feature_extractor
    print("Feature extractor loaded.\n")

    n_ok = n_skip = n_err = 0
    errors: list[str] = []
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {
            pool.submit(compute_one, p, feature_extractor, args.force, args.k_crops): p
            for p in paths
        }
        pbar = tqdm(
            as_completed(futs),
            total=total,
            unit="file",
            dynamic_ncols=True,
        )
        for fut in pbar:
            _, status = fut.result()
            if status == "ok":
                n_ok += 1
            elif status == "skip":
                n_skip += 1
            else:
                n_err += 1
                errors.append(status)
            pbar.set_postfix(ok=n_ok, skip=n_skip, err=n_err)

    elapsed = time.time() - t0
    print(
        f"\nDone.  computed={n_ok:,}  skipped={n_skip:,}  errors={n_err:,}  "
        f"({elapsed:.0f}s)"
    )
    if errors:
        print("First 10 errors:")
        for e in errors[:10]:
            print(f"  {e}")

    return 0 if n_err == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
