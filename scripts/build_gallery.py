#!/usr/bin/env python3
"""
Standalone gallery rebuild script.

Builds (or force-rebuilds) the CLAP search index (gallery_embeddings.pt)
without requiring the API server to be running.  Run this before starting
the server whenever you:
  - Change audio encoding settings (e.g. upgrading from centre-crop to
    full-duration fusion)
  - Add or remove recordings from audio_root
  - Swap the fine-tuned checkpoint

Usage (from repo root, with the ML venv active):

    python scripts/build_gallery.py
    python scripts/build_gallery.py --force          # delete old cache first
    python scripts/build_gallery.py --checkpoint checkpoints/finetune11/best_r1.pt

Environment variables (all optional, same as the server):
    MODEL_PROVIDER   must be "clap" (default)
    CHECKPOINT_PATH  path to .pt weights
    AUDIO_ROOT       root directory of audio files
    GALLERY_CACHE    path for the output .pt file
    METADATA_PATH    path to xc_metadata_unified.csv
    TAXONOMY_PATH    path to species_taxonomy.json
    VAL_PAIRS_PATH   path to clap_val_pairs.json
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
    force=True,
)


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def main() -> int:
    root = repo_root()

    ap = argparse.ArgumentParser(description="Build the BirdCLAP gallery embedding index.")
    ap.add_argument(
        "--checkpoint",
        default=None,
        help="Path to fine-tuned .pt checkpoint (overrides CHECKPOINT_PATH env var)",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="Delete existing gallery_embeddings.pt before building",
    )
    args = ap.parse_args()

    # ── Force MODEL_PROVIDER=clap ─────────────────────────────────────────────
    os.environ.setdefault("MODEL_PROVIDER", "clap")
    if os.environ["MODEL_PROVIDER"].lower() != "clap":
        print("[ERROR] MODEL_PROVIDER is not 'clap' — nothing to build.", file=sys.stderr)
        return 1

    if args.checkpoint:
        os.environ["CHECKPOINT_PATH"] = str(Path(args.checkpoint).resolve())

    # ── Resolve settings ──────────────────────────────────────────────────────
    sys.path.insert(0, str(root))
    from backend.config import load_settings

    settings = load_settings()

    cache_path = Path(settings.gallery_cache)
    audio_root = Path(settings.audio_root)

    print()
    print("=" * 60)
    print("  BirdCLAP Gallery Builder")
    print("=" * 60)
    print(f"  Checkpoint : {settings.checkpoint_path}")
    print(f"  Audio root : {audio_root}")
    print(f"  Cache path : {cache_path}")
    print(f"  Metadata   : {settings.metadata_path}")
    print()

    # ── Preflight checks ──────────────────────────────────────────────────────
    if not audio_root.is_dir():
        print(f"[ERROR] audio_root does not exist: {audio_root}", file=sys.stderr)
        return 1

    audio_files = list(audio_root.rglob("*.mp3")) + list(audio_root.rglob("*.wav"))
    print(f"  Audio files found : {len(audio_files):,}")
    if not audio_files:
        print("[ERROR] No audio files found under audio_root.", file=sys.stderr)
        return 1

    # ── Optionally wipe old cache ─────────────────────────────────────────────
    if args.force and cache_path.is_file():
        cache_path.unlink()
        print(f"[force] Deleted existing cache: {cache_path}")

    if cache_path.is_file():
        import torch
        cached = torch.load(str(cache_path), map_location="cpu", weights_only=False)
        existing_ver = cached.get("cache_version", 1)
        from backend.providers.clap_provider import CACHE_VERSION
        if existing_ver == CACHE_VERSION:
            print(
                f"[skip] Gallery cache is already v{CACHE_VERSION} "
                f"({len(cached['items']):,} items).  "
                "Use --force to rebuild anyway."
            )
            return 0
        else:
            print(
                f"[stale] Cache is v{existing_ver}, current encoding is v{CACHE_VERSION} — rebuilding."
            )
            cache_path.unlink()

    # ── Build ─────────────────────────────────────────────────────────────────
    print()
    print("Loading CLAP model — this takes 1-3 minutes the first time…")
    print("(subsequent starts load from the cache in seconds)\n")

    t0 = time.time()
    from backend.provider_factory import build_provider

    provider = build_provider(settings)

    elapsed = time.time() - t0
    minutes, seconds = divmod(int(elapsed), 60)
    print()
    print("=" * 60)
    print(f"  Gallery built successfully in {minutes}m {seconds}s")
    print(f"  Items indexed : {len(provider._meta):,}")
    print(f"  Cache written : {cache_path}")
    print("=" * 60)
    print()
    print("You can now start the server normally:")
    print("  powershell -ExecutionPolicy Bypass -File start.ps1")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
