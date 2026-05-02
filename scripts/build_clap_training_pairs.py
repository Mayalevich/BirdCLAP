"""
CLAP Training Pairs Builder
============================
Converts audio metadata + clap_all_labels.json into the training JSON
expected by LAION-CLAP and most CLAP fine-tuning frameworks.

Each audio file is paired with ALL text variants from its label pool
(5 taxonomy templates + up to 4 rich descriptions = ~9 variants per clip).
Each pair also carries a "combo" field (species||type) so the training
script can build a multi-positive mask — clips from the same combo are
treated as joint positives, avoiding false negatives in the contrastive loss.

Filters applied:
  - Birds only (taxonomic class == Aves); mammals, amphibians, insects removed.
  - Combos with fewer than MIN_CLIPS_PER_COMBO unique clips are dropped —
    they cannot contribute to multi-positive contrastive learning.

Output format (LAION-CLAP compatible):
    data/clap_train_pairs.json   — list of {audio, text, combo} dicts
    data/clap_val_pairs.json     — held-out 10% for validation

    [
        {"audio": "audio/xc/1060250.mp3", "text": "American Robin song",          "combo": "American Robin||song"},
        {"audio": "audio/xc/1060250.mp3", "text": "Turdus migratorius song",       "combo": "American Robin||song"},
        {"audio": "audio/xc/1060250.mp3", "text": "Animalia > ... song",           "combo": "American Robin||song"},
        {"audio": "audio/xc/1060250.mp3", "text": "A cheerful series of whistles", "combo": "American Robin||song"},
        ...
    ]

Balanced sampling: caps at MAX_PER_COMBO clips per (species, type) combo
so common species don't dominate. The cap applies to unique audio clips;
each clip still generates one pair per text variant.

Usage:
    conda activate birdclap
    python scripts/build_clap_training_pairs.py
    python scripts/build_clap_training_pairs.py --max-per-combo 50 --min-clips-per-combo 3 --val-split 0.1
"""

import argparse
import json
import math
import random
from pathlib import Path

import pandas as pd

# ── paths ─────────────────────────────────────────────────────────────────────
UNIFIED_META  = Path("data/xc_metadata_unified.csv")
LABELS_PATH   = Path("data/clap_all_labels.json")
TAXONOMY_PATH = Path("data/species_taxonomy.json")
TRAIN_OUT     = Path("data/clap_train_pairs.json")
VAL_OUT       = Path("data/clap_val_pairs.json")
HOLDOUT_OUT   = Path("data/clap_holdout_pairs.json")

SKIP_NAMES = frozenset({
    "soundscape", "identity unknown", "noise", "speech", "canine", "squirrel",
    "insects", "rooster", "other", "background", "unknown", "nan", "",
})
SKIP_TYPES = frozenset({"uncertain", "various", "various calls", "nan", ""})

MIN_CLIPS_PER_COMBO = 3   # combos below this threshold add no multi-positive signal
DEFAULT_QUALITY_MIN = 4       # XC quality_rating; §10 doc — disable with --quality-min 0
MIN_CLIPS_COMBO_LEVEL_VAL = 5 # combos with ≥ this many clips donate 3 clips to val (§9 doc)


def recording_context_suffix(row: pd.Series, duration_s: float | None = None) -> str:
    """Per-recording text suffix using whatever columns exist in the metadata row."""
    parts: list[str] = []
    for col in ("country", "recording_month", "month", "time_of_day",
                "habitat", "recording_location"):
        if col in row.index and pd.notna(row[col]) and str(row[col]).strip():
            v = str(row[col]).strip()
            if col in ("country", "recording_location"):
                parts.append(f"recorded in {v}")
            elif col in ("recording_month", "month"):
                parts.append(f"in {v}")
            elif col == "time_of_day":
                parts.append(f"at {v}")
            elif col == "habitat":
                parts.append(f"in {v}")

    ds = duration_s
    try:
        if ds is None and "duration" in row.index:
            dv = row["duration"]
            if pd.notna(dv):
                ds = float(dv)
    except (TypeError, ValueError):
        ds = ds

    if ds is not None and math.isfinite(float(ds)):
        sec = float(ds)
        if sec < 15:
            parts.append("(under 15s clip)")
        elif sec < 45:
            parts.append("(15–45s clip)")
        elif sec < 120:
            parts.append("(45s–2m clip)")
        else:
            parts.append("(long recording)")

    if "source" in row.index and pd.notna(row["source"]) and str(row["source"]).strip():
        parts.append(f"source: {str(row['source']).strip()}")
    elif "species_code" in row.index and pd.notna(row.get("species_code")):
        parts.append(f"xc:{str(row['species_code']).strip()}")

    if not parts:
        return ""
    return " " + ", ".join(parts)


def append_jitter_if_rich(labels_for_combo: list[str], filepath: str, rich_jitter_map: dict) -> list[str]:
    """Append per-recording context only to rich acoustic variants — not taxonomy templates."""
    jit = rich_jitter_map.get(filepath)
    if not jit:
        return labels_for_combo

    out: list[str] = []
    for i, s in enumerate(labels_for_combo):
        rich = i >= 5 or (len(s.split()) >= 12 and " > " not in s)
        out.append(s + jit if rich else s)
    return out


def build_pairs(metadata_paths: list[str],
                labels: dict,
                tax_db: dict,
                max_per_combo: int,
                min_clips_per_combo: int,
                top_n_species: int | None,
                seed: int,
                holdout_species: set[str] | None = None,
                quality_min: int = DEFAULT_QUALITY_MIN,
                ) -> tuple[list[dict], list[dict], dict[str, list[str]]]:
    """
    For each audio file, look up its (species, voc_type) key in labels and
    emit one {audio, text, combo} pair per text variant.

    Filters:
      - Birds only: species whose taxonomic class != 'Aves' are skipped.
      - top_n_species: if set, only keep the N species with the most clips.
      - min_clips_per_combo: combos with fewer unique clips are dropped after
        the first pass (they add no multi-positive signal).
      - max_per_combo cap: applied per combo to limit class imbalance.

    Returns ``(train_pairs, val_pairs)`` with combo-level validation split + per-recording
    suffixes on rich text. holdout clips are emitted separately via ``holdout_clips``.
    """
    rng = random.Random(seed)
    if holdout_species is None:
        holdout_species = set()

    # Pre-compute top-N species set if requested
    top_species: set[str] | None = None
    if top_n_species:
        for path in metadata_paths:
            p = Path(path)
            if not p.exists():
                continue
            df = pd.read_csv(p)
            name_col = next((c for c in ("common_name", "species", "name") if c in df.columns), None)
            if not name_col:
                continue
            counts = df[name_col].value_counts()
            top_species = set(counts.head(top_n_species).index.tolist())
        print(f"  Top-{top_n_species} species filter active ({len(top_species)} species)")

    # First pass: collect all valid clips per combo (respecting cap)
    combo_clips: dict[str, list[str]] = {}
    holdout_clips: dict[str, list[str]] = {}   # clips from held-out species
    recording_suffix: dict[str, str] = {}

    for path in metadata_paths:
        p = Path(path)
        if not p.exists():
            print(f"  [skip] {path} not found")
            continue
        df = pd.read_csv(p)

        name_col = next((c for c in ("common_name", "species", "name") if c in df.columns), None)
        type_col = next((c for c in ("vocalization_type", "type") if c in df.columns), None)
        path_col = next((c for c in ("filepath", "file_path", "path", "filename") if c in df.columns), None)

        if not name_col or not type_col or not path_col:
            print(f"  [skip] {path} — missing required columns "
                  f"(need name, type, filepath). Found: {list(df.columns)}")
            continue

        for _, row in df.iterrows():
            name    = str(row[name_col]).strip()
            voc_raw = str(row[type_col]).strip()
            fpath   = str(row[path_col]).strip()

            if name.lower() in SKIP_NAMES or not fpath or fpath == "nan":
                continue

            if quality_min and quality_min > 0:
                qr = row.get("quality_rating", None)
                try:
                    if qr is not None and int(float(qr)) < quality_min:
                        continue
                except (TypeError, ValueError):
                    pass

            # Birds only
            tax_class = tax_db.get(name, {}).get("class", "")
            if tax_class and tax_class != "Aves":
                continue

            # Top-N species filter
            if top_species is not None and name not in top_species:
                continue

            voc_type = voc_raw.split(",")[0].strip().lower()
            if voc_type in SKIP_TYPES:
                continue

            key = f"{name}||{voc_type}"
            if key not in labels:
                key_full = f"{name}||{voc_raw.lower()}"
                if key_full not in labels:
                    continue
                key = key_full

            # Route holdout species to separate dict (no cap needed for eval)
            if name in holdout_species:
                holdout_clips.setdefault(key, []).append(fpath)
                continue

            dur_sec: float | None = None
            if "duration" in row.index and pd.notna(row["duration"]):
                try:
                    dur_sec = float(row["duration"])
                except (TypeError, ValueError):
                    dur_sec = None

            sfx = recording_context_suffix(row, duration_s=dur_sec)
            if fpath not in recording_suffix:
                recording_suffix[fpath] = sfx or ""

            clips = combo_clips.setdefault(key, [])
            if len(clips) >= max_per_combo:
                continue
            if fpath not in clips:
                clips.append(fpath)

    # Second pass: drop combos below the minimum clip threshold
    dropped_combos = sum(1 for c in combo_clips.values() if len(c) < min_clips_per_combo)
    print(f"  Combos before min-clip filter : {len(combo_clips)}")
    print(f"  Combos dropped (<{min_clips_per_combo} clips)  : {dropped_combos}")
    combo_clips = {k: v for k, v in combo_clips.items() if len(v) >= min_clips_per_combo}
    print(f"  Combos after filter           : {len(combo_clips)}")

    # Partition clips — combos with fewer than MIN_CLIPS_COMBO_LEVEL_VAL stay train-only.
    train_paths: set[str] = set()
    val_paths: set[str] = set()
    for _, clips in combo_clips.items():
        paths = list(clips)
        rng.shuffle(paths)
        if len(paths) >= MIN_CLIPS_COMBO_LEVEL_VAL:
            val_paths.update(paths[:3])
            train_paths.update(paths[3:])
        else:
            train_paths.update(paths)

    def expand_for_paths(paths: set[str]) -> list[dict]:
        out: list[dict] = []
        for key, clips in combo_clips.items():
            variant_texts = labels.get(key)
            if not variant_texts:
                continue
            for fpath in clips:
                if fpath not in paths:
                    continue
                jittered = append_jitter_if_rich(
                    list(variant_texts),
                    fpath,
                    recording_suffix,
                )
                for text in jittered:
                    out.append({"audio": fpath, "text": text, "combo": key})
        return out

    train_pairs = expand_for_paths(train_paths)
    val_pairs   = expand_for_paths(val_paths)
    print(
        f"  Combo-level split: >={MIN_CLIPS_COMBO_LEVEL_VAL} clips -> 3 val; else train-only"
    )

    return train_pairs, val_pairs, holdout_clips


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata",  nargs="+", default=[str(UNIFIED_META)])
    parser.add_argument("--labels",    default=str(LABELS_PATH))
    parser.add_argument("--taxonomy",  default=str(TAXONOMY_PATH))
    parser.add_argument("--train-out", default=str(TRAIN_OUT))
    parser.add_argument("--val-out",   default=str(VAL_OUT))
    parser.add_argument("--max-per-combo", type=int, default=200,
                        help="Max audio clips per (species, type) combo (default 200; doc §10)")
    parser.add_argument(
        "--quality-min",
        type=int,
        default=DEFAULT_QUALITY_MIN,
        metavar="RATING",
        help="Skip rows with quality_rating below this (default 4). "
             "Set 0 to disable quality filtering.",
    )
    parser.add_argument("--min-clips-per-combo", type=int, default=MIN_CLIPS_PER_COMBO,
                        help=f"Drop combos with fewer than this many clips (default {MIN_CLIPS_PER_COMBO})")
    parser.add_argument("--top-n-species", type=int, default=None,
                        help="Only include the N species with the most clips (default: all)")
    parser.add_argument("--holdout-species-frac", type=float, default=0.2,
                        help="Fraction of species held out entirely from train/val for zero-shot eval "
                             "(default 0.2). Their clips are written to clap_holdout_pairs.json.")
    parser.add_argument("--holdout-out", default=str(HOLDOUT_OUT))
    parser.add_argument("--val-split", type=float, default=0.1,
                        help="Deprecated -- ignored; val clips come from combos with "
                             ">=5 clips (3 val each). Left for backwards CLI compatibility.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    labels = json.loads(Path(args.labels).read_text(encoding="utf-8"))
    print(f"Loaded {len(labels)} label keys from {args.labels}")

    tax_db: dict = {}
    tax_path = Path(args.taxonomy)
    if tax_path.exists():
        tax_db = json.loads(tax_path.read_text(encoding="utf-8"))
        print(f"Loaded taxonomy for {len(tax_db)} species (birds-only filter active)")
    else:
        print(f"[warn] Taxonomy file not found at {tax_path} — bird filter disabled")

    # Deterministic holdout species selection (sorted alphabetically → every Nth)
    holdout_species: set[str] = set()
    if args.holdout_species_frac > 0:
        all_species_set: set[str] = set()
        for meta_path in args.metadata:
            mp = Path(meta_path)
            if not mp.exists():
                continue
            df_h = pd.read_csv(mp)
            name_col = next(
                (c for c in ("common_name", "species", "name") if c in df_h.columns),
                None,
            )
            if not name_col:
                continue
            for _, row in df_h.iterrows():
                name = str(row[name_col]).strip()
                if tax_db.get(name, {}).get("class") == "Aves":
                    all_species_set.add(name)
        all_species = sorted(all_species_set)
        step = max(1, round(1.0 / args.holdout_species_frac))
        holdout_species = set(all_species[::step])
        print(f"Holdout species : {len(holdout_species)} / {len(all_species)} "
              f"({100*len(holdout_species)/max(len(all_species),1):.1f}%) "
              f"— excluded from train/val, written to {args.holdout_out}")

    train, val, holdout_clips = build_pairs(
        args.metadata,
        labels,
        tax_db,
        args.max_per_combo,
        args.min_clips_per_combo,
        args.top_n_species,
        args.seed,
        holdout_species=holdout_species,
        quality_min=max(0, args.quality_min),
    )

    n_train_clips = len({p["audio"] for p in train})
    n_val_clips   = len({p["audio"] for p in val})
    n_clips       = n_train_clips + n_val_clips

    print(f"Total unique clips: {n_clips:,} "
          f"(train: {n_train_clips:,}, val: {n_val_clips:,})")

    train_path = Path(args.train_out)
    val_path   = Path(args.val_out)
    train_path.parent.mkdir(parents=True, exist_ok=True)
    train_path.write_text(json.dumps(train, indent=2, ensure_ascii=False), encoding="utf-8")
    val_path.write_text(json.dumps(val, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n{'-'*60}")
    print(f"Done.")
    print(f"  Train : {len(train)} pairs ({n_train_clips} clips × avg "
          f"{len(train)/n_train_clips:.1f} variants) → {train_path}")
    print(f"  Val   : {len(val)} pairs ({n_val_clips} clips × avg "
          f"{len(val)/n_val_clips:.1f} variants) → {val_path}")

    # Text length stats
    from collections import Counter
    text_lengths = [len(p["text"].split()) for p in train]
    print(f"\n  Text length (words): "
          f"min={min(text_lengths)}  max={max(text_lengths)}  "
          f"avg={sum(text_lengths)/len(text_lengths):.1f}")

    # Show 3 sample clips (all variants for each)
    rng = random.Random(args.seed)
    sample_clips = rng.sample(list(set(p["audio"] for p in train)), min(3, n_train_clips))
    print(f"\n  Sample clips (all variants shown):")
    for clip in sample_clips:
        clip_pairs = [p for p in train if p["audio"] == clip]
        print(f"    audio: {clip}")
        for p in clip_pairs:
            print(f"      text: {p['text'][:80]}")
        print()

    # Write holdout pairs (clips from species never seen in training)
    if holdout_clips:
        holdout_pairs: list[dict] = []
        for key, clips in holdout_clips.items():
            for fpath in clips:
                for text in labels.get(key, []):
                    holdout_pairs.append({"audio": fpath, "text": text, "combo": key})
        holdout_path = Path(args.holdout_out)
        holdout_path.parent.mkdir(parents=True, exist_ok=True)
        holdout_path.write_text(json.dumps(holdout_pairs, indent=2, ensure_ascii=False), encoding="utf-8")
        n_holdout_clips = len({p["audio"] for p in holdout_pairs})
        n_holdout_combos = len(holdout_clips)
        print(f"  Holdout: {len(holdout_pairs)} pairs ({n_holdout_clips} clips, "
              f"{n_holdout_combos} combos) → {holdout_path}")

    print(f"Next step:")
    print(f"  Fine-tune CLAP with {train_path} and {val_path}")


if __name__ == "__main__":
    main()
