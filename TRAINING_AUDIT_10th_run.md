# Training Audit — Run 10 (finetune10)

**Status:** **Ready to train** after you run the **pre-train rebuild** steps below (one-time if `clap_descriptions.json` changed since last `build_clap_*`).  
**Last updated:** 2026-04-29 (evening)  
**Goal:** Push offline `all_variants` `R@1` above **10%** (step toward **20%**).

---

## Session log — what we did (2026-04-29)

### Training code (`scripts/train_clap.py`)

| Item | Purpose |
|------|--------|
| **`best_r1.pt`** | Saves when training **R@1** improves (separate from **`best.pt`** / val loss). Eval both; pick higher offline R@1. |
| **`--hard-neg-boost`** | Tunable same-genus logit multiplier (default `2.0`). |
| **`--label-smoothing`** | Softens multi-positive targets (default `0.0`). |
| **`--no-loss-weights`** | Turns off inverse-frequency weights **inside** the loss; sampler still balances classes. |
| **Warmup default `500`** | Was `200` — better for warm-start fine-tunes. |

### Description pipeline (`scripts/generate_clap_descriptions.py`)

| Item | Purpose |
|------|--------|
| **`if __name__ == "__main__"`** | Restored — script had been exiting with no work (missing entrypoint). |
| **UTF-8 reads** | `species_descriptions.json` read with `encoding="utf-8"` (fixes Windows `UnicodeDecodeError`). |
| **UTF-8 writes** | `clap_descriptions.json` written with `encoding="utf-8"`. |
| **`--provider gemini` / `--provider openai`** | Same RAG prompts; both inject `aab_text` from `data/species_descriptions.json`. |
| **Gemini rate limits** | Higher default **`--delay`** for Gemini (`5 s`), backoff + jitter, `--gemini-temperature`, longer retries. |
| **`--delay` default** | `None` → **5 s** (Gemini) or **0.35 s** (OpenAI) between combos. |
| **`.env` + `load_dotenv`** | Loads repo-root `.env` (`GEMINI_API_KEY` / `GOOGLE_API_KEY` / `OPENAI_API_KEY`). `.gitignore` already had `.env`. |

### Operational decisions

- **Gemini run:** Started then **cancelled** (rate limits). Tracked **`data/clap_descriptions.json`** was **unchanged** in git (working tree clean; single historical commit for that file) — **no Gemini-only rows** were committed; aborted job did not persist mixed provider data in the repo snapshot we verified.
- **OpenAI generation:** **Started** with **`--provider openai`** after **`OPENAI_API_KEY`** in `.env`. Continues the same RAG contract as always (species text → 4 JSON strings per combo).
- **Why ~22k “lines” vs ~28k audio:** Descriptions are **per (species × vocalization type)**, not per MP3. **`clap_descriptions.json`** line count is **pretty-printed JSON**, not “number of recordings.” See **§ Current dataset snapshot**.

---

## Current dataset snapshot (machine-measured)

| Artifact | Count | Notes |
|----------|------:|------|
| **`xc_metadata_unified.csv` rows** (recordings) | **27,913** | One row per catalogued clip. |
| **`clap_descriptions.json` keys** | **3,649** | Unique **`Species\|\|vocalization_type`** combos with rich lines. |
| **Complete keys** (≥ 4 strings) | **3,649** | Generator target is 4 strings per combo. |
| **`clap_all_labels.json` combos** | **3,664** | Taxonomy + rich pool (should be rebuilt after description changes). |
| **Train / val pairs** | **138,123 / 15,335** | Rebuild **`build_clap_training_pairs.py`** if labels changed. |

**Ratio:** ~27.9k recordings ÷ ~3.6k combos ≈ **7–8 clips per combo on average** — expected: many recordings share the same text variants.

**Stale labels check (2026-04-29):** `clap_descriptions.json` had **38** combo keys not yet present in **`clap_all_labels.json`**, and labels had **53** keys not in descriptions — **`build_clap_labels.py` + `build_clap_training_pairs.py` must be run** after finishing description generation **before** training.

---

## 0. Why Run 9 underperformed Run 6

| Metric | Run 6 | Run 9 | Delta |
|--------|------:|------:|------:|
| `all_variants` R@1 | **9.39%** | 7.4% | −2.0 pp |
| `all_variants` mAP | 0.247 | 0.197 | −0.050 |

Root causes (see earlier analyses): checkpoint picked by **val loss** vs better **training R@1**, double reweighting (sampler + loss), aggressive hard-negative boost, short warmup, missing **`.clap.pt`** sidecars for some pairs.

---

## 1. Dataset and label health (summary)

- **No train/val/holdout leakage** (audio or holdout combos) — verified earlier on pair JSONs.
- **Class imbalance** and **taxonomy-only combos** remain structural challenges; richer **`clap_descriptions.json`** coverage helps the text side.

---

## 2. Code reference — Run 10 training flags

Already documented in this file historically; defaults in code:

| Flag | Suggested Run 10 | Role |
|------|------------------|------|
| `--finetune-from` | `checkpoints/finetune9/best_r1.pt` **or** `checkpoints/sixth-fine-tune/best.pt` | Warm-start (avoid damaged Run 7/8). |
| `--lr` | `5e-6` | Stable continuation. |
| `--warmup-steps` | `500` | Matches new default; explicit flag optional. |
| `--hard-neg-boost` | `1.5` | Softer than `2.0`. |
| `--label-smoothing` | `0.1` | Mild regularisation. |
| `--no-loss-weights` | set | Avoid double reweighting with sampler. |
| `--epochs` | `15` | Longer than 10 if budget allows. |

Output checkpoints: **`best.pt`** (val loss), **`best_r1.pt`** (best training R@1), **`latest.pt`**, optional Pareto epoch snapshots.

---

## 3. Pre-train checklist — **do this before `train_clap.py`**

Run from repo root (`lets-solve-it/`), venv activated, `requirements-ml.txt` installed.

| Step | Command | Why |
|------|---------|-----|
| **1. Descriptions finished** | OpenAI script completed (or you accept current `data/clap_descriptions.json`). | All combos you want covered must have **4×** strings where applicable. |
| **2. Rebuild label pool** | `python scripts/build_clap_labels.py` | Merges taxonomy templates + **`clap_descriptions.json`** into **`clap_all_labels.json`**. |
| **3. Rebuild pairs** | `python scripts/build_clap_training_pairs.py` | Regenerates **`clap_train_pairs.json`** / **`clap_val_pairs.json`** from updated labels + metadata. |
| **4. Precompute coverage (strongly recommended)** | `python scripts/precompute_clap_features.py --dry-run` then run **without** `--dry-run` | Reduces skipped train pairs without **`.clap.pt`**. Aim **≥ 98%** sidecar coverage for training rows. |
| **5. Warm-start file exists** | On your machine: `checkpoints/finetune9/best_r1.pt` or `checkpoints/sixth-fine-tune/best.pt` | **`checkpoints/`** is gitignored — copy checkpoints locally if needed. |
| **6. First-batch sanity** | After start, first batch loss should be **≤ ~3.5**; abort if **> ~4.5** (bad checkpoint). | From project audits. |

**UTF-8:** Use `$env:PYTHONIOENCODING='utf-8'` on Windows when logging if you redirect output.

---

## 4. Recommended training command (Run 10)

```powershell
$env:PYTHONIOENCODING='utf-8'
cd "D:\Clap Training\1st-run\lets-solve-it"
.\.venv\Scripts\activate

python -u scripts/train_clap.py `
  --checkpoint-dir checkpoints/finetune10 `
  --finetune-from checkpoints/sixth-fine-tune/best.pt `
  --lr 5e-6 `
  --warmup-steps 500 `
  --hard-neg-boost 1.5 `
  --label-smoothing 0.1 `
  --no-loss-weights `
  --epochs 15
```

**Note:** Use **`checkpoints/finetune9/best_r1.pt`** instead of sixth-run **`best.pt`** only if that file exists on disk (Run 9 may predate `best_r1.pt` feature — if missing, sixth-run **`best.pt`** is the safe warm-start).

---

## 5. After training — evaluation

```powershell
python scripts/evaluate_clap.py `
  --checkpoint checkpoints/finetune10/best_r1.pt `
  --audio-root scripts/data/xc_audio `
  --output results/eval_results_finetune10.json `
  --figures-dir results/figures_finetune10
```

Repeat with **`best.pt`**; keep the checkpoint with higher **`all_variants` R@1**.

### Target metrics (offline)

| Metric | Run 9 | Run 10 target |
|--------|------:|---------------|
| `all_variants` R@1 | 7.4% | **≥ 10%** |
| `all_variants` mAP | 0.197 | ≥ 0.24 |

---

## 6. Epoch table *(fill after run)*

| Epoch | train_loss | val_loss | R@1 | Time | Notes |
|------:|-----------:|---------:|----:|-----:|-------|
| — | — | — | — | — | |

**Best val loss:** `checkpoints/finetune10/best.pt`  
**Best training R@1:** `checkpoints/finetune10/best_r1.pt`

---

## 7. If Run 10 still regresses

1. Compare **`best_r1.pt`** vs **`best.pt`** on eval.  
2. More epochs / adjust `--delay` during description gen / full precompute.  
3. Rich-description coverage for remaining weak combos (taxonomy-only).  
4. Escalation: fresh **`laion/clap-htsat-fused`** init with same flags (longer horizon).

---

*This file is the single place to append Run 10 notes, eval numbers, and follow-ups.*
