# Training Audit — Run 10 (finetune10)

**Status:** COMPLETE — training + evaluation complete (**2026-04-30**).  
**Last updated:** 2026-04-30  
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

Run status:
- Completed. Output JSON present at `results/eval_results_finetune10.json`.
- Figure outputs present at `results/figures_finetune10/`:
  - `strategy_comparison.pdf`
  - `class_breakdown_finetuned.pdf`
  - `class_breakdown_finetuned_zeroshot.pdf`
  - `hardest_easiest_finetuned.pdf`
  - `hardest_easiest_finetuned_zeroshot.pdf`
  - `rank_cdf_finetuned.pdf`
  - `rank_cdf_finetuned_zeroshot.pdf`

Note:
- `best.pt` vs `best_r1.pt` should still be compared in future runs, but this audit records the completed evaluation using `best_r1.pt`.

### Target metrics (offline)

| Metric | Run 9 | Run 10 target |
|--------|------:|---------------|
| `all_variants` R@1 | 7.4% | **≥ 10%** |
| `all_variants` mAP | 0.197 | ≥ 0.24 |

---

## 6. Epoch table *(completed)*

| Epoch | train_loss | val_loss | R@1 | Time | Notes |
|------:|-----------:|---------:|----:|-----:|-------|
| 00 | 2.3345 | 1.5751 | 0.2012 | 1120s | |
| 01 | 2.1524 | 1.4302 | 0.2207 | 1352s | |
| 02 | 2.0695 | 1.3878 | 0.2393 | 1242s | |
| 03 | 2.0226 | 1.3700 | 0.2588 | 1197s | |
| 04 | 1.9785 | 1.3446 | 0.2549 | 1199s | |
| 05 | 1.9464 | 1.3329 | 0.2803 | 1196s | |
| 06 | 1.9242 | 1.3285 | 0.2520 | 1198s | |
| 07 | 1.9119 | 1.3136 | 0.2549 | 1198s | |
| 08 | 1.8988 | 1.3070 | 0.2939 | 1198s | |
| 09 | 1.8824 | 1.3160 | 0.2627 | 1196s | |
| 10 | 1.8745 | 1.3018 | 0.2764 | 1200s | |
| 11 | 1.8737 | 1.2961 | 0.2725 | 1197s | best val loss |
| 12 | 1.8648 | 1.2975 | 0.2881 | 1199s | |
| 13 | 1.8654 | 1.3007 | 0.3018 | 1199s | best train R@1 |
| 14 | 1.8684 | 1.2998 | 0.2715 | 1197s | |

**Best val loss:** `checkpoints/finetune10/best.pt` (1.2961)  
**Best training R@1:** `checkpoints/finetune10/best_r1.pt` (epoch 13)

---

## 7. Post-run analysis and next actions

### Evaluation summary (from `results/eval_results_finetune10.json`)

Validation retrieval:

| Strategy | mAP | R@1 | R@5 | R@10 | Median first rank |
|----------|----:|----:|----:|-----:|------------------:|
| name | 0.223 | 0.088 | 0.269 | 0.373 | 7 |
| rich_holdout | 0.210 | 0.076 | 0.250 | 0.350 | 8 |
| **all_variants** | **0.227** | **0.084** | **0.278** | **0.390** | **6** |

Zero-shot holdout:

| Strategy | mAP | R@1 | R@5 | R@10 | Median first rank |
|----------|----:|----:|----:|-----:|------------------:|
| name | 0.119 | 0.0130 | 0.0458 | 0.0806 | 8 |
| **all_variants** | **0.129** | **0.0115** | **0.0532** | **0.0906** | **8** |

### What worked

1. Training converged stably across all 15 epochs; no crash after conservative dataloader settings.
2. Warm-start from Run 6 behaved correctly (early loss in healthy range, no collapse).
3. Full eval figure pack for finetuned + zeroshot was generated in `results/figures_finetune10/`.

### What did not

1. Primary target was missed: `all_variants` R@1 reached ~8.4%, below 10% target and below Run 6 benchmark.
2. Gains were stronger in rank-depth metrics (`R@10`) than strict top-1, indicating separability bottleneck for confusable classes.
3. Zero-shot remained weak in absolute terms.

### Action items (carried into Run 11+)

1. Improve low-specificity labels before training (targeted rewrite list).
2. Keep stable settings (`--no-loss-weights`, softer hard-neg pressure) and add ramping.
3. Continue full-coverage eval generation each run; compare both `best.pt` and `best_r1.pt`.

---

*Run 10 is completed; follow-up experiments proceeded in Run 11.*
