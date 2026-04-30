# Training Audit — Run 9 (finetune9)

**Status:** COMPLETE — training + retrieval eval logged (**2026-04-28**)
**Date started (prep):** 2026-04-28
**Author:** project log + manual notes

---

## 0. Pre-training preparation (done so far)

This section records data and code work **before** run 9 training. Update or replace with the actual training command and metrics after the run.

### 0a. Data layout and canonical CSV

- **`data/xc_metadata_unified.csv`** was aligned with the expanded export by **copying** from `scripts/xc_metadata_unified.csv` so all scripts that default to `data/xc_metadata_unified.csv` (label build, pair build, taxonomy tools) use the same **27,913**-row catalog.
- Rationale: avoid training or auditing against a stale path while the notebook still writes under `scripts/`.

### 0b. Regenerated labels (`build_clap_labels.py`)

- Command pattern: `python scripts/build_clap_labels.py --metadata data/xc_metadata_unified.csv`
- **Output:** `data/clap_all_labels.json` — **3,664** `(species||vocalization_type)` keys.
- **Per key:** up to **5** AnimalCLAP-style taxonomy templates (common, scientific, chain, …) **plus** rich description variants from `data/clap_descriptions.json` where that key already had generated text; **39** non-bird combos skipped; **0** “no taxonomy” species in the sense of the script’s counter (full template set where scientific name exists).
- **Holdout:** `data/clap_descriptions_holdout.json` rebuilt for held-out rich lines used in eval.
- **Intent vs Run 8 failure mode:** labels include **full taxonomy + rich text**, not vocal-type-only strings — consistent with `TRAINING_AUDIT_8th_run.md` (Run 7’s `--acoustic-only`-style damage is a **training** mistake; current `train_clap.py` has no such flag, but **labels must stay full**).

### 0c. Regenerated training / val / holdout pairs (`build_clap_training_pairs.py`)

- Command: `python scripts/build_clap_training_pairs.py --metadata data/xc_metadata_unified.csv` (defaults for labels, taxonomy, outputs unless overridden).
- **Results (from successful run on 2026-04-28):**
  - **Train:** **138,123** pairs (**17,295** unique clips × **8.0** variants avg) → `data/clap_train_pairs.json`
  - **Val:** **15,335** pairs (**1,921** clips × **8.0** variants avg) → `data/clap_val_pairs.json`
  - **Holdout:** **42,243** pairs (**5,298** clips, **284** combos) → `data/clap_holdout_pairs.json`
  - Holdout species: **81 / 405** bird species (**20.0%**) excluded from train/val for zero-shot-style evaluation.
  - Pipeline stats from builder: **1,116** combos before min-clip filter; **202** combos dropped (`<3` clips); **914** combos after filter.

### 0d. Code fixes applied during prep (worth keeping)

| Issue | Fix |
|-------|-----|
| Windows **UTF-8** when reading `clap_all_labels.json` | `Path.read_text(encoding="utf-8")` for labels in `scripts/build_clap_training_pairs.py`. |
| **Holdout-species selection** effectively re-read the CSV **per row** (extremely slow on ~28k rows) | Rewritten to **one `read_csv` per metadata file** and a single pass over rows. |
| Windows console **`UnicodeEncodeError`** on a box-drawing character in `print` | Separator line uses ASCII `-` instead of `─`. |

### 0e. Documentation / guardrails

- **`scripts/train_clap.py`** module docstring: added a **Tip** that training must use **full** label strings from `build_clap_labels.py` and pointed to **`TRAINING_AUDIT_8th_run.md`** for the damaged-checkpoint lesson.

### 0f. Optional follow-up (not done in this prep pass)

- **`scripts/generate_clap_descriptions.py`**: many **new** `(species||type)` combos still have **taxonomy-only** labels until GPT-generated lines are added under `data/clap_descriptions.json` and labels are rebuilt.
- **Downloads / sidecars:** some CSV paths may still lack MP3s or `.clap.pt` sidecars; training that uses precomputed features will skip missing files — confirm coverage before a long run.

### 0g. Launch blockers fixed immediately before training

- Pair JSON files contained **Windows-1252** bytes (e.g. `0x92` for typographic apostrophe in “Wilson’s”), which broke **`read_text(encoding="utf-8")`** in `train_clap.py`. Files were **re-decoded as CP1252** and **re-saved as UTF-8** (`data/clap_train_pairs.json`, `data/clap_val_pairs.json`, `data/clap_holdout_pairs.json`).
- **`build_clap_training_pairs.py`** now calls **`write_text(..., encoding="utf-8")`** so future rebuilds stay UTF-8 on Windows.

---

## 1. Training launch (Run 9 — fixed configuration)

**Warm-start source (explicit):** `checkpoints/sixth-fine-tune/best.pt` only.

**Not used (avoid Run 7 / Run 8 damage):** `checkpoints/seventh-fine-tune/best.pt`, `checkpoints/finetune8/best.pt`, or any resume from those runs.

**Command (repo root, PowerShell):**

```powershell
$env:PYTHONIOENCODING='utf-8'
python -u scripts/train_clap.py `
  --checkpoint-dir checkpoints/finetune9 `
  --finetune-from checkpoints/sixth-fine-tune/best.pt `
  --lr 5e-6
```

`--finetune-from` loads **weights only** and resets the optimiser (recommended when changing LR and continuing from another run’s `best.pt`).

| Parameter | Value |
|-----------|--------|
| Base model | `laion/clap-htsat-fused` |
| Checkpoint dir | `checkpoints/finetune9/` |
| Warm-start | `checkpoints/sixth-fine-tune/best.pt` |
| **Not used** | Run 7 / Run 8 checkpoints (see above) |
| Epochs | `10` (default) |
| Batch × accum | `16 × 8` → **effective 128** |
| Base LR | **`5e-6`** (audio ×0.1 = **5e-7**, text ×0.5 = **2.5e-6**) |
| Warmup steps | `200` |
| AMP | On (FP16) |
| Data mode | Pre-computed **`.clap.pt`** when present (`--no-precomputed` not set) |
| Train / val pairs (on disk JSON) | `data/clap_train_pairs.json` (**138,123** pairs) / `data/clap_val_pairs.json` (**15,335** pairs) |
| **Dataset rows actually used** (after skipping pairs without sidecars) | **131,460** train pairs \| **14,530** val pairs — logged at training start |
| Labels | `data/clap_all_labels.json` |
| Audio root | `scripts/data/xc_audio` |
| Training log | `training_9th_finetune.log` (tee from stdout) |
| Stub audit from script | `checkpoints/finetune9/TRAINING_AUDIT.md` |

### Warm-start sanity (epoch 0, observed)

- Rolling train loss in the first tens of batches stayed **~2.33–2.37** — consistent with a **healthy** sixth-run initialisation (contrast **Run 8** audit: damaged warm-start showed **~6.7** early loss).

### Pre-flight sanity (from Run 8 audit)

1. After warm-start, **first-batch loss** should be **≤ ~3.5** for a healthy checkpoint; if **> ~4.5**, stop and verify the checkpoint source.
2. Do **not** replicate Run 7’s behaviour of training on **stripped** text (vocal-type-only); full strings must remain in the pair JSON.

---

## 2. Epoch training curve

Sourced from **`checkpoints/finetune9/TRAINING_AUDIT.md`** (matches end-of-epoch summaries). **R@1** here is the **training loop’s** retrieval metric on the val loader, not the offline `evaluate_clap.py` suite.

| Epoch | train_loss | val_loss | R@1 | Time | Notes |
|------:|-----------:|---------:|----:|-----:|-------|
| 00 | 2.1023 | 1.3018 | 0.1553 | 1097s | |
| 01 | 1.8020 | 1.1182 | 0.1914 | 1219s | |
| 02 | 1.6333 | 1.0661 | 0.1982 | 1197s | |
| 03 | 1.5518 | 1.0570 | 0.1914 | 1199s | |
| 04 | 1.4900 | 1.0413 | 0.1836 | 1195s | |
| 05 | 1.4524 | 1.0287 | 0.1807 | 1196s | |
| 06 | 1.4276 | 1.0265 | 0.1963 | 1196s | |
| 07 | 1.4134 | 1.0183 | **0.2119** | 1198s | best training R@1 |
| 08 | 1.4027 | **1.0020** | 0.2061 | 1197s | best val_loss |
| 09 | 1.4075 | 1.0198 | 0.1807 | 1196s | |

**Checkpoint:** `checkpoints/finetune9/best.pt`.  
**Approx. wall time:** ~**3.3 h** total training (sum of epoch times above).

---

## 3. Evaluation results (`evaluate_clap.py`)

**Command:**

```text
python scripts/evaluate_clap.py --checkpoint checkpoints/finetune9/best.pt --audio-root scripts/data/xc_audio --output results/eval_results_finetune9.json --figures-dir results/figures_finetune9
```

### Val retrieval (seen species / val split)

Gallery **1,913** encoded clips (**8** failed: missing or bad MP3). **721** queries per strategy for `all_variants` (722 combos in metadata; one edge case).

| Strategy | mAP | MRR | R@1 | R@5 | R@10 | Median rank (first hit) |
|----------|----:|----:|----:|----:|-----:|--------------------------|
| name | 0.187 | 0.284 | 6.8% | 22.6% | 31.4% | 9 / 1913 |
| scientific | 0.172 | 0.255 | 6.2% | 20.1% | 28.2% | 12 / 1913 |
| chain | 0.177 | 0.262 | 6.6% | 19.7% | 29.1% | 12 / 1913 |
| sci_common | 0.189 | 0.283 | 7.1% | 23.0% | 32.2% | 9 / 1913 |
| chain_common | 0.191 | 0.283 | 6.9% | 23.4% | 32.5% | 8 / 1913 |
| rich | 0.184 | 0.277 | 6.8% | 22.4% | 31.5% | 9 / 1913 |
| rich_holdout | 0.176 | 0.266 | 6.3% | 20.7% | 30.9% | 10 / 1913 |
| **all_variants** | **0.197** | **0.292** | **7.4%** | 23.4% | 33.6% | 8 / 1913 |

### Zero-shot holdout (unseen species)

Gallery **5,294** clips (**4** failed). **284** combos.

| Strategy | mAP | MRR | R@1 |
|----------|----:|----:|----:|
| **all_variants** | **0.125** | 0.294 | **1.0%** |
| name | 0.119 | 0.296 | 1.6% |

**rich_holdout** on holdout: skipped (no valid combos in that split).

**Outputs:** `results/eval_results_finetune9.json`, **`results/figures_finetune9/`** (strategy comparison, class breakdown, hardest/easiest, rank CDFs).

### vs Run 8 (`TRAINING_AUDIT_8th_run.md`)

| Run | all_variants mAP | all_variants R@1 |
|-----|-----------------:|-----------------:|
| 8 | ~0.113 | ~4.6% |
| **9** | **0.197** | **7.4%** |

---

## 4. Post-run notes

- **Warm-start from Run 6** + **full labels** + **lr 5e-6** produced a large **retrieval gain** vs Run 8 (roughly **+75%** relative mAP, **+60%** relative R@1 on `all_variants`).
- **Eval caveats:** a few MP3s missing or corrupt; **`clap_val_pairs.json`** needed another **UTF-8** repair before `evaluate_clap.py` would load (see §0g — keep explicit UTF-8 on pair JSON writes).

---

## 5. Files to attach when closing the audit

| File | Role |
|------|------|
| Training log | e.g. `training_9th_finetune.log` or checkpoint dir log |
| `results/eval_results_*.json` | Eval metrics |
| `checkpoints/finetune9/best.pt` | Best checkpoint reference |
| This file | `TRAINING_AUDIT_9th_run.md` |

---

*Last updated: 2026-04-28 — training + offline eval complete.*
