# Training Audit — Run 11 (finetune11)

**Status:** COMPLETE — training + full eval + full PDF pack generated (**2026-04-30**)  
**Date started:** 2026-04-30  
**Author:** project log + manual notes

---

## 0. Pre-run changes and intent

Run 11 focused on two goals:

1. Improve label quality (low-specificity rewrite pass + richer generation prompt).
2. Improve training stability and retrieval quality with low-risk trainer changes.

### 0a. Label/data work completed before Run 11

- Built a rewrite target list for low-specificity combos:
  - `data/description_rewrite_priority_top200.json`
  - `data/description_rewrite_priority_top200.csv`
- Updated `scripts/generate_clap_descriptions.py` prompt to reduce repetitive lead-ins and force more varied sentence openings.
- Regenerated targeted descriptions (OpenAI) and rebuilt:
  - `data/clap_descriptions.json`
  - `data/clap_all_labels.json`
  - `data/clap_train_pairs.json`
  - `data/clap_val_pairs.json`
  - `data/clap_holdout_pairs.json`

### 0b. Trainer changes introduced for Run 11 (`scripts/train_clap.py`)

| Item | Purpose |
|------|---------|
| `--hard-neg-ramp-epochs` | Ramps hard-negative pressure from 1.0 to target value across early epochs for stability. |
| `--rich-text-prob` | Biases text augmentation toward richer sentence-like label variants. |
| hard-neg ramp in loop | Avoids over-aggressive early separation before embeddings settle. |
| rich label sampler helper | Prefer discriminative text exposure during training augmentation. |

---

## 1. Training launch (Run 11)

**Warm-start source:** `checkpoints/sixth-fine-tune/best.pt`

**Command (PowerShell):**

```powershell
$env:PYTHONIOENCODING='utf-8'
python -u scripts/train_clap.py `
  --checkpoint-dir checkpoints/finetune11 `
  --finetune-from checkpoints/sixth-fine-tune/best.pt `
  --lr 5e-6 `
  --warmup-steps 500 `
  --hard-neg-boost 1.5 `
  --hard-neg-ramp-epochs 3 `
  --label-smoothing 0.05 `
  --no-loss-weights `
  --rich-text-prob 0.8 `
  --epochs 15 `
  --workers 2 `
  --prefetch-factor 1 `
  --no-persistent-workers
```

| Parameter | Value |
|-----------|--------|
| Base model | `laion/clap-htsat-fused` |
| Checkpoint dir | `checkpoints/finetune11/` |
| Warm-start | `checkpoints/sixth-fine-tune/best.pt` |
| Epochs | `15` |
| Batch × accum | `16 × 8` → effective `128` |
| Base LR | `5e-6` (audio ×0.1, text ×0.5) |
| Warmup | `500` |
| Hard-neg | `1.5` with `3`-epoch ramp |
| Label smoothing | `0.05` |
| Loss weights | off (`--no-loss-weights`) |
| Rich text sampling | `0.8` |
| Data path | precomputed `.clap.pt` (100% coverage detected) |

---

## 2. Epoch training curve

| Epoch | train_loss | val_loss | R@1 | Time | Notes |
|------:|-----------:|---------:|----:|-----:|-------|
| 00 | 2.2665 | 1.4150 | 0.1943 | 1110s | |
| 01 | 2.0616 | 1.2448 | 0.2441 | 1303s | |
| 02 | 1.9582 | 1.1993 | 0.2393 | 1281s | |
| 03 | 1.8983 | 1.1704 | 0.2441 | 1281s | |
| 04 | 1.8509 | 1.1488 | 0.2695 | 1281s | |
| 05 | 1.8175 | 1.1389 | 0.2666 | 1277s | |
| 06 | 1.7878 | 1.1294 | 0.2461 | 1279s | |
| 07 | 1.7780 | 1.1190 | 0.2588 | 1278s | |
| 08 | 1.7612 | 1.1104 | 0.2930 | 1275s | |
| 09 | 1.7452 | 1.1116 | 0.2832 | 1284s | |
| 10 | 1.7405 | 1.1093 | 0.2637 | 1276s | |
| 11 | 1.7338 | 1.1019 | 0.2842 | 1278s | |
| 12 | 1.7227 | 1.1035 | 0.2891 | 1281s | |
| 13 | 1.7267 | 1.1020 | 0.2979 | 1279s | best train R@1 |
| 14 | 1.7214 | 1.0984 | 0.2754 | 1277s | best val loss |

**Best val checkpoint:** `checkpoints/finetune11/best.pt`  
**Best train-R@1 checkpoint:** `checkpoints/finetune11/best_r1.pt`

---

## 3. Evaluation results (`evaluate_clap.py`)

Primary eval JSON:
- `results/eval_results_finetune11_all.json`

Primary figures:
- `results/figures_finetune11_all/` (18 PDFs)

### 3a. Validation retrieval (seen-species val set)

| Strategy | mAP | MRR | R@1 | R@10 |
|----------|----:|----:|----:|-----:|
| name | 0.222 | 0.325 | 0.085 | 0.360 |
| scientific | 0.190 | 0.282 | 0.069 | 0.308 |
| chain | 0.194 | 0.288 | 0.077 | 0.310 |
| sci_common | 0.230 | 0.339 | 0.092 | 0.375 |
| chain_common | 0.228 | 0.340 | 0.089 | 0.382 |
| rich | 0.229 | 0.339 | 0.090 | 0.383 |
| rich_holdout | 0.211 | 0.313 | 0.076 | 0.358 |
| **all_variants** | **0.232** | **0.341** | **0.092** | **0.385** |

### 3b. Zero-shot holdout (unseen species)

| Model | all_variants mAP | all_variants R@1 | all_variants R@10 |
|-------|------------------:|-----------------:|------------------:|
| Fine-tuned | 0.129 | 0.0123 | 0.0917 |
| Base | 0.0116 | 0.0007 | 0.0047 |

---

## 4. What worked / what did not

- Training convergence was smooth and stable across all 15 epochs.
- New trainer knobs were stable (hard-neg ramp + rich-text probability + no-loss-weights).
- Retrieval quality improved strongly vs base on all strategies.
- Absolute top-1 remained low (`all_variants` R@1 around 9.2%), so the core bottleneck is still class separability for confusable bird/call combos.
- Eval reliability issues from earlier runs were fixed (holdout/semantic crash paths), and full figures now generate consistently.

---

## 5. Artifacts generated

| Artifact | Path |
|----------|------|
| Training log | `training_11th_finetune.log` |
| Checkpoints | `checkpoints/finetune11/` |
| Run-local audit | `checkpoints/finetune11/TRAINING_AUDIT.md` |
| Full eval JSON | `results/eval_results_finetune11_all.json` |
| Full figures | `results/figures_finetune11_all/` |

---

## 6. Action items for Run 12

1. Keep Run 11 stable trainer defaults as baseline.
2. Add confusion-mined hard negatives (nearest wrong classes), not just same-genus negatives.
3. Continue targeted label rewrites from hardest combos in `hardest_easiest_finetuned.pdf`.
4. Track progress on a fixed gold-eval subset to reduce variance and expose real top-1 movement.

---

*Last updated: 2026-04-30 — training + full eval + full PDF pack complete.*
