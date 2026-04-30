# CLAP Training Audits — Master Document

This document consolidates all run audits currently stored in `docs/Training Audits/`.

---

## Audit Index

- `TRAINING_AUDIT.md` (second fine-tune, completed 2026-04-13)
- `TRAINING_AUDIT_SECOND_FINE_TUNE.md` (second fine-tune, refreshed 2026-04-14)
- `TRAINING_AUDIT_6TH_RUN.md`
- `TRAINING_AUDIT_8th_run.md`
- `TRAINING_AUDIT_9th_run.md`
- `TRAINING_AUDIT_10th_run.md`
- `TRAINING_AUDIT_11th_run.md`

---

## Timeline Summary

### Early second fine-tune audits (`TRAINING_AUDIT.md`, `TRAINING_AUDIT_SECOND_FINE_TUNE.md`)
- Strong relative gains over base CLAP on then-current validation setup.
- Best strategy family centered around combined labels (`sci_common` / `chain_common`).
- Core lesson: training converged, but text/label structure quality remained decisive.

### Run 6 (`TRAINING_AUDIT_6TH_RUN.md`)
- Established the long-standing benchmark run.
- Reported strong in-distribution retrieval around ~9% `R@1` with `all_variants` on its eval setup.
- Highlighted that run-to-run comparisons require fixed eval pools and script parity.

### Run 8 (`TRAINING_AUDIT_8th_run.md`)
- Post-mortem run with major regression.
- Root cause identified as damaged warm-start lineage (Run 7 acoustic-only effects) and weak generalization.
- Produced process safeguards used in later runs (checkpoint hygiene, first-batch sanity checks, audit discipline).

### Run 9 (`TRAINING_AUDIT_9th_run.md`)
- Recovery run from a healthy Run 6 warm-start.
- Repaired data/encoding/build pipeline issues and regained a significant portion of lost performance.
- Still below Run 6 headline metrics.

### Run 10 (`TRAINING_AUDIT_10th_run.md`)
- Primarily a preparation/implementation audit:
  - training script improvements,
  - description generation pipeline fixes,
  - rebuild + precompute checklist.
- Served as the launchpad for Run 11 settings and process.

### Run 11 (`TRAINING_AUDIT_11th_run.md`)
- Included label quality intervention + trainer stability upgrades.
- Completed full eval and full PDF suite generation (including base/zeroshot/semantic plots).
- Final in-distribution `all_variants` stayed around ~9% `R@1`, with strong improvement over base but no breakthrough above Run 6 benchmark band.

---

## Cross-Run Patterns

## 1) What consistently helps
- Healthy warm-start checkpoints (or clean base initialization) matter a lot.
- Full label coverage (taxonomy + rich text) outperforms stripped acoustic-only supervision.
- Conservative, stable training settings reduce collapse/regression risk on Windows setups.
- Evaluating with both finetuned and base (`--also-base`) gives clearer signal.

## 2) What consistently hurts
- Damaged checkpoint lineage (especially when text semantics were previously degraded).
- Generic/repetitive rich descriptions that add little discrimination.
- Inconsistent eval pipelines (encoding issues, holdout failures, missing artifacts) obscuring true progress.

## 3) Why progress plateaus near ~9–10% `R@1`
- Retrieval remains bottlenecked by fine-grained class separability among acoustically similar species/call types.
- Many labels remain semantically overlapping enough to improve rank quality (`mAP`, `R@10`) more than strict top-1 (`R@1`).

---

## Current Best-Available View

- Run 11 is operationally the most complete audited run (training stability + full artifact generation).
- Run 6 remains the key historical benchmark for top-line retrieval.
- Direct 6 vs 11 leaderboard claims should be treated carefully unless both checkpoints are re-evaluated on the exact same frozen eval split and script version.

---

## Recommended Next Steps

1. Keep Run 11 trainer stability defaults as baseline.
2. Push label quality from broad regeneration to targeted hard-class rewrites.
3. Add confusion-mined hard negatives (nearest wrong classes), not only taxonomy hard negatives.
4. Standardize one frozen eval configuration for all future run-to-run comparisons.
5. Continue producing full artifact packs (JSON + all PDFs) for every run.

---

Last updated: 2026-04-30
