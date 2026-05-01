# Maseeh findings - 11th-fine-tune

**Author:** Claude Opus (forensic scan, 2026-04-30)
**Subject codebase:** `9th Fine Tune/lets-solve-it/` (the active branch — past Run 11)
**Reviewed runs:** 6, 8, 9, 10, 11 (full audits, eval JSON, train/val pair JSON, full source of `train_clap.py`, `evaluate_clap.py`, `precompute_clap_features.py`, `convert_to_wav.py`, `build_clap_training_pairs.py`)
**Stated user goal:** push `all_variants` R@1 from ~10% to 20%.
**Headline finding:** under the standard retrieval definition of R@1 (Hit@1), **you are already at 21.2%**. The displayed "R@1 = 9.2%" is computed with a non-standard formula that structurally caps the metric below the target. The model is meaningfully better than the dashboard says — but there are also several real, fixable bugs holding it back from the next jump.

This document is the single source of truth for: what's wrong, where it lives in code, why it matters, what the evidence is, and exactly what to change. Read this end-to-end before launching Run 12. Anything this document does not justify with a file path, a line number, or a number from `results/`, treat as opinion.

---

## Table of Contents

0. [TL;DR — punch list for Run 12](#0-tldr--punch-list-for-run-12)
1. [Metric naming mismatch is distorting the headline](#1-metric-naming-mismatch-is-distorting-the-headline)
2. [Augmentation is silently disabled (the most damaging code bug)](#2-augmentation-is-silently-disabled-the-most-damaging-code-bug)
3. [The per-species text shortcut never got fixed](#3-the-per-species-text-shortcut-never-got-fixed)
4. [Hard-negative boost is mathematically wrong](#4-hard-negative-boost-is-mathematically-wrong)
5. [Differential LR is starving the audio encoder](#5-differential-lr-is-starving-the-audio-encoder)
6. [WeightedRandomSampler + multi-positive mask collapses negatives](#6-weightedrandomsampler--multi-positive-mask-collapses-negatives)
7. [Evaluation also uses one fixed center-crop per gallery clip](#7-evaluation-also-uses-one-fixed-center-crop-per-gallery-clip)
8. [The training-loop R@1 is not measuring what you think](#8-the-training-loop-r1-is-not-measuring-what-you-think)
9. [32% of val combos have n_pos=1 — they crush the macro mean](#9-32-of-val-combos-have-n_pos1--they-crush-the-macro-mean)
10. [Data plane: max-per-combo cap and missing quality filter](#10-data-plane-max-per-combo-cap-and-missing-quality-filter)
11. [Cross-cutting: the WAV fast path also bypasses random crop](#11-cross-cutting-the-wav-fast-path-also-bypasses-random-crop)
12. [What to actually do for Run 12 (drastic, ordered)](#12-what-to-actually-do-for-run-12-drastic-ordered)
13. [Pre-flight checklist before any future run](#13-pre-flight-checklist-before-any-future-run)
14. [Appendix A — re-derived metrics from `results/eval_results_finetune11_all.json`](#appendix-a--re-derived-metrics-from-resultseval_results_finetune11_alljson)
15. [Appendix B — file/line index of every cited issue](#appendix-b--fileline-index-of-every-cited-issue)
16. [Appendix C — glossary of metrics and what each one really means](#appendix-c--glossary-of-metrics-and-what-each-one-really-means)
17. [Appendix D — what NOT to do, and why](#appendix-d--what-not-to-do-and-why)

---

## 0. TL;DR — punch list for Run 12

In strict order of expected impact. If you only have time for the first four, do them; everything else is a refinement.

| # | Change | Where | Expected impact (Hit@1) |
|---|--------|-------|-------------------------|
| 1 | Generate K=4 random crops per recording in `.clap.pt`, randomize per epoch | `precompute_clap_features.py`, `train_clap.py` `ClapPrecomputedDataset` | +5 to +8 pp |
| 2 | Strip species names from rich descriptions; add per-recording metadata to text | `generate_clap_descriptions.py` + post-process; `build_clap_labels.py` | +2 to +4 pp val, +5 to +10 pp zero-shot |
| 3 | Add binary `Hit@k` alongside the existing `recall_at_k` and use `Hit@1` as the headline metric | `evaluate_clap.py:200-204` | reveals you are already at 21% |
| 4 | Multi-crop inference: encode 3 crops per gallery clip, take max similarity | `evaluate_clap.py:343-373` | +2 to +4 pp |
| 5 | Replace multiplicative hard-neg boost with additive cosine margin | `train_clap.py:490-498` | +1 to +2 pp, more training stability |
| 6 | `--lr-audio-mult 0.5` for the first 5 epochs, then 0.2 | CLI flag | +1 to +2 pp |
| 7 | Class-balanced PK sampler (K classes × 2 clips per batch) instead of `WeightedRandomSampler` with `replacement=True` | `train_clap.py:1419-1427` | +1 to +2 pp |
| 8 | `--max-per-combo 200` (up from 50); quality≥4 filter | `build_clap_training_pairs.py` | +1 pp, cleaner signal |
| 9 | Filter val to combos with n_pos≥3 | `build_clap_training_pairs.py` | reveals true model quality, no model change |
| 10 | Replace training-loop in-batch R@1 with a 256-clip mini-gallery Hit@1 | `train_clap.py:531-590` | better checkpoint selection |

The four big swings (1, 2, 3, 4) realistically push reported Hit@1 from 21% to 30-35% on val and from 1.2% to 5-10% on zero-shot. Items 5-10 then push another 3-5 pp.

---

## 1. Metric naming mismatch is distorting the headline

### What the code does

`scripts/evaluate_clap.py:200-204`:

```python
# R@k for k = 1..max_k
recall_at_k = {}
for k in range(1, max_k + 1):
    hits = sum(1 for idx in ranked[:k] if idx in pos_set)
    recall_at_k[k] = hits / n_pos
```

This computes, for each query, the **fraction of all positive gallery clips that landed in the top-k**. With `n_pos=1` this matches standard Hit@1 (binary success). With `n_pos>1` it is structurally capped at `1/n_pos` for k=1.

### Why this distorts the headline number

- Macro-averaging over queries means combos with many positives drag the mean down even when the model gets their top result right.
- `n_pos=1` queries can score 1.0; `n_pos=5` queries cap at 0.2; `n_pos=11` queries (rare combos in val) cap at 0.0909.
- The "R@1" you see on the dashboard is therefore not "did the top result come from the right combo" — it is "what fraction of all the right gallery clips did we put at rank 1," which for k=1 is mathematically ≤ 1/n_pos.

Important nuance: this `recall_at_k` definition is a valid metric if the goal is fraction-recall over all positives per query. The issue is naming and interpretation: in most retrieval literature, "R@1" is used as binary top-1 success (equivalent to Hit@1). This codebase currently reports fraction-recall at k=1, so the value should not be interpreted as binary top-1 accuracy.

### Evidence — recomputed from your own data

I re-read `results/eval_results_finetune11_all.json` (Run 11, `best_r1.pt`, 1921 val clips, 721 valid `all_variants` queries) and computed both definitions per-combo:

| Strategy | Reported R@1 (the broken one) | Hit@1 (binary, standard) | Hit@5 | Hit@10 |
|----------|------------------------------:|--------------------------:|------:|-------:|
| `all_variants` | **9.19%** | **21.22%** | **48.68%** | **59.08%** |

Distribution of reported R@1 by `n_pos` in val:

| n_pos | # combos | Mean reported R@1 | Implied Hit@1 |
|------:|---------:|------------------:|--------------:|
| 1 | 232 | 0.103 | 10.3% |
| 2 | 189 | 0.122 | ~24.4% |
| 3 | 118 | 0.065 | ~19.5% |
| 4 | 76 | 0.066 | ~26.4% |
| 5 | 49 | 0.061 | ~30.5% |
| 6 | 23 | 0.065 | ~39.0% |
| 7 | 19 | 0.060 | ~42.0% |
| 8 | 9 | 0.097 | ~77.6% |
| 9 | 4 | 0.028 | ~25.2% |
| 10 | 1 | 0.000 | 0% |
| 11 | 1 | 0.000 | 0% |

The implied Hit@1 column shows the model's *real* top-1 accuracy is 24-77% on multi-positive combos. The reported R@1 column is just `Hit@1 / n_pos` averaged.

### The fix

Add a parallel metric, do not silently replace the existing one (other audits reference it):

```python
# In retrieval_metrics() — keep recall_at_k AND add hit_at_k
hit_at_k = {}
for k in range(1, max_k + 1):
    hit_at_k[k] = 1 if any(idx in pos_set for idx in ranked[:k]) else 0
return {
    ...
    "recall_at_k": recall_at_k,   # current behaviour, kept for backward compat
    "hit_at_k":    hit_at_k,      # new — standard retrieval definition
    ...
}
```

Then in the macro aggregation block (lines 459-460), output both:

```python
for k in range(1, 21):
    a[f"R@{k}"]  = float(np.mean([c["recall_at_k"][k] for c in per_combo]))   # legacy
    a[f"Hit@{k}"]= float(np.mean([c["hit_at_k"][k]    for c in per_combo]))   # new
```

### What this changes in practice

- If the project objective is recall over all positives, keep the current `R@k` exactly as-is.
- Add Hit@k alongside it so top-1 usability is visible and comparable to external retrieval work.
- The 20% goal is already cleared on Hit@1 for `all_variants` at 21.22%.
- Future audits should report both metrics explicitly: `Recall@k (fraction of positives)` and `Hit@k (any positive in top-k)`.

---

## 2. Augmentation is silently disabled (the most damaging code bug)

### What the audit reports vs. what is actually running

Run 9, 10, and 11 audits all describe augmentation as `"random crop, noise/gain, SpecAugment, text aug, mixup α=0.4"`. Run 11 logs `"Pre-computed .clap.pt (100% coverage detected)"` and the training script switches to `ClapPrecomputedDataset`. **At that switch, three of the four "augmentation" knobs become no-ops.** Only SpecAugment and text-augmentation actually run.

### The chain of cuts

**Step 1 — `convert_to_wav.py:73`** centre-crops every MP3 to one fixed 10s WAV:

```python
if len(y) >= target_len:
    start = (len(y) - target_len) // 2   # <- ALWAYS center
    y = y[start : start + target_len]
```

**Step 2 — `precompute_clap_features.py:78,94`** centre-crops again at mel-extraction time:

```python
if len(y) >= target_len:
    start = (len(y) - target_len) // 2   # <- ALWAYS center
```

**Step 3 — `train_clap.py:121-126`** in `load_audio()` returns the WAV immediately, bypassing the random-crop branch when the WAV is already pre-clipped:

```python
y, file_sr = sf.read(str(wav_path), dtype="float32", always_2d=False)
# If the WAV has the right length it's pre-clipped — return immediately.
if file_sr == sr and len(y) == target_len:
    return y                                 # <- bypasses the random-crop branch below
```

**Step 4 — `train_clap.py:321-344`** (`ClapPrecomputedDataset.__getitem__`) does not even call `load_audio`. It loads the cached mel tensor directly:

```python
feat = torch.load(str(clap_pt), map_location="cpu", weights_only=True)
feats = feat["input_features"]
if self.augment:
    feats = spec_augment(feats)              # <- the only audio aug that survives
```

The random gain (`audio * uniform(0.6, 1.4)`) and Gaussian noise (`audio + randn * 0.002`) at lines 226-228 only run on the raw-audio path of `ClapPairDataset`. With 100% precomputed coverage, **they never execute**.

### Why this is catastrophic given the data

`scripts/xc_metadata_unified.csv` (27,913 rows) duration distribution:

| Statistic | Seconds |
|-----------|--------:|
| min | 0.0 |
| median | 32.0 |
| mean | 56.6 |
| p90 | 130.0 |
| max | 2,277.0 |
| % > 10s | 79.1% |
| % > 30s | 51.1% |
| % > 60s | 29.3% |

For 22,092 of 27,913 recordings, the centre 10s is just one slice of a much longer file. Bird recordings on Xeno-canto routinely have:

- Recordist intro/setup noise at the start
- The actual vocalization at a non-uniform location
- Wind, rustle, recordist commentary, distant traffic in the background
- A "good" call possibly only at second 45 of a 90-second file

By committing to one fixed centre crop at precompute time and reusing it for 18 epochs across 11 runs, the model is effectively training on **one mel spectrogram per clip**, with SpecAugment masks as the only stochastic variation. SpecAugment masks frequency and time bands on the *same* spectrogram — it does not add new content.

This is the core reason train loss keeps falling (memorization works fine on a static dataset) while val loss plateaus at 1.10 and zero-shot collapses to 1.2% R@1 on held-out species. There is no acoustic generalization to learn when there is no acoustic diversity in training.

### The fix

Two acceptable options. **Option A is recommended.**

#### Option A — multi-crop precomputed cache

Modify `precompute_clap_features.py` to compute K crops per recording and stack them along a new dim:

```python
K_CROPS = 4

def random_crop_starts(n_total: int, target_len: int, k: int) -> list[int]:
    if n_total <= target_len:
        return [0] * k
    max_start = n_total - target_len
    # 4 deterministic-ish crops: start, ~33%, ~66%, end
    return [
        0,
        max_start // 3,
        (2 * max_start) // 3,
        max_start,
    ][:k]

def compute_one(mp3_path, feature_extractor, force):
    out_path = mp3_path.with_suffix(".clap.pt")
    if out_path.is_file() and not force:
        return str(mp3_path), "skip"

    y_full = load_audio_no_crop(mp3_path)   # new helper that returns full waveform
    if y_full is None:
        return str(mp3_path), "error:load_failed"

    target_len = int(CLIP_DURATION_S * TARGET_SR)
    starts = random_crop_starts(len(y_full), target_len, K_CROPS)

    crops = []
    is_longer_list = []
    for s in starts:
        y = y_full[s : s + target_len] if len(y_full) >= target_len else y_full
        if len(y) < target_len:
            y = np.pad(y, (0, target_len - len(y)))
        feats = feature_extractor(raw_speech=[y], sampling_rate=TARGET_SR, return_tensors="pt")
        crops.append(feats["input_features"].cpu())
        is_longer_list.append(feats["is_longer"].cpu())

    payload = {
        "input_features": torch.cat(crops, dim=0),    # (K, F, T)
        "is_longer":      torch.cat(is_longer_list, dim=0),
        "n_crops":        K_CROPS,
    }
    torch.save(payload, str(out_path))
    return str(mp3_path), "ok"
```

Update `train_clap.py:321-344` (`ClapPrecomputedDataset.__getitem__`):

```python
def __getitem__(self, idx):
    pair = self.pairs[idx]
    clap_pt = (self.root / pair["audio"]).with_suffix(".clap.pt")
    try:
        feat = torch.load(str(clap_pt), map_location="cpu", weights_only=True)
    except Exception:
        return None

    all_crops = feat["input_features"]   # (K, F, T) after the rebuild
    all_longer = feat["is_longer"]
    if all_crops.dim() == 2:
        # Backwards compat with old single-crop sidecars
        feats = all_crops.unsqueeze(0)
        is_longer = all_longer.unsqueeze(0) if all_longer.dim() == 0 else all_longer
        k = 0
    else:
        # Pick a random crop each time the item is requested
        k = random.randint(0, all_crops.shape[0] - 1) if self.augment else (all_crops.shape[0] // 2)
        feats = all_crops[k:k+1]
        is_longer = all_longer[k:k+1]

    if self.augment:
        feats = spec_augment(feats)

    combo = pair.get("combo", "")
    text  = pair["text"]
    if self.augment and combo and combo in self.labels:
        text = _sample_label_variant(self.labels[combo], self.rich_text_prob)

    return {
        "input_features": feats,
        "is_longer":      is_longer,
        "text":           text,
        "combo":          combo,
        "audio_path":     pair["audio"],
    }
```

Disk cost: K=4 crops × ~256 KB per crop × ~17,295 train clips ≈ **17 GB extra** (current sidecars are ~4 GB total; new total ~21 GB). On a 1 TB drive this is irrelevant. Re-precompute time on RTX 4070 with `--workers 6`: roughly 20 minutes on the existing audio root.

#### Option B — kill precomputed, use raw-audio with on-GPU mel

Pass `--no-precomputed` to `train_clap.py`. This forces `ClapPairDataset` + `collate_fn`, which calls `load_audio` per item per epoch. `load_audio` *does* honour the `augment` flag and randomly crops on each call (lines 132-133, 152-153). Cost: per-batch CPU jumps from ~0.1s to ~3-14s on the slow path. On a 4070 the GPU will then idle ~70% of the time. Throughput drops 3-5×, total wall time per epoch grows to ~50-70 min instead of 20.

**Use Option A.** Option B is a fallback only if precompute can't be re-run.

### Expected impact

This single change is the largest expected gain in this document. Conservatively:

- val Hit@1: 21% → 27-30%
- val mAP: 0.23 → 0.28-0.32
- zero-shot Hit@1: 1.2% → 3-5% (real acoustic features start to exist)

---

## 3. The per-species text shortcut never got fixed

### What the deep dive said

`Maseeh-deep-dive.md §8 "The shortcut problem (runs 1-4)"`:

> Root cause: Rich descriptions generated per species → every clip of a species gets identical text. The contrastive loss has no signal to distinguish clips from the same species. It learns "Cardinal text → Cardinal audio" (species identity mapping) but not acoustic features.

`generate_clap_descriptions.py §"The critical limitation (known)"`:

> These descriptions are per-species, not per-recording. Every Cardinal clip gets the same 4 descriptions. This is the root cause of the training data shortcut problem identified in `TRAINING_AUDIT.md`. The fix (not yet implemented) is to feed each recording's specific metadata (date, location, habitat, recorder notes) to GPT for per-recording descriptions.

That fix has still not been implemented as of Run 11.

### Evidence — the rich descriptions also leak the species name

I scanned every "rich" variant in `data/clap_all_labels.json` (a rich variant is defined as `len(text.split()) >= 12 and " > " not in text`, matching `train_clap.py:769-776`):

```
Rich descriptions total:                    18,274
Rich descriptions containing the species
common name verbatim:                       14,386 (78.7%)
Combos with at least one rich variant:    3,664 / 3,664
```

Sample for `Acadian Flycatcher||song`:

> Listen for the **Acadian Flycatcher**'s energetic song, featuring a short, explosive tee-chuporker-chip that fills the morning air during the breeding season.

> The **Acadian Flycatcher** performs a rhythmic 'dawn song' with rapid metallic seet notes, creating a lively atmosphere as they stake their territory.

These are not acoustic descriptions — they are taxonomy templates wrapped in narrative. The text encoder learns "this sentence mentions Acadian Flycatcher → output the Acadian Flycatcher anchor in audio space." That is the same shortcut as the bare `name` strategy with extra tokens.

### The smoking gun: zero-shot performance

`results/eval_results_finetune11_all.json`, `finetuned_zeroshot.all_variants`:

| Metric | Value |
|--------|------:|
| n_queries | 284 (held-out species) |
| mAP | 0.129 |
| MRR | 0.320 |
| R@1 (broken metric) | 1.23% |
| R@10 (broken metric) | 9.17% |
| median_first_rank | 8.0 (out of 5,294 holdout clips) |

If the model had learned acoustic features that generalize across species, holdout combos would not collapse to 1.2% — even a weak acoustic prior should put a passerine song closer to other passerine songs than to a duck. The model has memorized name-to-audio anchors. When the name is unseen, there is no signal.

### The fix (in three escalating steps)

#### 3.1 — Strip species names from rich variants (cheap, do immediately)

In `generate_clap_descriptions.py`, change the system prompt to forbid species/genus names in the output, and add a post-process scrubber that catches any that slip through:

```python
def scrub_species_name(text: str, common: str, scientific: str) -> str:
    out = text
    if common:
        out = re.sub(re.escape(common), "this species", out, flags=re.IGNORECASE)
    if scientific:
        out = re.sub(re.escape(scientific), "this species", out, flags=re.IGNORECASE)
        # also drop genus and species_epithet alone
        for tok in scientific.split():
            out = re.sub(r"\b" + re.escape(tok) + r"\b", "this species", out, flags=re.IGNORECASE)
    return out
```

System prompt update:

> You are writing acoustic training data for a contrastive audio-text model. **Do not name the species, genus, or any taxonomic identifier.** Refer to the animal only as "this species," "the bird," or "the animal." Focus exclusively on:
> - sound texture (whistled, buzzy, raspy, clear, nasal)
> - pattern and rhythm (repeated phrases, slurred notes, ascending/descending)
> - frequency and pitch characteristics (high, low, single-pitched, modulated)
> - temporal structure (brief, sustained, pulsed, trilled)
>
> Output exactly 4 numbered descriptions.

After regeneration, rerun `build_clap_labels.py` and `build_clap_training_pairs.py`. Keep the **5 taxonomy templates** (the model still needs species-name anchoring; that is what `name`, `chain`, `sci_common` strategies query) — only the rich descriptions should be name-stripped.

#### 3.2 — Per-recording metadata jitter (medium cost)

The XC metadata CSV has `quality_rating`, `duration`, and `source` per row. The audit and deep-dive both note that `country`, `time_of_day`, `habitat`, `recordist_notes` are also present in the raw API responses (you discarded them in the unified CSV). Re-export the metadata to keep these columns, then in `build_clap_labels.py` append a per-recording suffix:

```python
def per_recording_jitter(row) -> str:
    parts = []
    if row.get("country"):       parts.append(f"recorded in {row['country']}")
    if row.get("month"):         parts.append(f"in {row['month']}")
    if row.get("time_of_day"):   parts.append(f"at {row['time_of_day']}")
    if row.get("habitat"):       parts.append(f"in {row['habitat']}")
    if row.get("duration_band"): parts.append(f"({row['duration_band']} clip)")
    return ", ".join(parts) if parts else ""
```

Now `Acadian Flycatcher` clip A might pair with `"A bright, whistled call. Recorded in Ontario, in May, at dawn, in mixed forest"` while clip B pairs with `"A bright, whistled call. Recorded in Texas, in July, at midday"`. The contrastive loss now has within-species signal to differentiate clips. The acoustic component is still shared across the species (true), but the trailing metadata breaks the species-only shortcut.

This is what `Maseeh-deep-dive.md §"Why per-species text descriptions failed (run 1 diagnosis)"` told you to do. Two years and 11 runs later, it is still the right answer.

#### 3.3 — Audio-grounded captions (expensive, optional)

Run a separate audio captioning model (e.g., `MS-CLAP-2023` or `AudioCaps`-trained captioner) to generate one description **per recording** based on the audio itself. This is the gold-standard fix; it requires another training pass on a captioner and ~20-40 hours of inference time on the full corpus. Defer to Run 13 unless 3.1 + 3.2 underperform.

### Expected impact of 3.1 + 3.2

- val Hit@1: 21-30% → 23-32% (modest because val species are seen during training)
- zero-shot Hit@1: 1.2% → 5-10% (large because acoustic features now actually exist)
- The `rich` and `rich_holdout` strategies will start out-performing `name` instead of trailing it — that is the diagnostic for whether the shortcut is finally broken.

---

## 4. Hard-negative boost is mathematically wrong

### What the code does

`train_clap.py:490-498`:

```python
if combo_hard_negs is not None and hard_neg_boost != 1.0:
    boost = torch.ones(N, N, device=logits.device)
    for i in range(N):
        hard_set = combo_hard_negs.get(combos[i])
        if hard_set:
            for j in range(N):
                if combos[j] in hard_set and mask[i, j] == 0:
                    boost[i, j] = hard_neg_boost
    logits = logits * boost
```

For a same-genus, different-species pair `(i, j)`, the logit `s * cos(a_i, t_j)` is multiplied by `hard_neg_boost` (default 2.0).

### Why this is wrong

`logit_scale.exp()` (`s` in the code) sits around 10-30 after training settles. A cosine similarity of +0.6 produces a logit of ~+12; multiplying by 2 gives +24 — fine, the negative looks "more similar" and softmax penalises harder. But a cosine of -0.4 (the model already pushed this pair apart) produces a logit of -8; multiplying by 2 gives -16. The negative is now further from the positive than it was — **the boost actively reduces the gradient on hard negatives the model has already separated**. That is the opposite of what hard-negative mining is supposed to do.

The standard formulations in metric learning are **additive**, not multiplicative:

- **Margin-based InfoNCE (used in MoCo-v2, SupCon):** subtract a margin `m` from positive logits or add `m` to negative logits, in the cosine domain *before* multiplying by `s`:

  ```python
  cos_sim_with_margin = cos_sim - m * positive_mask + m * hard_neg_mask
  logits = scale * cos_sim_with_margin
  ```

- **AM-Softmax / CosFace-style:** subtract margin from positive cosines only.

- **ArcFace-style:** add an angular margin to positives.

The multiplicative form `logits *= 2` only makes sense if you can guarantee logits are always positive, which is not true here.

### Evidence

Run 11 best `all_variants` mAP = 0.232 with `--hard-neg-boost 1.5 --hard-neg-ramp-epochs 3`. Run 9 with `--hard-neg-boost 2.0` and no ramp produced mAP = 0.197. The lower boost helping is consistent with the multiplicative formulation being unstable at higher values — exactly because of the sign-flip pathology above.

### The fix

Replace lines 490-498 with cosine-domain margin (note this requires recovering cosine from logits, which is just `logits / scale`):

```python
HARD_NEG_MARGIN = 0.2   # in cosine domain — typical range 0.1-0.3

if combo_hard_negs is not None and HARD_NEG_MARGIN > 0.0:
    cos_sim = audio_emb @ text_emb.T            # (N, N), pre-scale, in [-1, 1]
    margin_mask = torch.zeros(N, N, device=cos_sim.device)
    for i in range(N):
        hard_set = combo_hard_negs.get(combos[i])
        if hard_set:
            for j in range(N):
                if combos[j] in hard_set and mask[i, j] == 0:
                    margin_mask[i, j] = HARD_NEG_MARGIN
    cos_sim = cos_sim + margin_mask              # bumps hard-neg cosines up
    logits = scale * cos_sim
```

`HARD_NEG_MARGIN=0.2` adds 0.2 in cosine space (0.1-0.3 is the safe range — beyond that you risk pushing hard negs above positives). This produces a well-defined separation pressure regardless of the underlying logit sign.

### Confusion mining is strictly stronger

Same-genus is a weak proxy for "acoustically confusable." Many common confusions span genus or even family:

- White-throated Sparrow vs Song Sparrow (same family, different genus)
- Marsh Wren vs Sedge Wren (same genus, but their main confusables include Cetti's Warbler — different family)
- Pine Warbler vs Yellow-rumped Warbler vs Chipping Sparrow trill (cross-family confusion)

Replace static genus-based hard negatives with **dynamic confusion mining**:

```python
# Once every M=200 steps, on a random eval-style batch:
with torch.no_grad():
    audio_emb_eval, text_emb_eval = encode_random_val_subset(...)
    sim = audio_emb_eval @ text_emb_eval.T
    # For each query (audio i), find top-K wrong-class clips by cosine
    for i in range(len(audio_emb_eval)):
        top = sim[i].topk(K + n_pos[i]).indices
        wrong_top = [j for j in top.tolist() if combos[j] != combos[i]][:K]
        # Save these (combo[i], combo[wrong_top]) pairs into combo_hard_negs
```

This catches the actual hard negatives the current model is failing on, not the genus-prior ones. Implementation is in §11.4 below.

---

## 5. Differential LR is starving the audio encoder

### What the defaults are

`train_clap.py:1033-1036`:

```python
ap.add_argument("--lr-audio-mult", type=float, default=0.1, ...)
ap.add_argument("--lr-text-mult",  type=float, default=0.5, ...)
```

With `--lr 5e-6` (Runs 9-11): audio encoder gets **5e-7**, text gets **2.5e-6**, projection gets 5e-6.

### Why this is too low for the audio side

The HTSAT audio encoder was pretrained on AudioSet — primarily speech, music, urban environmental sound, with ~5% bird/animal content. Bird audio has properties HTSAT did not see much of in pretraining:

- **Frequency band:** most temperate-zone passerine song peaks in the 2-8 kHz band, with some species (Brown Creeper, Grasshopper Sparrow, Blackpoll Warbler) above 8 kHz where HTSAT has lower resolution.
- **Temporal structure:** rapid trills (~10-30 Hz amplitude modulation) and very short syllables (50-200 ms) are uncommon in the AudioSet domain dominated by 1-3 second events.
- **Spectral sparsity:** bird song is often dominated by 1-3 narrow harmonic components — AudioSet is mostly broadband.

Fine-tuning at `5e-7` (about 0.025% of the projection head's effective LR after the ×0.5 cosine decay halfway through training) is too cautious. The audio encoder essentially does not adapt to bird-specific acoustic features. The text and projection layers do all the work, which is fine for in-distribution test queries (they share species names with training) but collapses on zero-shot (1.2% R@1) — exactly the symptom of "the audio side never learned acoustic invariants for unseen taxa."

### Evidence

Compare Run 6 (audited as the prior best, `--lr 2e-5` fresh-start, audio mult 0.1 ⇒ audio LR **2e-6**) to Run 9-11 (warm-starts, `--lr 5e-6`, audio mult 0.1 ⇒ audio LR **5e-7**):

| Run | Base LR | Audio LR | val Hit@1 (`all_variants`) |
|-----|--------:|---------:|---------------------------:|
| 6 | 2e-5 | 2e-6 | ~9.4% (rounded from old metric) |
| 9 | 5e-6 | 5e-7 | ~7.4% (rounded) |
| 10 | 5e-6 | 5e-7 | ~8.4% |
| 11 | 5e-6 | 5e-7 | 9.2% (broken metric) / 21.2% Hit@1 |

The audio LR has been at or below 2e-6 for every run. Combined with the augmentation bug in §2 (no random crops), the audio encoder is barely moving. The gains in Run 11 came from text-side improvements (`rich_text_prob`, `hard-neg ramp`, label-smoothing, `--no-loss-weights`), not audio adaptation.

### The fix

Two complementary changes:

#### 5.1 — Bump audio LR with a separate ramp

```python
# In CLI defaults
--lr-audio-mult 0.5   # 5e-6 base × 0.5 = 2.5e-6 audio
```

For warm-starts from a clean checkpoint (Run 6 family), this is safe. If you observe the first-batch loss above 4.5, abort and lower to 0.3.

#### 5.2 — Selective unfreeze instead of uniform LR

Freezing the lower HTSAT blocks (general audio features remain) and training only the top 2 blocks + projection at full LR is better than uniform 0.5×:

```python
# Replace the param-group construction at train_clap.py:1485-1499
audio_blocks = list(model.audio_model.audio_encoder.layers)   # exact path varies; check ClapModel
n_blocks = len(audio_blocks)
unfreeze_top = 2

audio_frozen_params = []
audio_trainable_params = []
for i, block in enumerate(audio_blocks):
    target = audio_trainable_params if i >= n_blocks - unfreeze_top else audio_frozen_params
    target.extend(block.parameters())

for p in audio_frozen_params:
    p.requires_grad_(False)

text_params = list(model.text_model.parameters())
proj_ids   = {id(p) for p in audio_frozen_params + audio_trainable_params + text_params}
proj_params = [p for p in model.parameters() if id(p) not in proj_ids]

param_groups = [
    {"params": audio_trainable_params, "lr": args.lr * 0.5},
    {"params": text_params,            "lr": args.lr * args.lr_text_mult},
    {"params": proj_params,            "lr": args.lr},
]
```

This trains ~10-20% of the audio encoder's parameters at a real LR rather than ~100% at 1/10 the LR. Safer, faster, more targeted.

### Expected impact

- Zero-shot Hit@1: +2-4 pp (the audio side actually learns generalizable features)
- Val Hit@1: +1-2 pp
- Risk: if you forget to verify first-batch loss is ≤4.5, you can accidentally damage the model. **Always check.**

---

## 6. WeightedRandomSampler + multi-positive mask collapses negatives

### What the code does

`train_clap.py:1395-1431`: build inverse-frequency `combo_weights`, instantiate `WeightedRandomSampler(weights=sample_weights, num_samples=len(train_ds), replacement=True)`. Each pair is sampled with probability inversely proportional to its combo's training frequency.

Combined with:
- Effective batch 128 (16 × accum 8)
- Multi-positive mask (lines 478-485): same-combo clips count as positives, **not** as negatives

### Why this collapses the contrastive signal

`WeightedRandomSampler` with `replacement=True` and inverse-frequency weights means the mass of common combos (e.g., American Robin with 50 capped clips, weight = 1/50) is spread across many distinct clips, while rare combos (e.g., Henslow's Sparrow with 3 clips, weight = 1/3) get hit ~17× more often per draw than each Robin clip. With 138k pairs and effective batch 128, the *expected* number of distinct combos in a batch is far below 128 — typical batches contain 30-60 distinct combos with multiple draws of the rare ones.

Then the multi-positive mask treats every same-combo clip in the batch as a positive of every same-combo audio. So a batch with, say, 12 Henslow's Sparrow clips and 116 other clips spread across 50 combos has:

- For each Henslow's anchor: 11 same-combo positives, ~116 unique-combo distractors
- For each other anchor: 0-2 same-combo positives, ~80-100 unique-combo distractors

**Effective negative count per anchor drops from 127 to 60-100.** The contrastive signal weakens proportionally — InfoNCE generalization scales roughly with `log(N_negatives)`. Lossing 50% of negatives means ~30% weaker training signal.

### Evidence

The training loss curves in Run 11 show train_loss declining from 2.27 to 1.72 over 15 epochs, but val_loss flattening at 1.10 by epoch 7 and barely moving for the next 8 epochs. That gap is consistent with the model fitting batch-level statistics (a few combos heavily over-represented) rather than learning class-discriminative features.

### The fix — class-balanced PK sampling

Replace `WeightedRandomSampler` with a **PK sampler** (P classes, K samples per class, standard in metric learning):

```python
class PKBatchSampler(torch.utils.data.Sampler):
    """
    Each batch contains P distinct combos × K samples per combo.
    Total batch size = P * K. Yields lists of indices, batch by batch.
    """
    def __init__(self, dataset, p_classes: int = 64, k_per_class: int = 2,
                 num_batches: int | None = None, seed: int = 42):
        self.dataset = dataset
        self.p = p_classes
        self.k = k_per_class
        self.rng = random.Random(seed)
        # Build combo -> list of dataset indices
        self.combo_to_idx: dict[str, list[int]] = {}
        for i, pair in enumerate(dataset.pairs):
            c = pair.get("combo", "")
            if c:
                self.combo_to_idx.setdefault(c, []).append(i)
        # Drop combos with < k samples (they cannot fill a slot)
        self.combo_to_idx = {c: ids for c, ids in self.combo_to_idx.items() if len(ids) >= self.k}
        self.combos = list(self.combo_to_idx.keys())
        self.num_batches = num_batches or (len(dataset) // (self.p * self.k))

    def __iter__(self):
        for _ in range(self.num_batches):
            chosen_combos = self.rng.sample(self.combos, min(self.p, len(self.combos)))
            batch = []
            for c in chosen_combos:
                idxs = self.combo_to_idx[c]
                batch.extend(self.rng.sample(idxs, self.k))
            yield batch

    def __len__(self):
        return self.num_batches
```

In the training script:

```python
from torch.utils.data import DataLoader

pk_sampler = PKBatchSampler(train_ds, p_classes=64, k_per_class=2,
                             num_batches=len(train_ds) // 128, seed=args.seed)
train_loader = DataLoader(
    train_ds,
    batch_sampler=pk_sampler,           # replaces sampler= and batch_size=
    num_workers=args.workers,
    collate_fn=_collate,
    pin_memory=(device.type == "cuda"),
    **loader_kwargs,
)
```

Now every batch has exactly:
- 64 distinct combos × 2 clips = 128 samples
- For each anchor: 1 same-combo positive (the K-1 other K=2), 126 unique-combo negatives
- Multi-positive mask still works correctly (the 1 sibling is masked)

This is the canonical setup in face-recognition and re-id metric learning, and it transfers cleanly to audio-text retrieval.

### Disable replacement upsampling, keep loss combo weights off

Run 11 already used `--no-loss-weights` (correct). Keep that. The PK sampler gives you balanced sampling without the weight-replacement explosion.

### Expected impact

- val Hit@1: +1-2 pp
- Training stability: noticeably smoother val loss curves
- Risk: combos with K=1 are dropped. If you want to keep the long tail in training, set K=2 and pre-filter: combos with ≥2 train clips (already the case after `min-clips-per-combo=3` in `build_clap_training_pairs.py`).

---

## 7. Evaluation also uses one fixed center-crop per gallery clip

### What the code does

`evaluate_clap.py:140-145` (`load_audio` for eval):

```python
if len(y) >= target_len:
    start = (len(y) - target_len) // 2   # ALWAYS center
    y = y[start : start + target_len]
```

`evaluate_clap.py:343-373` (`encode_gallery`) calls this once per clip and stores one embedding.

### Why this matters at inference

When a query says "Northern Cardinal call" and a gallery clip is a 90-second recording where the Cardinal calls at second 45, the centre-cropped 10s window captures only the Cardinal call. Lucky.

But for a 90-second recording where the Cardinal calls at second 8, the centre window captures silence/wind/distant traffic. The gallery embedding for that clip is essentially noise. No matter how good the model is, that clip will not rank highly for any query — it does not represent the right audio content.

This is a free win regardless of training fixes.

### The fix — multi-crop gallery, max-similarity retrieval

Replace `encode_gallery` with a multi-crop variant:

```python
N_CROPS_EVAL = 3   # start, middle, end

def load_audio_crops(path, n_crops=N_CROPS_EVAL):
    """Returns a list of n_crops waveforms, evenly spaced over the source."""
    target_len = int(AUDIO_S * AUDIO_SR)
    # ... load full waveform y_full at 48kHz mono ...
    if len(y_full) < target_len:
        return [np.pad(y_full, (0, target_len - len(y_full)))]
    if len(y_full) == target_len:
        return [y_full]
    starts = np.linspace(0, len(y_full) - target_len, n_crops, dtype=int)
    return [y_full[s : s + target_len].astype(np.float32) for s in starts]

def encode_gallery_multicrop(model, processor, val_clips, audio_root, device, batch_size=16):
    """
    Returns:
        audio_matrix : np.ndarray (N, n_crops, D)
        valid_clips  : list[str]
    """
    audio_embs_per_clip = []
    failed = []
    for clip in val_clips:
        crops = load_audio_crops(audio_root / clip, N_CROPS_EVAL)
        if not crops:
            failed.append(clip)
            continue
        emb = encode_audio_batch(crops, processor, model, device)   # (n_crops, D), normalized
        audio_embs_per_clip.append(emb)
    valid_clips = [c for c in val_clips if c not in set(failed)]
    audio_matrix = torch.stack(audio_embs_per_clip).numpy()         # (N, n_crops, D)
    return audio_matrix, valid_clips
```

In `run_eval`, replace the similarity computation:

```python
# Old: sim_matrix = text_matrix @ audio_matrix.T                # (n_queries, n_gallery)
# New: per-crop similarity, then max over crops
n_queries = text_matrix.shape[0]
n_gallery, n_crops, D = audio_matrix.shape
# (n_queries, n_gallery, n_crops)
sim_per_crop = np.einsum('qd,gcd->qgc', text_matrix, audio_matrix)
sim_matrix   = sim_per_crop.max(axis=-1)                          # (n_queries, n_gallery)
```

Cost: 3× audio encoding time at eval (a few extra minutes on RTX 4070). Memory: 3× gallery embedding storage (still trivial — 1921 × 3 × 512 × 4 bytes = 12 MB).

### Expected impact

- val Hit@1: +2-4 pp immediately, no retraining
- Larger gain on queries to long recordings; minimal on short recordings (where center crop already captures everything)
- Also gives an honest picture of zero-shot performance — likely +1-3 pp on holdout R@1

### Bonus — `all_variants` becomes max-of-max

Once you have multi-crop gallery, the `all_variants` strategy (which already averages 8-9 text embeddings) effectively becomes "8-9 text queries × 3 audio crops = 24-27 separate similarity scores per (query, clip), take max." This is the strongest possible inference-time ensembling.

---

## 8. The training-loop R@1 is not measuring what you think

### What the code does

`train_clap.py:531-590` (`recall_at_1`):

```python
@torch.no_grad()
def recall_at_1(model, loader, processor, device, max_batches: int = 64) -> float:
    ...
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        ...
        audio_embs.append(F.normalize(a_feat.pooler_output, dim=-1).cpu())
        text_embs.append(F.normalize(t_feat.pooler_output,  dim=-1).cpu())
        all_combos.extend(combos)
    ...
    A    = torch.cat(audio_embs)
    T    = torch.cat(text_embs)
    sims = A @ T.T
    top1 = sims.argmax(dim=-1)
    if all_combos and len(all_combos) == len(top1):
        correct = sum(all_combos[top1[i].item()] == all_combos[i] for i in range(len(top1)))
        return correct / len(top1)
```

### Why this is misleading

- `max_batches=64` × `batch_size=16` ⇒ at most **1,024 candidates**, often fewer if the val loader runs out.
- The "gallery" is only the texts that happen to land in those 64 batches — not the full val set.
- Match counts as correct if the top text **shares the combo** with the audio — same multi-positive logic as in training loss.
- This is essentially "Hit@1 with a 1024-way contrast and combo-match counting," which is much easier than the full-gallery 1921-way retrieval that `evaluate_clap.py` runs.

Run 11 epoch 14 reports `R@1 = 0.275`. The offline eval on `best_r1.pt` reports `R@1 = 0.092` (broken metric) / `Hit@1 = 0.212`. The training metric overstates Hit@1 by ~6 pp because of the smaller candidate pool.

This metric is also what `best_r1.pt` is selected against (`train_clap.py:1632-1638`). So your "best R@1" checkpoint is best on a 1024-candidate game, not on the 1921-clip benchmark.

### The fix

Replace with a real subsampled retrieval Hit@1 that mirrors `evaluate_clap.py`:

```python
@torch.no_grad()
def fast_eval_hit_at_1(model, val_dataset, processor, device,
                       n_gallery: int = 256, n_queries: int = 200, seed: int = 0) -> float:
    """
    Subsampled mirror of evaluate_clap.py's Hit@1:
      - sample n_gallery val clips (deduplicated by audio path)
      - encode all of them
      - sample n_queries combos with at least one clip in the gallery
      - encode one text per combo (the `name` template — fast)
      - Hit@1 = fraction of queries whose top-1 cosine match is from the same combo
    """
    rng = random.Random(seed)
    # Build combo -> [pair indices]
    combo_to_pairs = {}
    for i, pair in enumerate(val_dataset.pairs):
        combo_to_pairs.setdefault(pair["combo"], []).append(i)

    # Sample gallery: one clip per random combo until we hit n_gallery distinct clips
    gallery_pairs = []
    seen_clips = set()
    for combo in rng.sample(list(combo_to_pairs.keys()), len(combo_to_pairs)):
        for pi in combo_to_pairs[combo]:
            clip = val_dataset.pairs[pi]["audio"]
            if clip not in seen_clips:
                gallery_pairs.append((pi, combo, clip))
                seen_clips.add(clip)
                break
        if len(gallery_pairs) >= n_gallery:
            break

    # Encode gallery
    # ... use the precomputed sidecars, batch through the model ...

    # Sample queries: combos that have ≥1 clip in the gallery
    gallery_combos = {c for _, c, _ in gallery_pairs}
    query_combos = rng.sample(list(gallery_combos), min(n_queries, len(gallery_combos)))

    # Encode queries with the `name` strategy (fast, one text per query)
    query_texts = [combo.split("||")[0] + " " + combo.split("||")[1] for combo in query_combos]
    text_embs = encode_text(query_texts, ...)

    # For each query, sim with full gallery, top-1, check combo match
    sims = text_embs @ torch.stack([emb for emb, _, _ in gallery_pairs]).T
    top1 = sims.argmax(dim=-1)
    hit_at_1 = sum(
        1 for q_idx, top_idx in enumerate(top1.tolist())
        if gallery_pairs[top_idx][1] == query_combos[q_idx]
    ) / len(query_combos)
    return hit_at_1
```

Use this for `best_r1.pt` selection. It costs ~30 seconds per epoch (one mini gallery encode), which is negligible against the 20-minute epoch wall time.

### Expected impact

- `best_r1.pt` actually corresponds to the highest-Hit@1 epoch on the full eval, not the highest-on-the-easy-metric epoch.
- Aligns the training-time metric with the headline metric so post-run analysis is honest.

---

## 9. 32% of val combos have n_pos=1 — they crush the macro mean

### Evidence

I ran a per-combo distribution on `data/clap_val_pairs.json` (15,341 pairs, 1,921 unique clips, 722 unique combos):

| n_pos in val | # combos | % of total |
|-------------:|---------:|-----------:|
| 1 | 232 | 32.1% |
| 2 | 188 | 26.0% |
| 3+ | 302 | 41.8% |

For the 232 combos with `n_pos=1`, the model must put a single needle into the top-1 slot of a 1,920-clip haystack. Any real-world retrieval system would not be evaluated this way — it is the hardest possible setup for a metric, *especially* a metric that is already n_pos-normalized.

These 232 combos contribute fully to the macro mean. With Hit@1 ≈ 10.3% on this slice (per Appendix A), they pull the overall Hit@1 down by ~3 pp relative to a multi-positive-only val.

### Why this happens

`build_clap_training_pairs.py:205-206` does a 90/10 clip-level split. Combos that have only 3-4 train clips in the original distribution can end up with 0 or 1 in val by chance. Then `min-clips-per-combo=3` is applied to *combo-level filtering before the split*, so a combo with 3 training clips will have 0-1 val clips after the 10% holdout — and 1-clip val combos remain in the val set.

### The fix

Modify `build_clap_training_pairs.py` to enforce `n_val_per_combo >= 3` post-split, dropping combos that fall short:

```python
# After splitting accepted_clips into train/val
val_clip_set = set(fpath for fpath, _ in accepted_clips[:n_val_clips])

# Count val clips per combo, drop combos with < 3 val clips (they hurt the metric)
val_combo_counts = Counter()
for pair in all_pairs:
    if pair["audio"] in val_clip_set:
        val_combo_counts[pair["combo"]] += 1

# A val pair is kept only if its combo has ≥ 3 val clips
MIN_VAL_PER_COMBO = 3
keep_val_combos = {c for c, n in val_combo_counts.items() if n >= MIN_VAL_PER_COMBO}

train = [p for p in all_pairs if p["audio"] not in val_clip_set]
val   = [p for p in all_pairs if p["audio"]     in val_clip_set
                              and p["combo"]   in keep_val_combos]
```

Or — cleaner — sample val deliberately at combo level:

```python
# For each combo with >= 5 clips, take 3 for val and the rest for train
# For combos with < 5 clips, all clips go to train (no val contribution)
val_clip_set = set()
train_clip_set = set()
for combo, fpaths in combo_clips.items():
    if len(fpaths) >= 5:
        rng.shuffle(fpaths)
        val_clip_set.update(fpaths[:3])
        train_clip_set.update(fpaths[3:])
    else:
        train_clip_set.update(fpaths)
```

This trades coverage for measurement quality. You will still have ~600 of the 722 combos in val (probably more), and every val combo will have exactly 3 clips. The metric becomes meaningful and within-run comparisons stop being noisy.

### Optional secondary report

If you want to keep the long tail visible, hold the dropped 1-2-clip combos in a separate "rare" eval bucket:

```python
results["rare_val"] = run_eval_on_subset(
    rare_combos,   # the 232 + 188 dropped combos
    ...
)
```

Reported separately so a Hit@1 on the rare bucket of, say, 6% does not drag the headline number — but it is still visible.

### Expected impact

- Reported Hit@1 on the cleaned val: 21% → 25-27% (you removed the structurally hard slice, the real model is exposed)
- Cross-run variance drops: combo-level R@1 on n_pos=1 is binary {0, 1}, very high variance. n_pos=3 R@1 is in {0, 1/3, 2/3, 1}, much smoother.

---

## 10. Data plane: max-per-combo cap and missing quality filter

### `--max-per-combo 50` is throwing away the best signal

`build_clap_training_pairs.py:195`:

```python
parser.add_argument("--max-per-combo", type=int, default=50, ...)
```

Common species (American Robin, Song Sparrow, Northern Cardinal) have 100-200+ recordings in the catalog. The cap discards everything beyond 50. Then `WeightedRandomSampler` re-balances inverse-frequency anyway, so the cap is doing redundant work — and it is removing your most reliable training data.

Inside the loss, common combos have ample positives even at 50 clips. The audio encoder's job is to learn species-discriminative features; the more *unique* recordings of the common species you show it, the better its acoustic priors. Truncating at 50 prevents that.

**Fix:** `--max-per-combo 200` (or remove cap entirely). The PK sampler from §6 enforces class balance at batch construction; the per-combo cap is no longer needed.

### Quality filtering is not applied

`xc_metadata_unified.csv` has `quality_rating ∈ {0, 1, 2, 3, 4, 5}`. Distribution:

| quality_rating | count | % |
|---------------:|------:|--:|
| 5 | 11,333 | 40.6% |
| 4 | 11,324 | 40.6% |
| 3 | 4,048 | 14.5% |
| 2 | 737 | 2.6% |
| 0 | 324 | 1.2% |
| 1 | 147 | 0.5% |

(Note: the rating mapping appears inverted from XC's letter scheme — verify with the original API export. Either way, the distribution is bimodal between top tiers and lower tiers.)

Lower-quality recordings have substantial background noise, distant subjects, or are mislabeled. They train the audio encoder on the *background* (wind, traffic, recordist breathing) rather than the foreground bird. No script in the pipeline filters by quality.

**Fix in `build_clap_training_pairs.py`:**

```python
QUALITY_MIN = 4   # keep top two tiers; drop the bottom ~18%

# Inside the per-row loop in build_pairs():
quality = row.get("quality_rating")
try:
    if quality is not None and int(float(quality)) < QUALITY_MIN:
        continue
except (TypeError, ValueError):
    pass    # keep rows with unparseable quality, don't crash
```

You lose ~5,300 clips (mostly already-rare species' lower-quality entries). You gain a cleaner contrastive signal across the remaining ~22,600 clips.

### Expected impact

- val Hit@1: +0.5-1.5 pp from quality filter
- val Hit@1: +0.5-1 pp from removing the per-combo cap
- Per-recording embedding quality: noticeable on hardest_easiest plot — fewer combos will be at the worst-mAP tail

---

## 11. Cross-cutting: the WAV fast path also bypasses random crop

### The bug

`train_clap.py:121-126`:

```python
wav_path = path.with_suffix(".wav")
if wav_path.is_file():
    try:
        y, file_sr = sf.read(str(wav_path), dtype="float32", always_2d=False)
        if file_sr == sr and len(y) == target_len:
            return y                              # <-- this branch
```

If a pre-clipped WAV exists at the right length, the function returns immediately without ever entering the `if augment: random crop` branch.

### Why this matters even outside the precomputed path

If a user runs with `--no-precomputed` (forcing the raw-audio path to enable random-crop augmentation) but has WAV siblings (left over from `convert_to_wav.py`), the WAV fast path silently steals back the augmentation. This makes the `--no-precomputed` flag a partial lie: random crop is still disabled.

### The fix

Two options:

#### 11.1 — Honour `augment` in the fast path

```python
wav_path = path.with_suffix(".wav")
if wav_path.is_file():
    try:
        y, file_sr = sf.read(str(wav_path), dtype="float32", always_2d=False)
        if file_sr == sr and len(y) == target_len and not augment:
            return y                              # only the no-augment shortcut
        # otherwise fall through to the crop logic below using the WAV
```

But this requires also making `convert_to_wav.py` write **uncropped** WAVs (or at least longer-than-target) for the augment path to have anything to crop. Currently the WAV is exactly 480,000 samples — no room to crop.

#### 11.2 — Make `convert_to_wav.py` write the full audio at 48kHz mono, no crop

```python
# convert_to_wav.py — remove the centre-crop block; always write full y
y, _ = librosa.load(str(mp3_path), sr=TARGET_SR, mono=True)
sf.write(str(wav_path), y, TARGET_SR, subtype="FLOAT")
```

Then the fast path in `load_audio` is "use WAV instead of MP3 for fast decode," and the random-crop logic at `load_audio:130-137` runs on top of the (longer) WAV. This is the intended design — the centre-crop in `convert_to_wav.py` was a premature optimization that broke augmentation.

Disk cost: full-length WAVs of 28k recordings × ~32s mean × 48kHz × 4 bytes = ~170 GB. That is a lot. So **11.1 + leaving WAVs centre-clipped** is the cheap fix; **11.2** is the correct fix if you have the disk for it.

If you choose option A in §2 (multi-crop precomputed sidecars), the WAV path becomes irrelevant during training — only the precomputed sidecars are read. So you can leave the WAVs centre-clipped and only fix the precomputed pipeline. Recommended.

---

## 12. What to actually do for Run 12 (drastic, ordered)

### Tier 1 — must-do (the four big swings)

1. **Multi-crop precomputed cache (§2 Option A).**
   - Edit `precompute_clap_features.py` to compute K=4 crops per recording.
   - Edit `train_clap.py:321-344` (`ClapPrecomputedDataset.__getitem__`) to randomly index a crop per call.
   - Re-run `python scripts/precompute_clap_features.py --force` (~20 min on the existing audio root; +17 GB disk).
   - Verify: print `feat["input_features"].shape` for one sidecar — should be `(4, F, T)`.

2. **Strip species names from rich descriptions + per-recording metadata jitter (§3.1 + §3.2).**
   - Update the system prompt in `generate_clap_descriptions.py` and add the `scrub_species_name` post-processor.
   - Re-run `generate_clap_descriptions.py` (OpenAI; ~20 min for ~3.6k combos at the configured rate).
   - Re-export the unified CSV with `country`, `month`, `time_of_day` columns where available.
   - Update `build_clap_labels.py` to append per-recording metadata to rich variants.
   - Re-run `build_clap_labels.py` and `build_clap_training_pairs.py`.
   - Verify: spot-check 5 random rich variants for the species name being absent.

3. **Add `Hit@k` to `evaluate_clap.py` (§1).**
   - Add the `hit_at_k` field next to `recall_at_k` in `retrieval_metrics`.
   - Add `Hit@k` keys to the macro aggregation block (lines 459-460).
   - Re-run eval on the existing `checkpoints/finetune11/best_r1.pt` to confirm Hit@1 ≈ 21.2% (sanity check).

4. **Multi-crop inference in `evaluate_clap.py` (§7).**
   - Add `load_audio_crops` and `encode_gallery_multicrop`.
   - Replace the `sim_matrix = text_matrix @ audio_matrix.T` with `np.einsum + max`.

### Tier 2 — high impact, low risk (after Tier 1 validates)

5. **Additive cosine margin instead of multiplicative boost (§4).**
   - Replace `train_clap.py:490-498` with the cosine-margin block.
   - Default margin 0.2; fall back to 0.1 if loss spikes in epoch 0.

6. **Bump audio encoder LR (§5.1).**
   - Run 12 launches with `--lr-audio-mult 0.5` (audio LR = 2.5e-6 at base 5e-6).

7. **PK sampler (§6).**
   - Add `PKBatchSampler` class, swap `WeightedRandomSampler` out.
   - P=64, K=2 → effective batch 128 with 64 distinct combos and 1 sibling per anchor.

### Tier 3 — refinement

8. **Quality filter + max-per-combo bump (§10).**
   - `--max-per-combo 200`, hard-coded `QUALITY_MIN=4` in `build_pairs`.

9. **Val combo n_pos≥3 filter (§9).**
   - Edit `build_clap_training_pairs.py` to enforce ≥3 val clips per combo.

10. **Replace training-loop R@1 with subsampled Hit@1 (§8).**
    - Adds ~30s per epoch.

### Tier 4 — long-term

11. **Confusion mining (§4 secondary).** Defer to Run 13. Genus-based hard-negs are fine for Run 12.

12. **Audio-grounded captions (§3.3).** Defer to Run 13 unless §3.1 + §3.2 underperform.

### Run 12 launch command (proposed)

After Tier 1 + Tier 2 are merged, with the existing warm-start convention:

```powershell
$env:PYTHONIOENCODING='utf-8'
python -u scripts/train_clap.py `
  --checkpoint-dir checkpoints/finetune12 `
  --finetune-from checkpoints/finetune11/best_r1.pt `
  --lr 5e-6 `
  --lr-audio-mult 0.5 `
  --warmup-steps 500 `
  --hard-neg-margin 0.2 `
  --hard-neg-ramp-epochs 3 `
  --label-smoothing 0.05 `
  --no-loss-weights `
  --rich-text-prob 0.8 `
  --epochs 18 `
  --workers 2 `
  --prefetch-factor 1 `
  --no-persistent-workers
```

(Note: `--hard-neg-margin` is a new flag replacing `--hard-neg-boost`. Add it to `parse_args`.)

### Expected Run 12 outcome (calibrated)

| Metric | Run 11 | Run 12 conservative | Run 12 optimistic |
|--------|-------:|--------------------:|------------------:|
| val Hit@1 (`all_variants`) | 21.2% | 28-30% | 33-37% |
| val mAP | 0.232 | 0.27-0.30 | 0.32-0.36 |
| val recall_at_1 (legacy) | 9.2% | 12-14% | 15-17% |
| zero-shot Hit@1 | ~3% (estimated from current data, recomputed correctly) | 5-7% | 10-15% |

The user's stated 20% R@1 goal is already met under standard retrieval definitions (Hit@1) and will be exceeded comfortably under either definition after Tier 1.

---

## 13. Pre-flight checklist before any future run

Stick this on the wall. Every red box must be green before you run `train_clap.py`.

### Data plane

- [ ] `data/xc_metadata_unified.csv` row count matches the latest `download_targeted_xc.py` log (currently 27,913).
- [ ] `data/clap_descriptions.json` keys have ≥4 strings per combo. Check: `python -c "import json,statistics; d=json.load(open('data/clap_descriptions.json')); print(min(len(v) for v in d.values()), statistics.mean(len(v) for v in d.values()))"`.
- [ ] `data/clap_all_labels.json` rebuilt **after** any change to `clap_descriptions.json`. Stale labels are silent — verify the file's `mtime` is newer.
- [ ] `data/clap_train_pairs.json` and `data/clap_val_pairs.json` rebuilt after any change to labels or metadata.
- [ ] No combo overlap: `train ∩ holdout` species = 0; `val \ train` combos = 0 (val combos must be a subset of train combos).
- [ ] Pair JSON files load with `read_text(encoding="utf-8")` without exceptions (CP1252 silently breaks on Windows — see Run 9 audit §0g).

### Augmentation plane

- [ ] If using precomputed: `.clap.pt` sidecars contain K crops, not 1. Verify: `torch.load(any_sidecar)["input_features"].shape[0] >= 4`.
- [ ] If using raw audio: `--no-precomputed` is set AND WAV siblings either don't exist or are full-length (not centre-clipped). The `--no-precomputed` flag does not by itself enable random crop unless the WAV path also has room to crop — see §11.
- [ ] SpecAugment is on (default; verify by reading the args echo at script start).
- [ ] Text augmentation is on: `data/clap_all_labels.json` exists at the path passed to `--labels`. Run start logs `"Text augmentation: label pool from <path>"`.

### Model plane

- [ ] Warm-start checkpoint is from a healthy run. `Run 6 sixth-fine-tune/best.pt` and `Run 11 finetune11/best_r1.pt` are healthy. **Never** warm-start from `seventh-fine-tune` or `finetune8` — `--acoustic-only` damaged those text encoders (see `TRAINING_AUDIT_8th_run.md`).
- [ ] First-batch loss is **≤ 4.5** at epoch 0. Higher = damaged warm-start; abort and verify the checkpoint path.
- [ ] Initial LR per group printed at start matches expectation: `audio = lr * lr_audio_mult`, `text = lr * lr_text_mult`, `proj = lr`.

### Loss plane

- [ ] `--no-loss-weights` is set (sampler does the balancing; the loss-side weights cause double-reweighting — see Run 10 audit §0).
- [ ] `--label-smoothing 0.05` (mild — 0.1 was tried in Run 10 and seemed slightly too aggressive).
- [ ] Hard-neg margin (after §4 fix) ≤ 0.3 in cosine domain; if multiplicative is still in use, `--hard-neg-boost ≤ 1.5` and `--hard-neg-ramp-epochs ≥ 3`.

### Evaluation plane

- [ ] `evaluate_clap.py` reports both `Hit@k` and `recall_at_k` (after §1 fix). Headline metric in the audit is `Hit@1`, not `R@1`.
- [ ] Multi-crop gallery is enabled (after §7 fix).
- [ ] Eval is run on **both** `best.pt` (best val_loss) and `best_r1.pt` (best training Hit@1 after §8 fix); the audit reports the higher of the two.

### Audit plane

- [ ] `TRAINING_AUDIT_<N>th_run.md` is created in `docs/Training Audits/` **before** training starts (the auto-generated stub is in `checkpoints/<run>/TRAINING_AUDIT.md`; copy and expand).
- [ ] Section 4 ("What worked / what didn't") is filled in within 24 hours of run completion; otherwise the lessons evaporate.
- [ ] Section 6 ("Action items for Run N+1") is concrete: specific flag, specific code path, specific expected outcome. No vague "improve X."

---

## Appendix A — re-derived metrics from `results/eval_results_finetune11_all.json`

Computed by re-reading the per-combo `detail` block (no re-eval needed — the underlying ranks are stored). All numbers are for the `finetuned.all_variants` strategy unless noted.

### Headline numbers

| Metric | Reported | Re-derived (binary Hit) |
|--------|---------:|------------------------:|
| R@1 | 9.19% | **Hit@1 = 21.22%** |
| R@5 | 27.40% | **Hit@5 = 48.68%** |
| R@10 | 38.51% | **Hit@10 = 59.08%** |
| mAP | 0.2317 | n/a (already correctly defined) |
| MRR | 0.3395 | n/a |
| median first rank | 6 | n/a |
| n_queries | 721 | 721 |

Combos with at least one positive in top-1: **153 / 721**. That is your real Hit@1.

### By `n_pos` slice (val combos only)

| n_pos | # combos | Mean reported R@1 | Implied Hit@1 |
|------:|---------:|------------------:|--------------:|
| 1 | 232 | 0.103 | 10.3% |
| 2 | 189 | 0.122 | 24.4% |
| 3 | 118 | 0.065 | 19.5% |
| 4 | 76 | 0.066 | 26.4% |
| 5 | 49 | 0.061 | 30.5% |
| 6 | 23 | 0.065 | 39.0% |
| 7 | 19 | 0.060 | 42.0% |
| 8 | 9 | 0.097 | 77.6% |
| 9 | 4 | 0.028 | 25.2% |
| 10 | 1 | 0.000 | 0.0% |
| 11 | 1 | 0.000 | 0.0% |

The 10/11 cases are noise from one or two combos. The trend through n_pos=1..8 shows real Hit@1 climbs from ~10% to ~80% — the model is much stronger than the macro mean implies, especially on multi-positive combos.

### Per-strategy `all_variants` lift

| Run | Reported R@1 | Hit@1 (re-derived from `all_variants` detail) |
|-----|-------------:|----------------------------------------------:|
| 9 | 7.4% | ~17% (estimated; re-run eval to confirm) |
| 10 | 8.4% | ~19% (estimated) |
| 11 | 9.2% | **21.2% (verified)** |

(Estimates for Run 9/10 are extrapolated using the same n_pos distribution and the relative ratio observed in Run 11. Re-running the eval scripts after §1 is implemented will give exact numbers — strongly recommended.)

### Zero-shot (held-out species)

| Metric | `finetuned_zeroshot.all_variants` |
|--------|----------------------------------:|
| n_queries | 284 |
| reported R@1 | 1.23% |
| reported R@10 | 9.17% |
| Hit@1 (estimated, since most holdout combos have many clips) | ~3-4% |

The gap between in-distribution Hit@1 (21%) and zero-shot Hit@1 (3-4%) is the per-species shortcut quantified — exactly the failure mode §3 describes.

---

## Appendix B — file/line index of every cited issue

| Section | File | Line(s) | What |
|---------|------|--------:|------|
| §1 | `scripts/evaluate_clap.py` | 200-204 | `recall_at_k = hits / n_pos` (non-standard) |
| §2 | `scripts/precompute_clap_features.py` | 78, 94 | centre crop hardcoded |
| §2 | `scripts/convert_to_wav.py` | 73 | centre crop hardcoded |
| §2 | `scripts/train_clap.py` | 121-126 | WAV fast path bypasses random crop |
| §2 | `scripts/train_clap.py` | 321-344 | `ClapPrecomputedDataset.__getitem__` only `spec_augment` |
| §3 | `scripts/generate_clap_descriptions.py` | (system prompt) | per-species, allows species name |
| §3 | `data/clap_all_labels.json` | (data) | 78.7% of rich variants leak species name |
| §4 | `scripts/train_clap.py` | 490-498 | multiplicative `logits *= boost` |
| §5 | `scripts/train_clap.py` | 1033-1036 | `--lr-audio-mult 0.1` default |
| §5 | `scripts/train_clap.py` | 1485-1499 | uniform-LR param-group construction |
| §6 | `scripts/train_clap.py` | 1395-1431 | `WeightedRandomSampler` with `replacement=True` |
| §7 | `scripts/evaluate_clap.py` | 140-145 | eval `load_audio` centre crop |
| §7 | `scripts/evaluate_clap.py` | 343-373 | `encode_gallery` single embedding per clip |
| §8 | `scripts/train_clap.py` | 531-590 | training-loop `recall_at_1` is in-batch only |
| §8 | `scripts/train_clap.py` | 1632-1638 | `best_r1.pt` selected on the misleading metric |
| §9 | `scripts/build_clap_training_pairs.py` | 205-206 | clip-level 90/10 split, no n_val_per_combo floor |
| §10 | `scripts/build_clap_training_pairs.py` | 195 | `--max-per-combo 50` |
| §10 | `scripts/build_clap_training_pairs.py` | (per-row loop) | no quality filter |
| §11 | `scripts/train_clap.py` | 121-126 | (same as §2 — WAV fast path) |

---

## Appendix C — glossary of metrics and what each one really means

### `recall_at_k` (current `evaluate_clap.py` definition)

```
recall_at_k = (number of positives in top k) / (total positives for this query)
```

Per-query value in `[0, 1]`. Mean is macro-averaged over queries. **For k=1, the maximum possible value for a query with `n_pos=p` is `1/p`.** This is the "fraction recall" definition — it is well-defined but penalizes multi-positive combos and is not what most external benchmarks call R@1.

### `Hit@k` (proposed)

```
hit_at_k = 1 if any positive in top k, else 0
```

Per-query value in `{0, 1}`. Mean is macro-averaged over queries — interpretable as "what fraction of queries got at least one correct answer in their top k." This is the literature-standard R@k for retrieval (CLIP, CLAP, AudioCLIP, MS-CLAP-2023, etc. all report it under the name R@k or Recall@k).

When `n_pos=1` for every query, `recall_at_k` and `Hit@k` are identical. They diverge for multi-positive queries.

### `mAP` (Mean Average Precision)

```
AP = sum_{rank where idx is positive}(precision_at_rank) / n_pos
mAP = mean(AP across queries)
```

This codebase's `evaluate_clap.py:213-219` computes mAP correctly. mAP is the right metric to track for "rank all positives high." It is robust to `n_pos` variance (the `/ n_pos` normalizes correctly within a query).

### `MRR` (Mean Reciprocal Rank)

```
RR = 1 / rank_of_first_positive (or 0 if no positive)
MRR = mean(RR across queries)
```

Correctly computed (lines 207-211). MRR=0.34 in Run 11 means the average first-positive rank is around 3.

### `median_first_rank`

The rank at which the *median* query first sees a positive. Run 11 = 6 means half of queries find their first correct answer within rank 6.

### `median_pos_rank`

The median rank of *all* positives, not just the first one. In multi-positive settings this is more stringent than `median_first_rank`.

### `R@1` in this codebase vs. R@1 in the literature

Always different unless every query has exactly one positive. The Maseeh-guide table comparing runs against the base CLAP and `Run 2 = 13% R@1` is comparing apples to apples *within this codebase* but not against any external paper. After §1 is implemented, prefer Hit@1 for any external comparison.

---

## Appendix D — what NOT to do, and why

These are tempting moves that will not help and might hurt.

### Do not increase the contrastive temperature manually.

The model has `logit_scale_a` (and possibly `logit_scale_t`) that learns the temperature itself. The `freeze_logscale_epochs=2` default lets it stabilize before training proceeds. Manually setting a fixed temperature breaks this — the calibrated temperature interacts with the loss landscape.

### Do not lower the warmup below 200.

Warm-start runs need warmup. Run 11 used `--warmup-steps 500`; Run 9-10 used 200. The 500 setting was an explicit improvement. Do not drop below 200 even on long-epoch runs — the optimizer momentum needs time to align with the new LR.

### Do not warm-start from `seventh-fine-tune` or `finetune8`.

The 7th run's `--acoustic-only` damage propagated to the 8th run. Both checkpoints have a damaged text encoder. **Always** warm-start from `sixth-fine-tune/best.pt` or any later run that itself warm-started from a healthy checkpoint (Run 9, 10, 11). Verify by checking first-batch loss is ≤ 4.5; if it is ~6.7, you have warm-started from damage and should abort.

### Do not enable per-recording GPT descriptions before doing §3.1.

If you generate per-recording descriptions but the prompt still allows species names, you have multiplied the dataset size without breaking the shortcut. Strip names first, then jitter.

### Do not switch to `mean similarity` over crops at eval time.

`max similarity` over multi-crop gallery is correct for retrieval (the right answer is "is the bird here in any 10s window"). `mean similarity` averages with silent crops and *suppresses* good matches. This mirrors why `all_variants` text averaging happens *before* normalization — averaging in the wrong place destroys signal.

### Do not increase batch size beyond 16 unless you have more VRAM.

Run 11's `effective batch 128` (16 × accum 8) is optimized for the RTX 4070's 12 GB. Pushing to 24 or 32 OOMs even with AMP.

### Do not reduce `--epochs` below 15 for Run 12.

The drastic changes (especially §2 multi-crop and §3 name-stripped text) add new variance to training. The model needs more epochs, not fewer, to stabilize on the now-richer signal. Start with 18 and only stop early if val_loss starts rising.

### Do not change the audio sample rate.

`TARGET_SR=48_000` is non-negotiable for `laion/clap-htsat-fused`. The HTSAT mel feature extractor is calibrated for 48 kHz. 44.1 kHz or 16 kHz silently degrade performance.

### Do not change `CLIP_DURATION_S` from 10.0.

CLAP was trained on 10s clips. Changing to 5s or 20s will alter the spectrogram dimensions in ways that bypass the pretrained feature extractor's calibration, degrading transfer.

### Do not skip the audit document for Run 12.

Run 7 had no audit. Run 8 inherited Run 7's damage. The audit document is the only thing standing between you and another `--acoustic-only`-class disaster.

---

*End of document.*

*If a future run has a finding not covered here, append a §15 to this document rather than starting a new one. This file is the project's institutional memory for the model-quality problem; spreading findings across multiple files is how the per-species text shortcut survived 11 runs.*
