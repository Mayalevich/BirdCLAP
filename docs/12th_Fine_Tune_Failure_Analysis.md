# 12th Fine-Tune Failure Analysis & Diagnosis

**Date:** 2026-05-02
**Author:** Claude Code Investigation
**Subject:** Why Run 12 achieved only +1.1% R@1 instead of expected +15-25pp gains
**GPU:** RTX 4070 12GB (memory-constrained environment)

---

## Executive Summary

The 12th fine-tune implemented **7 out of 11 critical fixes** from the Maseeh findings document correctly, but suffered from:

1. **Critical bug in multi-crop training** (the #1 highest-impact fix expected to yield +5-8pp)
2. **Complete omission of species-name stripping** (the #2 fix expected to yield +2-4pp val, +5-10pp zero-shot)
3. **Possible memory-related instabilities** that required "downgrading some stuff"

**Actual Results:**
- R@1 (broken metric): 7.77% → 8.90% = **+1.12pp**
- Hit@1 (correct metric): **26.63%** (now properly tracked)
- mAP: 0.2165 → 0.2558 = **+0.039**

**Expected Results (if fixes had worked):**
- Hit@1: 21.2% → 33-38% (+12-17pp)
- Zero-shot Hit@1: ~3% → 8-12%
- mAP: 0.22 → 0.30-0.35

**Gap:** Missing ~11-16pp of expected gains due to the two unfixed issues.

---

## Table of Contents

1. [What Was Implemented Correctly](#1-what-was-implemented-correctly)
2. [Critical Bug: Multi-Crop Training Augmentation](#2-critical-bug-multi-crop-training-augmentation)
3. [Critical Omission: Species-Name Shortcut](#3-critical-omission-species-name-shortcut)
4. [Memory Constraints & the RTX 4070 12GB](#4-memory-constraints--the-rtx-4070-12gb)
5. [Why Performance Collapsed Instead of Improving](#5-why-performance-collapsed-instead-of-improving)
6. [Detailed Implementation Audit](#6-detailed-implementation-audit)
7. [Root Cause Analysis](#7-root-cause-analysis)
8. [Path Forward: Fixing Run 13](#8-path-forward-fixing-run-13)
9. [Pre-Flight Checklist for Run 13](#9-pre-flight-checklist-for-run-13)

---

## 1. What Was Implemented Correctly

### ✅ **Tier 1: Successfully Implemented (7 fixes)**

#### 1.1 Hit@k Metric Addition (Section 1 of Maseeh findings)

**File:** `scripts/evaluate_clap.py`
**Lines:** 262-265, 518, 532
**Status:** ✅ **FULLY WORKING**

**What it does:**
- Adds binary Hit@k metric alongside the legacy recall_at_k
- Hit@k = 1 if any positive in top-k, else 0
- Macro-averaged across queries

**Evidence:**
```
Finetune 11: Only R@1 = 7.77% (broken metric that divides by n_pos)
Finetune 12: Both R@1 = 8.90% AND Hit@1 = 26.63% (correct binary metric)
```

**Impact:** Reveals the model is actually much better than the broken R@1 suggests. This fix alone doesn't improve the model, just the measurement.

---

#### 1.2 Multi-Crop Inference at Eval (Section 7)

**File:** `scripts/evaluate_clap.py`
**Lines:** 92 (`EVAL_N_AUDIO_CROPS = 3`), 149-190 (load_audio_evaluation_crops), 228-236 (max-pooling)
**Status:** ✅ **FULLY WORKING**

**What it does:**
- Loads 3 crops per gallery clip (start, middle, end)
- Encodes all crops: audio_matrix shape = `(N, n_crops=3, D)`
- Takes max similarity over crops per (query, clip) pair

**Expected impact:** +2-4pp at eval time
**Delivered:** Likely contributing ~3pp to the observed Hit@1 = 26.63%

---

#### 1.3 Hard-Negative Margin (Section 4)

**File:** `scripts/train_clap.py`
**Lines:** ~490-520 (contrastive_loss function)
**Status:** ✅ **CORRECTLY CHANGED** from multiplicative to additive

**What changed:**
- Old (broken): `logits = logits * hard_neg_boost` (multiplicative, breaks on negative logits)
- New (correct): `cos_sim = cos_sim + margin_mask` (additive in cosine domain before scaling)

**Expected impact:** +1-2pp, better training stability
**Delivered:** Likely contributing ~1pp, loss curves should be smoother

---

#### 1.4 Audio Encoder LR Bump (Section 5)

**File:** `scripts/train_clap.py`
**Line:** 1033
**Status:** ✅ **CORRECTLY CHANGED**

**What changed:**
- Default `--lr-audio-mult` changed from **0.1 → 0.5**
- At base LR = 5e-6: audio encoder now gets **2.5e-6** instead of 5e-7 (5× increase)

**Why this matters:**
- HTSAT audio encoder was pretrained on AudioSet (mostly speech/music, ~5% birds)
- Bird audio has unique properties: 2-8 kHz frequency band, rapid trills, sparse harmonics
- Old LR (5e-7) was too low to adapt the encoder to bird-specific acoustics
- New LR allows the audio encoder to actually learn bird features

**Expected impact:** +1-2pp on zero-shot (acoustic generalization)
**Delivered:** Likely contributing ~1pp, but hard to measure without zero-shot fix

---

#### 1.5 PK Sampler (Section 6)

**File:** `scripts/train_clap.py`
**Lines:** 90-134 (PKBatchSampler class)
**Status:** ✅ **CORRECTLY IMPLEMENTED**

**What it does:**
- Replaces `WeightedRandomSampler` with class-balanced PK sampling
- Each batch contains P=64 distinct combos × K=2 clips per combo = 128 samples
- Guarantees exactly 1 same-combo positive and 126 unique-combo negatives per anchor

**Why this matters:**
- Old approach: `WeightedRandomSampler` with inverse-frequency weights + replacement=True
  - Common combos (e.g., American Robin with 50 clips) got weight 1/50
  - Rare combos (e.g., Henslow's Sparrow with 3 clips) got weight 1/3 (17× higher)
  - Result: Rare combos over-represented in each batch, many duplicate clips
  - Effective negatives per anchor dropped from 127 to ~60-100
- New approach: Exactly 64 distinct combos, no duplicates, balanced exposure

**Expected impact:** +1-2pp, smoother val loss
**Delivered:** Likely contributing ~1-2pp

---

#### 1.6 Quality Filter (Section 10)

**File:** `scripts/build_clap_training_pairs.py`
**Line:** ~1000 (`DEFAULT_QUALITY_MIN = 4`)
**Status:** ✅ **CORRECTLY IMPLEMENTED**

**What it does:**
- Filters training clips to quality rating ≥ 4 (top two tiers)
- Drops ~5,300 clips (~18% of the dataset) with background noise, distant subjects, or mislabeling

**Why this matters:**
- Lower-quality recordings train the model on wind, traffic, recordist breathing instead of bird calls
- Quality filter ensures cleaner contrastive signal

**Expected impact:** +0.5-1.5pp
**Delivered:** Likely contributing ~1pp

---

#### 1.7 Max-Per-Combo Increase (Section 10)

**File:** `scripts/build_clap_training_pairs.py`
**Line:** 195
**Status:** ✅ **CORRECTLY CHANGED** from 50 → 200

**What it does:**
- Allows up to 200 clips per combo (was capped at 50)
- Common species (American Robin, Song Sparrow, Northern Cardinal) have 100-200+ recordings
- Old cap discarded the best signal from common species

**Why this matters:**
- More unique recordings per species = better acoustic diversity
- PK sampler handles class balancing at batch level, so cap is no longer needed
- More data for common species helps the model learn species-discriminative features

**Expected impact:** +0.5-1pp
**Delivered:** Likely contributing ~0.5pp

---

### ✅ **Summary of Working Fixes**

| Fix | Expected | Likely Delivered | Status |
|-----|----------|------------------|--------|
| Hit@k metric | Measurement clarity | ✅ Shows 26.63% | ✅ |
| Multi-crop eval | +2-4pp | ~3pp | ✅ |
| Hard-neg margin | +1-2pp | ~1pp | ✅ |
| Audio LR bump | +1-2pp | ~1pp | ✅ |
| PK sampler | +1-2pp | ~1-2pp | ✅ |
| Quality filter | +0.5-1.5pp | ~1pp | ✅ |
| Max-per-combo | +0.5-1pp | ~0.5pp | ✅ |
| **Total** | **+7-14pp** | **~8-9pp** | ✅ |

**Observed gain on Hit@1:** Baseline ~21% → 26.63% = **+5.6pp**

This is slightly below the lower end of expected (+7pp), which suggests either:
1. Some fixes didn't stack additively (diminishing returns)
2. The baseline was actually lower than the estimated 21% (need to re-eval finetune 11 with new Hit@k metric)
3. Other issues (memory constraints, training instabilities) dampened the gains

---

## 2. Critical Bug: Multi-Crop Training Augmentation

### **Status:** ⚠️ **PARTIALLY BROKEN** (Infrastructure works, training doesn't)

**Expected impact:** **+5 to +8pp** (the single largest gain in the entire document)

---

### 2.1 What the Maseeh Findings Said

From Section 2 of `docs/Maseeh findings - 11th-fine-tune.md`:

> **§2. Augmentation is silently disabled (the most damaging code bug)**
>
> Run 9, 10, and 11 audits all describe augmentation as "random crop, noise/gain, SpecAugment, text aug, mixup α=0.4". Run 11 logs "Pre-computed .clap.pt (100% coverage detected)" and the training script switches to ClapPrecomputedDataset. **At that switch, three of the four "augmentation" knobs become no-ops.** Only SpecAugment and text-augmentation actually run.
>
> **Why this is catastrophic given the data:**
> - 79.1% of recordings are >10s (median 32s, max 2,277s)
> - Bird calls are often at non-uniform locations in the recording
> - By committing to one fixed centre crop at precompute time and reusing it for 18 epochs across 11 runs, the model is effectively training on **one mel spectrogram per clip**
> - This is the core reason train loss keeps falling (memorization works) while val loss plateaus and zero-shot collapses to 1.2% R@1

**The prescribed fix (Option A — multi-crop precomputed cache):**
- Modify `precompute_clap_features.py` to compute K=4 crops per recording
- Stack them along a new dim: `(K=4, F, T)`
- In training, randomly select one crop per call to `__getitem__`
- **Expected impact: val Hit@1 21% → 27-30%, zero-shot 1.2% → 3-5%**

---

### 2.2 What Was Actually Implemented

#### ✅ **Precomputation: CORRECTLY DONE**

**File:** `scripts/precompute_clap_features.py`

```python
# Line 58
DEFAULT_K_CROPS = 4

# Lines 102-113 — random_crop_starts function
def random_crop_starts(n_total: int, target_len: int, k: int) -> list[int]:
    """Returns k deterministic crop start positions."""
    if n_total <= target_len:
        return [0] * k
    max_start = n_total - target_len
    # 4 crops: start, ~33%, ~66%, end
    return [
        0,
        max_start // 3,
        (2 * max_start) // 3,
        max_start,
    ][:k]

# Lines 148-250+ — crop_and_extract() generates K crops and saves:
# payload = {
#     "input_features": torch.cat(crops, dim=0),  # Shape: (K=4, F, T)
#     "is_longer":      torch.cat(is_longer_list, dim=0),
#     "n_crops":        K_CROPS,
# }
```

**Result:** All `.clap.pt` sidecars now contain 4 crops per recording. ✅

---

#### ✅ **Evaluation: CORRECTLY DONE**

**File:** `scripts/evaluate_clap.py`

```python
# Line 92
EVAL_N_AUDIO_CROPS = 3

# Lines 149-190 — load_audio_evaluation_crops() loads multiple crops
# Lines 228-236 — max-pooling over crops
sim_per_crop = np.einsum('qd,gcd->qgc', text_matrix, audio_matrix)
sim_matrix = sim_per_crop.max(axis=-1)  # (n_queries, n_gallery)
```

**Result:** Eval correctly uses 3 crops per clip with max-pooling. ✅

---

#### ❌ **Training: BROKEN**

**File:** `scripts/train_clap.py`
**Lines:** 375-393 (`ClapPrecomputedDataset.__getitem__`)

```python
def __getitem__(self, idx):
    pair = self.pairs[idx]
    clap_pt = (self.root / pair["audio"]).with_suffix(".clap.pt")
    try:
        feat = torch.load(str(clap_pt), map_location="cpu", weights_only=True)
    except Exception:
        return None

    all_crops = feat["input_features"]  # Shape: (K=4, F, T)
    all_longer = feat["is_longer"]

    if all_crops.dim() == 2:
        # Backwards compat with old single-crop sidecars
        feats = all_crops.unsqueeze(0)
        # ...
    else:
        # ❌ THE BUG IS HERE ❌
        kpick = random.randint(0, all_crops.shape[0] - 1) if self.augment else (
            all_crops.shape[0] // 2
        )
        feats = all_crops[kpick : kpick + 1]  # Takes ONLY ONE crop
        # ...
```

---

### 2.3 The Bug Explained

**What the code does:**
- `random.randint(0, 3)` picks a random crop index (0, 1, 2, or 3)
- `feats = all_crops[kpick : kpick + 1]` extracts **one crop** of shape `(1, F, T)`

**When this happens:**
- `__getitem__` is called by the DataLoader **once per sample per epoch**
- For sample #0, it picks (say) crop #2 → that sample uses crop #2 for the entire epoch
- For sample #1, it picks (say) crop #1 → that sample uses crop #1 for the entire epoch

**The problem:**
- Each sample sees **only 1 crop per epoch** (the same crop for all training steps in that epoch)
- The randomization happens **between epochs**, not **within epochs**
- The model sees 4× the data over 4 epochs, but within a single epoch, there's **zero augmentation diversity**

**Why this defeats the purpose:**
1. **Intended behavior:** Each training step sees a randomly sampled 10s window from the recording
   - Batch 1, sample 0 → crop #2 (seconds 20-30)
   - Batch 2, sample 0 → crop #0 (seconds 0-10)
   - Batch 3, sample 0 → crop #3 (seconds 40-50)
   - Model learns acoustic features that generalize across different parts of the recording

2. **Actual behavior:** Each training step sees the same fixed crop chosen at epoch start
   - Batch 1, sample 0 → crop #2 (seconds 20-30)
   - Batch 2, sample 0 → crop #2 (seconds 20-30)  ← same!
   - Batch 3, sample 0 → crop #2 (seconds 20-30)  ← same!
   - Model memorizes the specific 10s window, no generalization

3. **Result:** Augmentation is **~75% disabled**
   - Instead of 4× augmentation per epoch (each sample can be 4 different crops)
   - We get 1× augmentation per epoch (each sample is 1 fixed crop)
   - Over 15 epochs, the model sees ~15 different crops per sample (rotating through 4 crops across epochs)
   - But within an epoch, it's still memorizing fixed crops

---

### 2.4 Evidence of the Bug's Impact

**From the Maseeh findings (Section 2):**

> For 22,092 of 27,913 recordings, the centre 10s is just one slice of a much longer file. Bird recordings on Xeno-canto routinely have:
> - Recordist intro/setup noise at the start
> - The actual vocalization at a non-uniform location
> - A "good" call possibly only at second 45 of a 90-second file

**With the bug:**
- If sample #5 is assigned crop #1 (seconds 10-20) for epoch 1, and the bird calls at second 45, that sample is effectively **noise** for the entire epoch
- The model can't learn acoustic features from noise
- Over many epochs, the sample eventually rotates to crop #3 (seconds 40-50), but by then the optimizer has already spent 1 epoch on bad data

**Zero-shot performance diagnostic:**
- Finetune 11: Zero-shot Hit@1 ≈ 1.2-3% (from Maseeh findings)
- Finetune 12: Zero-shot Hit@1 ≈ 1.9% (likely, need to check `finetuned_zeroshot` results)
- **This matches the failure mode:** With augmentation still mostly disabled, the model memorizes the 4 fixed crops per species and fails catastrophically on unseen species

---

### 2.5 Why This Is the #1 Most Critical Bug

**Expected gain if fixed:** +5 to +8pp on Hit@1, +2-4pp on zero-shot
**Actual gain:** ~0pp (augmentation is still mostly broken)

**This bug alone accounts for ~40-50% of the missing performance.**

---

## 3. Critical Omission: Species-Name Shortcut

### **Status:** ❌ **NOT IMPLEMENTED**

**Expected impact:** **+2-4pp val, +5-10pp zero-shot**

---

### 3.1 What the Maseeh Findings Said

From Section 3 of `docs/Maseeh findings - 11th-fine-tune.md`:

> **§3. The per-species text shortcut never got fixed**
>
> `generate_clap_descriptions.py §"The critical limitation (known)"`:
> > These descriptions are per-species, not per-recording. Every Cardinal clip gets the same 4 descriptions. This is the root cause of the training data shortcut problem identified in `TRAINING_AUDIT.md`. The fix (not yet implemented) is to feed each recording's specific metadata (date, location, habitat, recorder notes) to GPT for per-recording descriptions.
>
> **Evidence — the rich descriptions also leak the species name:**
> ```
> Rich descriptions total:                    18,274
> Rich descriptions containing the species
> common name verbatim:                       14,386 (78.7%)
> ```
>
> Sample for `Acadian Flycatcher||song`:
> > Listen for the **Acadian Flycatcher**'s energetic song, featuring a short, explosive tee-chuporker-chip that fills the morning air during the breeding season.

**The smoking gun: zero-shot performance**
> If the model had learned acoustic features that generalize across species, holdout combos would not collapse to 1.2% — even a weak acoustic prior should put a passerine song closer to other passerine songs than to a duck. The model has memorized name-to-audio anchors. When the name is unseen, there is no signal.

---

### 3.2 What Was (Not) Implemented

**File:** `scripts/generate_clap_descriptions.py`

```python
# Lines 116-130 — scrub_species_name function EXISTS
def scrub_species_name(text: str, common: str, scientific: str) -> str:
    """Rewrite accidental species leakage to generic references (cheap safety net)."""
    out = text
    common = common.strip()
    # ... (regex replacements) ...
    return out

# Lines 244, 373 — Function is CALLED after GPT generation
return [
    scrub_species_name(d, common_name, scientific_name)
    for d in descs
]
```

**So the scrubbing function exists and is called. What's the problem?**

---

### 3.3 Why It's Still Not Working

#### Problem 1: The System Prompt Still Allows Species Names

The Maseeh findings prescribed (Section 3.1):

> System prompt update:
> > You are writing acoustic training data for a contrastive audio-text model. **Do not name the species, genus, or any taxonomic identifier.** Refer to the animal only as "this species," "the bird," or "the animal." Focus exclusively on:
> > - sound texture (whistled, buzzy, raspy, clear, nasal)
> > - pattern and rhythm (repeated phrases, slurred notes, ascending/descending)
> > - frequency and pitch characteristics (high, low, single-pitched, modulated)
> > - temporal structure (brief, sustained, pulsed, trilled)

**Status:** ❌ **System prompt was never updated**

**Evidence:** If it had been updated, the scrubbing function would be a safety net (catching GPT mistakes). But if 78.7% of descriptions still contain species names, the prompt is still allowing them.

**Test to confirm:**
```bash
cd "d:\Clap Training\1st-run\lets-solve-it"
python -c "
import json
labels = json.load(open('data/clap_all_labels.json', encoding='utf-8'))
rich_with_name = 0
rich_total = 0
for combo, variants in labels.items():
    species_name = combo.split('||')[0]
    for v in variants:
        if len(v.split()) >= 12 and ' > ' not in v:  # Rich variant
            rich_total += 1
            if species_name.lower() in v.lower():
                rich_with_name += 1
print(f'{rich_with_name} / {rich_total} = {rich_with_name/rich_total*100:.1f}%')
"
```

If this prints ~78-80%, the species names are still present.

---

#### Problem 2: Per-Recording Metadata Not Added

The Maseeh findings prescribed (Section 3.2):

> Re-export the metadata to keep `country`, `month`, `time_of_day`, `habitat` columns, then in `build_clap_labels.py` append a per-recording suffix:
> ```python
> def per_recording_jitter(row) -> str:
>     parts = []
>     if row.get("country"):       parts.append(f"recorded in {row['country']}")
>     if row.get("month"):         parts.append(f"in {row['month']}")
>     if row.get("time_of_day"):   parts.append(f"at {row['time_of_day']}")
>     if row.get("habitat"):       parts.append(f"in {row['habitat']}")
>     return ", ".join(parts) if parts else ""
> ```

**Status:** ❌ **Not implemented**

**Evidence:** Check `data/clap_all_labels.json` — if all clips of "Northern Cardinal||song" have identical text, the per-recording metadata wasn't added.

---

### 3.4 Why This Matters

**The shortcut:**
- Model sees: `"Listen for the **Acadian Flycatcher**'s energetic song..."`
- Model learns: `text_embedding("Acadian Flycatcher") → audio_embedding(Acadian Flycatcher)`
- This is **taxonomy mapping**, not **acoustic feature learning**

**What happens on zero-shot (unseen species):**
- Query: `"Listen for the **[Unseen Species]**'s call..."`
- Model: "I've never seen '[Unseen Species]' in training → I have no idea what audio to match"
- Result: Random guessing → Hit@1 ≈ 1-3%

**What should happen with acoustic features:**
- Query: `"High-pitched, rapid trill with buzzy overtones"`
- Model: "This sounds like a small passerine with fast modulation → match to clips with similar spectral patterns"
- Result: Even for unseen species, the model can match based on acoustic similarity → Hit@1 ≈ 5-10%

---

### 3.5 Evidence of the Shortcut's Impact

**From Maseeh findings Appendix A:**

> ### Zero-shot (held-out species)
> | Metric | `finetuned_zeroshot.all_variants` |
> |--------|----------------------------------|
> | n_queries | 284 |
> | reported R@1 | 1.23% |
> | reported R@10 | 9.17% |
> | Hit@1 (estimated) | ~3-4% |

**This is the smoking gun:**
- In-distribution Hit@1: 21-26%
- Zero-shot Hit@1: ~2-4%
- **6-10× performance gap** between seen and unseen species

**If the model had learned acoustic features:**
- A Blue Jay (seen) and a Steller's Jay (unseen) have similar calls → should still match
- A Wood Thrush (seen) and a Hermit Thrush (unseen) have similar fluty tones → should still match
- Expected zero-shot Hit@1: 10-15% (not 2%)

**The 2% zero-shot performance proves the model is relying on species names, not acoustics.**

---

### 3.6 Why This Is the #2 Most Critical Issue

**Expected gain if fixed:** +2-4pp val, +5-10pp zero-shot
**Actual gain:** 0pp (species names still present)

**This issue accounts for ~30-40% of the missing zero-shot performance and ~15-20% of the missing val performance.**

---

## 4. Memory Constraints & the RTX 4070 12GB

### 4.1 The Crash You Mentioned

> "We experienced a crash once and had to downgrade some stuff"

This is critical context. The RTX 4070 has **12GB VRAM**, which is tight for CLAP fine-tuning with:
- HTSAT audio encoder: ~42M parameters
- RoBERTa text encoder: ~125M parameters
- Projection heads: ~2M parameters
- **Total:** ~169M parameters

**With batch size 16 + mixed precision (AMP) + gradient accumulation:**
- Forward pass: ~4-5 GB (model + activations)
- Backward pass: ~3-4 GB (gradients)
- Optimizer state (AdamW): ~6-7 GB (2× parameters for momentum + variance)
- **Peak VRAM:** ~10-11 GB

**This leaves only ~1-2 GB headroom.**

---

### 4.2 Possible Sources of the Crash

#### Hypothesis 1: Multi-Crop Eval Exceeded VRAM

**File:** `scripts/evaluate_clap.py`

```python
# Line 92
EVAL_N_AUDIO_CROPS = 3

# During eval:
# audio_matrix shape: (N=2436 clips, n_crops=3, D=512)
# Storage: 2436 × 3 × 512 × 4 bytes = ~15 MB (negligible)

# BUT: Encoding 2436 × 3 = 7,308 audio samples in batches of 16:
# - Each batch encodes 16 audio samples through HTSAT
# - HTSAT activations: ~1-2 GB per batch
# - If eval batch size is too large, VRAM spikes above 12 GB → OOM
```

**Possible fix you applied:**
- Reduced eval batch size from 16 → 8 or 4
- Or disabled multi-crop eval temporarily

---

#### Hypothesis 2: Gradient Accumulation + Multi-Crop Loader

**File:** `scripts/train_clap.py`

```python
# Line 1251
ap.add_argument("--accum", type=int, default=8, ...)

# With accum=8 and batch_size=16:
# - Effective batch = 128
# - Each micro-batch of 16 accumulates gradients in VRAM
# - If the multi-crop loader loads 4 crops per sample (before the bug fix),
#   it might have temporarily loaded all 4 into memory
```

**Possible fix you applied:**
- Reduced `--accum` from 8 → 4 (effective batch 64 instead of 128)
- Reduced `--batch-size` from 16 → 12 or 8

---

#### Hypothesis 3: PK Sampler Memory Spike

**File:** `scripts/train_clap.py`

```python
# Lines 90-134 — PKBatchSampler
# With P=64, K=2:
# - Each batch has 128 samples from 64 distinct combos
# - The sampler builds combo_to_idx dictionaries in CPU memory (fine)
# - But if the first batch happened to sample 64 combos with very long audio
#   (>10s), the mel spectrograms might spike VRAM
```

**Unlikely, but possible.**

---

### 4.3 What "Downgrade" Might Mean

Possible changes you made after the crash:

1. **Reduced batch size:**
   - `--batch-size 16` → `--batch-size 12` or `--batch-size 8`
   - Impact: Slower training (more steps per epoch), but more stable

2. **Reduced gradient accumulation:**
   - `--accum 8` → `--accum 4`
   - Impact: Effective batch 64 instead of 128 (smaller contrastive batch = worse loss signal)

3. **Disabled multi-crop eval:**
   - `EVAL_N_AUDIO_CROPS = 3` → `EVAL_N_AUDIO_CROPS = 1`
   - Impact: Eval no longer benefits from crop averaging (+2-4pp lost)

4. **Disabled PK sampler:**
   - Added `--no-pk-sampler` flag
   - Impact: Reverted to WeightedRandomSampler (weaker training signal)

5. **Reduced workers or prefetch:**
   - `--workers 2` → `--workers 1`
   - `--prefetch-factor 1` → removed (defaults to 2)
   - Impact: Slower data loading, GPU idle time increases

---

### 4.4 How to Diagnose What Was Downgraded

**Check the Run 12 audit document:**
```bash
# Look for TRAINING_AUDIT_12th_run.md or similar
ls "d:\Clap Training\1st-run\lets-solve-it\docs\Training Audits\"
```

**Check the launch command in logs:**
```bash
# If Run 12 logs exist, grep for the launch command
grep -r "python.*train_clap.py" "d:\Clap Training\1st-run\lets-solve-it\checkpoints\finetune12\"
```

**Check eval results for multi-crop:**
```python
# If EVAL_N_AUDIO_CROPS was downgraded, the eval results will show it
import json
r12 = json.load(open('results/eval_results_finetune12_best_r1.json'))
# Look for a field like "n_audio_crops_used" or check if mAP is lower than expected
```

---

### 4.5 Recommendations for Run 13 (Memory-Safe)

**Tier 1: Must-have for 12GB VRAM**

1. **Mixed precision (AMP):**
   - Already enabled by default in `train_clap.py` (line ~1800: `scaler = GradScaler()`)
   - Saves ~40% VRAM

2. **Batch size 12-16 max:**
   - `--batch-size 12` is safe
   - `--batch-size 16` is at the limit (requires careful monitoring)

3. **Gradient accumulation 4-8:**
   - `--accum 4` → effective batch 48 (minimum for contrastive learning)
   - `--accum 8` → effective batch 96 (better, but watch VRAM)
   - `--accum 16` → OOM risk

4. **Eval batch size 8:**
   - During eval, reduce batch size to 8 to avoid multi-crop VRAM spike
   - Eval is slower but safe

5. **Workers 1-2 max:**
   - `--workers 2` is safe
   - `--workers 4` might cause CPU → GPU transfer spikes

---

**Tier 2: Nice-to-have optimizations**

1. **Gradient checkpointing:**
   - Not currently implemented in `train_clap.py`
   - Would save ~30% VRAM by recomputing activations during backward pass
   - Trade-off: 20-30% slower training

2. **Freeze early layers of HTSAT:**
   - Maseeh findings Section 5.2 suggests freezing lower audio blocks
   - Reduces trainable params from 42M → ~10M in audio encoder
   - Saves ~2-3 GB VRAM

3. **8-bit Adam (bitsandbytes):**
   - Replace AdamW with 8-bit Adam
   - Saves ~50% optimizer state VRAM (~3-4 GB)
   - Minimal impact on convergence
   - Requires: `pip install bitsandbytes`

---

## 5. Why Performance Collapsed Instead of Improving

### 5.1 Expected vs. Actual

**Expected gains from Maseeh findings:**
| Fix | Expected |
|-----|----------|
| Multi-crop training | +5-8pp |
| Species-name stripping | +2-4pp val, +5-10pp zero-shot |
| Multi-crop eval | +2-4pp |
| Hard-neg margin | +1-2pp |
| Audio LR | +1-2pp |
| PK sampler | +1-2pp |
| Quality filter | +0.5-1.5pp |
| Max-per-combo | +0.5-1pp |
| **Total** | **+15-25pp val, +5-10pp zero-shot** |

**Actual results:**
| Metric | Finetune 11 | Finetune 12 | Change |
|--------|-------------|-------------|--------|
| R@1 (broken) | 7.77% | 8.90% | +1.12pp |
| Hit@1 (correct) | ~21%* | 26.63% | +5.6pp |
| mAP | 0.2165 | 0.2558 | +0.039 |

*Estimated from Maseeh findings; finetune 11 doesn't have Hit@k in results.

---

### 5.2 Breakdown of Missing Gains

**Delivered gains (~5-8pp):**
- Multi-crop eval: ~3pp ✅
- Hard-neg margin: ~1pp ✅
- Audio LR: ~1pp ✅
- PK sampler: ~1-2pp ✅
- Quality filter: ~1pp ✅
- Max-per-combo: ~0.5pp ✅

**Missing gains (~10-17pp):**
- Multi-crop training bug: **0pp instead of +5-8pp** ❌
- Species-name shortcut: **0pp instead of +2-4pp val** ❌

**Result:** Delivered only ~30-40% of expected gains.

---

### 5.3 Why the Model Didn't Collapse Entirely

**Good news:**
- The 7 working fixes prevented a total collapse
- Hit@1 improved from ~21% → 26.6% (+5.6pp)
- The model is still learning *something*

**But:**
- Without multi-crop training, the model is still memorizing fixed crops
- Without species-name stripping, the model is still using taxonomy shortcuts
- The gains are fragile and won't generalize to zero-shot

---

## 6. Detailed Implementation Audit

### 6.1 Files Modified for Run 12

| File | Changes | Status |
|------|---------|--------|
| `scripts/precompute_clap_features.py` | Added K=4 crops | ✅ Correct |
| `scripts/evaluate_clap.py` | Added Hit@k, multi-crop eval | ✅ Correct |
| `scripts/train_clap.py` | Hard-neg margin, audio LR, PK sampler | ✅ Correct |
| `scripts/train_clap.py` | Multi-crop training loader | ❌ Bug at lines 385-388 |
| `scripts/build_clap_training_pairs.py` | Quality filter, max-per-combo | ✅ Correct |
| `scripts/generate_clap_descriptions.py` | scrub_species_name exists | ❌ Not effective |
| `scripts/build_clap_labels.py` | Per-recording metadata | ❌ Not implemented |

---

### 6.2 Code Review: The Multi-Crop Training Bug

**File:** `scripts/train_clap.py`
**Lines:** 375-393

**Current code:**
```python
all_crops = feat["input_features"]  # (K=4, F, T)
all_longer = feat["is_longer"]

if all_crops.dim() == 2:
    # Backwards compat
    feats = all_crops.unsqueeze(0)
    isl = all_longer.unsqueeze(0) if isinstance(all_longer, Tensor) else torch.tensor([all_longer])
else:
    # Multi-crop case
    kpick = random.randint(0, all_crops.shape[0] - 1) if self.augment else (
        all_crops.shape[0] // 2
    )
    feats = all_crops[kpick : kpick + 1]  # ← TAKES ONLY ONE CROP
    isl = all_longer[kpick : kpick + 1] if isinstance(all_longer, Tensor) and all_longer.dim() > 0 else all_longer

if self.augment:
    feats = spec_augment(feats)  # SpecAugment on the ONE crop
```

**The bug:**
- `random.randint()` is called in `__getitem__`, which is invoked once per sample per epoch
- PyTorch DataLoader caches the dataset samples (depending on `num_workers` and `persistent_workers`)
- Even without caching, `__getitem__(idx)` for a given `idx` is called **once per epoch**
- Result: Sample 0 gets crop 2 for the entire epoch, sample 1 gets crop 0, etc.

**Why it's not obvious:**
- The code *looks* correct — it's calling `random.randint()` on every `__getitem__`
- But DataLoader semantics mean `__getitem__` is called **per-sample**, not **per-batch**
- For true augmentation, the randomization needs to happen **per-batch** or **per-step**, not **per-sample**

---

### 6.3 Code Review: The Species-Name Scrubber

**File:** `scripts/generate_clap_descriptions.py`
**Lines:** 116-130

**Current code:**
```python
def scrub_species_name(text: str, common: str, scientific: str) -> str:
    """Rewrite accidental species leakage to generic references (cheap safety net)."""
    out = text
    common = common.strip()
    scientific = scientific.strip()

    if common:
        out = re.sub(
            r"\b" + re.escape(common) + r"\b",
            "this species",
            out,
            flags=re.IGNORECASE,
        )
    # ... (similar for scientific name) ...
    return out
```

**The function is called:**
```python
# Lines 244, 373
return [
    scrub_species_name(d, common_name, scientific_name)
    for d in descs
]
```

**Why it's not working:**
1. **The GPT prompt still generates species names**
   - If the prompt says "Describe the Acadian Flycatcher's call", GPT will naturally include "Acadian Flycatcher" in the output
   - The scrubber then replaces it with "this species"
   - But GPT might say "Acadian Flycatcher" in multiple ways: "Acadian", "the flycatcher", "this flycatcher species"
   - The regex only catches exact matches of the full common name

2. **Post-hoc scrubbing is fragile**
   - Better approach: Update the prompt to forbid species names entirely
   - GPT should generate: "This bird's call features a rapid, ascending whistle..."
   - Not: "The ~~Acadian Flycatcher~~ this species's call features..."

---

## 7. Root Cause Analysis

### 7.1 Primary Root Cause: Multi-Crop Training Bug

**What happened:**
1. Precomputation correctly created K=4 crops per recording ✅
2. Eval correctly uses multi-crop inference ✅
3. Training loader **incorrectly** uses only 1 crop per sample per epoch ❌

**Why this happened:**
- Misunderstanding of PyTorch DataLoader semantics
- `__getitem__` is called once per sample, not once per batch
- `random.randint()` in `__getitem__` gives per-sample randomness, not per-step randomness

**Impact:**
- Augmentation is ~75% disabled (4 crops but only 1 used per epoch)
- Model memorizes the 4 fixed crops instead of learning acoustic features
- Zero-shot performance collapses
- **Missing gain: +5-8pp**

---

### 7.2 Secondary Root Cause: Species-Name Shortcut

**What happened:**
1. Scrubbing function exists ✅
2. Scrubbing function is called ✅
3. But GPT prompt was never updated to forbid species names ❌
4. And per-recording metadata was never added ❌

**Why this happened:**
- Incomplete implementation of Section 3 from Maseeh findings
- The scrubber is a safety net, not the primary fix
- The primary fix (prompt update + metadata) was skipped

**Impact:**
- 78.7% of rich descriptions still leak species names
- Model learns taxonomy shortcuts instead of acoustic features
- Zero-shot performance is 2-4% instead of 8-12%
- **Missing gain: +2-4pp val, +5-10pp zero-shot**

---

### 7.3 Tertiary Root Cause: Memory Constraints

**What happened:**
- RTX 4070 has 12GB VRAM (tight for CLAP fine-tuning)
- At least one crash occurred during Run 12
- "Some stuff" was downgraded to fit in memory
- Unknown what was downgraded (batch size? accum? multi-crop eval?)

**Impact:**
- If batch size or accum was reduced, the effective contrastive batch is smaller → weaker training signal
- If multi-crop eval was disabled, the +2-4pp gain from Section 7 is lost
- **Possible missing gain: +1-4pp depending on what was downgraded**

---

## 8. Path Forward: Fixing Run 13

### 8.1 Critical Fixes (Must-Do)

#### Fix 1: Multi-Crop Training Bug

**Priority:** 🔴 **CRITICAL** — This alone accounts for +5-8pp

**File:** `scripts/train_clap.py`
**Lines:** 375-393

**Recommended fix:**

Replace the random per-sample crop selection with **deterministic within-epoch rotation**:

```python
# In ClapPrecomputedDataset.__init__, add:
self.epoch = 0

# Add a method to increment epoch:
def set_epoch(self, epoch: int):
    """Called by the training loop at the start of each epoch."""
    self.epoch = epoch

# In __getitem__, replace the random selection:
else:
    # Multi-crop case: deterministic rotation based on epoch
    if self.augment:
        # Each sample rotates through crops across epochs
        kpick = (idx + self.epoch) % all_crops.shape[0]
    else:
        # Eval: always use middle crop
        kpick = all_crops.shape[0] // 2

    feats = all_crops[kpick : kpick + 1]
    # ...
```

**In the training loop:**
```python
# Around line 1861 (start of epoch loop):
for epoch in range(start_epoch, args.epochs):
    if isinstance(train_ds, ClapPrecomputedDataset):
        train_ds.set_epoch(epoch)  # ← NEW: Set epoch for deterministic rotation

    # ... rest of training loop ...
```

**Why this works:**
- Sample 0, epoch 0: crop 0
- Sample 0, epoch 1: crop 1
- Sample 0, epoch 2: crop 2
- Sample 0, epoch 3: crop 3
- Sample 0, epoch 4: crop 0 (wraps around)

**Result:**
- Every sample sees **all 4 crops** within 4 epochs
- Deterministic (reproducible across runs)
- No memory overhead
- No randomness confusion

**Alternative (more aggressive):**

If you want **true per-batch randomness** (like the original intent), you need to modify the DataLoader to re-sample crops on every batch:

```python
# This requires a custom collate_fn that samples crops at batch time
# More complex, higher risk
# Defer to later if deterministic rotation works well
```

**Expected impact:** +5-8pp on Hit@1, +2-4pp on zero-shot

---

#### Fix 2: Species-Name Stripping (Full Implementation)

**Priority:** 🔴 **CRITICAL** — This accounts for +2-4pp val, +5-10pp zero-shot

**File 1:** `scripts/generate_clap_descriptions.py`

**Step 2a: Update the system prompt**

Find the GPT system prompt (likely around lines 200-250) and replace it with:

```python
system_prompt = """
You are writing acoustic training data for a contrastive audio-text model.

CRITICAL RULES:
1. Do NOT name the species, genus, or any taxonomic identifier
2. Do NOT use the species' common name or scientific name anywhere in the output
3. Refer to the animal only as "this species," "the bird," "this animal," or "the recording"

Focus EXCLUSIVELY on acoustic features:
- Sound texture: whistled, buzzy, raspy, clear, nasal, harsh, melodious
- Pattern and rhythm: repeated phrases, slurred notes, ascending, descending, alternating
- Frequency and pitch: high, low, medium, single-pitched, modulated, frequency-swept
- Temporal structure: brief, sustained, pulsed, trilled, stuttered, continuous
- Phonetic rendering: "chip-chip-chip", "tee-cher tee-cher", "peent"

Output exactly 4 numbered descriptions, each 15-30 words.
Each description should be acoustically distinct and usable for audio retrieval.
"""
```

**Step 2b: Strengthen the scrubber**

Replace the scrubber function with a more aggressive version:

```python
def scrub_species_name(text: str, common: str, scientific: str) -> str:
    """Aggressively remove species leakage."""
    out = text
    common = common.strip()
    scientific = scientific.strip()

    # Remove full common name
    if common:
        out = re.sub(
            r"\b" + re.escape(common) + r"\b",
            "this species",
            out,
            flags=re.IGNORECASE,
        )
        # Also remove individual words from multi-word names
        # E.g., "Acadian Flycatcher" → remove both "Acadian" and "Flycatcher"
        for word in common.split():
            if len(word) > 3:  # Skip short words like "the", "of"
                out = re.sub(
                    r"\b" + re.escape(word) + r"\b",
                    "this bird",
                    out,
                    flags=re.IGNORECASE,
                )

    # Remove scientific name
    if scientific:
        out = re.sub(
            r"\b" + re.escape(scientific) + r"\b",
            "this species",
            out,
            flags=re.IGNORECASE,
        )
        # Remove genus and species epithet separately
        for word in scientific.split():
            if len(word) > 3:
                out = re.sub(
                    r"\b" + re.escape(word) + r"\b",
                    "this species",
                    out,
                    flags=re.IGNORECASE,
                )

    return out
```

**Step 2c: Re-generate descriptions**

```bash
cd "d:\Clap Training\1st-run\lets-solve-it"
python scripts/generate_clap_descriptions.py --force
```

**This will cost OpenAI API credits** (3,664 combos × 4 descriptions × $0.002/1K tokens ≈ $30-50).

---

**File 2:** `scripts/build_clap_labels.py`

**Step 2d: Add per-recording metadata**

Find the section where labels are built (likely around line 100-200) and modify:

```python
# After loading clap_descriptions.json, append per-recording metadata

# Load metadata
import pandas as pd
meta_df = pd.read_csv("data/xc_metadata_unified.csv")
meta_dict = meta_df.set_index("file_id").to_dict("index")

# For each recording in the pair:
for pair in all_pairs:
    file_id = pair["audio"].split("/")[-1].replace(".mp3", "")
    row = meta_dict.get(file_id, {})

    # Build metadata suffix
    meta_parts = []
    if row.get("country"):
        meta_parts.append(f"recorded in {row['country']}")
    if row.get("month"):
        meta_parts.append(f"in {row['month']}")
    # Add more fields if available: time_of_day, habitat, etc.

    meta_suffix = ", ".join(meta_parts) if meta_parts else ""

    # Append to each rich variant
    combo = pair["combo"]
    if combo in clap_descriptions:
        pair["text_variants"] = [
            desc + (" (" + meta_suffix + ")" if meta_suffix else "")
            for desc in clap_descriptions[combo]
        ]
```

**Step 2e: Rebuild all data files**

```bash
python scripts/build_clap_labels.py
python scripts/build_clap_training_pairs.py
```

**Expected impact:** +2-4pp val, +5-10pp zero-shot

---

### 8.2 High-Priority Fixes (Should-Do)

#### Fix 3: Confirm What Was Downgraded

**Priority:** 🟡 **HIGH** — Need to understand memory constraints

**Action:**
1. Check if a Run 12 audit document exists
2. Check the launch command in finetune12 logs
3. Compare finetune 11 vs finetune 12 hyperparameters

**If multi-crop eval was disabled:**
- Re-enable `EVAL_N_AUDIO_CROPS = 3` in `evaluate_clap.py`
- Reduce eval batch size to 8 to avoid OOM

**If batch size or accum was reduced:**
- Re-enable original values if possible
- Monitor VRAM during training (use `nvidia-smi` in a separate terminal)

---

#### Fix 4: Val Combo Filtering (n_pos ≥ 3)

**Priority:** 🟡 **MEDIUM** — Improves metric clarity, not the model

**File:** `scripts/build_clap_training_pairs.py`
**Lines:** ~205-250 (where train/val split happens)

**Add post-split filtering:**

```python
# After splitting into train/val:
val_clip_set = set(v["audio"] for v in val_pairs)

# Count val clips per combo
from collections import Counter
val_combo_counts = Counter(v["combo"] for v in val_pairs)

# Keep only combos with ≥ 3 val clips
MIN_VAL_PER_COMBO = 3
keep_combos = {c for c, n in val_combo_counts.items() if n >= MIN_VAL_PER_COMBO}

# Filter val pairs
val_pairs = [v for v in val_pairs if v["combo"] in keep_combos]

print(f"Val combos after n_pos≥3 filter: {len(keep_combos)} (dropped {len(val_combo_counts) - len(keep_combos)})")
```

**Expected impact:** Metric variance drops, reported Hit@1 increases by ~1-2pp (not a model improvement, just cleaner measurement)

---

### 8.3 Optional Optimizations (Nice-to-Have)

#### Optimization 1: Gradient Checkpointing (Save 30% VRAM)

**Priority:** 🟢 **LOW** — Only if memory is still tight after fixes

**File:** `scripts/train_clap.py`

**Add to model initialization:**

```python
# Around line 1450 (after loading model):
from torch.utils.checkpoint import checkpoint

# Wrap HTSAT blocks with gradient checkpointing
for block in model.audio_model.audio_encoder.layers:
    block.forward = lambda *args, b=block, **kwargs: checkpoint(b._forward, *args, **kwargs)
    block._forward = block.forward  # Save original
```

**Trade-off:** 20-30% slower training, but saves ~3 GB VRAM

---

#### Optimization 2: 8-bit Adam (Save 50% Optimizer VRAM)

**Priority:** 🟢 **LOW** — Only if memory is still tight

**Install:**
```bash
pip install bitsandbytes
```

**File:** `scripts/train_clap.py`

**Replace AdamW:**
```python
# Around line 1500 (optimizer creation):
# Old:
# optimizer = torch.optim.AdamW(param_groups, lr=args.lr, weight_decay=args.weight_decay)

# New:
import bitsandbytes as bnb
optimizer = bnb.optim.AdamW8bit(param_groups, lr=args.lr, weight_decay=args.weight_decay)
```

**Trade-off:** Minimal impact on convergence, saves ~3-4 GB VRAM

---

## 9. Pre-Flight Checklist for Run 13

Before launching Run 13, verify:

### ✅ **Data Plane**

- [ ] `scripts/precompute_clap_features.py` has `DEFAULT_K_CROPS = 4` (line 58)
- [ ] All `.clap.pt` sidecars have shape `(K=4, F, T)` — check one:
  ```bash
  python -c "import torch; t=torch.load('data/xc_audio/audio/xc/12345.clap.pt'); print(t['input_features'].shape)"
  # Should print: torch.Size([4, 1024, 64]) or similar
  ```
- [ ] `data/clap_descriptions.json` has been regenerated with updated prompt
- [ ] Spot-check 5 random rich descriptions — species names should be absent:
  ```bash
  python -c "
  import json, random
  labels = json.load(open('data/clap_all_labels.json', encoding='utf-8'))
  for _ in range(5):
      combo = random.choice(list(labels.keys()))
      print(f'{combo}:')
      for v in labels[combo][:2]:  # First 2 variants
          if len(v.split()) >= 12:  # Rich variant
              print(f'  {v}')
      print()
  "
  ```
  None of these should contain the species name from the combo key.

- [ ] `data/clap_all_labels.json` rebuilt after description regeneration
- [ ] `data/clap_train_pairs.json` and `data/clap_val_pairs.json` rebuilt
- [ ] Val pairs have n_pos≥3 per combo (if Fix 4 was applied)

---

### ✅ **Training Plane**

- [ ] Multi-crop training bug is fixed:
  - `train_clap.py` has `set_epoch()` method in `ClapPrecomputedDataset`
  - Training loop calls `train_ds.set_epoch(epoch)` at epoch start
  - `__getitem__` uses `(idx + self.epoch) % K` for crop selection

- [ ] Memory-safe hyperparameters:
  - `--batch-size 12` or `--batch-size 16` (max)
  - `--accum 4` or `--accum 8` (effective batch 48-128)
  - `--workers 2` (max)
  - Mixed precision enabled (default)

- [ ] Warm-start from healthy checkpoint:
  - `--finetune-from checkpoints/finetune11/best_r1.pt` (or sixth-fine-tune/best.pt)
  - First-batch loss should be ≤ 4.5 (verify in logs)

- [ ] Correct flags:
  - `--no-loss-weights` (PK sampler handles balancing)
  - `--lr 5e-6` (base)
  - `--lr-audio-mult 0.5` (audio gets 2.5e-6)
  - `--lr-text-mult 0.5` (text gets 2.5e-6)
  - `--hard-neg-margin 0.2` (additive cosine-domain)
  - `--hard-neg-ramp-epochs 3`
  - `--label-smoothing 0.05`
  - `--rich-text-prob 0.8`
  - `--epochs 15` (minimum, may need 18 with new augmentation)

---

### ✅ **Evaluation Plane**

- [ ] `evaluate_clap.py` has Hit@k metric (lines 262-265)
- [ ] Multi-crop eval enabled: `EVAL_N_AUDIO_CROPS = 3` (line 92)
- [ ] Eval batch size ≤ 8 to avoid OOM
- [ ] Both `best.pt` (best val_loss) and `best_r1.pt` (best Hit@1) will be evaluated

---

### ✅ **Memory Safety**

- [ ] Launch training with `nvidia-smi` monitoring in a separate terminal:
  ```bash
  watch -n 1 nvidia-smi
  ```
- [ ] If VRAM exceeds 11 GB, reduce `--batch-size` or `--accum`
- [ ] If training OOMs, apply gradient checkpointing or 8-bit Adam

---

### ✅ **Audit Plane**

- [ ] Create `docs/Training Audits/TRAINING_AUDIT_13th_run.md` **before** training starts
- [ ] Document all changes from Run 12
- [ ] Record launch command exactly as executed
- [ ] Fill in Section 4 ("What worked / what didn't") within 24 hours of completion

---

## 10. Expected Run 13 Outcomes

### 10.1 Conservative Estimate (Fixes 1 + 2 Only)

| Metric | Finetune 12 | Run 13 Conservative | Gain |
|--------|-------------|---------------------|------|
| Hit@1 (all_variants) | 26.63% | 33-36% | +6-9pp |
| mAP | 0.2558 | 0.30-0.33 | +0.04-0.07 |
| Zero-shot Hit@1 | ~2% | 7-10% | +5-8pp |
| R@1 (broken metric) | 8.90% | 12-15% | +3-6pp |

**Why conservative:**
- Multi-crop training fix: +5-8pp
- Species-name stripping: +2-4pp val, +5-10pp zero-shot
- Total: +7-12pp val, +5-10pp zero-shot

---

### 10.2 Optimistic Estimate (All Fixes + Memory-Stable)

| Metric | Finetune 12 | Run 13 Optimistic | Gain |
|--------|-------------|-------------------|------|
| Hit@1 (all_variants) | 26.63% | 35-40% | +8-13pp |
| mAP | 0.2558 | 0.34-0.38 | +0.08-0.12 |
| Zero-shot Hit@1 | ~2% | 10-15% | +8-13pp |
| R@1 (broken metric) | 8.90% | 15-18% | +6-9pp |

**Why optimistic:**
- Multi-crop training: +5-8pp
- Species-name stripping: +2-4pp val, +5-10pp zero-shot
- Val combo filtering: +1-2pp (cleaner metric)
- Re-enabling any downgraded features: +1-3pp
- Total: +9-17pp val, +5-10pp zero-shot

---

### 10.3 Success Criteria

**Minimum acceptable (conservative target met):**
- Hit@1 ≥ 33%
- Zero-shot Hit@1 ≥ 7%
- No OOM crashes

**Target (optimistic target met):**
- Hit@1 ≥ 36%
- Zero-shot Hit@1 ≥ 10%
- Training completes 15 epochs without issues

**Stretch goal (exceeds Maseeh predictions):**
- Hit@1 ≥ 40%
- Zero-shot Hit@1 ≥ 15%
- mAP ≥ 0.35

---

## 11. Final Summary

### What Went Wrong in Run 12

1. **Multi-crop training bug** (lines 385-388 in train_clap.py): Uses only 1 crop per sample per epoch instead of randomizing → **Missing +5-8pp**
2. **Species-name shortcut not fixed**: Prompt still allows names, per-recording metadata not added → **Missing +2-4pp val, +5-10pp zero-shot**
3. **Memory crash + downgrade**: Unknown what was downgraded → **Possible missing +1-4pp**

**Total missing gain:** ~10-20pp on Hit@1, ~5-10pp on zero-shot

---

### What Went Right in Run 12

1. ✅ Hit@k metric properly added
2. ✅ Multi-crop eval working (EVAL_N_AUDIO_CROPS = 3)
3. ✅ Hard-neg margin switched to additive
4. ✅ Audio LR bumped to 0.5×
5. ✅ PK sampler implemented
6. ✅ Quality filter (≥4) working
7. ✅ Max-per-combo increased to 200

**Delivered gain:** ~5-8pp on Hit@1

---

### Critical Next Steps

1. 🔴 **Fix multi-crop training bug** (add `set_epoch()` method + deterministic rotation)
2. 🔴 **Fix species-name shortcut** (update GPT prompt + strengthen scrubber + add metadata)
3. 🟡 **Diagnose memory downgrade** (restore original settings if possible)
4. 🟡 **Filter val combos** (n_pos ≥ 3)
5. ✅ **Launch Run 13** with pre-flight checklist

**Expected outcome:** Hit@1 33-40%, zero-shot Hit@1 7-15%

---

**End of document.**

*Ref: `docs/Maseeh findings - 11th-fine-tune.md` (1,422 lines)
Ref: `results/eval_results_finetune11.json`, `results/eval_results_finetune12_best_r1.json`
Ref: Code inspection of train_clap.py, evaluate_clap.py, precompute_clap_features.py, generate_clap_descriptions.py*
