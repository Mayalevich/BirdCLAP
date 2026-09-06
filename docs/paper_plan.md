# Name-dropout paper — analysis, plan, and open questions

Working notes on the budget-controlled sweep (Sir7s/sweep-test-result, `sweep2`).
Purpose: lock the story we can honestly defend for ICASSP, list what only the
person with the checkpoints can resolve, and give a concrete next direction.

Deadline anchor: ICASSP 2027 full paper **Sep 16, 2026**.

---

## 1. What the budget-controlled sweep actually shows (seen species, val gallery)

Same clips, same splits, same optimizer steps at every `p`; only the caption
distribution changes. `p=0` name-ful (baseline), `p=1` name-free (masked).

| p | name-following (diff-fam) ↓ | acoustic-only mAP ↑ | name-only top-1 | synonym ret. | kNN top-1 |
|---|---|---|---|---|---|
| 0.0  | 63.5% | 0.0216 | 4.07% | 55.2% | 15.8% |
| 0.25 | 52.3% | 0.0253 | 2.84% | 55.1% | 15.4% |
| 0.347| 47.6% | **0.0319** | 3.95% | 55.2% | 15.6% |
| 0.5  | 38.9% | 0.0317 | 2.96% | 52.9% | 15.7% |
| 0.75 | **31.8%** | 0.0301 | 2.59% | **60.2%** | 15.1% |
| 1.0  | 40.2% | 0.0174 | 1.48% | 55.6% | 12.7% |

Holdout (80 unseen species): name-following falls 58.9% → 38.8%, acoustic mAP
peaks 0.0473 at p=0.347, same qualitative shape.

**The three claims that survive and are defensible:**
1. **Budget confound eliminated.** With pairs/steps held constant, name-following
   still falls monotonically 63.5% → 31.8%. The earlier "namedrop wins" could not
   be separated from "namedrop trained ~3× longer"; now it can.
2. **Generalizes to unseen species.** The trend holds on 80 held-out species the
   model never trained on — a stronger claim than a single in-distribution point.
3. **Partial beats full removal.** p=1.0 is worse than p=0.75 on every metric
   (name-following rebounds, mAP is the lowest of all six, kNN drops). Fully
   masking the name is *not* the answer; a moderate dropout rate is. This is the
   scientific hook.

**Noise floor:** run1 vs run2 at p=0 differ 0.65pp vs 5–12pp steps between
adjacent p — the trend is well outside run-to-run noise.

**A subtle, publishable observation:** the two objectives peak at *different* p.
Acoustic mAP peaks at p≈0.35; name-following bottoms at p≈0.75. Grounding and
shortcut-suppression are not the same axis — and it hands Max a concrete target
(push the mAP peak, currently 0.032 @ p≈0.35, higher).

---

## 2. Open questions for whoever holds the runs (resolve BEFORE writing numbers)

These are not nitpicks — they decide what we can claim. Two numbers in this honest
sweep contradict the earlier (unbudgeted) run that was shared, by large margins:

| metric | earlier run (shared) | this budget-controlled sweep |
|---|---|---|
| acoustic-only mAP (best) | **0.129**, "beats the paper's 0.116" | **0.032** (p=0.347) |
| name-following baseline | ~79% | 63.5% |
| **name-only top-1, baseline** | **25.03%** | **4.07%** |
| name-only top-1, masked | 1.85% | 1.48% |

**Q1 — Why is acoustic mAP 4× lower here (0.032 vs 0.129)?** Most likely the
earlier namedrop run simply trained on more pairs/longer (the confound). If so,
**the "beats BirdCLAP's 0.116" claim does not survive budget control** and must be
dropped from the paper. Need confirmation of what differed (epochs, pairs, gallery).

**Q2 — Why is baseline name-only top-1 25% earlier but 4% here?** This is a
different gallery/probe definition, not just budget. It matters because the
earlier story "namedrop *preserves* name search (24% vs 25%)" cannot be told with
this data — here name-only is ~4% everywhere, so there is no name-search ability
to preserve/lose. We need to know which probe/gallery is the one we report.

**Q3 — Which run is canonical for the paper?** We should report *one* internally
consistent setup end-to-end. My recommendation: the budget-controlled `sweep2`,
because it is the one that is defensible. That means smaller headline numbers but a
claim that holds up under reproduction.

---

## 3. Recommended paper story (honest version)

Not "name-dropout makes performance jump." Instead:

> **A minimal, budget-neutral training intervention — randomly dropping the species
> name from a fraction of captions — monotonically suppresses the taxonomic-name
> shortcut in contrastive audio-text models, generalizes to unseen species, and a
> partial drop rate strictly dominates full name masking.**

Smaller than the flashy version, but true, reproducible, and reviewer-proof. The
"part beats full mask" result is the genuinely interesting, non-obvious finding.

Drop from the paper (unless Q1/Q2 resolve in our favor): "beats BirdCLAP 0.116",
and "preserves name search at 24%".

---

## 4. Four-page outline (IEEE double column)

1. **Intro** (¾ col): shortcut learning in audio-text models; the taxonomic-name
   shortcut; contribution bullets (the 3 defensible claims in §1).
2. **Related work** (½ col): CLAP/BirdCLAP; shortcut/spurious-correlation
   mitigation; caption/word augmentation. Delta sentence each. (Draft: TODO, see §6.)
3. **Method** (¾ col): the name-dropout knob as one continuous axis; name-ful vs
   name-free pools; the measurement suite (name-following, acoustics-win,
   acoustic-only mAP, name-only, synonym retention, kNN).
4. **Experiments** (1¼ col): budget-control setup; the seen + holdout dose-response
   tables/figure; run-to-run noise floor; p=1 vs p=0.75.
5. **Discussion / Limitations** (½ col): two objectives peak at different p; single
   dataset; state the confound explicitly (effect tracks presence of name-free
   variants, budget held constant).
6. **Conclusion** (¼ col).

---

## 5. Direction for Max (scope = name-dropping variants only)

North star: **acoustic query accuracy** (acoustic mAP / acoustics-win), with a hard
guardrail that name/synonym ability must not collapse. Lever: variants of
name-dropping only — not new architectures, not extra data.

Concrete target: **push the acoustic-mAP peak above the current 0.032 @ p≈0.35.**
Candidates, in priority order:
1. **Hypernym replacement** instead of deletion: replace the name with genus /
   family / "a bird" rather than "this species". Best novelty bet.
2. **Curriculum on p**: anneal p low→high over training vs fixed p.
3. **Selective drop**: drop scientific vs common name separately; or drop more for
   high-frequency/over-sampled species.
4. **Adaptive p** per species/frequency.

Evaluate every variant with the same suite so we can show a Pareto push, not a
single-metric win.

---

## 6. What is still to produce (independent of the checkpoints)

- [ ] Related-work draft with citations (CLAP/BirdCLAP, shortcut mitigation,
      caption augmentation) + the delta sentence for each. — next
- [ ] Method + Experiments prose once Q1–Q3 are answered.
- [ ] Figure polish: the dose-response plot exists; may want seen+holdout as one
      2-panel figure for the paper.
