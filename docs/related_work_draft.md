# Related Work — draft with citations and delta sentences

Draft for the paper's §2. Each entry: the work, what it does, and **our delta**
(how we differ). Grouped into the three buckets a reviewer will place us in.
BibTeX keys are placeholders — fill from the arXiv/venue pages linked.

> Prose is intentionally terse and is meant as scaffolding — rewrite in your own
> voice for submission.

---

## A. Contrastive language-audio models & bioacoustics (our foundation)

- **CLAP / LAION-CLAP** — Elizalde et al. 2023; Wu et al. 2023 (LAION-CLAP,
  feature fusion + keyword-to-caption augmentation). arXiv 2211.06687 / 2211.03687.
  The audio-text contrastive backbone we build on (`laion/clap-htsat-fused`).
- **BioLingual** — Robinson et al., "Transferable Models for Bioacoustics with
  Human Language Supervision," arXiv **2308.04978**. CLAP adapted to bioacoustics
  via the AnimalSpeak caption corpus; zero-shot species ID and text-audio retrieval.
- **AnimalCLAP** — arXiv **2603.22053**, "Taxonomy-Aware Language-Audio Pretraining
  for Species Recognition and Trait Inference." **Our primary point of contrast:**
  it *deliberately* builds text from taxonomic templates (common/scientific name,
  taxonomy chain) — exactly the caption construction our repo inherits. It uses the
  name to establish alignment; **we show that same construction installs a
  taxonomic-name shortcut, and we remove it with a targeted intervention.**
- **NatureLM-audio** — arXiv 2411.07186; **BirdSet** — arXiv 2403.10380 (benchmark);
  **What Matters for Bioacoustic Encoding** — arXiv 2508.11845. Context for the
  bioacoustics setting and evaluation.

**Delta for the bucket:** these establish audio-text alignment for animal sound,
largely *using* names/labels in the caption. None diagnose or measure the
name-as-shortcut failure, and none report a shortcut-vs-capability trade-off.

## B. Shortcut learning & spurious-correlation mitigation

- **Geirhos et al. 2020**, "Shortcut Learning in Deep Neural Networks,"
  arXiv **2004.07780** — the framing (decision rules that win on-benchmark, fail
  off-distribution). We cite for the problem definition.
- **Group DRO** — Sagawa et al. 2020, arXiv **1911.08731** — worst-group
  optimization; **requires group/shortcut annotations.**
- **DFR / last-layer retraining** — Kirichenko et al. 2023, arXiv **2204.02937**
  (and follow-ups arXiv 2308.00473) — retrain the final layer on group-balanced
  data; **a post-hoc classifier fix, assumes a held-out balanced set.**
- **Language-guided augmentation for robustness** — e.g. ASPIRE, arXiv 2308.10103;
  "Mitigating Spurious Correlations in Multi-modal Models during Fine-tuning,"
  arXiv 2304.03916.

**Delta for the bucket:** these mitigate shortcuts but (i) need group labels or a
balanced held-out set, and (ii) target a downstream classifier, trading capability
for robustness. Ours needs **no annotations**, acts **during contrastive
pretraining via a one-line caption intervention**, and — crucially — we measure
**both** shortcut suppression **and** capability retention, showing a Pareto move
rather than a trade-off.

## C. Caption / token augmentation in vision-/audio-language training (most adjacent)

- **Keyword-to-caption augmentation** (LAION-CLAP, above) — expands captions for
  coverage.
- **RobustCLAP** — paraphrase / multi-view contrastive for query-variation
  robustness (arXiv 2404.17806, T-CLAP temporal augmentation).
- **Word/token dropout** in language and multimodal training — generic
  regularization that drops random tokens.

**Delta for the bucket — the one reviewers will press on:** these add caption
diversity or robustness with **undirected** augmentation, and do not target an
identified shortcut feature. Our intervention is (1) **directed** at a *diagnosed*
shortcut (the species name), and (2) evaluated on a **bidirectional** suite that
separates shortcut suppression from capability loss. We further find (3) a
**partial** drop strictly dominates **full** removal (masking) — the non-obvious
result that generic dropout framing does not predict.

---

## The three contribution sentences (for the intro; keep consistent with §3 of paper_plan)

1. We diagnose the taxonomic **name as a shortcut** in contrastive audio-text
   models and remove it with a directed, annotation-free, budget-neutral caption
   intervention (name-dropout).
2. A **bidirectional measurement suite** lets us show name-dropout suppresses the
   shortcut **and** generalizes to unseen species, under a controlled training
   budget (ruling out the dataset-size confound).
3. **Partial dropout strictly dominates full masking** — removing the name entirely
   is worse on every metric than dropping it part of the time.

---

## To verify before submission

- [ ] Confirm exact CLAP citation(s) and years (Elizalde 2023 vs Wu/LAION 2023).
- [ ] Pull correct DFR arXiv id (2204.02937) and Group DRO (1911.08731) BibTeX.
- [ ] Confirm AnimalCLAP (2603.22053) is the template source our repo actually
      copied, and cite it as such in Method too.
- [ ] Decide if BirdSet/NatureLM belong in Related Work or just the setup.
