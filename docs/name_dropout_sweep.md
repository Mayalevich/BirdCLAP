# Name-dropout rate sweep (confound control for the ICASSP paper)

**TL;DR for whoever runs training:** `train_clap.py` now has a runtime
`--name-dropout-p` knob. Train it at `p = 0, 0.25, 0.5, 0.75, 1.0` with everything
else identical to reproduce baseline / namedropout / namemasked as *one* continuous
axis. This gives us the dose-response curve that closes the reviewer's main
objection.

## Why we need this

Reviewers will ask whether namedropout wins because of the **presence of name-free
text variants** (our claimed mechanism) or simply because of a **dataset-size /
caption-variety artifact**. A dropout-rate sweep answers this cleanly: every `p`
uses the **same clips, same steps, same audio, same splits** — only the fraction of
name-free captions changes. Any trend across `p` is therefore attributable to
name-free variants, not data size. The sweep *is* the confound control.

It also turns the three separate runs (baseline / namemasked / namedropout) into
endpoints + interior of a single knob, and produces a dose-response figure, which
is far more convincing than three isolated points.

## What changed (code)

All in `scripts/train_clap.py`, **non-breaking** — no label-file format change, no
change to `build_clap_labels.py`, reuses the label files we already have:

- New helper `sample_text_variant(labels, labels_free, combo, p)`: picks a text
  variant for a combo; with probability `p` it draws from the **name-free** pool
  instead of the name-ful pool.
- Both dataset classes (`ClapPairDataset` and `ClapPrecomputedDataset` — the fast
  path used in training) gained two args: `labels_namefree_path` and
  `name_dropout_p`, and now call the helper in `__getitem__`.
- Two new CLI flags: `--labels-namefree` and `--name-dropout-p`.

Semantics:

| `p`      | behaviour                                  | equivalent to        |
|----------|--------------------------------------------|----------------------|
| `0.0`    | always name-ful text                       | **baseline**         |
| `1.0`    | always name-free text                      | **namemasked**       |
| `0<p<1`  | name dropped per-clip with prob `p`        | **namedropout**      |

`p=0` reproduces `data/clap_all_labels.json`; `p=1` reproduces
`data/clap_all_labels_namemasked.json`. Verified in a unit test that the observed
name-free sampling fraction matches `p` (0.25 → 0.249, etc.), and that `p=0` is
purely name-ful and `p=1` purely name-free.

## How to run the sweep

Fill in the rest of the flags (`--seed`, `--epochs`, `--batch-size`, `--lr`, …) so
they are **identical to the original three runs** — only `--name-dropout-p` varies.

```bash
for p in 0.0 0.25 0.5 0.75 1.0; do
  python scripts/train_clap.py \
    --labels          data/clap_all_labels.json \
    --labels-namefree data/clap_all_labels_namemasked.json \
    --name-dropout-p  "$p" \
    --checkpoint-dir  "checkpoints/sweep_p${p}" \
    # ... all other flags identical to the baseline/masked/dropout runs ...
done
```

## Validation (please check before trusting the sweep)

1. **`p=0.0` must reproduce the baseline numbers** (it draws only from
   `clap_all_labels.json`).
2. **`p=1.0` must reproduce the namemasked numbers** (it draws only from
   `clap_all_labels_namemasked.json`).
3. **Cross-check:** the static `clap_all_labels_namedropout.json` file corresponds
   to some name-free rate `r`; running `--name-dropout-p r` should reproduce the
   namedropout numbers. If it does, the runtime knob ≡ the static file, and the
   static dropout file becomes redundant (the knob generates it continuously).

If `p=0`/`p=1` don't match, it's almost certainly a seed or flag mismatch, not the
sampling logic (that's unit-tested).

## What to log per checkpoint

Run the usual eval/probe suite on each `sweep_p*` checkpoint and collect, per `p`:

- name-following (diff-family, same-family)
- acoustics-win (diff-family)
- acoustic-only mAP
- name-only top-1 correct
- synonym retention
- audio kNN top-1 / top-5

Assemble a `p × metric` table and plot the dose-response curve (x = `p`, twin
y-axes: name-following ↓ and acoustic-only mAP ↑). That curve is the headline
confound-control figure for the paper.

## Note for the writeup

State the confound explicitly (per the earlier feedback): the effect tracks the
**presence of name-free variants**, and the sweep controls for dataset size by
holding the training budget constant across `p`.
