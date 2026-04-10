# Codebase guide — `lets-solve-it`

This document is for **new teammates** and for **LLM-assisted coding**: it states what the repo is, what is stable, what to avoid breaking, and sensible next steps.

---

## 1. What this project is (product vs repo)

**Product direction (team vision)**  
Use **audio** (especially birds and wildlife) together with **text** so users can **search or retrieve clips** using natural language — often discussed in terms of **joint audio–text embeddings** (e.g. CLAP-style models), separate from classic **species-only classifiers**.

**What this repository actually implements today**  
A **data + experimentation** workspace:

1. **Fetch metadata** from [Xeno-canto](https://xeno-canto.org/) API v3 for a fixed query (`cnt:canada`) and export a **unified CSV**.
2. **Explore** that CSV (counts, missing fields, bird vs non-bird heuristics) in a notebook.
3. **Optional smoke test**: sample rows, **download** a few MP3s, build short **text captions** from CSV columns, run **Hugging Face LAION CLAP** and print **audio–text similarity** scores.

There is **no production web app**, **no training pipeline**, and **no Cornell / RAG “enriched descriptions”** implemented in code here yet — only the **thin captions** built in `mini_clap_xc_sample.py` (`Recording of {common_name}: {vocalization_type}`).

---

## 2. Repository layout

| Path | Purpose |
|------|--------|
| `README.md` | Minimal setup: venv, `pip install`, env check, optional CLAP test |
| `requirements.txt` | Pinned **core** stack (pandas, requests, jupyter, …) |
| `requirements-ml.txt` | **Optional** PyTorch / `transformers` / librosa for CLAP script |
| `scripts/get_xenocanto.ipynb` | **Source of truth** for building `xc_metadata_unified.csv` from API v3 |
| `scripts/eda_xc_metadata.ipynb` | EDA on the CSV;-documents taxon mix (birds, frogs, bats, soundscapes, …) |
| `scripts/check_environment.py` | Validates Python, imports, CSV presence, `.env` / `XC_API_KEY`, live API ping |
| `scripts/mini_clap_xc_sample.py` | End-to-end **sample**: CSV → download MP3s → HF CLAP similarities |
| `scripts/xc_metadata_unified.csv` | **Committed artifact** (~18k rows) — full metadata export (when present) |
| `scripts/data/` | **Gitignored** — default download dir for `mini_clap_xc_sample.py` (`xc_mini/`) |

**Notebooks may write CSVs relative to the Jupyter **current working directory** (often `scripts/` if the server was started there). The fetch notebook saves `xc_metadata_unified.csv` next to the CWD; `check_environment.py` and `mini_clap_xc_sample.py` look in **`scripts/` first**, then repo root.

---

## 3. Data contract (do not break without updating all consumers)

The unified metadata file is **`xc_metadata_unified.csv`** with header:

```text
filepath,species_code,common_name,vocalization_type,quality_rating,duration,source
```

| Column | Meaning | Constraints |
|--------|---------|--------------|
| `filepath` | Logical path / filename pattern | **Must** match `audio/xc/<numeric_id>.<ext>` so scripts can parse Xeno-canto **recording id** (regex in `mini_clap_xc_sample.py`: `audio/xc/(\d+)\.`) |
| `species_code` | Slug (often from English name) | Used for grouping; may be empty for some rows |
| `common_name` | Human-readable name | `mini_clap` **drops rows** missing `filepath`, `species_code`, or `common_name` |
| `vocalization_type` | Call/song/etc. | Free text; may contain commas (quoted in CSV) |
| `quality_rating` | Numeric (mapped from XC letter grades in notebook) | Integer-ish |
| `duration` | String like `M:SS` | Not normalized to seconds in CSV |
| `source` | Provenance | Currently `xeno-canto` |

**If you add or rename columns**, update:

- `scripts/get_xenocanto.ipynb` (`clean_to_unified_schema` or equivalent),
- `scripts/mini_clap_xc_sample.py` (column reads and `build_label`),
- `scripts/eda_xc_metadata.ipynb` (any hard-coded column lists),
- This guide.

---

## 4. What is done vs not done

### Done (working in repo)

- **API v3 ingestion** with `XC_API_KEY`, pagination, unified schema → CSV (`get_xenocanto.ipynb`).
- **Large CSV** checked in (or regenerable) for Canada-tagged recordings — includes **birds and non-birds** (amphibians, mammals, insects, soundscapes) because **`cnt:canada` is geographic, not taxonomic**.
- **Environment verification** (`check_environment.py`) including optional HF CLAP processor download (`--with-ml`).
- **EDA notebook** for distribution and data-quality questions.
- **CLAP smoke test** using **Transformers** (`laion/clap-htsat-fused`), not the older `laion_clap` package.

### Not done (out of scope or future work)

- Bird-only filtering (explicit taxon rules or allowlist).
- Bulk audio download for the full 18k rows (would be heavy; respect Xeno-canto **terms and rate limits**).
- Fine-tuning CLAP or training custom models.
- Dashboard / API server.
- Cornell / external **enriched text** pipeline (RAG, scraping) — **strategy TBD**; current captions are **template-only**.

---

## 5. How to run (quick reference)

From repo root (`lets-solve-it/`):

```bash
python -m venv .venv
# Windows: .\.venv\Scripts\activate
# Unix:    source .venv/bin/activate
pip install -r requirements.txt
```

Create **`.env`** in repo root (gitignored):

```env
XC_API_KEY=your_key_here
```

Verify:

```bash
python scripts/check_environment.py
python scripts/check_environment.py --with-ml   # after: pip install -r requirements-ml.txt
```

Optional CLAP sample (needs **ffmpeg** on PATH for MP3 on many systems):

```bash
pip install -r requirements-ml.txt
python scripts/mini_clap_xc_sample.py --sample 6
```

Jupyter: open `scripts/get_xenocanto.ipynb` or `scripts/eda_xc_metadata.ipynb` (ensure kernel uses the same venv).

---

## 6. Pitfalls (read before you change code)

### Secrets and sharing

- **Never commit** `.env` or API keys. **Never paste keys** in Discord/slack screenshots.
- Each developer should use their **own** Xeno-canto key where possible.

### API and scraping etiquette

- The mini script sleeps **~0.35s** between downloads; **do not** remove throttling or parallelize aggressively without team + site policy agreement.
- Use a **identifying User-Agent** when adding new HTTP clients (see `HEADERS` in `mini_clap_xc_sample.py`).
- `check_environment.py` performs a **minimal** API call when `XC_API_KEY` is set; avoid wrapping it in tight CI loops that hammer Xeno-canto.

### Regenerating CSV

- Re-running `get_xenocanto.ipynb` **overwrites** `xc_metadata_unified.csv` in the notebook’s CWD. Commit diffs deliberately; row counts and ordering can change as Xeno-canto grows.

### Notebook working directory

- EDA notebook tries `xc_metadata_unified.csv` then `scripts/xc_metadata_unified.csv`. If paths fail, set `CSV_PATH` explicitly or start Jupyter from a consistent folder.

### ML stack size and GPU

- `requirements-ml.txt` pulls **PyTorch** and **transformers**; first CLAP run **downloads large weights**. CI without GPU should skip `--with-ml` or cache models.
- CUDA is optional; CPU runs work but are slower.

### `filepath` format

- Downstream code **depends** on `audio/xc/<id>.<ext>`. If you change the pattern, **update** `xc_id_from_row` and any download URLs (`https://xeno-canto.org/{id}/download`).

### Two CLAP stacks in the wild

- This repo’s **supported** path is **Hugging Face** `ClapModel` / `ClapProcessor` in `mini_clap_xc_sample.py`.
- Older snippets may use **`laion_clap`**; mixing both in one environment can cause confusion — prefer one stack per branch unless you document why.

---

## 7. Conventions for writing code that does not break the repo

1. **Treat the CSV schema as a public API** — version or migrate columns explicitly (see §3).
2. **Keep scripts runnable from repo root** with paths derived from `Path(__file__)` (see `repo_root()` in existing scripts).
3. **Large or downloaded data** goes under `scripts/data/` (gitignored) or a path passed via CLI flags — **do not** commit MP3s or model weights.
4. **Pin or bound** new dependencies: add to `requirements.txt` or `requirements-ml.txt` with a short comment if optional.
5. **Extend `check_environment.py`** when you add mandatory imports or services (so newcomers fail fast with a clear message).
6. **Notebooks for exploration**; **promote** stable logic to `.py` modules if multiple entry points need it (future refactor).

---

## 8. Suggested next steps (prioritized)

1. **Product decision:** Canada **all-taxa** vs **birds-only** — if birds-only, add a **filtering step** (post-CSV or API query) and document counts in EDA.
2. **Caption / “enriched description” v1:** define a **single function** (e.g. `build_caption(row) -> str`) used by both notebooks and CLAP code; add unit tests for edge cases (`soundscape`, empty `common_name`, commas).
3. **Download pipeline:** optional script to fetch audio for a **subset** (with disk quota, resume, and rate limits) producing a manifest alongside CSV.
4. **Evaluation harness:** small set of labeled pairs or human spot-checks for retrieval quality before scaling fine-tuning.
5. **Module split:** move `fetch_xc_page` / `clean_to_unified_schema` from notebook into `scripts/xc_api.py` (or similar) to reduce notebook drift.

---

## 9. Who to ask

- **Data semantics** (what a column means, whether to drop soundscapes): team + EDA notebook conclusions.
- **Xeno-canto policy / keys**: [xeno-canto.org](https://xeno-canto.org/) account and API docs.
- **Model choice** (HF checkpoint vs fine-tune): team ML lead / mentor.

This guide should be **updated** when the CSV schema, default queries, or primary CLAP stack changes.
