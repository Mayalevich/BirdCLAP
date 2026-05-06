# Codebase Guide — `lets-solve-it`

For new teammates and LLM-assisted coding. Describes what the repo is, what is stable, what to avoid breaking, and sensible next steps.

---

## 1. What this project is

**Product direction:** Use audio (especially birds and wildlife) with text so users can search or retrieve clips using natural language — joint audio-text embeddings (CLAP-style), not species-only classifiers.

**What this repo implements today:**

1. **Data pipeline** — fetch metadata from Xeno-Canto, bulk-download ~17k MP3s, build training labels.
2. **Fine-tuning** — CLAP (`laion/clap-htsat-fused`) on 121k (audio, text) pairs with contrastive loss.
3. **Live backend API** — FastAPI service wrapping the fine-tuned model; handles text search, audio similarity, classification, and audio file serving.
4. **Web UI** — Vite + React SPA: catalog search, upload, classify/similar via the real backend, 3-D sound visualisation, saved list, compare slots.

The web app is **no longer mock-only**. It talks to a live FastAPI backend that runs the fine-tuned CLAP model. See `docs/BACKEND.md` for the API reference and `SETUP.md` for running the stack.

---

## 2. Repository layout

| Path | Purpose |
|------|--------|
| `README.md` | Minimal setup: venv, pip install, env check |
| `SETUP.md` | **Full setup guide** — start here for a new machine |
| `CHANGES.md` | Cumulative changelog — what changed and why |
| `requirements.txt` | Core Python stack (pandas, requests, jupyter, …) |
| `requirements-ml.txt` | PyTorch / transformers / librosa for training |
| `start.ps1` | **One-command launcher** — starts FastAPI + Vite; `-RebuildGallery` flag |
| `backend/` | FastAPI service + CLAP provider (see `docs/BACKEND.md`) |
| `web/` | React + Vite SPA (see `web/docs/`) |
| `scripts/` | Data pipeline, training, evaluation scripts |
| `data/` | Training pairs, metadata CSV, species descriptions, gallery cache |
| `checkpoints/` | **Gitignored** — `best.pt` fine-tuned weights |
| `docs/` | Extended documentation for backend, training, scripts |

### Backend layout

| File | Role |
|------|------|
| `backend/app.py` | FastAPI app, route handlers, startup audio map |
| `backend/config.py` | Environment variables → `Settings` dataclass |
| `backend/provider_factory.py` | Selects and constructs the active provider |
| `backend/schemas.py` | Pydantic request/response models |
| `backend/errors.py` | `BackendError` (HTTP status + code + message) |
| `backend/providers/base.py` | `InferenceProvider` abstract class |
| `backend/providers/placeholder.py` | Dev stub — returns 501 |
| `backend/providers/clap_provider.py` | Real CLAP model inference |

### Scripts layout

| Script | What it does |
|--------|-------------|
| `scripts/download_xc_audio.py` | Bulk downloader — ~17k MP3s, adaptive rate limiting |
| `scripts/build_gallery.py` | Standalone gallery builder with progress output |
| `scripts/train_clap.py` | Fine-tunes CLAP with contrastive loss, AMP, checkpointing |
| `scripts/evaluate_clap.py` | Retrieval metrics + figures against held-out set |
| `scripts/generate_clap_descriptions.py` | LLM-generated acoustic descriptions → `clap_descriptions.json` |
| `scripts/build_clap_labels.py` | Label pool: taxonomy templates + rich descriptions |
| `scripts/build_clap_training_pairs.py` | Expands labels → train/val JSON pair files |
| `scripts/build_taxonomy_db.py` | GBIF taxonomy lookup → `species_taxonomy.json` |
| `scripts/scrape_species_descriptions.py` | Scrapes AllAboutBirds + ADW → `species_descriptions.json` |

---

## 3. Data contract (do not break without updating all consumers)

### `xc_metadata_unified.csv`

```
filepath, species_code, common_name, vocalization_type, quality_rating, duration, source
```

| Column | Constraints |
|--------|-------------|
| `filepath` | Must match `audio/xc/<numeric_id>.<ext>` — recording ID parsed by regex |
| `common_name` | Must match keys in `species_descriptions.json` for text search to work |
| `vocalization_type` | Free text; may contain commas (quoted in CSV) |

**If you add or rename columns**, update: `get_xenocanto.ipynb`, `mini_clap_xc_sample.py`, `eda_xc_metadata.ipynb`, and `backend/providers/clap_provider.py`.

### Gallery cache (`data/gallery_embeddings.pt`)

```python
{
    "embeddings": FloatTensor[N, D],   # L2-normalised audio embeddings
    "items": [dict, ...],              # N metadata dicts, same order
    "cache_version": int,              # must match CACHE_VERSION in clap_provider.py
}
```

Current `CACHE_VERSION = 2` (10-second centre-crop encoding to match fine-tuning). If you change the encoding strategy, bump this constant — the provider detects the mismatch on startup and rebuilds automatically.

### `species_descriptions.json`

Keys must exactly match `common_name` values in `xc_metadata_unified.csv`. These keys drive:
- The `_desc_map` that populates `species_description` on search results.
- The `_species_text_embs` matrix used for text-to-text catalog search.
- The "What CLAP heard" panel content.

---

## 4. How search works (critical architecture)

### Audio similarity search (`/api/search-by-audio`)

```
user upload
  │ decode → mono 48kHz
  │ centre-crop to 10 seconds (matches gallery encoding)
  ▼
CLAP audio encoder → L2-normalised [1, D] query embedding
  │ cosine similarity against gallery [N, D]
  ▼
top-k results with audio_url, species_description
```

### Text catalog search (`/api/search`)

**Important:** Cross-modal text→audio comparison is **not used** for catalog search. Fine-tuning on (audio, text) pairs shifted the embedding spaces — text queries mapped to wrong audio neighbors.

**Actual path:**

```
text query
  │ CLAP text encoder → L2-normalised [1, D] query embedding
  ▼
cosine similarity against _species_text_embs [S, D]
  (text embeddings of "Species Name — acoustic description")
  ▼
top-K species by text-to-text similarity
  │ _species_to_indices lookup → gallery recording indices
  ▼
up to 2 recordings per species → result list
```

Both sides use the same text encoder — no cross-modal gap. This is why searching for "sharppeeknote similar to Downy Woodpecker" correctly returns Hairy Woodpecker.

### Why the 10-second crop

The fine-tuning used 10-second WAV sidecars (created by `convert_to_wav.py`). If you encode a 60-second recording without cropping, the embedding lands in a completely different region of the space than anything in the gallery — results become random. Both gallery build and user-upload encoding must use the same 10-second crop for the similarity math to be valid.

The browser's **analysis window picker** lets users choose *which* 10 seconds to send, so noise at the start of a recording doesn't corrupt the search.

---

## 5. What is done vs not done

### Done (working in repo)

- Fine-tuned CLAP checkpoint (`best.pt`) with 12+ training runs — 62.8% Hit@5 on held-out set.
- Live FastAPI backend with text search, audio similarity, classification, and audio file serving.
- Real spectrograms and audio playback in result cards (fetched from the backend).
- 3-D sound visualisation using the actual catalog audio file.
- Analysis window picker — lets users select the cleanest 10-second window from long recordings.
- Text-to-text catalog search using species acoustic descriptions.
- "What CLAP heard" panel showing acoustic profiles of matched species.
- Gallery rebuild on demand (`-RebuildGallery` flag or `scripts/build_gallery.py`).
- ~17k MP3s downloaded and indexed.

### Not done / future work

- **Per-recording descriptions** — rich text labels are per-species, so many different audio clips share identical text targets. This is the main driver of training quality plateau.
- **VAD pre-pass** — automatic selection of the best 10-second window by bird-audio energy, removing the need for manual window picking.
- **LoRA / PEFT** — currently full fine-tune of both encoders; parameter-efficient tuning could preserve base model cross-modal alignment while still specialising for birds.
- **Multi-GPU / distributed training.**
- **Pagination** — UI renders only the first response batch (`top_k`); no "load more".

---

## 6. How to run (quick reference)

```powershell
# Install backend dependencies (after PyTorch)
pip install -r backend/requirements.txt

# Install frontend
cd web && npm install && cd ..

# Start both services
powershell -ExecutionPolicy Bypass -File start.ps1

# Rebuild gallery explicitly
python scripts/build_gallery.py
```

See `SETUP.md` for the full setup guide including PyTorch CUDA installation.

---

## 7. Pitfalls

### Audio encoding consistency

The gallery MUST be built with the same crop strategy that live queries use. If you change `_decode_bytes` (the query path), you must also change `_load_file` (the gallery build path) and bump `CACHE_VERSION`. Otherwise query and gallery embeddings live in different spaces and all similarity scores are meaningless.

### Cross-modal alignment after fine-tuning

Both the text and audio encoders were updated during fine-tuning. Do not use raw cosine similarity between text query embeddings and audio gallery embeddings (cross-modal) — the spaces are no longer aligned. Use text→text (current `search_text`) or audio→audio (current `search_by_audio`).

### MP3 vs WAV for audio serving

The backend prefers MP3 over WAV when building the audio serving map. This is intentional — the 10-second WAV sidecars are only for embedding; users should hear the full original recording. Do not change this preference without also updating the audio serving logic.

### `species_descriptions.json` key matching

The text search builds `_species_to_indices` with lowercase keys from `meta["species"]` (which is `common_name` from the metadata CSV). The `_species_names_list` comes from `_desc_map.keys()` (from `species_descriptions.json`). The lookup uses `.lower()` on both sides — but if a species name in the JSON has a different spelling than in the CSV, it will silently produce zero results for that species.

### Regenerating training CSV

Re-running `get_xenocanto.ipynb` **overwrites** `xc_metadata_unified.csv`. Commit diffs deliberately — row counts and ordering can change as Xeno-Canto grows.

### Secrets

Never commit `.env` or API keys. Each developer should use their own Xeno-Canto key.

---

## 8. Adding a new backend provider

1. Create `backend/providers/your_provider.py` implementing `InferenceProvider`.
2. Add an `elif settings.model_provider == "yourname":` branch in `backend/provider_factory.py`.
3. Add any required env vars to `backend/config.py` `Settings`.
4. Set `MODEL_PROVIDER=yourname` and start the server.

The HTTP layer, request validation, and response serialisation are untouched.

---

## 9. Web client (`web/`) — pointer for pipeline developers

If you only touch Python and data, you can ignore `web/` until you need to demo retrieval UX.

- **Onboarding:** `web/docs/README.md` → `web/docs/FEATURES.md`.
- **Run locally:** `cd web && npm install && npm run dev`.
- **API contract:** `web/src/api/types.ts` defines `SearchResult`, `ClassificationHit`, etc. These map to the Pydantic schemas in `backend/schemas.py`.
- **Backend client:** `web/src/api/backend.ts` — all fetch calls live here.

Web documentation belongs under `web/docs/`; this root guide stays focused on data and training.

---

## 10. Suggested next steps

1. **Per-recording descriptions** — regenerate `clap_descriptions.json` with clip-specific metadata (date, location, habitat, notes) instead of species-level templates.
2. **VAD window auto-selection** — compute per-frame RMS on upload, auto-set the window start to the highest-energy 10-second region.
3. **Balanced resampling** — cap dominant species and upsample rare ones before rebuilding training pairs to reduce shortcut learning.
4. **LoRA fine-tuning** — fine-tune only lightweight adapter layers to preserve base cross-modal alignment while still specialising for birds; this would restore the original text→audio search quality.
5. **Evaluation cadence** — after each future training run, compare mAP/MRR/Hit@K using `evaluate_clap.py` before deploying the checkpoint.
