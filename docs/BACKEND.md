# Backend — Technical Reference

## Overview

The backend is a **FastAPI service** that wraps the fine-tuned CLAP model behind HTTP endpoints. The ML layer is swappable via a provider interface: `PlaceholderProvider` (dev stub) and `ClapProvider` (real inference) both implement the same `InferenceProvider` abstract class.

```
HTTP request
    │
    ▼
FastAPI (backend/app.py)
    │  Pydantic request validation
    │  startup: audio file map built independently (no model needed)
    ▼
InferenceProvider (backend/providers/base.py)
    ├─ PlaceholderProvider  → 501 Not Implemented
    └─ ClapProvider         → fine-tuned CLAP model
```

---

## File Map

| File | Role |
|------|------|
| `backend/app.py` | FastAPI app, route handlers, CORS, startup audio map |
| `backend/config.py` | Reads env vars into frozen `Settings` dataclass |
| `backend/provider_factory.py` | Selects and constructs the active provider at startup |
| `backend/schemas.py` | Pydantic request/response models |
| `backend/errors.py` | `BackendError` exception (HTTP status + code + message) |
| `backend/providers/base.py` | `InferenceProvider` abstract class |
| `backend/providers/placeholder.py` | Dev stub — always raises 501 |
| `backend/providers/clap_provider.py` | Real CLAP model inference |

---

## API Endpoints

All inference endpoints return `501` when running with the placeholder provider, and the structured error response when CLAP fails to decode audio.

### `GET /health`

Returns service status, active provider, and audio file index count.

```json
{
  "status": "ok",
  "provider": "clap",
  "audio_files_indexed": 17432
}
```

`audio_files_indexed` reflects the startup audio map scan — this is populated even before the CLAP model loads.

### `GET /api/audio/{recording_id}`

Streams the audio file for a recording. Returns the full MP3 (preferred) or WAV with `Accept-Ranges` headers for seeking.

- **Never stalls on model load** — reads from the startup audio map built at FastAPI startup.
- Returns `404` if no file found for that ID.
- Supports range requests for the browser's `<audio>` element.

### `POST /api/search`

Text-to-catalog retrieval via **text-to-text species matching** (not cross-modal text→audio).

**Request body (JSON):**
```json
{ "query": "sharppeeknote lower than Downy Woodpecker", "top_k": 10 }
```

**Response:**
```json
{
  "query": "sharppeeknote lower than Downy Woodpecker",
  "count": 10,
  "results": [
    {
      "id": "112580",
      "title": "Hairy Woodpecker call",
      "species": "Hairy Woodpecker",
      "scientific_name": "Dryobates villosus",
      "vocalization_type": "call",
      "duration": "1:36",
      "species_code": "HAWO",
      "score": 0.7230,
      "audio_url": "/api/audio/112580",
      "image_url": null,
      "species_description": "The most common call is a short, sharppeeknote…"
    }
  ]
}
```

**How it works internally:**

1. Query encoded as text by the CLAP text encoder.
2. Cosine similarity against `_species_text_embs` — text embeddings of every species' `"Name — description"` string.
3. Top-N species by text-to-text similarity.
4. Up to 2 recordings per species returned from the gallery.

Cross-modal text→audio comparison is **not used** — fine-tuning shifted the audio embedding space away from where the text encoder expects it.

### `POST /api/search-by-audio`

Audio-to-audio similarity search.

**Request:** `multipart/form-data` with field `file` (WAV or MP3). Optional `top_k` query param (default 10).

**Audio processing:**
1. Decode bytes → mono float32 at 48 kHz.
2. Centre-crop (or zero-pad) to exactly 10 seconds — matches the gallery encoding strategy.
3. CLAP audio encoder → L2-normalised [1, D] embedding.
4. Cosine similarity against gallery [N, D].
5. Top-k results with `audio_url` and `species_description`.

**Response:** Same shape as `/api/search`.

### `POST /api/classify-audio`

Species classification. Same encoding as `search-by-audio`; groups by species and returns the top-k species by maximum similarity score.

**Request:** Same multipart upload.

**Response:**
```json
{
  "count": 5,
  "results": [
    { "label": "Hairy Woodpecker", "scientificName": "Dryobates villosus", "score": 0.8142 },
    { "label": "Downy Woodpecker", "scientificName": "Dryobates pubescens",  "score": 0.7601 }
  ]
}
```

### `POST /api/describe-audio`

Returns acoustic descriptions for the species acoustically nearest to the uploaded audio. Used to power the "What CLAP heard" UI panel.

**Request:** Same multipart upload.

**How it works:** Routes through `_top_k` (same audio→audio path as `search-by-audio`), collects the top unique species, then returns their `_desc_map` entries. Does **not** use cross-modal text-to-audio comparison.

**Response:**
```json
{
  "descriptions": [
    "The most common call is a short, sharppeeknote very similar to Downy Woodpeckers…",
    "A loud, sharp peek call repeated rapidly; also a long descending whinny rattle…"
  ]
}
```

### Error response shape

All errors return this body regardless of HTTP status:
```json
{ "code": "AUDIO_DECODE_FAILED", "message": "Human-readable description." }
```

Common codes: `MODEL_NOT_CONNECTED` (501), `AUDIO_DECODE_FAILED` (422).

---

## ClapProvider Internals

### Startup sequence

When the first API call triggers model load:

1. **Load processor and model** from HuggingFace (`laion/clap-htsat-fused`).
2. **Load fine-tuned weights** from `checkpoints/best.pt` (strict=False; missing/unexpected keys logged as warnings).
3. **Load or build gallery:**
   - If `data/gallery_embeddings.pt` exists and `cache_version` matches `CACHE_VERSION`, load it (~1 second).
   - Otherwise, build from scratch: read `xc_metadata_unified.csv`, load each audio file (WAV sidecar if present, MP3 fallback), encode in batches, save.
4. **Build audio path map** (`_audio_map`): scan gallery metadata for known paths; fallback to directory stem scan. MP3 preferred over WAV.
5. **Load species descriptions** from `species_descriptions.json` → `_desc_map`.
6. **Build species index** (`_species_to_indices`): `{species_name_lower: [meta_index, ...]}` for text search routing.
7. **Build acoustic text gallery** from `clap_descriptions.json` → `_text_embs` / `_text_strings` (for `describe_audio`).
8. **Build species text gallery** from `_desc_map` → `_species_text_embs` / `_species_names_list` (for `search_text`).

**Separately (at FastAPI startup, before model load):**
- `_build_startup_audio_map()` scans `audio_root` for all `.mp3`, `.wav`, `.ogg`, `.flac` files by stem. This makes `/api/audio/{id}` fast and available immediately.

### Audio encoding pipeline

```
uploaded bytes / file on disk
  │ torchaudio.load (preferred, handles MP3/WAV without audioread)
  │ fallback: librosa.load
  │ resample to 48 kHz, downmix to mono
  │ centre-crop OR zero-pad to exactly 10 seconds
  │   (ONLY for embeddings — serving uses full file)
  ▼
ClapProcessor → mel-spectrogram input
  ▼
HTSAT audio encoder → pooler_output
  ▼
L2 normalise → [1, D] embedding
```

### Why centre-crop (not full-duration fusion)

The checkpoint was fine-tuned on 10-second WAV sidecars. Full-duration fusion (`is_longer=True`) produces embeddings in a different region of the space — gallery embeddings built with 10-second crops and query embeddings built with full-duration fusion are incompatible. Both must use the same strategy.

### Gallery cache versioning

```python
CACHE_VERSION = 2  # v1=centre-crop-10s (original), v2=10s WAV sidecars + this crop
```

If the loaded cache has a different version, the provider logs a warning and rebuilds automatically. Bump `CACHE_VERSION` whenever you change `_decode_bytes` or `_load_file`.

### Gallery item metadata schema

| Field | Source |
|-------|--------|
| `id`, `recording_id` | Filename stem of the audio path (e.g. `"256526"`) |
| `title` | `"{common_name} {vocalization_type}"` |
| `species` | `common_name` from metadata CSV |
| `scientific_name` | Looked up from `species_taxonomy.json` |
| `vocalization_type` | From metadata CSV |
| `duration` | From metadata CSV |
| `species_code` | From metadata CSV |
| `audio_rel` | Relative path stored in cache for `_build_audio_map` |
| `audio_url` | Set to `/api/audio/{id}` in `_top_k` if file exists on disk |
| `species_description` | From `_desc_map` — attached in `_top_k` |

---

## Configuration Reference

All configuration is via environment variables. `Settings` in `backend/config.py` holds all values.

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PROVIDER` | `placeholder` | Set to `clap` to enable the real model |
| `CORS_ORIGINS` | `http://localhost:5173,http://127.0.0.1:5173,…` | Comma-separated allowed origins |
| `CHECKPOINT_PATH` | `checkpoints/best.pt` | Fine-tuned `.pt` checkpoint |
| `AUDIO_ROOT` | `scripts/data/xc_audio` | Root directory containing `.mp3` / `.wav` files |
| `GALLERY_CACHE` | `data/gallery_embeddings.pt` | Gallery save/load path |
| `BASE_MODEL` | `laion/clap-htsat-fused` | HuggingFace model ID |
| `METADATA_PATH` | `data/xc_metadata_unified.csv` | Master catalog CSV |
| `TAXONOMY_PATH` | `data/species_taxonomy.json` | Scientific name lookup |
| `VAL_PAIRS_PATH` | `data/clap_val_pairs.json` | Fallback gallery source (if CSV absent) |
| `SPECIES_DESCRIPTIONS_PATH` | `data/species_descriptions.json` | Per-species acoustic descriptions |
| `CLAP_DESCRIPTIONS_PATH` | `data/clap_descriptions.json` | Training-corpus descriptions |

---

## Adding a New Provider

1. Create `backend/providers/your_provider.py` implementing `InferenceProvider`.
2. Add `elif settings.model_provider == "yourname":` in `backend/provider_factory.py`.
3. Add required env vars to `backend/config.py`.
4. Set `MODEL_PROVIDER=yourname` and start the server.

HTTP layer, request validation, and response serialisation are untouched.
