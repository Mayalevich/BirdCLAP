# Backend — Technical Reference

## Overview

The backend is a **FastAPI service** that wraps a fine-tuned CLAP model behind three HTTP endpoints. The ML layer is fully swappable via a provider interface: the placeholder stub (used in development) and the real `ClapProvider` both implement the same `InferenceProvider` abstract class, so swapping them requires only an environment variable change.

```
HTTP request
    │
    ▼
FastAPI (backend/app.py)
    │  validates request shape via Pydantic schemas
    ▼
InferenceProvider (backend/providers/base.py)
    │  abstract interface: search_text / search_by_audio / classify_audio
    ├─ PlaceholderProvider  →  returns 501 Not Implemented
    └─ ClapProvider         →  runs fine-tuned CLAP model
```

---

## File Map

| File | Role |
|------|------|
| `backend/app.py` | FastAPI app, route handlers, CORS, error handler |
| `backend/config.py` | Reads all env vars into a frozen `Settings` dataclass |
| `backend/provider_factory.py` | Selects and constructs the right provider at startup |
| `backend/schemas.py` | Pydantic request/response models |
| `backend/errors.py` | `BackendError` exception (carries HTTP status, code, message) |
| `backend/providers/base.py` | `InferenceProvider` abstract class |
| `backend/providers/placeholder.py` | Stub — always raises 501 |
| `backend/providers/clap_provider.py` | Real CLAP model inference |

---

## API Endpoints

All inference endpoints return `501` when running with the placeholder provider and the structured error response when the CLAP provider fails to decode audio.

### `GET /health`
Returns the service status and active provider name.

```json
{ "status": "ok", "provider": "clap" }
```

### `POST /api/search`
Text-to-audio retrieval. Encodes the query string with the CLAP text encoder, computes cosine similarity against the pre-built audio gallery, returns the top-k recordings.

**Request body (JSON):**
```json
{ "query": "Northern Cardinal call", "top_k": 10 }
```

**Response:**
```json
{
  "query": "Northern Cardinal call",
  "count": 10,
  "results": [
    {
      "id": "256526",
      "title": "Northern Cardinal call",
      "species": "Northern Cardinal",
      "score": 0.8421,
      "recording_id": "256526",
      "scientific_name": "Cardinalis cardinalis",
      "vocalization_type": "call",
      "duration": "0:12",
      "species_code": "northern_cardinal",
      "image_url": null,
      "audio_url": null
    }
  ]
}
```

### `POST /api/search-by-audio`
Audio-to-audio retrieval. Accepts a multipart file upload, encodes it with the CLAP audio encoder, returns the top-k similar recordings from the gallery.

**Request:** `multipart/form-data` with a field named `file` (WAV or MP3). The `top_k` query parameter (default 10) controls result count.

**Response:** Same shape as `/api/search`.

### `POST /api/classify-audio`
Species classification. Encodes the uploaded audio, scores it against every species in the gallery, and returns the top-k species by maximum similarity.

**Request:** Same multipart upload as `/api/search-by-audio`.

**Response:**
```json
{
  "count": 5,
  "results": [
    { "label": "Northern Cardinal", "scientificName": "Cardinalis cardinalis", "score": 0.7814 },
    { "label": "House Finch",       "scientificName": "Haemorhous mexicanus",  "score": 0.5102 }
  ]
}
```

### Error response shape
All errors return this body regardless of HTTP status:
```json
{ "code": "MODEL_NOT_CONNECTED", "message": "Human-readable description." }
```

Common codes: `MODEL_NOT_CONNECTED` (501), `AUDIO_DECODE_FAILED` (422).

---

## ClapProvider Internals

### Startup sequence

1. **Load processor and model** from the HuggingFace base checkpoint (`laion/clap-htsat-fused`).
2. **Load fine-tuned weights** from the `.pt` checkpoint file. Only `model_state` is restored — optimizer and scheduler state is ignored. Missing/unexpected keys are logged as warnings.
3. **Build or load the gallery.**

The gallery is the retrieval database. On the first run it is built by:
- Reading every unique audio path from `clap_val_pairs.json`.
- Loading each audio file from disk (same fast-path logic as `train_clap.py`: prefers the pre-clipped `.wav` sibling, falls back to librosa MP3 decoding).
- Running audio through the CLAP audio encoder in batches of 16.
- Saving `{"embeddings": FloatTensor[N, D], "items": [metadata…]}` to the gallery cache file.

On subsequent runs the cache is loaded directly — no audio files or model inference needed at startup.

### Text query encoding (`search_text`)

```
query string
    │ ClapProcessor.tokenize
    ▼
BERT text encoder → pooler_output → L2 normalize → [1, D] query embedding
    │ cosine similarity against gallery [N, D]
    ▼
argsort → top-k indices → metadata lookup → SearchResultItem list
```

### Audio query encoding (`search_by_audio`, `classify_audio`)

```
uploaded bytes
    │ soundfile (fast, WAV) or librosa (MP3 fallback)
    │ resample to 48 kHz, mono
    │ centre-crop or zero-pad to 10 s
    ▼
ClapProcessor → mel-spectrogram → HTSAT audio encoder → pooler_output → L2 normalize
    │ cosine similarity against gallery [N, D]
    ▼
top-k recordings  (search_by_audio)
  OR
group by species → max sim per species → top-k species  (classify_audio)
```

### Gallery item metadata

Each gallery item is a dict that maps directly to `SearchResultItem`:

| Field | Source |
|-------|--------|
| `id`, `recording_id` | Filename stem of the audio path (e.g. `"256526"`) |
| `title` | `"{common_name} {vocalization_type}"` |
| `species` | Common name from metadata CSV, fallback from val pairs combo |
| `scientific_name` | Looked up from `species_taxonomy.json` by common name |
| `vocalization_type` | From val pairs combo or metadata CSV |
| `duration` | From metadata CSV |
| `species_code` | From metadata CSV |
| `audio_url`, `image_url` | Always `null` (not served by backend; future work) |

---

## Configuration Reference

All configuration is via environment variables. The `Settings` dataclass in `backend/config.py` holds all values; no mutable state at module level.

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PROVIDER` | `placeholder` | Set to `clap` to enable the real model |
| `CORS_ORIGIN` | `http://localhost:5173` | Frontend origin allowed by CORS |
| `CHECKPOINT_PATH` | `checkpoints/finetune11/best_r1.pt` | Path to the fine-tuned `.pt` checkpoint |
| `AUDIO_ROOT` | `scripts/data/xc_audio` | Root directory containing `.mp3` / `.wav` files |
| `GALLERY_CACHE` | `data/gallery_embeddings.pt` | Where gallery embeddings are saved/loaded |
| `BASE_MODEL` | `laion/clap-htsat-fused` | HuggingFace model ID for base weights |
| `METADATA_PATH` | `data/xc_metadata_unified.csv` | Metadata CSV (species, duration, etc.) |
| `TAXONOMY_PATH` | `data/species_taxonomy.json` | Taxonomy JSON for scientific names |
| `VAL_PAIRS_PATH` | `data/clap_val_pairs.json` | Val pairs JSON (defines the gallery) |

---

## Adding a New Provider

1. Create `backend/providers/your_provider.py` implementing `InferenceProvider`.
2. Add an `elif settings.model_provider == "yourname":` branch in `backend/provider_factory.py`.
3. Add any required env vars to `backend/config.py` `Settings`.
4. Set `MODEL_PROVIDER=yourname` and start the server.

The HTTP layer, request validation, and response serialization are untouched.

---

## Gallery Cache

The gallery cache (`data/gallery_embeddings.pt`) is a plain `torch.save` dict:

```python
{
    "embeddings": torch.FloatTensor,  # shape [N, embedding_dim]
    "items": [                        # length N, same order as embeddings
        {
            "id": str,
            "recording_id": str,
            "title": str,
            "species": str,
            "scientific_name": str | None,
            "vocalization_type": str | None,
            "duration": str | None,
            "species_code": str | None,
            "audio_url": None,
            "image_url": None,
        },
        ...
    ]
}
```

The cache is model-specific. If you load a different checkpoint or change the base model, delete the cache file so it is rebuilt with the new weights.
