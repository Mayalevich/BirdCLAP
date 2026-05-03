# System Architecture: How CLAP, Backend, and Frontend Connect

This document explains exactly how a user action in the browser ends up as a result from the fine-tuned CLAP model. Follow the layers top to bottom.

---

## Overview

```
Browser (React SPA)
  └─ fetch("/api/...")           ← same-origin request in dev
       │
  Vite dev server proxy          ← transparent HTTP forward (dev only)
       │
  FastAPI  (uvicorn, port 8000)  ← receives the real HTTP request
       │
  ClapProvider                   ← Python class wrapping the model
       │
  ClapModel (HuggingFace)        ← fine-tuned weights from checkpoints/best.pt
       │
  Gallery embeddings             ← pre-computed float32 tensor (data/gallery_embeddings.pt)
```

---

## Layer 1 — The Browser (React + TypeScript)

All API calls live in **`web/src/api/backend.ts`**. There are three:

| Function | HTTP call | When used |
|---|---|---|
| `searchDataset(query)` | `POST /api/search` | User types a text query |
| `searchSimilarToUpload(file)` | `POST /api/search-by-audio` | User uploads an audio file |
| `classifyUpload(file)` | `POST /api/classify-audio` | "What species is this?" tab |

### URL construction

In **development** (`npm run dev`), `getApiOriginForFetch()` returns an empty string, producing a relative URL like `/api/search`. The Vite dev server intercepts that request before it ever leaves the machine.

In **production** (built with `npm run build`), the full origin comes from the `VITE_API_BASE_URL` environment variable (e.g. `http://my-server:8000`), producing an absolute URL.

### Result caching

Every API response is written into a `Map<id, SearchResult>` in memory and also saved to `localStorage` (key: `lets-solve-it:result-cache`). This means pages like Compare and Viz still work after a browser refresh even when no new search has been run.

---

## Layer 2 — The Vite Dev Proxy

**File:** `web/vite.config.ts`

```ts
server: {
  proxy: {
    "/api": { target: "http://127.0.0.1:8000", changeOrigin: true },
    "/health": { target: "http://127.0.0.1:8000", changeOrigin: true },
  },
}
```

The browser sends `POST /api/search` to `http://localhost:5173/api/search` (same origin — no CORS check). Vite silently rewrites and forwards it to `http://127.0.0.1:8000/api/search`. The browser never talks directly to FastAPI in dev. This also eliminates the `localhost` vs `127.0.0.1` CORS mismatch that browsers treat as different origins.

In production there is no proxy; the browser calls `VITE_API_BASE_URL` directly.

---

## Layer 3 — FastAPI (the Backend)

**File:** `backend/app.py`  
Started with: `uvicorn backend.app:app --host 127.0.0.1 --port 8000`

### Routes

| Route | Body | Returns |
|---|---|---|
| `GET /health` | — | `{status, provider, model_ready}` |
| `POST /api/search` | `{query: string, top_k: int}` | `SearchResponse` |
| `POST /api/search-by-audio` | multipart audio file | `SearchResponse` |
| `POST /api/classify-audio` | multipart audio file | `ClassificationResponse` |

### Lazy model loading

The CLAP model is **not loaded at startup**. The first real API call triggers `_get_provider()`, which acquires a thread lock and calls `build_provider(_settings)`. This means:
- Uvicorn binds port 8000 immediately (no 2-minute wait while weights load).
- `/health` responds instantly and shows `model_ready: false` until the first call.
- The first search/classify call takes ~30–90 seconds while weights and gallery load. Subsequent calls are fast.

### Environment variables read at startup

| Variable | Default | Effect |
|---|---|---|
| `MODEL_PROVIDER` | `placeholder` | `clap` loads the real model; anything else returns stub results |
| `CHECKPOINT_PATH` | `checkpoints/finetune11/best_r1.pt` | Path to your `.pt` fine-tuned weights |
| `AUDIO_ROOT` | `scripts/data/xc_audio` | Root folder containing `audio/xc/*.mp3` |
| `GALLERY_CACHE` | `data/gallery_embeddings.pt` | Where to save/load pre-computed embeddings |
| `CORS_ORIGINS` | localhost 5173/5174 | Comma-separated list of allowed browser origins |

---

## Layer 4 — ClapProvider

**File:** `backend/providers/clap_provider.py`

This class does everything model-related. It is constructed once and reused for every request.

### Startup sequence (first API call)

```
1. Load ClapProcessor from HuggingFace ("laion/clap-htsat-fused")
2. Load ClapModel architecture from HuggingFace
3. Load fine-tuned weights from checkpoints/best.pt
   - reads "model_state" key from the checkpoint dict
   - calls model.load_state_dict(sd, strict=False)
4. model.eval() — disables dropout, etc.
5. Move model to GPU if available, else CPU
6. Check for gallery_embeddings.pt
   - EXISTS  → load tensor + metadata list from disk (fast, ~1 second)
   - MISSING → call _build_gallery() (slow, see below)
```

### Gallery build (first ever run, or after -RebuildGallery)

```
1. Read data/xc_metadata_unified.csv
   - 27,913 rows: filepath, common_name, vocalization_type, species_code, duration
2. For each row, resolve the file: scripts/data/xc_audio/<filepath>
3. Decode audio → 48 kHz mono float32, centre-cropped to 10 seconds
4. Batch 16 clips at a time through ClapModel.get_audio_features()
5. L2-normalise each 512-dim embedding vector
6. Save tensor + metadata list to data/gallery_embeddings.pt
```

Result: a matrix of shape `[N, 512]` where N ≈ 17,765 (however many files are on disk).

### Inference: text search

```
query string
  → ClapProcessor (tokenise)
  → ClapModel.get_text_features()
  → L2-normalise → query vector [1, 512]
  → dot product with gallery matrix [N, 512]
  → top-k cosine similarity scores
  → return metadata rows sorted by score
```

### Inference: audio search / classify

```
uploaded bytes
  → librosa.load() at 48 kHz
  → centre-crop to 10 seconds
  → ClapProcessor (mel spectrogram)
  → ClapModel.get_audio_features()
  → L2-normalise → query vector [1, 512]
  → dot product with gallery matrix
  → top-k (search) OR per-species max + sort (classify)
```

The key point: **text and audio embeddings live in the same 512-dimensional space** because that is what contrastive training (CLAP) achieves. Similarity between a text query and an audio clip is a cosine dot product.

---

## Layer 5 — The Fine-Tuned Weights

**File:** `checkpoints/best.pt`

This is a standard PyTorch checkpoint dict with these keys:

| Key | Contents |
|---|---|
| `model_state` | `OrderedDict` of 477 named weight tensors |
| `optim_state` | AdamW optimiser state (not used at inference) |
| `scaler_state` | AMP GradScaler state (not used at inference) |
| `scheduler_state` | LR scheduler state (not used at inference) |
| `epoch` | Last completed training epoch (12) |
| `best_val_loss` | Best validation loss achieved (1.12) |

The base architecture (`laion/clap-htsat-fused`) is downloaded from HuggingFace on first use. The checkpoint only stores the weight deltas from fine-tuning — which is why `load_state_dict(strict=False)` is used (the checkpoint may omit some layers that were frozen during training).

---

## Data Flow Diagram — Text Search End-to-End

```
User types "robin song" → QueryPage.tsx
  │
  └─ searchDataset("robin song")          [backend.ts]
       │
       └─ fetch POST /api/search          [Vite proxy forwards to FastAPI]
            │
            └─ search() handler           [app.py]
                 │
                 └─ _get_provider()       [lazy-loads ClapProvider]
                      │
                      └─ search_text("robin song", top_k=10)   [clap_provider.py]
                           │
                           ├─ _encode_text(["robin song"])
                           │    └─ ClapModel.get_text_features()
                           │         └─ returns [1, 512] tensor
                           │
                           └─ _top_k(query_emb, 10)
                                └─ query_emb @ gallery_embs.T   → [N] cosine sims
                                     └─ torch.topk(k=10)
                                          └─ [{id, title, species, score, ...}, ...]

 FastAPI returns SearchResponse JSON
  │
 Vite proxy forwards response to browser
  │
 backend.ts mapItem() converts snake_case → camelCase SearchResult objects
  │
 cacheResults() writes to localStorage
  │
 QueryPage renders ResultCard grid
```

---

## Starting Everything

### Development (two terminals)

```powershell
# Terminal 1 — backend
powershell -ExecutionPolicy Bypass -File scripts\start_backend_clap.ps1

# Terminal 2 — frontend
cd web && npm run dev
```

### Development (one command)

```powershell
powershell -ExecutionPolicy Bypass -File start.ps1
```

This opens two separate PowerShell windows automatically.

### Gallery is stale / newly downloaded audio

```powershell
powershell -ExecutionPolicy Bypass -File start.ps1 -RebuildGallery
```

Deletes `data/gallery_embeddings.pt` before starting. The first search after startup will rebuild it from the full metadata CSV.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `/health` returns `model_ready: false` | Normal — first request hasn't arrived yet | Make one search; wait for gallery load |
| First search takes 5–20 minutes | Gallery is being rebuilt from ~17k recordings | Normal; subsequent searches are instant |
| Results look unrelated | Old gallery built from eval split only | Delete `data/gallery_embeddings.pt` and restart |
| `501 Not Implemented` from API | `MODEL_PROVIDER` not set to `clap` | Check env var; use `start_backend_clap.ps1` |
| `Failed to fetch` in browser | CORS or proxy not running | Confirm `npm run dev` is running; check Vite proxy config |
| Port 8000 already in use | Previous uvicorn still alive | `start_backend_clap.ps1` kills it automatically |
