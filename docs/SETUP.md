# Setup & Run Guide

This guide covers everything needed to get both the backend (FastAPI + CLAP model) and the frontend (React/Vite) running locally. Follow it top to bottom the first time; subsequent starts only need the "Start servers" section.

---

## What the system does

- **Backend** — a FastAPI server that loads your fine-tuned CLAP model and exposes three endpoints: text-to-audio search, audio-to-audio similarity, and species classification.
- **Frontend** — a React/Vite app with a search workspace, upload interface, and a 3D frequency visualizer. The visualizer runs entirely in the browser using the Web Audio API; it does not call the backend.
- **Gallery** — on the first backend start, CLAP embeds every audio file in the validation set and caches the result to `data/gallery_embeddings.pt`. This is the retrieval database. Subsequent starts load from cache in seconds.

---

## Prerequisites

| Tool | Version | Notes |
|------|---------|-------|
| Python | 3.10+ | 3.11 recommended |
| Node.js | 18+ | 20 LTS recommended |
| npm | 9+ | comes with Node |
| ffmpeg | any | required by librosa for MP3 decoding; must be on PATH |
| PyTorch | 2.x | GPU optional but strongly recommended for gallery build |

**Install Python ML dependencies:**
```powershell
pip install -r requirements-ml.txt
```

**Install Python web dependencies (FastAPI, etc.):**
```powershell
pip install fastapi uvicorn[standard] pydantic python-multipart
```

**Install frontend dependencies:**
```powershell
cd web
npm install
cd ..
```

---

## Required Files

Before starting the backend you need these files in place. Paths are relative to the repo root.

### Always required

| File | How to get it |
|------|--------------|
| `data/clap_val_pairs.json` | Run `python scripts/build_clap_training_pairs.py` or copy from a previous run |
| `data/xc_metadata_unified.csv` | Run `python get_xenocanto.ipynb` or copy from a previous run |

### Required on first start (to build the gallery)

| File | How to get it |
|------|--------------|
| `checkpoints/finetune11/best_r1.pt` (or your chosen checkpoint) | Output of `python scripts/train_clap.py` |
| Audio files under `scripts/data/xc_audio/` | Run `python scripts/download_xc_audio.py` |

> **If audio files are on a different machine:** Build the gallery cache there (`data/gallery_embeddings.pt`) and copy that single file to this machine. The backend will load it directly without needing any audio files.

### Optional (enriches metadata)

| File | How to get it | What it adds |
|------|--------------|--------------|
| `data/species_taxonomy.json` | `python scripts/build_taxonomy_db.py` | Scientific names in results |

---

## Backend Setup

### 1. Configure environment variables

Copy `.env.example` (at repo root, if present) or set variables directly in your shell. The minimum to enable the real model:

```powershell
$env:MODEL_PROVIDER   = "clap"
$env:CHECKPOINT_PATH  = "checkpoints/finetune11/best_r1.pt"
$env:AUDIO_ROOT       = "scripts/data/xc_audio"
```

All variables with their defaults:

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PROVIDER` | `placeholder` | Set to `clap` to load the real model |
| `CHECKPOINT_PATH` | `checkpoints/finetune11/best_r1.pt` | Path to your `.pt` checkpoint |
| `AUDIO_ROOT` | `scripts/data/xc_audio` | Root folder containing the `.mp3` / `.wav` audio files |
| `GALLERY_CACHE` | `data/gallery_embeddings.pt` | Where to save/load the embedding cache |
| `BASE_MODEL` | `laion/clap-htsat-fused` | HuggingFace base model ID |
| `METADATA_PATH` | `data/xc_metadata_unified.csv` | Metadata CSV |
| `TAXONOMY_PATH` | `data/species_taxonomy.json` | Taxonomy JSON (optional) |
| `VAL_PAIRS_PATH` | `data/clap_val_pairs.json` | Val pairs JSON (defines the gallery) |
| `CORS_ORIGIN` | `http://localhost:5173` | Frontend origin allowed by CORS |

If you leave `MODEL_PROVIDER` as `placeholder`, the backend starts immediately but all inference endpoints return `501 Not Implemented`.

### 2. Start the backend

Run from the repo root:

```powershell
uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload
```

**What happens on first start (with `MODEL_PROVIDER=clap`):**

1. The HuggingFace base model downloads to the local cache (~1.8 GB, one-time).
2. Your fine-tuned checkpoint weights are loaded on top.
3. The gallery is built by loading every unique audio file from `clap_val_pairs.json`, running it through the audio encoder, and saving the result to `data/gallery_embeddings.pt`. This takes **5–20 minutes** depending on hardware (GPU strongly recommended).
4. The server becomes ready and logs: `Gallery ready: N embeddings, dim=512`.

**On subsequent starts:** Step 3 is replaced by loading `data/gallery_embeddings.pt` directly, which takes a few seconds.

Verify it's up:
```powershell
curl http://localhost:8000/health
# {"status":"ok","provider":"clap"}
```

---

## Frontend Setup

### 1. Create the environment file

```powershell
Copy-Item web\.env.example web\.env
```

The default points to `http://localhost:8000` which is correct if the backend is running locally. Edit `web/.env` if your backend is on a different host or port:

```
VITE_API_BASE_URL=http://localhost:8000
```

### 2. Start the frontend dev server

```powershell
cd web
npm run dev
```

Open `http://localhost:5173` in your browser.

---

## Verify End-to-End

With both servers running:

1. Open `http://localhost:5173/query`.
2. Type `Northern Cardinal call` in the search box and press Enter. You should see real results with similarity scores returned from the backend.
3. Upload a `.wav` or `.mp3` file using "Choose audio", then click "Classify" — you should see species names with scores.
4. Click any result card's visualizer link. The 3D frequency visualization opens and animates using your audio file — **this runs entirely in the browser** and does not call the backend.

---

## The 3D Frequency Visualizer

The visualizer (`/viz/:id`) is fully client-side and works independently of the backend:

- Your uploaded audio file is decoded locally by the browser's Web Audio API.
- A custom chirp-detection algorithm finds contiguous amplitude bursts and identifies their dominant frequency (2–8 kHz range).
- Each chirp becomes a glowing point in a 3D point cloud, coloured by frequency (low = red, high = blue), rendered with Three.js bloom post-processing.
- If no file is uploaded, a synthetic bird chirp is generated from a seed for demonstration.

There is nothing to configure — it works as long as the frontend is running.

---

## Gallery Cache Management

The gallery cache (`data/gallery_embeddings.pt`) is tied to the model weights. **Delete it and let it rebuild whenever you:**

- Load a different checkpoint.
- Change `BASE_MODEL` to a different architecture.
- Want to rebuild from a different `VAL_PAIRS_PATH`.

To pre-build the cache on a GPU machine and copy it elsewhere:

```powershell
# On the GPU machine — just start the backend once with CLAP configured
$env:MODEL_PROVIDER = "clap"
uvicorn backend.app:app --port 8000
# Wait for "Gallery ready" in the logs, then Ctrl+C

# Copy the cache to the target machine
scp data/gallery_embeddings.pt user@target:/path/to/repo/data/
```

The target machine does **not** need the audio files or GPU — only the cache file and the checkpoint.

---

## Troubleshooting

**`Missing VITE_API_BASE_URL` error in browser console**
→ `web/.env` does not exist. Run `Copy-Item web\.env.example web\.env`.

**Backend returns `501 Not Implemented` on all inference calls**
→ `MODEL_PROVIDER` is still `placeholder`. Set `$env:MODEL_PROVIDER = "clap"` before starting uvicorn.

**`Gallery empty — no audio loaded` on startup**
→ `AUDIO_ROOT` does not point to the folder containing your `.mp3`/`.wav` files, or the files have not been downloaded yet.

**`FileNotFoundError: Val pairs JSON not found`**
→ `data/clap_val_pairs.json` is missing. Run `python scripts/build_clap_training_pairs.py`.

**`No checkpoint at ... — using base model weights`**
→ `CHECKPOINT_PATH` points to a file that does not exist. Verify the path. The backend will still start and use the zero-shot base model weights.

**Gallery build is very slow (CPU only)**
→ Install PyTorch with CUDA support. Even a modest GPU (e.g. RTX 3060) cuts gallery build time from hours to minutes.

**CORS error in browser**
→ The frontend origin doesn't match `CORS_ORIGIN`. Set `$env:CORS_ORIGIN = "http://localhost:5173"` (or whatever URL your Vite dev server is using) before starting the backend.

**`AUDIO_DECODE_FAILED` on audio upload**
→ The uploaded file could not be decoded. Ensure ffmpeg is installed and on your PATH (required for MP3). WAV files work without ffmpeg.
