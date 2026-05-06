# BirdCLAP — Setup Guide

Everything a new developer needs to go from zero to a running BirdCLAP stack. Read top to bottom the first time.

---

## What You Need

### 1. The repository

Clone or copy the full repo. Every path below is relative to the repo root (`lets-solve-it/`).

### 2. Python 3.11+

Python 3.14 is what the original development used. 3.11–3.14 all work.

### 3. PyTorch with CUDA (strongly recommended)

Gallery rebuilds on CPU take 4–8 hours for ~17k recordings. On GPU they take 5–20 minutes.

Find your CUDA version with `nvidia-smi`, then install the matching wheel:

```powershell
# CUDA 12.6 / 12.7 / 12.8 (all compatible)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# CPU-only fallback
pip install torch torchvision torchaudio
```

Verify:

```powershell
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

### 4. Backend Python dependencies

```powershell
pip install fastapi uvicorn python-multipart pydantic
pip install transformers accelerate soundfile librosa pandas tqdm
```

Or from the requirements file (API layer only — PyTorch must be installed first):

```powershell
pip install -r backend/requirements.txt
```

### 5. Node.js 18+ (frontend)

```powershell
cd web && npm install
```

---

## Required Data and Model Files

### Files already in the repo

| File | Size | What it is |
|---|---|---|
| `data/xc_metadata_unified.csv` | ~2.4 MB | Master catalog — one row per recording with `filepath`, `common_name`, `vocalization_type`, `species_code`, `duration` |
| `data/species_taxonomy.json` | ~115 KB | 502 species → scientific name |
| `data/species_descriptions.json` | ~410 KB | Acoustic descriptions per species (scraped from AllAboutBirds + Animal Diversity Web) |
| `data/clap_descriptions.json` | ~2.5 MB | LLM-generated rich acoustic descriptions used during fine-tuning |
| `data/clap_all_labels.json` | ~3 MB | All text labels used during training |
| `data/clap_train_pairs.json` | ~25 MB | 121k audio–text training pairs |
| `data/clap_val_pairs.json` | ~4 MB | 19k validation pairs |
| `data/clap_holdout_pairs.json` | ~7 MB | Held-out eval pairs (never seen during training) |

### Files you need to obtain separately

#### `checkpoints/best.pt` — fine-tuned weights (~1.6 GB)

This is the result of 12+ fine-tuning runs. Obtain from shared storage.

Place at: `checkpoints/best.pt`

To use a different checkpoint, pass `-CheckpointPath` to the startup script.

#### `scripts/data/xc_audio/audio/xc/*.mp3` — bird recordings

MP3 files from [Xeno-Canto](https://xeno-canto.org). The gallery is built from whatever is on disk — you do not need all ~27k. Even a few hundred will make the backend functional; more recordings = richer search results. The current production setup has ~17,765 files.

To download:

```powershell
# Full download (takes hours; safe to interrupt and resume)
python scripts/download_xc_audio.py

# Quick start — 500 recordings only
python scripts/download_xc_audio.py --limit 500
```

Files should resolve as `scripts/data/xc_audio/audio/xc/<id>.mp3`.

#### `data/gallery_embeddings.pt` — search index (auto-generated)

Built automatically on the first search request, or explicitly via:

```powershell
python scripts/build_gallery.py
```

If you receive a pre-built gallery from someone else, verify it was made with the same checkpoint — otherwise delete it and rebuild.

---

## Directory Layout

```
lets-solve-it/
├── checkpoints/
│   └── best.pt                          ← obtain from shared storage
├── data/
│   ├── xc_metadata_unified.csv          ← in repo
│   ├── species_taxonomy.json            ← in repo
│   ├── species_descriptions.json        ← in repo
│   ├── clap_descriptions.json           ← in repo
│   ├── clap_train_pairs.json            ← in repo
│   ├── clap_val_pairs.json              ← in repo
│   ├── clap_holdout_pairs.json          ← in repo
│   └── gallery_embeddings.pt            ← auto-generated
├── scripts/
│   └── data/
│       └── xc_audio/
│           └── audio/
│               └── xc/
│                   ├── 433216.mp3       ← any number of .mp3 files
│                   └── ...
├── backend/
├── web/
└── start.ps1
```

---

## Running the Stack

### One command (opens two windows)

```powershell
powershell -ExecutionPolicy Bypass -File start.ps1
```

Kills anything on port 8000, launches FastAPI in one window and Vite in another. Set `MODEL_PROVIDER=clap` (the script does this automatically when `checkpoints/best.pt` exists).

### Two terminals manually

**Terminal 1 — Backend:**

```powershell
$env:MODEL_PROVIDER = "clap"
$env:CHECKPOINT_PATH = "checkpoints/best.pt"
uvicorn backend.app:app --reload --port 8000
```

**Terminal 2 — Frontend:**

```powershell
cd web
npm run dev
```

### URLs

- Frontend: `http://localhost:5173`
- Backend health: `http://localhost:8000/health`

---

## What Happens on First Run

1. Uvicorn starts and binds port 8000 immediately.
2. The audio file map is scanned at startup (fast, no model needed) — `/api/audio/{id}` is available immediately.
3. You open the frontend at `http://localhost:5173`.
4. You run the first search or upload a file.
5. **At this point** the CLAP model loads lazily:
   - Downloads the base architecture from HuggingFace (`laion/clap-htsat-fused`) if not cached — ~1.5 GB, once only.
   - Loads fine-tuned weights from `checkpoints/best.pt`.
   - If `data/gallery_embeddings.pt` exists and matches the current `CACHE_VERSION`, loads it (~1 second).
   - Otherwise builds the gallery by encoding all audio files through the model — **5–20 minutes** on GPU, several hours on CPU. Progress is printed in the backend terminal.
6. Once the gallery is saved, every subsequent start loads it instantly.
7. Additionally, at model load time:
   - Species text gallery built (~509 species text embeddings — a few seconds).
   - Acoustic text gallery built from `clap_descriptions.json`.
   - These enable the text-to-text catalog search and "What CLAP heard" features.

---

## Rebuilding the Gallery

Rebuild when:
- You switch to a different checkpoint.
- You add new audio files to `scripts/data/xc_audio/audio/xc/`.

```powershell
# Via start.ps1 flag (deletes old cache, rebuilds on next request)
powershell -ExecutionPolicy Bypass -File start.ps1 -RebuildGallery

# Explicitly with progress (recommended for large rebuilds)
python scripts/build_gallery.py

# Manual delete then restart
Remove-Item data/gallery_embeddings.pt
```

---

## Features at a Glance

| Feature | How it works |
|---|---|
| **Catalog text search** | Text query → text-to-text cosine similarity against species acoustic descriptions → returns recordings for top-matched species |
| **Audio similarity search** | Upload → 10 s centre-crop → CLAP audio encoder → cosine sim against gallery |
| **Classification** | Same encoding as similarity; groups by species, returns per-species max sim |
| **Real spectrograms** | `ResultCard` fetches from `/api/audio/{id}`, decodes with Web Audio API |
| **Audio playback** | Native `<audio>` element per result card; full MP3 served (not the 10 s training sidecar) |
| **3-D sound map** | Catalog results fetch their real audio file; Three.js viz uses the actual recording |
| **Analysis window picker** | For uploads > 10 s: drag a window, preview 10 s, search uses that exact slice |
| **What CLAP heard** | Top-4 unique species from the similarity search, with their acoustic profiles |

---

## Evaluating the Model

```powershell
# Retrieval metrics against the held-out set
python scripts/evaluate_clap.py

# Compare fine-tuned vs base model
python scripts/evaluate_clap.py --checkpoint checkpoints/best.pt --also-base
```

Results are written to `results/`. The key metric is **Hit@5** — fraction of queries where the correct species appears in the top 5 results.

---

## Troubleshooting

### Backend says `model_ready: false`

Normal. Model loads lazily on the first API call. Make one search and wait. The audio serving endpoint works immediately regardless.

### Gallery build takes forever or seems stuck

Check the backend terminal for `Gallery progress: N embedded so far`. If nothing appears after 5 minutes:

```powershell
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

`False` means CPU. Reinstall PyTorch with the CUDA index URL.

### Text search returns unrelated species

The species text gallery may not have been built yet (appears in backend logs as `Species text gallery ready: 509 species embeddings`). Restart the backend and wait for the model to load fully.

### Similarity search returns wrong species

The gallery was probably built with a different checkpoint. Delete `data/gallery_embeddings.pt` and restart.

### `WinError 10048` — port already in use

```powershell
netstat -ano | findstr :8000
taskkill /PID <pid> /F
```

`start.ps1` kills the old process automatically; this only happens if you start manually.

### HuggingFace download is slow or fails

```powershell
$env:HF_TOKEN = "hf_your_token_here"
```

Or download the base model manually and set `BASE_MODEL` to a local path.

### Audio plays for only 10 seconds

You are being served a 10-second training WAV sidecar instead of the full MP3. Check that `_build_startup_audio_map` in `backend/app.py` is finding the MP3 files — it prefers `.mp3` over `.wav`. Verify the MP3s exist in `scripts/data/xc_audio/audio/xc/`.
