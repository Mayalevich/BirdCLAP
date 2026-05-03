# Setup Guide — Getting CLAP Running on a New Machine

This guide covers everything a new developer needs to go from zero to a running backend with the fine-tuned CLAP model. Read it top to bottom the first time.

---

## What You Need

### 1. The repository

Clone or copy the full repo. Everything below assumes you are in the repo root (`lets-solve-it/`).

### 2. Python

Python **3.11 or newer** is required. Python 3.14 is what this was originally developed on.

### 3. PyTorch with CUDA (recommended)

If you have an NVIDIA GPU, install the CUDA build — gallery rebuilds are 10–30x faster.

Find your CUDA version with `nvidia-smi`, then install the matching wheel:

```powershell
# CUDA 12.6 / 12.7 / 12.8 (all compatible)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# CPU-only fallback (slow gallery rebuild, fine for inference after cache is warm)
pip install torch torchvision torchaudio
```

### 4. Backend Python dependencies

```powershell
pip install fastapi uvicorn python-multipart pydantic
pip install transformers accelerate soundfile librosa pandas tqdm
```

Or from the backend requirements file (covers the API layer):

```powershell
pip install -r backend/requirements.txt
```

The ML packages (`transformers`, `librosa`, etc.) are intentionally not in that file because PyTorch must be installed first with the right CUDA variant.

### 5. Node.js 18+ (frontend only)

```powershell
cd web && npm install
```

---

## Required Data Files

All of these live under `data/` and `scripts/data/`. Some are small config files, some are large binary caches.

### Files already in the repo (committed or shared)

| File | Size | What it is |
|---|---|---|
| `data/xc_metadata_unified.csv` | ~2 MB | Master catalog: one row per recording, with `filepath`, `common_name`, `vocalization_type`, `species_code`, `duration` |
| `data/species_taxonomy.json` | ~115 KB | 502 species → scientific name + taxonomic tree |
| `data/clap_all_labels.json` | ~3 MB | All text labels used during training |
| `data/clap_train_pairs.json` | ~25 MB | 121,090 audio–text pairs used for training |
| `data/clap_val_pairs.json` | ~4 MB | 19,434 pairs used for validation during training |
| `data/clap_holdout_pairs.json` | ~7 MB | Held-out pairs never seen during training (used for final eval) |
| `data/species_descriptions.json` | ~410 KB | Long-form descriptions per species |
| `data/clap_descriptions.json` | ~2.5 MB | Generated text descriptions used to augment training |

### Files you need to obtain separately

#### `checkpoints/best.pt` — the fine-tuned weights

This is the result of 12 fine-tuning runs. Get it from whoever owns the training output (shared drive, cloud storage, etc.). It should be ~1.6 GB.

Put it at exactly: `checkpoints/best.pt`

If you want to use a different checkpoint, pass `-CheckpointPath` to the startup scripts.

#### `scripts/data/xc_audio/audio/xc/*.mp3` — the audio recordings

MP3 files downloaded from [Xeno-Canto](https://xeno-canto.org). The full metadata catalog lists ~27,900 recordings, but **the gallery is built from whatever is on disk** — you don't need all of them. The more you have, the richer the search results. The original setup has ~17,765 files, but even a few hundred will get the backend running and returning results.

**To download recordings**, run:

```powershell
python scripts/download_xc_audio.py
```

The script reads `scripts/xc_metadata_unified.csv`, skips files already on disk, and saves to `scripts/data/xc_audio/audio/xc/<id>.mp3`. It supports parallel workers and adaptive rate-limiting. You can cap the download with `--limit N` if you only want a subset:

```powershell
# Download only 500 recordings as a quick start
python scripts/download_xc_audio.py --limit 500
```

If you already have an audio folder from someone else, place it so files resolve as `scripts/data/xc_audio/audio/xc/<id>.mp3`.

#### `data/gallery_embeddings.pt` — pre-computed search index

This is built automatically the first time you run the backend and make a search request. You do **not** need to obtain this manually — it will be generated.

If you receive a pre-built gallery from someone else (saves ~10–20 minutes of GPU time), put it at `data/gallery_embeddings.pt`. Make sure it was built with the same checkpoint you are using; otherwise delete it and let it rebuild.

---

## Directory Structure You Need

```
lets-solve-it/
├── checkpoints/
│   └── best.pt                          ← get from shared storage
├── data/
│   ├── xc_metadata_unified.csv          ← in repo
│   ├── species_taxonomy.json            ← in repo
│   ├── clap_train_pairs.json            ← in repo
│   ├── clap_val_pairs.json              ← in repo
│   ├── clap_holdout_pairs.json          ← in repo
│   └── gallery_embeddings.pt            ← auto-generated on first run
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

### Option A — one command (opens two windows)

```powershell
powershell -ExecutionPolicy Bypass -File start.ps1
```

This kills anything on port 8000, starts FastAPI in one window and Vite in another.

### Option B — two terminals

**Terminal 1 — Backend:**

```powershell
powershell -ExecutionPolicy Bypass -File scripts/start_backend_clap.ps1
```

**Terminal 2 — Frontend:**

```powershell
cd web
npm run dev
```

### URLs

- Frontend: `http://localhost:5173`
- Backend health check: `http://localhost:8000/health`

---

## What Happens on First Run

1. Uvicorn starts and binds port 8000 immediately.
2. You open the frontend at `http://localhost:5173`.
3. You run the first search or upload a file.
4. **At this point** the CLAP model loads:
   - Downloads the base architecture from HuggingFace (`laion/clap-htsat-fused`) if not cached — this is a ~1.5 GB download, once only.
   - Loads your fine-tuned weights from `checkpoints/best.pt`.
   - Builds `data/gallery_embeddings.pt` by running all ~17,765 MP3 files through the model — **takes 5–20 minutes** depending on GPU/CPU. Progress is printed in the backend terminal.
5. Once the gallery is saved, every subsequent start loads it in ~1 second.

---

## Rebuilding the Gallery

You need to rebuild the gallery when:
- You switch to a different checkpoint
- You add new audio files to `scripts/data/xc_audio/audio/xc/`

```powershell
# Deletes the old gallery and rebuilds on next search
powershell -ExecutionPolicy Bypass -File start.ps1 -RebuildGallery
```

Or manually:

```powershell
Remove-Item data/gallery_embeddings.pt
```

Then restart the backend normally — the rebuild happens automatically on the first request.

---

## Evaluating the Model

To run the zero-shot evaluation against the held-out set:

```powershell
python scripts/evaluate_clap.py
```

Results are written to `results/`. The key metric is **Hit@5** — the fraction of queries where the correct species appears in the top 5 results. The current checkpoint (epoch 12) achieves 62.8% Hit@5 on the held-out set.

---

## Troubleshooting

### Backend says `model_ready: false`

Normal. The model loads on the first API call, not at startup. Make one search and wait.

### Gallery rebuild takes forever / seems stuck

Check the backend terminal window for progress logs (`Gallery progress: N embedded so far`). If there are no logs after 5 minutes, the model may be running on CPU — verify CUDA is available:

```powershell
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

If it prints `False`, reinstall PyTorch with the CUDA index URL (see step 3 above).

### `WinError 10048` — port already in use

The startup scripts kill the existing process automatically. If it still fails:

```powershell
netstat -ano | findstr :8000
taskkill /PID <pid> /F
```

### HuggingFace download is slow or fails

Set a token for higher rate limits:

```powershell
$env:HF_TOKEN = "hf_your_token_here"
```

Or download the base model manually and set `BASE_MODEL` to a local path.

### Results look wrong / unrelated species

The gallery may have been built with a different checkpoint. Delete `data/gallery_embeddings.pt` and restart.
