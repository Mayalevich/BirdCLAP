# BirdCLAP — Development Log

Cumulative plain-English changelog for the BirdCLAP web platform. Each section records what changed, why, and which files were touched. Does not replace the code — it is the "why" layer on top of it.

---

## Session 1 — Rebranding, UI polish, and rich descriptions

### 1.1 Rebranding to BirdCLAP

**Files:** `web/index.html`, `web/src/layout/AppShell.tsx`, `web/src/index.css`

- Document `<title>` → **BirdCLAP**.
- App header title → **BirdCLAP**; subtitle rewritten to emphasise the CLAP model ("contrastive audio-language embeddings") rather than the 3-D visualisation.
- Removed the "API OK" status badge from the header (backend probe and error banners still work — the badge was noise).
- Footer description line removed entirely.

### 1.2 UI polish

**Files:** `web/src/index.css`, `web/src/pages/HomePage.tsx`

- Tighter border radii, deeper panel shadows, consistent header/nav styling via CSS variables.
- Spectrogram preview card gained a drop-shadow and better proportions.
- Dedicated `audio-intake__cta` wrapper around the search button — top-border separator, generous padding, `btn--cta-primary` style.
- Animated placeholder spectrogram bars added to the empty state on `HomePage`.

### 1.3 Species description cards

**Files:** `backend/config.py`, `backend/schemas.py`, `backend/providers/base.py`, `backend/providers/clap_provider.py`, `backend/provider_factory.py`, `web/src/api/types.ts`, `web/src/api/backend.ts`, `web/src/components/ResultCard.tsx`, `web/src/index.css`

- Config: `species_descriptions_path` and `clap_descriptions_path` fields added.
- Schema: `SearchResultItem` gained `species_description: str | None`.
- `ClapProvider._load_species_descriptions` reads `data/species_descriptions.json` and truncates to ≤ 280 chars (1–2 sentences).
- `_top_k` attaches the matching species description to every result via `_desc_map`.
- `ResultCard` renders a collapsible `<details>` / `<summary>` block labelled **Vocalization notes** when a description is present.

### 1.4 "What CLAP heard" panel

**Files:** `backend/schemas.py`, `backend/providers/base.py`, `backend/providers/placeholder.py`, `backend/providers/clap_provider.py`, `backend/app.py`, `web/src/api/backend.ts`, `web/src/pages/QueryPage.tsx`, `web/src/index.css`

- `POST /api/describe-audio` endpoint added.
- `ClapProvider._build_text_gallery` pre-encodes all `clap_descriptions.json` entries into a text embedding matrix at startup.
- `QueryPage` fires `describe_audio` in parallel with `search_by_audio`.
- Results appear in a **What CLAP heard** panel below classification — plain acoustic descriptions, no species names originally; later overhauled (see §3.3).

### 1.5 Acoustic text search hints

**Files:** `web/src/pages/QueryPage.tsx`

- Text search placeholder updated to suggest acoustic queries.
- Six clickable example chips populate the search box on click.

---

## Session 2 — Catalog audio: real files, real spectrograms, real 3-D map

### Problem

Search results showed mock spectrograms because `audio_url` was never populated. The `_audio_map` inside `ClapProvider` was built from `audio_rel` keys that the stale gallery cache didn't have.

### 2.1 Backend: audio path resolution

**Files:** `backend/providers/base.py`, `backend/providers/clap_provider.py`

- `BaseProvider.get_audio_path(recording_id)` added as an overridable hook.
- `ClapProvider._build_audio_map` uses a two-pass strategy:
  1. `audio_rel` stored per meta entry (new caches).
  2. Full directory scan by file stem (old caches — no rebuild required).
- **MP3 preferred over WAV** so the full original recording is served rather than a 10-second training sidecar.

### 2.2 Backend: `/api/audio/{recording_id}` endpoint

**Files:** `backend/app.py`

- `GET /api/audio/{recording_id}` returns the file via `FileResponse` with `Accept-Ranges`.
- **Critical fix:** audio map built at FastAPI startup, independent of the CLAP model. `serve_audio` reads the map directly — never calls `_get_provider()` — so the first request is instant instead of blocking for several minutes while the model loads.
- `/health` now reports `audio_files_indexed`.

### 2.3 Frontend: real spectrograms in result cards

**Files:** `web/src/lib/spectrogramCanvas.ts`, `web/src/components/ResultCard.tsx`, `web/src/index.css`

- `drawSpectrogramFromAudioBuffer` extracted and **exported** from `spectrogramCanvas.ts`.
- New `drawSpectrogramFromUrl(url, canvas)` fetches audio, decodes with Web Audio API, delegates to the buffer renderer.
- `ResultCard` uses `drawSpectrogramFromUrl` when `result.audioUrl` is present; renders a native `<audio controls>` player.

### 2.4 Frontend: 3-D visualisation uses catalog audio

**Files:** `web/src/pages/VizPage.tsx`

- For catalog results `VizPage` fetches `result.audioUrl` as a Blob → `File` and passes it to `BirdSoundEmbeddingViz`.
- `audioFetchState` (`idle | loading | done | error`) drives the source label so users always see the real fetch status.

---

## Session 3 — Audio encoding fix, gallery rebuild, analysis window picker

### 3.1 The 10-second centre-crop decision

**Files:** `backend/providers/clap_provider.py`

**Context:** The model was fine-tuned on 10-second WAV sidecars pre-cut by `convert_to_wav.py`. Switching to full-duration fusion (attempted earlier and reverted) produced gallery embeddings that no longer matched the fine-tuned checkpoint's learned representations, causing completely wrong results.

**Final strategy (CACHE_VERSION = 2):**

- `_load_file` (gallery build): loads the 10-second `.wav` sidecar if present, falls back to the full MP3.
- `_decode_bytes` (query upload): centre-crops user uploads to 10 seconds to match the gallery space.
- `_build_audio_map`: **MP3 preferred over WAV for serving** — full original recordings are streamed to the browser; the 10-second WAVs are only used for embedding.
- `CACHE_VERSION` bumped to 2; stale caches are detected on startup and rebuilt automatically.

### 3.2 Standalone gallery build script

**Files:** `scripts/build_gallery.py`

- Independent of the FastAPI server; shows progress and exits cleanly.
- Use when you want visible rebuild progress rather than waiting on the first API call:

```powershell
python scripts/build_gallery.py
```

### 3.3 "What CLAP heard" consistency fix

**Files:** `backend/providers/clap_provider.py`, `web/src/pages/QueryPage.tsx`

**Problem:** `describe_audio` was performing *audio → text embedding* cross-modal comparison. The fine-tuning modified the audio encoder significantly, so the audio embeddings no longer aligned with the text encoder's space — the returned descriptions were about completely unrelated species.

**Fix:**
- Backend: `describe_audio` now routes through `_top_k` (the same audio gallery path as `search_by_audio`) and returns `_desc_map` entries for the matched species. The cross-modal text gallery is still built but no longer used for this endpoint.
- Frontend: `runSimilarSearch` derives "What CLAP heard" directly from the result cards' `speciesDescription` fields — no second API round-trip, guaranteed to match the displayed results.
- UI: each item now shows the **species name** as a labelled header above the acoustic description.

### 3.4 Analysis window picker

**Files:** `web/src/lib/audioUtils.ts` (new), `web/src/components/AudioWindowPicker.tsx` (new), `web/src/pages/QueryPage.tsx`, `web/src/index.css`

**Problem:** Field recordings often start with noise (dog barks, wind, car doors) that confused the model because it centre-cropped the first available 10 seconds regardless of content.

**Feature:**

When an uploaded file is longer than 10 seconds, a **window picker** appears between the action buttons and the spectrogram:

- Draggable range slider to position the 10-second analysis window anywhere in the recording.
- Live time display: `0:08 – 0:18 / 1:23`.
- **▶ Preview 10 s** button — plays the selected window from the original file; auto-stops at the boundary. Click again to stop early.
- The spectrogram updates live as the slider moves (shows only the selected window).
- **Classify** and **Search similar** both encode only the selected 10-second slice as a WAV and send it to the backend. Full pipeline: `decodeAudioFile → sliceAudioBuffer → audioBufferToWavFile` entirely in the browser, no server round-trip for cutting.
- "What CLAP heard" and `describe_audio` also receive the windowed file.
- Short clips (≤ 10 s) are unaffected; the picker is hidden.

### 3.5 Catalog text search — text-to-text species matching

**Files:** `backend/providers/clap_provider.py`

**Problem:** Catalog text search (`POST /api/search`) was doing *text query → audio gallery* cross-modal comparison. Fine-tuning modified both the audio and text encoders on audio-text pairs. After fine-tuning, the shared cross-modal embedding space shifted, so "sharppeeknote similar to Downy Woodpecker" returned Red Crossbill instead of Hairy Woodpecker.

**Root cause confirmed:** Verified in isolation — the fine-tuned text encoder *does* correctly rank Hairy Woodpecker at 0.72 cosine similarity for that exact query when compared against other text strings (text-to-text). The failure was only on the text-to-audio cross-modal path.

**Fix — `_build_species_text_gallery` (new standalone method):**

At model load time, encodes every species entry from `_desc_map` as `"{species name} — {acoustic description}"` using the CLAP text encoder. Stored as `_species_text_embs` (matrix) + `_species_names_list` (index → name).

**`search_text` rewritten:**

1. Encode the query as text (same encoder).
2. Compare against `_species_text_embs` (text→text, no cross-modal gap).
3. Collect top-N unique species; look up recording indices via `_species_to_indices` (built at startup, keyed case-insensitively).
4. Return up to 2 recordings per matched species.
5. Falls back to the original cross-modal path only if the species gallery is absent.

**Key details:**

- `_species_to_indices` is keyed by `species_name.lower()` to absorb any capitalisation differences between `species_descriptions.json` and the metadata CSV.
- The method runs independent of `clap_descriptions.json` (uses `_desc_map` from `species_descriptions.json` which is always present).
- 434 of 436 metadata-CSV species have entries in `species_descriptions.json` — near-complete coverage.

---

## Configuration reference (current defaults)

| Variable | Default | Notes |
|---|---|---|
| `MODEL_PROVIDER` | `placeholder` | Set to `clap` for real inference |
| `CHECKPOINT_PATH` | `checkpoints/best.pt` | Fine-tuned weights |
| `AUDIO_ROOT` | `scripts/data/xc_audio` | Where MP3s live |
| `GALLERY_CACHE` | `data/gallery_embeddings.pt` | Auto-built on first run |
| `BASE_MODEL` | `laion/clap-htsat-fused` | HuggingFace base |
| `METADATA_PATH` | `data/xc_metadata_unified.csv` | Master catalog |
| `TAXONOMY_PATH` | `data/species_taxonomy.json` | Scientific names |
| `VAL_PAIRS_PATH` | `data/clap_val_pairs.json` | Fallback gallery source |
| `SPECIES_DESCRIPTIONS_PATH` | `data/species_descriptions.json` | Per-species text |
| `CLAP_DESCRIPTIONS_PATH` | `data/clap_descriptions.json` | Training descriptions |

---

## Known limitations and future work

- **Text search quality** is bounded by the acoustic richness of `species_descriptions.json`. Species whose AllAboutBirds page has thin descriptions will rank below their true acoustic similarity.
- **Mnemonic queries** ("drink your teeeea") work only if the corresponding species description explicitly mentions the mnemonic or the CLAP text encoder generalises to it.
- **Gallery coverage** depends entirely on which MP3 files are on disk — species with no audio files on disk will never appear in any result set even if they match the query perfectly.
- **Audio window picker** only shows when the file is >10 s; very short clips receive no segmentation help.
- **VAD (voice-activity detection)** pre-pass would improve automatic window selection by finding the segment with the highest bird-audio energy, rather than relying on user judgement.
- **Gallery rebuild** is still required when switching checkpoints or adding new audio files. Use `start.ps1 -RebuildGallery` or `python scripts/build_gallery.py`.
