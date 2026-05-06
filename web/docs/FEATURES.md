# Web application — feature reference

This document describes every user-visible feature and the code that implements it.

Runtime search/classify/similarity use `src/api/backend.ts` against a live FastAPI backend. `src/api/mock.ts` exists as an offline reference but is not used at runtime.

---

## 1. Product scope

The web client is a full-featured bird-audio exploration workspace:

- Search a species catalog via the real backend (`POST /api/search`).
- Upload an audio file — decoded locally for spectrograms and the 3-D viz.
- Run classification and audio similarity against the backend.
- Select a specific 10-second analysis window from long recordings before searching.
- See "What CLAP heard" — acoustic profiles of the top-matched species.
- Save rows to `localStorage`, compare two slots, open the 3-D audio visualisation.

---

## 2. Technology stack

| Layer | Choice |
|-------|--------|
| Build | Vite 6 |
| UI | React 18 + TypeScript |
| Routing | React Router 6 |
| 3D | Three.js (WebGLRenderer, OrbitControls, CSS2D labels, EffectComposer + UnrealBloomPass) |
| Styling | Single global `index.css` (CSS variables for design tokens) |

---

## 3. Application entry and routing

**`src/main.tsx`** — mounts React root.

**`src/App.tsx`** — wraps tree in:
- `BrowserRouter`
- `AppPreferencesProvider` — upload + vocab + compare slots (in-memory except vocab)
- `SavedProvider` — saved list (backed by `localStorage`)

| Route | Component | Purpose |
|-------|-----------|---------|
| `/` | `HomePage` | Workspace overview, upload entry, quick links |
| `/query` | `QueryPage` | Catalog search, upload, classify/similar, analysis window, results grid |
| `/saved` | `SavedPage` | Grid of saved `SearchResult` rows |
| `/compare` | `ComparePage` | Two-slot side-by-side comparison |
| `/viz/:id` | `VizPage` | 3-D audio visualisation; `/viz/upload` uses user clip; other IDs use catalog audio |
| `*` | `Navigate → /` | Unknown paths bounce home |

---

## 4. Shell, navigation, and branding

**`src/layout/AppShell.tsx`**

- **Brand block:** Title **BirdCLAP** + subtitle emphasising contrastive audio-language embeddings.
- **Navigation:** `NavLink`s — Overview (`/`), Query, Saved, Compare. Compare shows badge `2` when both slots are filled.
- **Skip link:** keyboard "Skip to content" → `#main-content`.

---

## 5. Overview page (`/`)

**`src/pages/HomePage.tsx`**

| UI block | Behaviour |
|----------|-----------|
| **Audio intake** | File input; `setUploadedFile` from `AppPreferences` |
| **Spectrogram preview** | `useSpectrogram(uploadedFile, canvas)` + STFT drawing; **3-D sound map** link when a file is chosen |
| **Search with this clip** | `Link` to `/query` — carries the in-memory upload |
| **Catalog search** | `Link` to `/query?source=dataset` |
| **Display names** | Segmented control: common / scientific names; persisted in `localStorage` |

---

## 6. Query workspace (`/query`)

**`src/pages/QueryPage.tsx`**

### 6.1 Vocabulary mode

Segmented control sets `vocabMode` (`common | scientific`). Sent with `searchDataset` for backend context. Persisted as `lets-solve-it:vocab`.

### 6.2 Catalog search

- Text input; Enter or button triggers `searchDataset` → `POST /api/search` (top_k 10).
- Six clickable **example chips** populate the search box (acoustic mnemonics and descriptions).
- Successful rows cached in `localStorage` under `lets-solve-it:result-cache`.
- Error banner + Retry button on failure.

### 6.3 Upload and CLAP outputs

- File input updates `uploadedFile`; clears prior hits.
- On upload, audio is **decoded in the browser** to get an `AudioBuffer` and total duration.

**Buttons:**
- **Classify audio** → `POST /api/classify-audio` (multipart).
- **Search similar** → `POST /api/search-by-audio` (multipart).

Both buttons use the **effective file** — either the original upload (if ≤ 10 s) or a WAV-encoded slice of the selected analysis window (if > 10 s). The encoding happens entirely in the browser via `audioUtils.ts`.

### 6.4 Analysis window picker

Shown only when the uploaded file is **longer than 10 seconds**.

**`src/components/AudioWindowPicker.tsx`**

| Element | Behaviour |
|---------|-----------|
| **Time display** | `0:08 – 0:18 / 1:23` — updates live as slider moves |
| **Range slider** | Positions the 10-second analysis window anywhere in the recording; step 0.5 s |
| **Window highlight** | Coloured overlay on the track showing the selected 10-second region |
| **▶ Preview 10 s** | Creates an object URL for the original file; sets `audio.currentTime = windowStart`; auto-stops at `windowStart + 10 s` |

When **Classify** or **Search similar** is clicked with a window active:
1. `sliceAudioBuffer(buffer, windowStart, 10)` → new `AudioBuffer`
2. `audioBufferToWavFile(buffer, name)` → `File` object (32-bit float PCM WAV, encoded in-browser)
3. That `File` is sent to the backend as if it were the original upload

The spectrogram re-renders to show the selected 10-second window whenever the slider moves.

**Utility functions — `src/lib/audioUtils.ts`:**
- `decodeAudioFile(file)` → `AudioBuffer` (Web Audio API)
- `sliceAudioBuffer(buffer, startS, durationS)` → `AudioBuffer` (mono, no copy of the original)
- `audioBufferToWavFile(buffer, name)` → `File` (RIFF WAV, IEEE float32, mono)
- `fmtSeconds(s)` → `"m:ss"` string

### 6.5 Spectrogram preview

When `audioDuration > 10`:
- The `useSpectrogram` hook is bypassed.
- A `useEffect` calls `drawSpectrogramFromAudioBuffer(specCanvas, slicedBuffer)` whenever `windowStart` changes.

When `audioDuration ≤ 10`:
- `useSpectrogram(uploadedFile, specCanvas)` draws the full file.

### 6.6 "What CLAP heard" panel

After a successful **similarity search**, the result cards' `speciesDescription` fields are used to populate a panel below the classification results.

- Derived directly from the result set — no separate API call, guaranteed to match the displayed species.
- Displays up to 4 unique species with their name in a monospace label above the acoustic description.
- Cleared when a new file is chosen.

### 6.7 Classification results

`ClassificationHit[]` — species name, optional scientific name, similarity score as %.

### 6.8 Result grid

`ResultCard` components. Empty state links to seeded catalog query.

---

## 7. Result cards

**`src/components/ResultCard.tsx`**

| Element | Description |
|---------|-------------|
| **Header** | Species initials avatar (coloured by species), common + scientific name, vocalization type |
| **Metadata row** | Duration, recording ID, similarity % |
| **Spectrogram** | Fetched from `result.audioUrl` via `drawSpectrogramFromUrl`; falls back to `drawMockSpectrogram` for catalog results with no audio on disk |
| **Audio player** | Native `<audio controls>` with `result.audioUrl`; full recording (not a 10-second clip) |
| **Vocalization notes** | Collapsible `<details>` block with `speciesDescription` |
| **3-D map** | Link to `/viz/{result.id}` |
| **Compare** | Fills a compare slot; navigates to `/compare` when both slots are filled |
| **Save** | Toggles `localStorage` saved state |

---

## 8. Spectrogram pipeline

**`src/lib/spectrogramCanvas.ts`**

| Function | Use |
|----------|-----|
| `drawSpectrogramFromAudioBuffer(canvas, buffer)` | Core STFT renderer — exported for direct use |
| `drawSpectrogramFromFile(canvas, file)` | Decodes file → calls buffer renderer |
| `drawSpectrogramFromUrl(canvas, url, signal?)` | Fetches URL → decodes → calls buffer renderer |
| `drawMockSpectrogram(canvas, seed)` | Deterministic fake for cards without audio |

**`src/hooks/useSpectrogram.ts`** — wraps `drawSpectrogramFromFile` in a React effect; exposes `loading` and `error`.

---

## 9. 3-D audio visualisation (`/viz/:id`)

> Full deep-dive: [`VISUALIZATION.md`](./VISUALIZATION.md)

**`src/pages/VizPage.tsx`**, **`src/components/BirdSoundEmbeddingViz.tsx`**, **`src/lib/audioDrivenPointCloud.ts`**

### Audio source

- **`/viz/upload`** — uses `uploadedFile` from `AppPreferences`.
- **`/viz/:id` (catalog)** — fetches `result.audioUrl` as a `Blob` → `File`; `audioFetchState` drives the source label:
  - `· fetching audio…` — request in flight
  - `· audio unavailable` — fetch failed
  - `· no audio on disk` — `audioUrl` is null (gallery entry without a matching file)
  - File name once done

### Visualization pipeline

Each ~60 fps audio frame → square data point in 3-D space. Frequency-coloured border ring + white center dot. As audio plays, the chirp chain lights up sequentially. Silence stays dim. Post-processing: UnrealBloomPass (ghost borders don't bloom; lit chain does).

Point coordinates:
- **X** = progress within the current chirp segment (calls overlay the same corridor)
- **Y** ≈ dominant frequency (2–8 kHz)
- **Z** = amplitude + oscillations of within-call phase

---

## 10. Saved and compare

**Saved (`src/pages/SavedPage.tsx`):** `localStorage` key `lets-solve-it:saved` — JSON array of `SearchResult`. Toggled via `ResultCard`.

**Compare (`src/pages/ComparePage.tsx`):** Two `[string | null, string | null]` slots storing result IDs. Resolved via memory → `lets-solve-it:result-cache` → saved list.

---

## 11. Type definitions

**`src/api/types.ts`**

```typescript
interface SearchResult {
  id: string;
  title: string;
  species: string;
  scientificName: string | null;
  commonName: string;
  vocalizationType: string | null;
  duration: string | null;
  recordingId: string;
  speciesCode: string | null;
  score: number | null;
  audioUrl: string | null;
  imageUrl: string | null;
  speciesDescription?: string;   // acoustic profile for "What CLAP heard" and Vocalization notes
}

interface ClassificationHit {
  label: string;
  scientificName: string | null;
  score: number;
}
```

---

## 12. Styling

**`src/index.css`** — single global stylesheet with CSS custom properties.

Design tokens: `--bg`, `--surface`, `--surface2`, `--border`, `--accent`, `--accent-line`, `--danger`, `--radius`, `--radius-panel`, `--font-sans`, `--font-display`, `--font-mono`.

Key component classes:
- `.panel`, `.page-header`, `.btn`, `.btn--primary`, `.btn--outline`, `.btn--small`
- `.result-card`, `.spectrogram-preview`, `.window-picker`
- `.model-heard`, `.model-heard__species`, `.model-heard__desc`
- `.query-examples`, `.query-chip`
- `.classify-hits`, `.file-input`, `.file-name`
- `.viz-sound-*`, `.embedding-viz*` (viz chrome)

### Window picker classes

| Class | Purpose |
|-------|---------|
| `.window-picker` | Container with accent left-border |
| `.window-picker__header` | Label + time display row |
| `.window-picker__track` | Relative container for rail + highlight + slider |
| `.window-picker__rail` | Full-width dim background bar |
| `.window-picker__window` | Coloured highlight showing the selected 10 s region |
| `.window-picker__slider` | Range input with styled thumb only (transparent track) |
| `.window-picker__actions` | Hint text + preview button row |

---

## 13. Known limitations

| Area | Limitation |
|------|------------|
| Text search | Quality bounded by `species_descriptions.json` richness; mnemonics work only if the description mentions them |
| Gallery coverage | Species with no audio files on disk never appear in results |
| Pagination | UI renders only the first `top_k` batch |
| Compare slots | Reset on page refresh; resolved via `result-cache` + saved |
| Viz | Synthetic fallback when catalog audio is unavailable; frequency estimate is heuristic |
| Window picker | Only shown for uploads > 10 s; no automatic noise detection |

---

## 14. File-to-feature quick index

| If you care about… | Open… |
|---------------------|--------|
| Routes / providers | `App.tsx` |
| Nav + footer | `layout/AppShell.tsx` |
| Backend API client | `api/backend.ts` |
| Result-cache persistence | `api/resultCachePersistence.ts` |
| Friendly error copy | `lib/demoErrors.ts` |
| Shared DTOs | `api/types.ts` |
| Upload + vocab persistence | `context/AppPreferences.tsx` |
| Saved list | `context/SavedContext.tsx`, `saved/savedStore.ts` |
| Audio decode / slice / WAV encode | `lib/audioUtils.ts` |
| Spectrogram drawing | `lib/spectrogramCanvas.ts` |
| Analysis window picker | `components/AudioWindowPicker.tsx` |
| Result card | `components/ResultCard.tsx` |
| Frame features + chirp chains | `lib/audioDrivenPointCloud.ts` |
| 3D + bloom + narrative | `components/BirdSoundEmbeddingViz.tsx` |
| Global styles | `index.css` |

---

## 15. Future direction

- **VAD auto-window** — compute per-frame RMS on upload, auto-set the window picker to the highest-energy 10-second region so the model always sees the cleanest bird call.
- **MFCC + PCA visualisation** — replace the oscillator-driven 3-D layout with a fully data-grounded approach: 40 MFCCs per frame → PCA → 3-D positions reflect timbral similarity.
- **Pagination** — "Load more" button for catalog search results.
