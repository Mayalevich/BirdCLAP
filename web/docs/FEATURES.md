# Web application — feature reference

This document describes **every user-visible feature** and the **code that implements it**, as of the current repository. It is aimed at onboarding engineers who will extend or replace the mock layer with real services.

---

## 1. Product scope (what the web app is)

The web client is a **research-style prototype workspace** for exploring bird-audio workflows **without a server**:

- Browse a **fixed mock catalog** of species rows (text search).
- **Upload** an audio file in the browser (decoded with Web Audio API).
- Run **mock** “classification” and “similarity” that ignore audio content but simulate latency.
- **Save** rows to `localStorage`, **compare** two slots, and open a **3D visualization** of uploaded audio (frame-level analysis; synthetic only when no upload).

Nothing here trains models or calls Xeno-canto; it is a **UI and pipeline sketch** aligned with the team’s longer-term audio–text retrieval direction.

---

## 2. Technology stack

| Layer | Choice |
|-------|--------|
| Build | Vite 6 |
| UI | React 18 + TypeScript |
| Routing | React Router 6 |
| 3D | Three.js (WebGLRenderer, OrbitControls, CSS2D labels, EffectComposer + UnrealBloomPass) |
| Styling | Single global `index.css` (design tokens as CSS variables) |

---

## 3. Application entry and routing

**File:** `src/main.tsx`  
Mounts React root.

**File:** `src/App.tsx`  
Wraps the tree in:

- `BrowserRouter`
- `AppPreferencesProvider` — upload + vocab + compare slots (in-memory except vocab).
- `SavedProvider` — saved list (backed by `localStorage`).

**Routes** (all except `*` use `AppShell` layout):

| Path | Page component | Purpose |
|------|----------------|---------|
| `/` | `HomePage` | Workspace overview, upload entry, links into Query |
| `/query` | `QueryPage` | Catalog search, upload, mock classify/similarity, results grid |
| `/saved` | `SavedPage` | Grid of saved `SearchResult` rows |
| `/compare` | `ComparePage` | Two-slot side-by-side comparison |
| `/viz/:id` | `VizPage` | Visualization: **`/viz/upload`** = user clip (`id`=`upload`). Other ids = catalog row + optional upload in context (`getResultById`) |
| `*` | `Navigate` → `/` | Unknown paths bounce home |

**Param:** `id` from the URL — use **`upload`** with an in-memory uploaded file (`AppPreferences`). Other values resolve through `getResultById`; without an upload, audio can fall back to synthetic while still showing species context.

---

## 4. Shell, navigation, and footer

**File:** `src/layout/AppShell.tsx`

- **Brand block:** Title “Bird audio analysis” + subtitle noting client-side prototype / mock catalog.
- **Navigation:** `NavLink`s — Overview (`/`), Query, Saved, Compare. Compare shows a badge `2` when both compare slots are filled (`useAppPreferences().compareSlots`).
- **Main:** `<Outlet />` renders the active page.
- **Footer:** Disclaimer that embeddings and search are simulated in-browser.

Styling for header/nav/footer lives in `src/index.css` (classes prefixed `app-shell__`).

---

## 5. Feature: Overview (Home)

**File:** `src/pages/HomePage.tsx`

| UI block | Behaviour |
|----------|-----------|
| **Instrument strip** | Mono banner stating prototype / mock / no server. |
| **Page header** | “Overview” + paragraph describing the workspace. |
| **Audio intake** | File input (`accept` audio). On change, calls `setUploadedFile` from `AppPreferences`. |
| **Spectrogram preview** | `useSpectrogram(uploadedFile, canvas)` + `drawSpectrogramFromFile` pipeline (see §10); **`Visualization`** link to `/viz/upload` beside the spectrogram when a file is chosen. |
| **Continue to Query** | `Link` to `/query` carrying the in-memory upload (same session). |
| **Catalog search** | Link to `/query?source=dataset` — Query page reads this to seed the search field (see §6). |
| **Taxonomic display** | Segmented control toggles `vocabMode` (`common` \| `scientific`) persisted in `localStorage` (`AppPreferences`). |

**Hook:** `src/hooks/useSpectrogram.ts` — redraws when `file` or `canvas` changes; surfaces decode errors (not heavily surfaced in Home UI today).

---

## 6. Feature: Query workspace

**File:** `src/pages/QueryPage.tsx`  
**API:** `src/api/mock.ts`  
**Types:** `src/api/types.ts`

### 6.1 Vocabulary mode

- Same segmented control as Home: `vocabMode` affects **dataset search** string matching (common vs scientific name primary field).
- Persisted key: `lets-solve-it:vocab` (see `AppPreferences.tsx`).

### 6.2 Catalog (dataset) search

- Input: text search; Enter or button triggers `searchDataset(query, vocabMode)`.
- **Mock logic:** substring match on normalized common name, scientific name, or `speciesCode`. Empty query returns full catalog filter pass; if filter is empty, returns first 8 rows (see `mock.ts`).
- Artificial delay ~400 ms to mimic network.
- Results rendered as `ResultCard` grid.

**URL hint:** `?source=dataset` in `useEffect` seeds `query` to “Turdus” or “sparrow” depending on vocab mode.

### 6.3 Upload and mock classification

- File input updates `uploadedFile` and clears prior `classifyHits`.
- **Spectrogram row:** canvas (`useSpectrogram`) + **`Visualization`** link (`/viz/upload`) shown only when a file is loaded; aligned beside the spectrogram (`.spectrogram-preview-row`).
- **Classify button:** `classifyUpload(uploadedFile)` — **ignores file content**; returns fixed chickadee-heavy label list after ~600 ms.
- Renders ordered list “Posterior over labels (mock)” with scores.

### 6.4 Mock similarity search

- **Search similar** button: `searchSimilarToUpload(uploadedFile)` — ignores file; returns first 6 catalog rows with descending fake `similarity` after ~500 ms.

### 6.5 Result set

- Grid of `ResultCard` for current `results` state.
- Link to `/saved` for saved list.

---

## 7. Feature: Saved specimens

**File:** `src/pages/SavedPage.tsx`  
**State:** `src/context/SavedContext.tsx`  
**Persistence:** `src/saved/savedStore.ts`

- Reads `saved` array from context (derived from `loadSaved()`).
- **Storage key:** `lets-solve-it:saved` — JSON array of `SearchResult` objects.
- **Toggle:** implemented on `ResultCard` via `toggle(result)`; dedupes by `result.id`.
- Empty state: short message when no rows saved.

---

## 8. Feature: Paired comparison

**Files:** `src/pages/ComparePage.tsx`, preferences in `AppPreferences.tsx`

- **Slots:** `[string | null, string | null]` storing **result ids** (`SearchResult.id`).
- **Add:** `ResultCard` calls `addToCompare(result.id)`:
  - Fills slot A, then B; duplicates ignored; if both full, replaces slot A with new id (`CompareAddResult` enum).
  - When second slot is filled (`added_second`), card navigates to `/compare`.
- **Resolve rows:** `getResultById` from mock catalog.
- **Clear:** `clearCompare()` resets both slots.
- Empty slots show dashed placeholder copy.

---

## 9. Feature: Result cards

**File:** `src/components/ResultCard.tsx`  
**Drawing:** `src/lib/spectrogramCanvas.ts`

| Element | Description |
|---------|-------------|
| **Image** | `imageUrl` from mock (currently ui-avatars.com placeholder). |
| **Metadata** | Common + scientific name, vocalization, duration, recording id, optional similarity %. |
| **Spectrogram canvas** | If `spectrogramFile` prop set, draws from file; else `drawMockSpectrogram(canvas, result.id)` deterministic fake pattern. |
| **Compare slot** | `addToCompare` (see §8). |
| **Save** | `toggle` saved state; primary style when saved. |

---

## 10. Feature: Spectrogram preview (shared)

**Files:** `src/hooks/useSpectrogram.ts`, `src/lib/spectrogramCanvas.ts`

- **Real file path:** decode audio with Web Audio (`AudioContext.decodeAudioData`), downmix to mono, compute magnitude STFT-like bins, draw greyscale canvas.
- **Mock path:** hash `result.id` into a stable fake pattern for cards without upload.
- **Errors:** Hook sets error string on failure; not all pages surface it in UI (could be improved).

Used on **Home** and **Query** for the upload preview. When a file is chosen, **`Visualization`** (link beside the spectrogram) goes to **`/viz/upload`**.

---

## 11. Feature: 3D audio visualization

**Route:** `/viz/:id` (primary entry: **`/viz/upload`** with an uploaded clip)  
**Files:** `src/pages/VizPage.tsx`, `src/components/BirdSoundEmbeddingViz.tsx`, `src/lib/audioDrivenPointCloud.ts`

### 11.1 Page chrome

- Breadcrumb back to Query.
- Title **Visualization**; header explains frame-wise analysis at ~60 Hz and playback-driven highlighting.
- **Stack:** Dark “stage” panel with stats line + WebGL canvas; below, species **footer card** (image + names + recording meta).
- **Data guide:** Static frequency axis legend + short methodology blurb.

### 11.2 Lazy loading

`BirdSoundEmbeddingViz` is `React.lazy` + `Suspense` so Three.js loads only when visiting viz.

### 11.3 Audio → points pipeline (`audioDrivenPointCloud.ts`)

- **`buildAudioDrivenPoints(seed, file?)`:** If `file` provided, decode to mono PCM, else **`makeSyntheticAudio(seed)`** (short separated chirp-like bursts for demo). **`syntheticAudioToWavBlob(seed)`** exposes the same synthetic signal as a WAV for `<audio>` playback in the viz.
- **Framing:** `frameAudioData` — sliding Hann-windowed chunks at **TARGET_FPS 60**, per-frame **amplitude** (RMS) and **dominant frequency** via coarse bank of complex sinusoids (not a full FFT library). By default it spans the **entire decoded buffer** (older 12&nbsp;s cap was removed so long clips stay in sync with `<audio>.currentTime`).
- **`extractChirpChains(amplitudes, fps, opts?)`:** Segments “chirps” from amplitude envelope. Optional **`ChirpChainOptions`** tweak gap / minimum length / **`maxNodes`** (viz uses tighter gaps than the defaults for finer call boundaries).
- **`freqToColor` / `enrichChirpRgb`:** Jewel-tone ramp + saturation boost for lit points.
- **Exports** also include `computeHighlightFrameIndices` (legacy / unused by current viz narrative).

Each point represents:
- **`emissionTime`**: Time in seconds when this audio frame occurs (from 0 to audio duration)
- **`amplitude`**: Signal strength (0-1) at this time
- **`freqHz`**: Dominant frequency (2000-8000 Hz) detected at this time

### 11.4 Three.js scene — Frequency chirp visualization (`BirdSoundEmbeddingViz.tsx`)

**Concept:** Points sit in a **spectrogram-inspired volume** (time × band × modulation), matching the “60&nbsp;fps features in 3D” storyboard copy. Highlights are **not** a blunt global time-window over the clip: they march **inside energy segments** identified by **`extractChirpChains`** so silence stays dark and bursts line up with detected calls.

**Spatial positioning (call-centered, not slideshow time on X):**
- Frames **inside** `extractChirpChains` segments: **X** = progress **through that call** only (0 → 1 from segment start → end). **Repeated identical chirps stack in the same spatial corridor**; they do not march left‑to‑right with global clip timeline.
- **Y** ≈ band (~2–8&nbsp;kHz); **Z** mixes **amplitude** with oscillations driven by the same **within-call** phase so the ribbon shape is stable per gesture.
- Frames **outside** segments (quieter gaps): **X** stays in a **narrow wiggle** (`sin`/`cos` of clip time) so background ghosts do not scan sideways across the stage.

**Playhead lighting (matches “one band lights with the chirp”):**
- Reads **`HTMLAudioElement.currentTime`**, clamped to the analyzed timeline (**no modulo** against a shortened buffer—modulo combined with an old **12&nbsp;s analysis cap** made late chirps wrap to wrong times / miss entirely).
- Picks the **single analysis frame** closest on the ~60&nbsp;Hz grid (`floor`-biased search). **Color and intensity** = that frame’s **`freqHz` and RMS amplitude**.
- Lights whenever that frame clears an **adaptive noise floor** (≈15th percentile of clip RMS)—**not** chirp-envelope gated, so the 3rd / 4th softer repeat still gets a beacon if it’s above room noise.
- **~75&nbsp;ms decay** behind the playhead for afterglow.
- **Synthetic fallback** uses the same **WAV blob** path as uploads so the clock matches what you hear.

**Visual elements:**
- **Scene:** Dark background, fog, hidden grid for clean appearance
- **Points:** `BufferGeometry` + `PointsMaterial` with vertex colors and **additive blending**
- **Lines:** at most a **short temporal streak**—only **consecutive frame indices** `(i, i+1)` when both are in the tiny lit decay set (connects the playhead dot to its immediate past)
- **Labels:** top line **amplitude normalized 0–1** (clip peak); second line **emission time (s)** and **elapsed seconds since emission** (“life”)
- **Post-processing:** `EffectComposer` + `UnrealBloomPass` (stronger bloom) + `ACESFilmicToneMapping` on renderer
- **Stats line:** Shows frame count and detected chirp chains

**Color coding:**
- Color represents frequency band (low frequencies = warm colors, high = cool colors)
- Brightness scales with amplitude (louder = brighter)
- Ghost grey points (barely visible) when not currently playing

### 11.5 Viz + upload coupling and audio playback

`VizPage` passes `audioFile={uploadedFile}` from preferences.

**With uploaded audio:**
- Creates an `HTMLAudioElement` with `loop: true` and `URL.createObjectURL(file)` (revoked on teardown).
- Attempts autoplay; if the browser blocks it, the first click on the stage starts playback.
- Volume is 70%.
- The WebGL playhead always follows `audio.currentTime` (not wall clock).

**Without uploaded audio:**
- Builds the same synthetic PCM used for analysis, wraps it as **16-bit mono WAV** (`syntheticAudioToWavBlob` in `audioDrivenPointCloud.ts`), and plays it through the same audio path so the viz stays locked to what you hear.
- Object URLs are revoked on unmount / effect cleanup.

---

## 12. Type definitions (contract for UI and future API)

**File:** `src/api/types.ts`

- `VocabMode` — used throughout search UI.
- `QuerySource` — exported for future API/source toggles; **not referenced** by page components today.
- `ClassificationHit`, `SearchResult` — shape of mock rows and classifier lines.

Any backend should aim to preserve these fields or provide adapters before the UI is rewritten.

---

## 13. Styling and design language

**File:** `src/index.css`

- CSS variables for **paper-like neutrals**, **steel blue accent**, IBM Plex font stacks.
- Components: `.panel`, `.page-header`, `.result-card`, `.viz-sound-*`, `.embedding-viz*`, etc.
- **Viz** top line uses dark chrome; labels use mono stack variables.

---

## 14. Known limitations (honest list for planning)

| Area | Limitation |
|------|------------|
| Search | Substring only on static array; no pagination/facets. |
| Classify / similarity | No model; fixed outputs independent of audio. |
| Saved | Full JSON of rows in `localStorage` — size and privacy considerations. |
| Compare | In-memory only until refresh; not deep-linked. |
| Viz | Simplified physics of frequency estimation; bloom + additive art direction, not calibrated science viz. |
| A11y | Viz is WebGL-heavy; labels have some ARIA on container but deep accessibility not audited. |

---

## 15. File-to-feature quick index

| If you care about… | Open… |
|---------------------|--------|
| Routes / providers | `App.tsx` |
| Nav + footer | `layout/AppShell.tsx` |
| Mock HTTP-shaped API | `api/mock.ts` |
| Shared DTOs | `api/types.ts` |
| Upload + vocab persistence | `context/AppPreferences.tsx` |
| Starred list | `context/SavedContext.tsx`, `saved/savedStore.ts` |
| Spectrogram drawing | `lib/spectrogramCanvas.ts` |
| Frame features + chirp chains | `lib/audioDrivenPointCloud.ts` |
| 3D + bloom + narrative | `components/BirdSoundEmbeddingViz.tsx` |
| Global look | `index.css`, `index.html` (fonts) |

---

When you add a feature, append a subsection here (or add a linked doc) so the next developer does not reverse-engineer the UI.
