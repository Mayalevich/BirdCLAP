# Web application — feature reference

This document describes **every user-visible feature** and the **code that implements it**, as of the current repository. Runtime search/classify/similarity use **`src/api/backend.ts`** against **`VITE_API_BASE_URL`**; **`src/api/mock.ts`** remains an offline reference catalog.

---

## 1. Product scope (what the web app is)

The web client is a **demo-first workspace** for bird-audio exploration with a backend API plus the **3-D spatiotemporal sound visualization** in-browser:

- Browse/search a **species catalog via the backend** (`POST /api/search`).
- **Upload** an audio file (decoded with Web Audio API for spectrograms and viz).
- Run **classification** and **similarity** against the backend (`/api/classify-audio`, `/api/search-by-audio`).
- **Save** rows to `localStorage`, **compare** two slots (persisted catalog cache aids lookup after reload), open **Visualization** (`/viz/...`).
- **3-D viz:** Every bird verse (detected chirp segment) becomes a network of square data points in 3-D space — an ever-changing generative sculpture. As a chirp plays its full chain of frames accumulates simultaneously, forming a spatial structure you can orbit and read like a score.

Nothing here trains models or calls Xeno-canto; it is a **UI and pipeline sketch** aligned with the team's longer-term audio-text retrieval direction.

### Reference project context

The visualization model is inspired by the original **"Visualizing Bird Songs"** project — a data visualization work where bird song recordings are analyzed frame-by-frame and translated into 3-D networks. The spatial distribution closely mirrors oscillatory patterns found in the spectrogram: amplitude, frequency, and temporal characteristics are all encoded in the geometry.

> *"Every verse becomes a 3D network of points, forming ever-changing generative sculptures. It's possible to see through the birds singing and read it in terms of frequency as well as any evolution happening in the sound tissue — like some sort of spatial visual score."*

The original was created in TouchDesigner using audio-driven oscillators. Our implementation approximates this with Web Audio API frame analysis + Three.js WebGL rendering, entirely in-browser.

A more advanced version of the original project uses **40 Mel-Frequency Cepstral Coefficients (MFCCs)** per frame, reduced to 3 dimensions via **Principal Component Analysis (PCA)** — a "Spatiotemporal Acoustic Manifold" where vocal timbres unfold as trajectories in a reproducible, data-rich representation space. See §16 for a roadmap note on this approach.
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
| `/query` | `QueryPage` | Catalog search, upload, classify/similar via API, results grid |
| `/saved` | `SavedPage` | Grid of saved `SearchResult` rows |
| `/compare` | `ComparePage` | Two-slot side-by-side comparison |
| `/viz/:id` | `VizPage` | Visualization: **`/viz/upload`** = user clip (`id`=`upload`). Other ids = catalog row + optional upload in context (`getResultById`) |
| `*` | `Navigate` → `/` | Unknown paths bounce home |

**Param:** `id` — **`upload`** needs an in-memory file (`AppPreferences`). Other IDs resolve via **`getResultById`** (in-memory cache + **`localStorage`** result cache **`lets-solve-it:result-cache`** + **saved list**). If metadata is cold, `/viz/:id` still renders a synthetic timeline with an on-page hint; upload audio overrides when present.

---

## 4. Shell, navigation, and footer

**File:** `src/layout/AppShell.tsx`

- **Brand block:** Title “Bird audio analysis” + subtitle · compact **API** badge (**No API**, **API…**, **API OK**, **API down**) from a lightweight `POST /api/search` probe when `VITE_API_BASE_URL` is set.
- **Banners:** Yellow warning when the API URL is missing; red alert when the probe fails (likely down / CORS).
- **Skip link:** keyboard “Skip to content” jumps to **`#main-content`**.
- **Navigation:** `NavLink`s — Overview (`/`), Query, Saved, Compare. Compare shows a badge `2` when both compare slots are filled (`useAppPreferences().compareSlots`).
- **Main:** `<Outlet />` renders the active page (`id="main-content"`).
- **Footer:** Short note on the viz metaphor.

Styling for header/nav/footer lives in `src/index.css` (classes prefixed `app-shell__`).

---

## 5. Feature: Overview (Home)

**File:** `src/pages/HomePage.tsx`

| UI block | Behaviour |
|----------|-----------|
| **Instrument strip** | Banner reflects whether **`VITE_API_BASE_URL`** is configured. |
| **Page header** | “Overview” + paragraph describing the workspace. |
| **Audio intake** | File input (`accept` audio). On change, calls `setUploadedFile` from `AppPreferences`. |
| **Spectrogram preview** | `useSpectrogram(uploadedFile, canvas)` + `drawSpectrogramFromFile` pipeline (see §10); **`Visualization`** link to `/viz/upload` beside the spectrogram when a file is chosen. |
| **Continue to Query** | `Link` to `/query` carrying the in-memory upload (same session). |
| **Catalog search** | Link to `/query?source=dataset` — Query page reads this to seed the search field (see §6). |
| **Taxonomic display** | Segmented control toggles `vocabMode` (`common` \| `scientific`) persisted in `localStorage` (`AppPreferences`). |

**Hook:** `src/hooks/useSpectrogram.ts` — redraws when `file` or `canvas` changes; decode failures show as a **`panel-alert`** on Home (and Query).

---

## 6. Feature: Query workspace

**File:** `src/pages/QueryPage.tsx`  
**API:** `src/api/backend.ts` (needs **`web/.env`** → **`VITE_API_BASE_URL`**)  
**Types:** `src/api/types.ts`

### 6.1 Vocabulary mode

- Same segmented control as Home; `vocabMode` is sent with **`searchDataset`** for backend/context (exact semantics depend on the server).
- Persisted key: `lets-solve-it:vocab` (see `AppPreferences.tsx`).

### 6.2 Catalog (dataset) search

- Input: text search; Enter or button triggers **`searchDataset`** → **`POST {base}/api/search`** (`top_k` 10 server-side unless changed).
- Successful rows are **cached** into `localStorage` under **`lets-solve-it:result-cache`** via `cacheResults()` for Compare/Viz id resolution across reloads.
- **Errored requests** show **`formatUserFacingDemoError`** copy plus **Retry search**; **rapid double-submit** guarded with in-flight refs.
- Results rendered as **`ResultCard`** grid.

**URL hint:** `?source=dataset` in `useEffect` seeds `query` to “Turdus” or “sparrow” depending on vocab mode.

### 6.3 Upload and classification

- File input updates `uploadedFile` and clears prior `classifyHits`.
- **Spectrogram row:** canvas (`useSpectrogram`) shows **Decoding spectrogram…** while busy; **`Visualization`** (`/viz/upload`) when a file exists.
- **Classify:** **`classifyUpload`** → multipart **`POST /api/classify-audio`** (`top_k` 5).
- Hits are **normalized**: invalid entries dropped; empty list yields a readable panel message.

### 6.4 Similarity search

- **`searchSimilarToUpload`** → **`POST /api/search-by-audio`** (multipart clip + top_k).

### 6.5 Result set

- **`ResultCard`** exposes **Visualize** → `/viz/{id}`, **Compare slot** (calls **`rememberResult`**), **Save**.
- Empty state hints link to seeded dataset query.

---

## 7. Feature: Saved specimens

**File:** `src/pages/SavedPage.tsx`  
**State:** `src/context/SavedContext.tsx`  
**Persistence:** `src/saved/savedStore.ts`

- Reads `saved` array from context (derived from `loadSaved()`).
- **Storage key:** `lets-solve-it:saved` — JSON array of `SearchResult` objects.
- **Toggle:** implemented on `ResultCard` via `toggle(result)`; dedupes by `result.id` (also calls **`rememberResult`** so ids stay in **`lets-solve-it:result-cache`**).
- Empty state: directs user to **Query** with next-step copy.

---

## 8. Feature: Paired comparison

**Files:** `src/pages/ComparePage.tsx`, preferences in `AppPreferences.tsx`

- **Slots:** `[string | null, string | null]` storing **result ids** (`SearchResult.id`).
- **Add:** `ResultCard` calls **`rememberResult(result)`** then `addToCompare(result.id)`:
  - Fills slot A, then B; duplicates ignored; if both full, replaces slot A with new id (`CompareAddResult` enum).
  - When second slot is filled (`added_second`), card navigates to `/compare`.
- **Resolve rows:** **`getResultById`** merges **memory**, **`lets-solve-it:result-cache`**, then **saved** specimens.
- **Clear:** `clearCompare()` resets both slots.
- Missing metadata (stale id) shows actionable copy + link back to **Query**.

---

## 9. Feature: Result cards

**File:** `src/components/ResultCard.tsx`  
**Drawing:** `src/lib/spectrogramCanvas.ts`

| Element | Description |
|---------|-------------|
| **Image** | `imageUrl` from backend or placeholder service. |
| **Metadata** | Common + scientific name, vocalization, duration, recording id, optional similarity %. |
| **Spectrogram canvas** | If `spectrogramFile` prop set, draws from file; else `drawMockSpectrogram(canvas, result.id)` deterministic fake pattern. |
| **Visualize** | Links to `/viz/{result.id}`. |
| **Compare slot** | `rememberResult` + `addToCompare` (see §8). |
| **Save** | `rememberResult` + `toggle` saved state; primary style when saved. |

---

## 10. Feature: Spectrogram preview (shared)

**Files:** `src/hooks/useSpectrogram.ts`, `src/lib/spectrogramCanvas.ts`

- **Real file path:** decode audio with Web Audio (`AudioContext.decodeAudioData`), downmix to mono, compute magnitude STFT-like bins, draw greyscale canvas.
- **Mock path:** hash `result.id` into a stable fake pattern for cards without upload.
- **Errors:** Hook sets error string on failure; **Home** + **Query** show **`panel-alert`** on decode failure.

Used on **Home** and **Query** for the upload preview. When a file is chosen, **`Visualization`** (link beside the spectrogram) goes to **`/viz/upload`**.

---

## 11. Feature: 3D audio visualization

> **Full deep-dive:** [`VISUALIZATION.md`](./VISUALIZATION.md) — plain-language walkthrough of every concept, the data pipeline, what each label means, and the MFCC+PCA roadmap.

**Route:** `/viz/:id` (primary entry: **`/viz/upload`** with an uploaded clip)  
**Files:** `src/pages/VizPage.tsx`, `src/components/BirdSoundEmbeddingViz.tsx`, `src/lib/audioDrivenPointCloud.ts`

### 11.1 Page chrome

- Breadcrumb back to Query.
- Title **Spatiotemporal Sound Visualization**; header explains frame-wise analysis at ~60 Hz and playback-driven highlighting.
- **Stack:** Dark "stage" panel with stats line + WebGL canvas; below, species **footer card** (image + names + recording meta).
- **Data guide:** Frequency color bar (violet 2 kHz → red 8 kHz) + structured legend table explaining Color/Frequency, Signal Amplitude, Emission Time, Lifetime, and Red Border indicators.

### 11.2 Lazy loading

`BirdSoundEmbeddingViz` is `React.lazy` + `Suspense` so Three.js loads only when visiting viz.

### 11.3 Audio → points pipeline (`audioDrivenPointCloud.ts`)

- **`buildAudioDrivenPoints(seed, file?)`:** If `file` provided, decode to mono PCM, else **`makeSyntheticAudio(seed)`** (short separated chirp-like bursts for demo). **`syntheticAudioToWavBlob(seed)`** exposes the same synthetic signal as a WAV for `<audio>` playback in the viz.
- **Framing:** `frameAudioData` — sliding Hann-windowed chunks at **TARGET_FPS 60**, per-frame **amplitude** (RMS) and **dominant frequency** via coarse bank of complex sinusoids (not a full FFT library). Spans the **entire decoded buffer** so viz time matches `<audio>.currentTime` for long clips.
- **`extractChirpChains(amplitudes, fps, opts?)`:** Segments "chirps" from the amplitude envelope. Optional **`ChirpChainOptions`** tweak gap / minimum length / **`maxNodes`** (viz uses tighter gaps than defaults for finer call boundaries).
- **`freqToColor`:** Classic **spectral / rainbow ramp** — violet at 2 kHz through blue, cyan, green, yellow, orange to red at 8 kHz, matching standard spectrogram color convention. **`enrichChirpRgb`** pushes the result away from grey for richer hues under additive blending.
- **Exports** also include `computeHighlightFrameIndices` and `buildTemporalEdges` (legacy / unused by current viz).

Each point represents:

- **`emissionTime`**: Time in seconds when this audio frame occurs (from 0 to audio duration)
- **`amplitude`**: Signal strength (0–1) at this time
- **`freqHz`**: Dominant frequency (2000–8000 Hz) detected at this time

### 11.4 Three.js scene — Square chirp-chain visualization (`BirdSoundEmbeddingViz.tsx`)

**Core concept:** Every ~60 fps audio frame becomes a **square data point** in 3D space. The scene is dark and the ghost cloud is nearly invisible. As the bird **chirps**, the corresponding chain of squares **lights up sequentially in sync with the audio** — the dominant visual effect. Silence stays dark.

**Point rendering — custom GLSL ShaderMaterial:**

Each point is rendered via a custom `ShaderMaterial` using `gl_PointCoord` in the fragment shader to produce a **square with a colored border ring and a white center dot**:

| State | Border | Center dot | Fill |
|-------|--------|-----------|------|
| **Ghost** (not playing) | Frequency color × ~0.05, α ≈ 0.30 | Tiny dim dot | Discarded (transparent) |
| **Lit / decay** | Frequency color × (1.9 + litWeight × 2.1), α 0.93 | White, α 0.88 | Dim colored fill |
| **Active playhead** | **Red** `rgb(255, 18, 8)`, α 1.0 | White, α 1.0 | Dim colored fill |

`aLitness` float attribute (per-point, updated each frame) drives shader state: `0` = ghost, `0–1` = decay weight, `2` = active playhead.

**Spatial positioning (call-centered):**

- Frames **inside** a chirp segment: **X** = progress through that call only (0 → 1). Repeated calls overlay the same corridor — they do not march left-to-right with global clip time.
- **Y** ≈ frequency band (2–8 kHz mapped to the Y range).
- **Z** mixes amplitude with multi-frequency oscillations of the within-call phase, creating stable ribbon shapes per chirp gesture.
- Frames **outside** segments (silence): **X** is a compact sine/cosine wiggle of global clip time so ghost points don't scan sideways across the stage.

**Chirp chain lighting — the main visual effect:**

- Reads **`HTMLAudioElement.currentTime`**, clamped to the analyzed timeline (no modulo wrapping).
- Picks the single analysis frame closest on the ~60 Hz grid (`floor`-biased ±2 frame search).
- Lights when that frame clears an **adaptive noise floor** (≈15th percentile of clip RMS) so softer repeat chirps still register.
- **Full verse accumulation:** when the playhead enters a chirp segment, every frame from the **chain start to the current playhead** is lit simultaneously. Frames at the start of the verse are dimmer (weight 0.20); frames just behind the playhead are near full brightness (weight ~1.0). The full verse network builds in 3-D space as the bird sings.
- **Verse fade-out (~0.9 s):** after the chirp ends, all lit intensities decay at a constant rate via a persistent `litIntensities` Float32Array. The sculpture lingers briefly and then fades before the next verse begins.
- **Gap / silence frames:** if the playhead is between chirp segments, only the immediate 6-frame window is lit (short dim beacon, no chain accumulation).
- The brightest square (intensity 1.0, red border) is always the **current playhead**; all other squares in the verse inherit their brightness from their position within the chain.

**Visual layers (draw order):**

1. **Skeleton lines** — static `LineSegments` connecting consecutive frames within every chirp chain; very dim (color × 0.03), always present; subtly reveal cloud structure between chirps.
2. **Point cloud** — `THREE.Points` with `ShaderMaterial`; additive blending; ghost squares nearly invisible, lit chain pops in color.
3. **Dynamic lit lines** — `LineSegments` rebuilt each frame connecting only the frames in the current decay window; bright additive colors matching each frame's frequency.
4. **CSS2D labels** — pool of 8 `CSS2DObject` elements shown only on lit/active frames; each displays **amplitude** (0.xxxx), **lifetime** (animated seconds since emission), and **emission time** (seconds from clip start). Border color matches the point's frequency color via CSS `--lc` custom property.

**Post-processing:** `EffectComposer` + `UnrealBloomPass` (bloom threshold 0.22 — ghost borders stay below threshold and don't bloom; lit chain and playhead bloom strongly) + `ACESFilmicToneMapping` on the renderer.

**Stats line:** Shows frame count, analyzed duration, and number of detected chirp chains.

### 11.5 Viz + upload coupling and audio playback

`VizPage` passes `audioFile={uploadedFile}` from preferences (`null` clears per global upload state).

**With uploaded audio:**

- Creates an `HTMLAudioElement` with `loop: true` and `URL.createObjectURL(file)` (revoked on teardown).
- Attempts autoplay; if the browser blocks it, the first click on the stage starts playback.
- Volume is 70%.
- The WebGL playhead always follows `audio.currentTime` (not wall clock).

**Catalog route (`/viz/:id`):**

- `seed = "${id}:${recordingId}"` when `getResultById` succeeds.
- Seeds `catalog:${id}` when catalog metadata is **cold**, so visualization still renders; an inline advisory tells the presenter how to reload metadata (**Query**/result-card **Visualize**).

**Fullscreen:** stage panel **`requestFullscreen`** + CSS `::fullscreen` tweaks (toolbar stays visible).

**Without uploaded audio (any route):**

- Builds the synthetic PCM backing that `seed`, wraps it as **16-bit mono WAV** (`syntheticAudioToWavBlob` in `audioDrivenPointCloud.ts`).
- Object URLs revoked on teardown.

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
- Components: `.panel`, `.page-header`, `.result-card`, `.viz-sound-*`, `.embedding-viz*`, `.evl-*`, `.vdgl-*`, etc.
- **Viz** top line uses dark chrome; labels use mono stack variables.
- **Viz label classes:** `.evl-amp` (amplitude), `.evl-life` (lifetime), `.evl-time` (emission time), `.evl-static` / `.evl-dyn` / `.evl-active` modifiers. Label border color driven by `--lc` CSS custom property set per-point to match frequency.
- **Frequency bar:** `.viz-data-guide__bar` spans use inline background colors matching the spectral rainbow ramp (violet → red).

---

## 14. Known limitations (honest list for planning)

| Area | Limitation |
|------|------------|
| Search | Backend-defined ranking/pagination — UI renders first response batch only (`top_k` on client requests). |
| Classify / similarity | Quality fully depends on deployed models and API wiring. |
| Saved + cache | Full JSON blobs in **`localStorage`** — size/privacy/host-specific clearing. |
| Compare | Slots are **React state until refresh**; rows resolve via **`result-cache`** + **saved**. |
| Viz | Synthetic fallback when catalog metadata is cold; frequency estimate is heuristic. |
| A11y | Viz is WebGL-heavy; baseline skip-link + alerts; dense audit optional. |

---

## 15. File-to-feature quick index

| If you care about… | Open… |
|---------------------|--------|
| Routes / providers | `App.tsx` |
| Nav + footer | `layout/AppShell.tsx` |
| Live demo API client | `api/backend.ts` |
| Result-cache persistence | `api/resultCachePersistence.ts` (`lets-solve-it:result-cache`) |
| Friendly error copy | `lib/demoErrors.ts` |
| Offline mock catalog fixture | `api/mock.ts` |
| Shared DTOs | `api/types.ts` |
| Upload + vocab persistence | `context/AppPreferences.tsx` |
| Starred list | `context/SavedContext.tsx`, `saved/savedStore.ts` |
| Spectrogram drawing | `lib/spectrogramCanvas.ts` |
| Frame features + chirp chains | `lib/audioDrivenPointCloud.ts` |
| 3D + bloom + narrative | `components/BirdSoundEmbeddingViz.tsx` |
| Global look | `index.css`, `index.html` (fonts) |

---

When you add a feature, append a subsection here (or add a linked doc) so the next developer does not reverse-engineer the UI.

---

## 16. Future direction: MFCC + PCA spatiotemporal manifold

The reference project's more advanced visualization replaces the oscillator-driven layout with a fully data-grounded approach:

1. **Feature extraction:** Compute **40 Mel-Frequency Cepstral Coefficients (MFCCs)** for every ~60 fps frame using a proper mel filterbank over the decoded PCM.
2. **Dimensionality reduction:** Apply **Principal Component Analysis (PCA)** across all frame feature vectors to find the 3 principal axes of variance in the timbral space.
3. **3-D layout:** Each frame is placed at its 3-D PCA coordinate. Distances between points reflect **timbral similarity** (frames with similar spectral envelopes cluster together). Trajectories through the cloud reveal temporal evolution of the sound.
4. **Additional metadata:** Each point still carries `amplitude`, `emissionTime`, and `freqHz` for color and label display.

**Why this matters:** The PCA-based layout is reproducible and data-rich — similar bird species will produce similar manifold shapes regardless of when the analysis runs. The oscillator model (current implementation) produces aesthetically compelling forms that closely resemble the MFCC manifold through tuned parameters, but varies with implementation choices.

**Implementation path:**
- Replace `analyzeFrame` in `audioDrivenPointCloud.ts` with a mel filterbank + DCT producing 40 MFCC coefficients per frame.
- Collect all frame coefficient vectors into a matrix; compute covariance; extract top 3 eigenvectors (power iteration or a small linear algebra library).
- Project all frames onto the 3 eigenvectors to get `[x, y, z]` positions.
- The rest of the `BirdSoundEmbeddingViz.tsx` pipeline (shader, lighting, labels) remains unchanged.
