# 3-D Sound Visualization — how it works

**Route:** `/viz/upload` (uploaded clip) or `/viz/:id` (catalog recording)  
**Key files:** `src/pages/VizPage.tsx` · `src/components/BirdSoundEmbeddingViz.tsx` · `src/lib/audioDrivenPointCloud.ts`

---

## The core idea in one sentence

Every fraction of a second of bird audio becomes a glowing square in 3-D space, and as the bird sings a phrase, those squares light up one after another to form a floating network — a spatial version of the song you can rotate and read.

---

## What you are looking at

When you open the visualization and press play, you see a dark 3-D stage. Almost nothing is visible at first.

Then the bird sings.

As each phrase (chirp, trill, call) plays, a chain of colored squares assembles in mid-air. The chain grows square by square, in sync with the audio, until the entire phrase is hanging in 3-D space as a connected network. Lines link consecutive squares. Labels float beside the brightest ones, showing numbers. When the phrase ends the network slowly fades — then the next one forms somewhere else in the cloud.

You can orbit around the whole structure with your mouse while it plays.

---

## Where each square comes from

The audio file is decoded into raw samples and then divided into **~60 frames per second**. Each frame is a tiny slice of the recording — about 4 milliseconds of sound.

For each frame, two things are measured:

| Measurement | How | What it means |
|-------------|-----|---------------|
| **Amplitude** | Root-mean-square of the waveform samples | How loud the audio is at that instant (0–1) |
| **Dominant frequency** | Coarse DFT scan across 56 frequency targets in 2–8 kHz | Which pitch/band the audio is loudest in at that instant |

Those two numbers, combined with the frame's position in time, determine everything about how the square is placed and colored.

---

## How a square is positioned in 3-D space

Each frame's `(x, y, z)` position is calculated from its data:

```
Y  ≈  frequency band
       low frequency (2 kHz) → bottom
       high frequency (8 kHz) → top

Z  ≈  amplitude + oscillation
       loud frames push outward in Z
       a multi-frequency wobble shapes each phrase into a ribbon

X  ≈  progress through the current chirp
       0 = start of this phrase
       1 = end of this phrase
       (identical phrases stack in the same spatial corridor)
```

Frames that fall between phrases (silence, quiet gaps) stay in a compact cluster near the center using a slow sine/cosine drift — they don't march across the stage.

The result is that each detected phrase carves out its own region of 3-D space. The shape of that region encodes the timbral and dynamic evolution of the phrase.

---

## How a square is colored

Color maps to **dominant frequency** using a spectral rainbow ramp:

| Frequency | Color |
|-----------|-------|
| 2 kHz | Violet |
| 3 kHz | Blue |
| 4 kHz | Cyan |
| 5 kHz | Green |
| 6 kHz | Yellow |
| 7 kHz | Orange |
| 8 kHz | Red |

This is the same convention as a standard spectrogram — a higher-pitched moment is warmer/redder, a lower-pitched moment is cooler/bluer.

---

## The three states a square can be in

Every square in the cloud is always one of three things:

### 1. Ghost (not currently part of a playing phrase)
Nearly invisible. The square's border is drawn at about 5% brightness. You can barely see it. The whole cloud is full of ghosts — one for every frame in the entire recording.

### 2. Lit (part of the currently-playing phrase)
The square's border glows in its full frequency color. Brightness depends on how close the frame is to the current playhead:
- Frame at the **start of the phrase** → dimmer (weight ~0.20)
- Frame **just behind the playhead** → near full brightness (weight ~1.0)

Lines connecting consecutive lit squares are also drawn in matching colors.

### 3. Active playhead (the single frame closest to `audio.currentTime`)
The square gets a **red border** instead of its frequency color. This is the "now" indicator — it shows exactly where in the song you are at this moment.

---

## What the labels show

When squares are lit, up to 8 of them get floating labels. The label for each square shows:

```
0.1355        ← AMPLITUDE  (0 to 1, relative to the clip's loudest moment)
4.97          ← LIFETIME   (seconds since this frame was first emitted — counts up)
37.54         ← EMISSION TIME  (seconds from the start of the clip)
```

The label border color matches the square's frequency color.

---

## The full-chain accumulation effect

This is the most important thing to understand about how the lighting works.

**Old mental model (wrong):** a spotlight that moves through the song, lighting just the current moment.

**Correct mental model:** an ink pen drawing a path. As the phrase plays, the pen moves forward and the line it drew stays visible behind it. By the time the phrase finishes, the whole path is drawn — a complete 3-D network representing that one verse.

In code terms: when the playhead enters a phrase, every frame from the **phrase start** up to the current position is lit simultaneously, with a brightness gradient from dim (start) to bright (now). The `litIntensities` array holds each frame's current brightness and persists between animation frames.

After the phrase ends, all those brightnesses decay at a constant rate (~0.9 seconds to fade completely). The network lingers briefly, then disappears before the next phrase starts building.

---

## The static skeleton

In addition to the animated lighting, a very dim set of lines is always drawn connecting consecutive frames within each detected phrase. This is the **skeleton** — it shows the underlying cloud structure even when nothing is playing. You have to look carefully to see it.

---

## How phrases are detected

The system detects phrases automatically by looking at the amplitude over time:

1. Scan the amplitude envelope for runs of frames above a threshold (~25th–45th percentile of the clip's RMS distribution).
2. Group consecutive above-threshold frames into segments, allowing small gaps (< 68 ms) to be bridged.
3. Discard segments shorter than 48 ms (too short to be a real call).
4. Each surviving segment becomes a **chirp chain** — an ordered list of frame indices.

The stats line above the canvas shows how many chains were found (e.g., `"480 frames · ≤8.0s · 3 chirp chains"`).

---

## How audio playback stays in sync

The visualization does **not** use a wall clock. It reads `HTMLAudioElement.currentTime` directly on every animation frame (~60 fps). This means:

- Scrubbing, pausing, and looping all work correctly.
- The red-border playhead always points at the frame mathematically closest to wherever the audio actually is.
- There is no drift.

If autoplay is blocked by the browser, clicking anywhere on the canvas starts the audio.

---

## Post-processing (bloom)

The Three.js scene runs through an `EffectComposer` with an `UnrealBloomPass`. This adds the soft glow around bright squares and lines. The bloom threshold is set to 0.22 — ghost squares (brightness ~0.05) stay below it and render crisply without blooming. Lit squares (brightness > 0.20) and the red playhead bloom noticeably.

`ACESFilmicToneMapping` is applied on the renderer for consistent HDR-to-display mapping.

---

## Synthetic fallback

If no audio file is uploaded, the system generates a short synthetic signal (three chirp-like bursts at different frequencies) and uses it for both the visualization and the `<audio>` playback. The analysis pipeline is identical — you just get a demo instead of a real recording.

---

## Controls

| Input | Action |
|-------|--------|
| Left-drag | Orbit the camera |
| Right-drag | Pan |
| Scroll | Zoom |
| Click | Start audio (if autoplay was blocked) |

The camera auto-rotates slowly when idle.

---

## For developers: the data pipeline

```
File upload
    ↓
AudioContext.decodeAudioData → mono Float32Array (PCM)
    ↓
frameAudioData()
  → sliding Hann-windowed 256-sample frames at 60 fps
  → per-frame: RMS amplitude + coarse DFT dominant frequency
  → returns AudioDrivenPoint[] { index, emissionTime, amplitude, freqHz }
    ↓
extractChirpChains()
  → groups frames above adaptive threshold into segments
  → returns number[][] (one array of frame indices per phrase)
    ↓
spectrogramSampleToPosition()
  → maps (emissionTime, freqHz, amplitude, normWithinSegment) → (x, y, z)
    ↓
Three.js scene
  → ShaderMaterial on Points  → square rendering with aLitness attribute
  → static skeleton LineSegments  → always-on dim chain structure
  → dynamic lit LineSegments  → rebuilt every frame for the active verse
  → CSS2DRenderer labels  → 8-slot pool on brightest lit frames
  → EffectComposer + UnrealBloomPass
```

### Key attributes updated per frame

| Attribute | Type | Purpose |
|-----------|------|---------|
| `aLitness` | `float` per point | `0` = ghost, `0–1` = decay weight, `2` = active playhead |
| `litIntensities` | `Float32Array` (CPU) | Persists between frames; holds each frame's current glow level; decays at `1 / (60 × 0.9)` per frame |

Only `aLitness` is uploaded to the GPU each frame. `litIntensities` is the source of truth on the CPU side.

---

## Future direction: MFCC + PCA

The current layout is driven by amplitude and a coarse frequency estimate — it produces visually compelling forms that closely resemble the original artist's work.

A more rigorous version would:

1. Compute **40 Mel-Frequency Cepstral Coefficients (MFCCs)** per frame using a proper mel filterbank + DCT.
2. Collect all frame MFCC vectors into a matrix and run **Principal Component Analysis (PCA)** to find the top 3 axes of variance.
3. Project each frame onto those 3 axes → `(x, y, z)`.

In this model — called a **Spatiotemporal Acoustic Manifold** — spatial distance between two points directly reflects **timbral similarity**. Trajectories through the cloud show how the bird's vocal timbre evolves. Different species produce reproducibly different manifold shapes.

The rest of the pipeline (shader, lighting, labels, bloom) stays exactly the same. Only `spectrogramSampleToPosition()` in `BirdSoundEmbeddingViz.tsx` and the frame analysis in `audioDrivenPointCloud.ts` need to change. See `FEATURES.md §16` for more detail.
