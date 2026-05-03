import { Suspense, lazy, useState, useCallback, useEffect, useRef } from "react";
import { Link, useParams } from "react-router-dom";
import { getResultById } from "@/api/backend";
import { useAppPreferences } from "@/context/AppPreferences";

const BirdSoundEmbeddingViz = lazy(async () => {
  const m = await import("@/components/BirdSoundEmbeddingViz");
  return { default: m.BirdSoundEmbeddingViz };
});

export function VizPage() {
  const { id } = useParams<{ id: string }>();
  const { uploadedFile } = useAppPreferences();

  // ── Fullscreen ──────────────────────────────────────────────────────────────
  const stageRef = useRef<HTMLDivElement>(null);
  const [isFullscreen, setIsFullscreen] = useState(false);
  useEffect(() => {
    const onChange = () => setIsFullscreen(!!document.fullscreenElement);
    document.addEventListener("fullscreenchange", onChange);
    return () => document.removeEventListener("fullscreenchange", onChange);
  }, []);
  const toggleFullscreen = useCallback(() => {
    if (!document.fullscreenElement) {
      stageRef.current?.requestFullscreen().catch(() => {});
    } else {
      document.exitFullscreen().catch(() => {});
    }
  }, []);
  const isUploadRoute = id === "upload";
  const result = id && !isUploadRoute ? getResultById(id) : undefined;

  const seed =
    isUploadRoute && uploadedFile
      ? `upload:${uploadedFile.name}:${uploadedFile.lastModified}`
      : result
        ? `${result.id}:${result.recordingId}`
        : "";

  /** Upload route requires a file in context; `/viz/:id` can still render with synthetic audio only. */
  const showViz = isUploadRoute ? Boolean(uploadedFile) : Boolean(result);

  let emptyMessage: string | null = null;
  if (isUploadRoute && !uploadedFile) {
    emptyMessage =
      "No uploaded file in memory. Choose audio on Query or Home, then open Visualization.";
  } else if (!isUploadRoute && id && !result) {
    emptyMessage = "Unknown recording id.";
  }

  return (
    <div className="page viz-page">
      <p className="breadcrumb">
        <Link to="/query">← Back to query</Link>
      </p>
      <header className="page-header">
        <h1>Spatiotemporal Sound Visualization</h1>
        <p className="muted">
          Each audio frame becomes a square data point in 3-D space. As the bird sings, every verse
          assembles into a glowing network — a spatial score you can orbit and read in terms of
          frequency, amplitude, and temporal evolution. Silence stays dark.
        </p>
      </header>

      {showViz ? (
        <>
          <div className="viz-sound-stack">
            <div className="viz-sound-panel viz-sound-panel--stage" ref={stageRef}>
              <div className="embedding-viz-topline">
                <span className="embedding-viz-stats">Preparing stream…</span>
                <div className="embedding-viz-topline__right">
                  <span className="embedding-viz-source">
                    Source:{" "}
                    {uploadedFile
                      ? `upload (${uploadedFile.name})`
                      : isUploadRoute
                        ? "—"
                        : "synthetic timeline (no upload)"}
                  </span>
                  <button
                    className="viz-fullscreen-btn"
                    onClick={toggleFullscreen}
                    title={isFullscreen ? "Exit fullscreen" : "Enter fullscreen"}
                    aria-label={isFullscreen ? "Exit fullscreen" : "Enter fullscreen"}
                  >
                    {isFullscreen ? (
                      <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
                        <path d="M8 3v3a2 2 0 0 1-2 2H3"/><path d="M21 8h-3a2 2 0 0 1-2-2V3"/><path d="M3 16h3a2 2 0 0 1 2 2v3"/><path d="M16 21v-3a2 2 0 0 1 2-2h3"/>
                      </svg>
                    ) : (
                      <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
                        <path d="M8 3H5a2 2 0 0 0-2 2v3"/><path d="M21 8V5a2 2 0 0 0-2-2h-3"/><path d="M3 16v3a2 2 0 0 0 2 2h3"/><path d="M16 21h3a2 2 0 0 0 2-2v-3"/>
                      </svg>
                    )}
                    {isFullscreen ? "Exit" : "Fullscreen"}
                  </button>
                </div>
              </div>
              <Suspense
                fallback={
                  <div className="embedding-viz embedding-viz--loading muted" aria-busy="true">
                    Loading WebGL view…
                  </div>
                }
              >
                <BirdSoundEmbeddingViz seed={seed} audioFile={uploadedFile} />
              </Suspense>
            </div>
            {result ? (
              <footer className="viz-sound-species-card">
                <img
                  className="viz-sound-species-card__thumb"
                  src={result.imageUrl}
                  alt=""
                  width={80}
                  height={80}
                  loading="lazy"
                />
                <div className="viz-sound-species-card__meta">
                  <div className="viz-sound-species-card__title">{result.commonName}</div>
                  <div className="viz-sound-species-card__sci muted">{result.scientificName}</div>
                  <div className="viz-sound-species-card__rec small muted">
                    Recording {result.recordingId} · {result.vocalizationType}
                  </div>
                </div>
              </footer>
            ) : (
              <footer className="viz-upload-footer muted small">{uploadedFile!.name}</footer>
            )}
          </div>
          <section className="viz-data-guide">
            <h2>Data for every point</h2>
            <div className="viz-data-guide__bar" aria-hidden>
              <span>2 kHz</span>
              <span>3 kHz</span>
              <span>4 kHz</span>
              <span>5 kHz</span>
              <span>6 kHz</span>
              <span>7 kHz</span>
              <span>8 kHz</span>
            </div>
            <div className="viz-data-guide__legend">
              <div className="vdgl-row">
                <span className="vdgl-key">COLOR / FREQUENCY</span>
                <span className="vdgl-val muted">Most active frequency band at emission time (violet 2&nbsp;kHz → red 8&nbsp;kHz)</span>
              </div>
              <div className="vdgl-row">
                <span className="vdgl-key">SIGNAL AMPLITUDE</span>
                <span className="vdgl-val muted">Top label number — RMS level scaled 0→1 to the clip peak</span>
              </div>
              <div className="vdgl-row">
                <span className="vdgl-key">EMISSION TIME</span>
                <span className="vdgl-val muted">Bottom label number — seconds from clip start (t&nbsp;=&nbsp;0)</span>
              </div>
              <div className="vdgl-row">
                <span className="vdgl-key">LIFETIME</span>
                <span className="vdgl-val muted">Middle label number — seconds elapsed since emission, animated during playback</span>
              </div>
              <div className="vdgl-row">
                <span className="vdgl-key">RED BORDER</span>
                <span className="vdgl-val muted">Active playhead — the single frame closest to <code>audio.currentTime</code></span>
              </div>
            </div>
            <p className="small muted" style={{ marginTop: "0.55rem" }}>
              Amplitude from each audio frame is sampled at ≈60&nbsp;Hz and translated into data points distributed
              in 3-D space. The spatial distribution is shaped by audio-driven oscillatory patterns closely mirroring
              the amplitude and temporal characteristics found in the spectrogram. Every verse forms an ever-changing
              generative structure — a three-dimensional, data-driven visualization of sound.
            </p>
          </section>
          <p className="muted small viz-page__hint">
            Drag to orbit · scroll to zoom · right-drag to pan · click to start audio.
          </p>
        </>
      ) : (
        <p className="muted">{emptyMessage}</p>
      )}
    </div>
  );
}
