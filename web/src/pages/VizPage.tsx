import { Suspense, lazy } from "react";
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
        <h1>Visualization</h1>
        <p className="muted">
          Amplitude sampled at ≈60&nbsp;Hz; color reflects the dominant band (~2–8&nbsp;kHz),
          brightness the level—grounded on your uploaded clip below. Click the stage if playback
          does not start.
        </p>
      </header>

      {showViz ? (
        <>
          <div className="viz-sound-stack">
            <div className="viz-sound-panel viz-sound-panel--stage">
              <div className="embedding-viz-topline">
                <span className="embedding-viz-stats">Preparing stream…</span>
                <span className="embedding-viz-source">
                  Source:{" "}
                  {uploadedFile
                    ? `upload (${uploadedFile.name})`
                    : isUploadRoute
                      ? "—"
                      : "synthetic timeline (no upload)"}
                </span>
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
            <h2>Data for each point</h2>
            <div className="viz-data-guide__bar" aria-hidden>
              <span>2 kHz</span>
              <span>3 kHz</span>
              <span>4 kHz</span>
              <span>5 kHz</span>
              <span>6 kHz</span>
              <span>7 kHz</span>
              <span>8 kHz</span>
            </div>
            <p className="small muted">
              <strong>Top number</strong> = signal amplitude for that frame, scaled 0–1 to the clip peak ·{" "}
              <strong>Second line</strong> = emission time in seconds (t&nbsp;=&nbsp;0 at clip start) and{" "}
              <em>life</em> = seconds since that emission · <strong>Color</strong> = loudest band (~2–8&nbsp;kHz) at
              that instant · Click the stage if audio does not auto-start.
            </p>
          </section>
          <p className="muted small viz-page__hint">
            Drag to orbit · scroll to zoom · right-drag to pan. One marker tracks the clock: dominant band and level;
            silence stays dark; short decay only.
          </p>
        </>
      ) : (
        <p className="muted">{emptyMessage}</p>
      )}
    </div>
  );
}
