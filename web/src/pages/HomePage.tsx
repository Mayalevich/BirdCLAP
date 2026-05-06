import { Link } from "react-router-dom";
import { useState } from "react";
import { useAppPreferences } from "@/context/AppPreferences";
import { useSpectrogram } from "@/hooks/useSpectrogram";

export function HomePage() {
  const { vocabMode, setVocabMode, uploadedFile, setUploadedFile } = useAppPreferences();
  const [specCanvas, setSpecCanvas] = useState<HTMLCanvasElement | null>(null);
  const { error: specError } = useSpectrogram(uploadedFile, specCanvas);

  return (
    <div className="page home-page">
      <header className="page-header">
        <h1>Overview</h1>
        <p className="muted">
          Upload a recording and run CLAP-backed species classification and acoustic similarity against the catalog.
          Search by text aligns audio with language in the same embedding space. Saved specimens and preferences stay
          in your browser.
        </p>
      </header>

      <section className="panel panel--intake">
        <h2>Audio intake</h2>
        <p className="muted">
          Choose a clip for the embedding pipeline. A quick spectrogram preview is computed locally—model calls run on
          the backend when you classify or search.
        </p>
        <div className="row gap">
          <label className="file-input">
            <input
              type="file"
              accept="audio/*,.mp3,.wav,.ogg,.webm,.m4a"
              onChange={(e) => {
                const f = e.target.files?.[0];
                setUploadedFile(f ?? null);
              }}
            />
            <span className="btn btn--outline">Choose file</span>
          </label>
          {uploadedFile ? (
            <span className="file-name">{uploadedFile.name}</span>
          ) : null}
        </div>
        <div className="spectrogram-preview-row">
          <div className={`spectrogram-preview${uploadedFile ? "" : " spectrogram-preview--idle"}`}>
            {uploadedFile ? (
              <canvas ref={setSpecCanvas} width={320} height={96} aria-label="Spectrogram preview" />
            ) : (
              <div className="spectrogram-preview__idle" aria-hidden>
                <div className="spectrogram-preview__wave-bars">
                  {[12, 20, 8, 24, 14, 28, 10, 22, 16, 18, 6, 20].map((h, i) => (
                    <span key={i} className="spectrogram-preview__bar" style={{ height: `${h}px` }} />
                  ))}
                </div>
              </div>
            )}
            {!uploadedFile ? (
              <p className="muted spectrogram-preview__hint">
                Pick a file to decode the waveform. CLAP embeddings are built when you search or classify.
              </p>
            ) : null}
          </div>
          {uploadedFile ? (
            <Link to="/viz/upload" className="btn btn--outline btn--narrow">
              3-D sound map
            </Link>
          ) : null}
        </div>
        {specError ? (
          <p className="panel-alert panel-alert--error" role="alert">
            {specError}
          </p>
        ) : null}
        <div className="audio-intake__cta">
          <Link to="/query" className="btn btn--primary btn--cta-primary">
            Search with this clip
          </Link>
        </div>
      </section>

      <section className="panel">
        <h2>Catalog search</h2>
        <p className="muted">
          Search the bird recording catalog by species name, vocalization type, or any free-text description.
        </p>
        <Link to="/query?source=dataset" className="btn btn--primary">
          Search catalog
        </Link>
      </section>

      <section className="panel">
        <h2>CLAP classification &amp; similarity</h2>
        <p className="muted">
          The backend scores your clip against label text and compares embedding space neighbors to rank similar
          recordings.
        </p>
        <Link to="/query" className="btn btn--outline">
          Open query
        </Link>
      </section>

      <section className="panel">
        <h2>Display names</h2>
        <p className="muted">
          Choose whether species are shown by common name or scientific name throughout the app.
        </p>
        <div className="segmented" role="group" aria-label="Vocabulary mode">
          <button
            type="button"
            className={vocabMode === "common" ? "segmented__btn is-on" : "segmented__btn"}
            onClick={() => setVocabMode("common")}
          >
            Common names
          </button>
          <button
            type="button"
            className={vocabMode === "scientific" ? "segmented__btn is-on" : "segmented__btn"}
            onClick={() => setVocabMode("scientific")}
          >
            Scientific names
          </button>
        </div>
      </section>
    </div>
  );
}
