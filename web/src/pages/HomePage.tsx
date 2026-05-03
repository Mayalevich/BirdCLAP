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
          Upload a bird recording, search the catalog by text or acoustic similarity, and open the
          3-D visualization. Saved specimens and preferences persist in your browser.
        </p>
      </header>

      <section className="panel">
        <h2>Audio intake</h2>
        <p className="muted">
          Select a recording. The clip is decoded in the browser and a spectrogram is drawn for orientation.
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
          <div className="spectrogram-preview">
            <canvas ref={setSpecCanvas} width={320} height={96} aria-label="Spectrogram preview" />
            {!uploadedFile ? (
              <p className="muted spectrogram-preview__hint">Spectrogram appears after you choose a file.</p>
            ) : null}
          </div>
          {uploadedFile ? (
            <Link to="/viz/upload" className="btn btn--outline">
              Visualize
            </Link>
          ) : null}
        </div>
        {specError ? (
          <p className="panel-alert panel-alert--error" role="alert">
            {specError}
          </p>
        ) : null}
        <p>
          <Link to="/query" className="btn btn--primary">
            Search with this clip
          </Link>
        </p>
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
        <h2>Classification &amp; similarity</h2>
        <p className="muted">
          Upload a clip to identify the species and find acoustically similar recordings in the catalog.
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
