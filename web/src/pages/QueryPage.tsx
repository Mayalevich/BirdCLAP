import { useEffect, useRef, useState } from "react";
import { Link, useSearchParams } from "react-router-dom";
import {
  classifyUpload,
  formatUserFacingDemoError,
  searchDataset,
  searchSimilarToUpload,
} from "@/api/backend";
import type { ClassificationHit, SearchResult } from "@/api/types";
import { useAppPreferences } from "@/context/AppPreferences";
import { ResultCard } from "@/components/ResultCard";
import { useSpectrogram } from "@/hooks/useSpectrogram";

export function QueryPage() {
  const [params] = useSearchParams();
  const sourceHint = params.get("source");

  const {
    vocabMode,
    setVocabMode,
    uploadedFile,
    setUploadedFile,
  } = useAppPreferences();

  const [query, setQuery] = useState("");
  const [results, setResults] = useState<SearchResult[]>([]);
  const [loading, setLoading] = useState(false);
  const [classifyHits, setClassifyHits] = useState<ClassificationHit[] | null>(null);
  const [classifyLoading, setClassifyLoading] = useState(false);
  const [specCanvas, setSpecCanvas] = useState<HTMLCanvasElement | null>(null);

  const [searchError, setSearchError] = useState<string | null>(null);
  const [uploadError, setUploadError] = useState<string | null>(null);

  const searchInFlight = useRef(false);
  const similarInFlight = useRef(false);
  const classifyInFlight = useRef(false);

  const { loading: specLoading, error: specError } = useSpectrogram(uploadedFile, specCanvas);

  useEffect(() => {
    if (sourceHint !== "dataset") return;
    setQuery((q) => q || (vocabMode === "scientific" ? "Turdus" : "sparrow"));
  }, [sourceHint, vocabMode]);

  const runDatasetSearch = async () => {
    if (searchInFlight.current) return;
    searchInFlight.current = true;
    setLoading(true);
    setSearchError(null);
    try {
      const r = await searchDataset(query, vocabMode);
      setResults(r);
    } catch (err) {
      setResults([]);
      setSearchError(formatUserFacingDemoError(err));
    } finally {
      setLoading(false);
      searchInFlight.current = false;
    }
  };

  const runSimilarSearch = async () => {
    if (!uploadedFile || similarInFlight.current) return;
    similarInFlight.current = true;
    setLoading(true);
    setUploadError(null);
    try {
      const r = await searchSimilarToUpload(uploadedFile);
      setResults(r);
    } catch (err) {
      setResults([]);
      setUploadError(formatUserFacingDemoError(err));
    } finally {
      setLoading(false);
      similarInFlight.current = false;
    }
  };

  const runClassify = async () => {
    if (!uploadedFile || classifyInFlight.current) return;
    classifyInFlight.current = true;
    setClassifyLoading(true);
    setUploadError(null);
    try {
      const h = await classifyUpload(uploadedFile);
      setClassifyHits(h.length ? h : null);
      if (!h.length) setUploadError("Classification returned no labels. Check the API response.");
    } catch (err) {
      setClassifyHits(null);
      setUploadError(formatUserFacingDemoError(err));
    } finally {
      setClassifyLoading(false);
      classifyInFlight.current = false;
    }
  };

  return (
    <div className="page query-page">
      <header className="page-header">
        <h1>Query workspace</h1>
        <p className="muted">
          Search the catalog by text, upload a clip for species classification, or find acoustically
          similar recordings.
        </p>
      </header>

      <section className="panel">
        <h2>Vocabulary</h2>
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
        <p className="muted small">
          Passed as context to the backend search query (for example “Turdus” vs “robin”).
        </p>
      </section>

      <section className="panel">
        <h2>Catalog search</h2>
        <div className="row gap wrap">
          <input
            type="search"
            className="input"
            placeholder={
              vocabMode === "scientific"
                ? "Try Turdus, Poecile, …"
                : "Try sparrow, chickadee, …"
            }
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            aria-describedby={searchError ? "query-search-error" : undefined}
            onKeyDown={(e) => e.key === "Enter" && runDatasetSearch()}
          />
          <button type="button" className="btn btn--primary" onClick={runDatasetSearch} disabled={loading}>
            {loading ? "Searching…" : "Search dataset"}
          </button>
          {searchError ? (
            <button type="button" className="btn btn--outline" onClick={runDatasetSearch} disabled={loading}>
              Retry search
            </button>
          ) : null}
        </div>
        {searchError ? (
          <p className="panel-alert panel-alert--error" id="query-search-error" role="alert">
            {searchError}
          </p>
        ) : null}
      </section>

      <section className="panel">
        <h2>Upload &amp; classifier output</h2>
        <div className="row gap wrap">
          <label className="file-input">
            <input
              type="file"
              accept="audio/*,.mp3,.wav,.ogg,.webm,.m4a"
              onChange={(e) => {
                const f = e.target.files?.[0];
                setUploadedFile(f ?? null);
                setClassifyHits(null);
                setUploadError(null);
              }}
            />
            <span className="btn btn--outline">Choose audio</span>
          </label>
          {uploadedFile ? <span className="file-name">{uploadedFile.name}</span> : null}
          <button
            type="button"
            className="btn btn--outline"
            disabled={!uploadedFile || classifyLoading}
            onClick={runClassify}
          >
            {classifyLoading ? "Classifying…" : "Classify audio"}
          </button>
          <button
            type="button"
            className="btn btn--primary"
            disabled={!uploadedFile || loading}
            onClick={runSimilarSearch}
          >
            {loading ? "Searching…" : "Search similar"}
          </button>
          {uploadError ? (
            <>
              <button
                type="button"
                className="btn btn--outline"
                disabled={classifyLoading}
                onClick={runClassify}
              >
                Retry classify
              </button>
              <button type="button" className="btn btn--outline" disabled={loading} onClick={runSimilarSearch}>
                Retry similar
              </button>
            </>
          ) : null}
        </div>
        <div className="spectrogram-preview-row">
          <div className="spectrogram-preview">
            <canvas ref={setSpecCanvas} width={320} height={96} aria-label="Upload spectrogram" />
          </div>
          {uploadedFile ? (
            <Link to="/viz/upload" className="btn btn--outline">
              Visualization
            </Link>
          ) : null}
        </div>
        {specLoading ? <p className="muted small">Decoding spectrogram…</p> : null}
        {uploadError ? (
          <p className="panel-alert panel-alert--error" role="alert">
            {uploadError}
          </p>
        ) : null}
        {specError ? (
          <p className="panel-alert panel-alert--error" role="alert">
            {specError}
          </p>
        ) : null}
        {classifyHits ? (
          <div className="classify-hits">
            <h3>Classification results</h3>
            <ol>
              {classifyHits.map((h) => (
                <li key={h.label}>
                  <strong>{h.label}</strong>
                  {h.scientificName ? (
                    <span className="muted"> — {h.scientificName}</span>
                  ) : null}{" "}
                  <span className="score">{(h.score * 100).toFixed(1)}%</span>
                </li>
              ))}
            </ol>
          </div>
        ) : null}
      </section>

      <section className="panel">
        <div className="row spread">
          <h2>Result set</h2>
          <Link to="/saved" className="muted small">
            Saved list →
          </Link>
        </div>
        {results.length === 0 ? (
          <div className="empty-panel-hint muted">
            <p>Run a dataset search or similarity search to see cards here.</p>
            <p className="small">
              Tip: jump to Dataset search from the Overview, or{" "}
              <Link to="/query?source=dataset">open a seeded catalog query</Link>.
            </p>
          </div>
        ) : (
          <div className="results-grid">
            {results.map((r) => (
              <ResultCard key={r.id} result={r} />
            ))}
          </div>
        )}
      </section>
    </div>
  );
}
