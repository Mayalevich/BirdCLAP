import { useCallback, useEffect, useRef, useState } from "react";
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
import { AudioWindowPicker } from "@/components/AudioWindowPicker";
import { useSpectrogram } from "@/hooks/useSpectrogram";
import { decodeAudioFile, sliceAudioBuffer, audioBufferToWavFile } from "@/lib/audioUtils";
import { drawSpectrogramFromAudioBuffer } from "@/lib/spectrogramCanvas";

/** Example queries drawn from acoustic descriptions and mnemonics — no species names. */
const EXAMPLE_QUERIES = [
  "sounds like who cooks for you",
  "teacher teacher teacher, louder each time",
  "drink your teeeea with a rising trill",
  "rich melodic phrases, flute-like and varied",
  "rapid descending trill slowing at the end",
  "sharp raspy alarm call repeated",
] as const;

const WINDOW_S = 10;

export function QueryPage() {
  const [params] = useSearchParams();
  const sourceHint = params.get("source");

  const { vocabMode, setVocabMode, uploadedFile, setUploadedFile } = useAppPreferences();

  const [query, setQuery] = useState("");
  const [results, setResults] = useState<SearchResult[]>([]);
  const [loading, setLoading] = useState(false);
  const [classifyHits, setClassifyHits] = useState<ClassificationHit[] | null>(null);
  const [classifyLoading, setClassifyLoading] = useState(false);
  const [describeHits, setDescribeHits] = useState<{ species: string; description: string }[] | null>(null);
  const [specCanvas, setSpecCanvas] = useState<HTMLCanvasElement | null>(null);
  const [searchError, setSearchError] = useState<string | null>(null);
  const [uploadError, setUploadError] = useState<string | null>(null);

  // ── Audio window state ───────────────────────────────────────────────────────
  const [decodedBuffer, setDecodedBuffer] = useState<AudioBuffer | null>(null);
  const [audioDuration, setAudioDuration] = useState<number | null>(null);
  const [windowStart, setWindowStart] = useState(0);
  const [decoding, setDecoding] = useState(false);

  const isWindowed = audioDuration !== null && audioDuration > WINDOW_S;

  // Decode the uploaded file to get duration and AudioBuffer for slicing
  useEffect(() => {
    if (!uploadedFile) {
      setDecodedBuffer(null);
      setAudioDuration(null);
      setWindowStart(0);
      return;
    }
    let cancelled = false;
    setDecoding(true);
    decodeAudioFile(uploadedFile)
      .then((buf) => {
        if (!cancelled) {
          setDecodedBuffer(buf);
          setAudioDuration(buf.duration);
          setWindowStart(0);
        }
      })
      .catch(() => { /* spectrogram hook already handles decode errors */ })
      .finally(() => { if (!cancelled) setDecoding(false); });
    return () => { cancelled = true; };
  }, [uploadedFile]);

  // Draw spectrogram for the selected window (overrides the hook when windowed)
  useEffect(() => {
    if (!isWindowed || !decodedBuffer || !specCanvas) return;
    const sliced = sliceAudioBuffer(decodedBuffer, windowStart, WINDOW_S);
    drawSpectrogramFromAudioBuffer(specCanvas, sliced);
  }, [isWindowed, decodedBuffer, windowStart, specCanvas]);

  // When not windowed, use the hook for a normal full-file spectrogram
  const { loading: specLoading, error: specError } = useSpectrogram(
    isWindowed ? null : uploadedFile,
    specCanvas,
  );

  // ── API helpers ──────────────────────────────────────────────────────────────

  /**
   * Return the file that CLAP should analyse — either the uploaded file
   * directly (short clips) or a WAV slice of the selected 10-second window.
   */
  const getAnalysisFile = useCallback((): File | null => {
    if (!uploadedFile) return null;
    if (!isWindowed || !decodedBuffer) return uploadedFile;
    const sliced = sliceAudioBuffer(decodedBuffer, windowStart, WINDOW_S);
    const baseName = uploadedFile.name.replace(/\.[^.]+$/, "");
    return audioBufferToWavFile(sliced, `${baseName}_window.wav`);
  }, [uploadedFile, isWindowed, decodedBuffer, windowStart]);

  const searchInFlight = useRef(false);
  const similarInFlight = useRef(false);
  const classifyInFlight = useRef(false);

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
    const file = getAnalysisFile();
    if (!file || similarInFlight.current) return;
    similarInFlight.current = true;
    setLoading(true);
    setUploadError(null);
    try {
      const r = await searchSimilarToUpload(file);
      setResults(r);
      // Derive "What CLAP heard" from top unique species in the result set.
      // This is always consistent with the similarity search because it IS
      // the same ranking — no cross-modal mismatch possible.
      const seen = new Set<string>();
      const descs: { species: string; description: string }[] = [];
      for (const item of r) {
        const key = item.commonName || item.id;
        if (!seen.has(key) && item.speciesDescription) {
          seen.add(key);
          descs.push({ species: item.commonName || key, description: item.speciesDescription });
        }
        if (descs.length >= 4) break;
      }
      setDescribeHits(descs.length ? descs : null);
    } catch (err) {
      setResults([]);
      setDescribeHits(null);
      setUploadError(formatUserFacingDemoError(err));
    } finally {
      setLoading(false);
      similarInFlight.current = false;
    }
  };

  const runClassify = async () => {
    const file = getAnalysisFile();
    if (!file || classifyInFlight.current) return;
    classifyInFlight.current = true;
    setClassifyLoading(true);
    setUploadError(null);
    try {
      const h = await classifyUpload(file);
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
          Text search binds natural language to audio via shared CLAP embeddings. Upload a clip to classify against
          label space and retrieve nearest neighbors by acoustic similarity.
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
          Passed as context to the backend search query (for example "Turdus" vs "robin").
        </p>
      </section>

      <section className="panel">
        <h2>Catalog search</h2>
        <p className="muted small">
          Type a species name <em>or describe the sound</em> — CLAP understands acoustic language.
        </p>
        <div className="row gap wrap">
          <input
            type="search"
            className="input"
            placeholder={
              vocabMode === "scientific"
                ? 'Try Turdus, or "descending flute-like phrases"\u2026'
                : 'Try "raspy alarm call" or "who cooks for you"\u2026'
            }
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            aria-describedby={searchError ? "query-search-error" : undefined}
            onKeyDown={(e) => e.key === "Enter" && runDatasetSearch()}
          />
          <button type="button" className="btn btn--primary" onClick={runDatasetSearch} disabled={loading}>
            {loading ? "Searching…" : "Search"}
          </button>
          {searchError ? (
            <button type="button" className="btn btn--outline" onClick={runDatasetSearch} disabled={loading}>
              Retry
            </button>
          ) : null}
        </div>
        <div className="query-examples" aria-label="Example queries">
          <span className="query-examples__label">Try:</span>
          {EXAMPLE_QUERIES.map((q) => (
            <button
              key={q}
              type="button"
              className="query-chip"
              onClick={() => setQuery(q)}
            >
              {q}
            </button>
          ))}
        </div>
        {searchError ? (
          <p className="panel-alert panel-alert--error" id="query-search-error" role="alert">
            {searchError}
          </p>
        ) : null}
      </section>

      <section className="panel">
        <h2>Upload &amp; CLAP outputs</h2>
        <div className="row gap wrap">
          <label className="file-input">
            <input
              type="file"
              accept="audio/*,.mp3,.wav,.ogg,.webm,.m4a"
              onChange={(e) => {
                const f = e.target.files?.[0];
                setUploadedFile(f ?? null);
                setClassifyHits(null);
                setDescribeHits(null);
                setUploadError(null);
              }}
            />
            <span className="btn btn--outline">Choose audio</span>
          </label>
          {uploadedFile ? <span className="file-name">{uploadedFile.name}</span> : null}
          <button
            type="button"
            className="btn btn--outline"
            disabled={!uploadedFile || classifyLoading || decoding}
            onClick={runClassify}
          >
            {classifyLoading ? "Classifying…" : "Classify audio"}
          </button>
          <button
            type="button"
            className="btn btn--primary"
            disabled={!uploadedFile || loading || decoding}
            onClick={runSimilarSearch}
          >
            {loading ? "Searching…" : "Search similar"}
          </button>
          {uploadError ? (
            <>
              <button type="button" className="btn btn--outline" disabled={classifyLoading} onClick={runClassify}>
                Retry classify
              </button>
              <button type="button" className="btn btn--outline" disabled={loading} onClick={runSimilarSearch}>
                Retry similar
              </button>
            </>
          ) : null}
        </div>

        {/* Window picker — only visible for recordings longer than 10 s */}
        {isWindowed && uploadedFile && audioDuration ? (
          <AudioWindowPicker
            file={uploadedFile}
            duration={audioDuration}
            windowStart={windowStart}
            onChange={setWindowStart}
          />
        ) : null}

        <div className="spectrogram-preview-row">
          <div className="spectrogram-preview">
            <canvas ref={setSpecCanvas} width={320} height={96} aria-label="Upload spectrogram" />
          </div>
          {uploadedFile ? (
            <Link to="/viz/upload" className="btn btn--outline btn--narrow">
              3-D sound map
            </Link>
          ) : null}
        </div>
        {(specLoading || decoding) ? <p className="muted small">Decoding audio…</p> : null}
        {isWindowed ? (
          <p className="muted small window-picker__spec-note">
            Spectrogram shows the selected 10 s window.
          </p>
        ) : null}
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
        {describeHits && describeHits.length > 0 ? (
          <div className="model-heard" role="region" aria-label="What CLAP heard">
            <h3 className="model-heard__heading">
              <span className="model-heard__icon" aria-hidden>◈</span>
              What CLAP heard
            </h3>
            <p className="model-heard__note muted small">
              Acoustic profiles of the closest-matching species, in similarity order.
            </p>
            <ul className="model-heard__list">
              {describeHits.map(({ species, description }, i) => (
                <li key={i} className="model-heard__item">
                  <span className="model-heard__species">{species}</span>
                  <span className="model-heard__desc">{description}</span>
                </li>
              ))}
            </ul>
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
