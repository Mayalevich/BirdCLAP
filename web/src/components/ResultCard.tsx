import { useEffect, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import type { SearchResult } from "@/api/types";
import { rememberResult } from "@/api/backend";
import {
  drawMockSpectrogram,
  drawSpectrogramFromFile,
  drawSpectrogramFromUrl,
} from "@/lib/spectrogramCanvas";
import { useAppPreferences } from "@/context/AppPreferences";
import { useSaved } from "@/context/SavedContext";

interface ResultCardProps {
  result: SearchResult;
  /** When set, draws real spectrogram from upload (e.g. preview clip). */
  spectrogramFile?: File | null;
}

export function ResultCard({ result, spectrogramFile }: ResultCardProps) {
  const navigate = useNavigate();
  const { addToCompare } = useAppPreferences();
  const { toggle, isSaved } = useSaved();
  const [canvasEl, setCanvasEl] = useState<HTMLCanvasElement | null>(null);
  const saved = isSaved(result.id);
  const imgAlt = `${result.commonName} placeholder`;

  useEffect(() => {
    if (!canvasEl) return;
    const ac = new AbortController();
    if (spectrogramFile) {
      drawSpectrogramFromFile(canvasEl, spectrogramFile).catch(() => {
        if (!ac.signal.aborted) drawMockSpectrogram(canvasEl, result.id);
      });
    } else if (result.audioUrl) {
      drawSpectrogramFromUrl(canvasEl, result.audioUrl, 320, 96, ac.signal).catch(() => {
        if (!ac.signal.aborted) drawMockSpectrogram(canvasEl, result.id);
      });
    } else {
      drawMockSpectrogram(canvasEl, result.id);
    }
    return () => ac.abort();
  }, [canvasEl, result.id, result.audioUrl, spectrogramFile]);

  const handleCompare = () => {
    rememberResult(result);
    const r = addToCompare(result.id);
    if (r === "added_second") navigate("/compare");
  };

  const handleToggleSave = () => {
    rememberResult(result);
    toggle(result);
  };

  return (
    <article className="result-card">
      <div className="result-card__top">
        <div className="result-card__image-wrap">
          <img
            src={result.imageUrl}
            alt={imgAlt}
            className="result-card__image"
            width={96}
            height={96}
            loading="lazy"
          />
        </div>
        <div className="result-card__meta">
          <h3 className="result-card__title">{result.commonName}</h3>
          <p className="result-card__sci">{result.scientificName}</p>
          <ul className="result-card__details">
            <li>
              <span className="muted">Vocalization</span> {result.vocalizationType}
            </li>
            <li>
              <span className="muted">Duration</span> {result.duration}
            </li>
            <li>
              <span className="muted">Recording</span> {result.recordingId}
            </li>
            {result.similarity != null && (
              <li>
                <span className="muted">Similarity</span>{" "}
                {(result.similarity * 100).toFixed(1)}%
              </li>
            )}
          </ul>
        </div>
      </div>
      <div className="result-card__spec">
        <canvas ref={setCanvasEl} className="result-card__canvas" aria-hidden />
        {result.audioUrl ? (
          <audio
            className="result-card__audio"
            controls
            preload="none"
            src={result.audioUrl}
            aria-label={`Audio playback for ${result.commonName} ${result.vocalizationType}`}
          />
        ) : null}
      </div>
      {result.speciesDescription ? (
        <details className="result-card__voc-desc">
          <summary className="result-card__voc-desc-toggle">Vocalization notes</summary>
          <p className="result-card__voc-desc-text">{result.speciesDescription}</p>
        </details>
      ) : null}
      <div className="result-card__actions">
        <Link to={`/viz/${encodeURIComponent(result.id)}`} className="btn btn--outline btn--narrow">
          3-D map
        </Link>
        <button type="button" className="btn btn--ghost" onClick={handleCompare}>
          Compare
        </button>
        <button
          type="button"
          className={`btn ${saved ? "btn--primary" : "btn--outline"}`}
          onClick={handleToggleSave}
          aria-pressed={saved}
        >
          {saved ? "Saved" : "Save"}
        </button>
      </div>
    </article>
  );
}
