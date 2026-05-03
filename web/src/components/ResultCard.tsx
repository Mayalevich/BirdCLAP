import { useEffect, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import type { SearchResult } from "@/api/types";
import { rememberResult } from "@/api/backend";
import { drawMockSpectrogram, drawSpectrogramFromFile } from "@/lib/spectrogramCanvas";
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
    let cancelled = false;
    if (spectrogramFile) {
      drawSpectrogramFromFile(canvasEl, spectrogramFile).catch(() => {
        if (!cancelled) drawMockSpectrogram(canvasEl, result.id);
      });
    } else {
      drawMockSpectrogram(canvasEl, result.id);
    }
    return () => {
      cancelled = true;
    };
  }, [canvasEl, result.id, spectrogramFile]);

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
      </div>
      <div className="result-card__actions">
        <Link to={`/viz/${encodeURIComponent(result.id)}`} className="btn btn--outline">
          Visualize
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
