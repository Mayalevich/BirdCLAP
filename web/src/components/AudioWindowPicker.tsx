import { useEffect, useRef, useState } from "react";
import { fmtSeconds } from "@/lib/audioUtils";

interface Props {
  file: File;
  duration: number;
  windowStart: number;
  onChange: (start: number) => void;
}

const WINDOW_S = 10;

export function AudioWindowPicker({ file, duration, windowStart, onChange }: Props) {
  const audioRef = useRef<HTMLAudioElement>(null);
  const [objUrl, setObjUrl] = useState("");
  const [previewing, setPreviewing] = useState(false);

  useEffect(() => {
    const url = URL.createObjectURL(file);
    setObjUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  // Stop preview when window moves
  useEffect(() => {
    const el = audioRef.current;
    if (!el || !previewing) return;
    el.pause();
    setPreviewing(false);
  }, [windowStart]); // eslint-disable-line react-hooks/exhaustive-deps

  const handlePreview = () => {
    const el = audioRef.current;
    if (!el) return;
    el.currentTime = windowStart;
    el.play();
    setPreviewing(true);
  };

  const handleStop = () => {
    const el = audioRef.current;
    if (!el) return;
    el.pause();
    setPreviewing(false);
  };

  // Auto-stop after 10 s
  useEffect(() => {
    const el = audioRef.current;
    if (!el) return;
    const check = () => {
      if (el.currentTime >= windowStart + WINDOW_S) {
        el.pause();
        setPreviewing(false);
      }
    };
    el.addEventListener("timeupdate", check);
    return () => el.removeEventListener("timeupdate", check);
  }, [windowStart]);

  const maxStart = Math.max(0, duration - WINDOW_S);
  const endTime = Math.min(windowStart + WINDOW_S, duration);
  const windowPct = (windowStart / duration) * 100;
  const widthPct = (WINDOW_S / duration) * 100;

  return (
    <div className="window-picker">
      <audio ref={audioRef} src={objUrl} preload="metadata" />

      <div className="window-picker__header">
        <span className="window-picker__label">Analysis window</span>
        <span className="window-picker__time">
          <strong>{fmtSeconds(windowStart)}</strong>
          {" – "}
          <strong>{fmtSeconds(endTime)}</strong>
          <span className="muted"> / {fmtSeconds(duration)}</span>
        </span>
      </div>

      <div className="window-picker__track">
        <div className="window-picker__rail" />
        <div
          className="window-picker__window"
          style={{ left: `${windowPct}%`, width: `${widthPct}%` }}
          aria-hidden
        />
        <input
          type="range"
          className="window-picker__slider"
          min={0}
          max={maxStart}
          step={0.5}
          value={windowStart}
          onChange={(e) => onChange(Number(e.target.value))}
          aria-label="Select analysis window start time"
        />
      </div>

      <div className="window-picker__actions">
        <span className="window-picker__hint muted small">
          Drag to select which 10 s CLAP analyzes
        </span>
        <button
          type="button"
          className={`btn btn--small ${previewing ? "btn--primary" : "btn--outline"}`}
          onClick={previewing ? handleStop : handlePreview}
        >
          {previewing ? "■ Stop" : "▶ Preview 10 s"}
        </button>
      </div>
    </div>
  );
}
