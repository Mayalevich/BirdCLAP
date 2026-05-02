from __future__ import annotations

import io
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from backend.providers.base import InferenceProvider

log = logging.getLogger(__name__)

TARGET_SR = 48_000
CLIP_DURATION_S = 10.0
MIN_DURATION_S = 0.5
GALLERY_BATCH = 16


def _decode_bytes(data: bytes) -> np.ndarray | None:
    """Decode audio bytes → float32 array at 48 kHz, 10s centre-crop."""
    import librosa
    import soundfile as sf

    target_len = int(CLIP_DURATION_S * TARGET_SR)
    min_len = int(MIN_DURATION_S * TARGET_SR)

    try:
        y, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
        if y.ndim > 1:
            y = y.mean(axis=1)
        if sr != TARGET_SR:
            y = librosa.resample(y, orig_sr=sr, target_sr=TARGET_SR)
    except Exception:
        try:
            y, _ = librosa.load(io.BytesIO(data), sr=TARGET_SR, mono=True)
        except Exception:
            return None

    if len(y) < min_len:
        return None
    if len(y) >= target_len:
        start = (len(y) - target_len) // 2
        y = y[start : start + target_len]
    else:
        y = np.pad(y, (0, target_len - len(y)))
    return y.astype(np.float32)


def _load_file(path: Path) -> np.ndarray | None:
    """Load audio from disk (mirrors train_clap.py fast-path logic)."""
    import librosa
    import soundfile as sf

    target_len = int(CLIP_DURATION_S * TARGET_SR)
    min_len = int(MIN_DURATION_S * TARGET_SR)

    wav_path = path.with_suffix(".wav")
    if wav_path.is_file():
        try:
            y, sr = sf.read(str(wav_path), dtype="float32", always_2d=False)
            if sr == TARGET_SR and len(y) == target_len:
                return y
            if len(y) < min_len:
                return None
            if len(y) >= target_len:
                start = (len(y) - target_len) // 2
                y = y[start : start + target_len]
            else:
                y = np.pad(y, (0, target_len - len(y)))
            return y.astype(np.float32)
        except Exception:
            pass

    try:
        y, _ = librosa.load(str(path), sr=TARGET_SR, mono=True)
    except Exception:
        return None

    if len(y) < min_len:
        return None
    if len(y) >= target_len:
        start = (len(y) - target_len) // 2
        y = y[start : start + target_len]
    else:
        y = np.pad(y, (0, target_len - len(y)))
    return y.astype(np.float32)


class ClapProvider(InferenceProvider):
    def __init__(
        self,
        checkpoint_path: str,
        audio_root: str,
        gallery_cache: str,
        base_model: str = "laion/clap-htsat-fused",
        metadata_path: str = "",
        taxonomy_path: str = "",
        val_pairs_path: str = "",
    ) -> None:
        from transformers import ClapModel, ClapProcessor

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log.info("Loading CLAP on %s", self.device)

        self.processor = ClapProcessor.from_pretrained(base_model)
        self.model: ClapModel = ClapModel.from_pretrained(base_model)

        ckpt_path = Path(checkpoint_path)
        if ckpt_path.is_file():
            ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
            sd = ckpt.get("model_state", ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt)))
            missing, unexpected = self.model.load_state_dict(sd, strict=False)
            if missing:
                log.warning("%d missing keys in checkpoint", len(missing))
            if unexpected:
                log.warning("%d unexpected keys in checkpoint", len(unexpected))
            log.info("Loaded checkpoint: %s", ckpt_path)
        else:
            log.warning("No checkpoint at %r — using base model weights", checkpoint_path)

        self.model.eval().to(self.device)

        cache_path = Path(gallery_cache)
        if cache_path.is_file():
            log.info("Loading gallery cache from %s", cache_path)
            cached = torch.load(str(cache_path), map_location="cpu", weights_only=False)
            self._embs: torch.Tensor = cached["embeddings"]
            self._meta: list[dict] = cached["items"]
        else:
            log.info("Building gallery — this may take several minutes")
            self._embs, self._meta = self._build_gallery(
                audio_root=audio_root,
                val_pairs_path=val_pairs_path,
                metadata_path=metadata_path,
                taxonomy_path=taxonomy_path,
            )
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"embeddings": self._embs, "items": self._meta}, str(cache_path))
            log.info("Gallery cached to %s (%d items)", cache_path, len(self._meta))

        log.info("Gallery ready: %d embeddings, dim=%d", self._embs.shape[0], self._embs.shape[1])

    # ── gallery construction ────────────────────────────────────────────────────

    def _build_gallery(
        self,
        audio_root: str,
        val_pairs_path: str,
        metadata_path: str,
        taxonomy_path: str,
    ) -> tuple[torch.Tensor, list[dict]]:
        import pandas as pd

        root = Path(audio_root)

        pairs_path = Path(val_pairs_path) if val_pairs_path else Path("data/clap_val_pairs.json")
        if not pairs_path.is_file():
            raise FileNotFoundError(f"Val pairs JSON not found: {pairs_path}")

        with open(str(pairs_path), encoding="utf-8") as f:
            val_pairs = json.load(f)

        # Metadata CSV lookup: filepath → row dict
        meta_lookup: dict[str, dict] = {}
        if metadata_path and Path(metadata_path).is_file():
            df = pd.read_csv(metadata_path, encoding="utf-8")
            for _, row in df.iterrows():
                fp = str(row.get("filepath", ""))
                meta_lookup[fp] = {
                    "species_code": str(row.get("species_code", "")),
                    "common_name": str(row.get("common_name", "")),
                    "vocalization_type": str(row.get("vocalization_type", "")),
                    "duration": str(row.get("duration", "")),
                }

        # Taxonomy lookup: common_name → scientific name
        sci_lookup: dict[str, str] = {}
        if taxonomy_path and Path(taxonomy_path).is_file():
            with open(taxonomy_path, encoding="utf-8") as f:
                taxonomy = json.load(f)
            for _key, info in taxonomy.items():
                if isinstance(info, dict):
                    common = info.get("common_name", "") or _key
                    sci = info.get("species", "")
                    if common and sci:
                        sci_lookup[common] = sci

        # Deduplicate val pairs by audio path (same file has multiple text variants)
        seen: dict[str, dict] = {}
        for pair in val_pairs:
            audio_rel = pair["audio"]
            if audio_rel not in seen:
                combo = pair.get("combo", "")
                if "||" in combo:
                    species_name, voc_type = combo.split("||", 1)
                else:
                    species_name, voc_type = combo, ""
                seen[audio_rel] = {"audio_rel": audio_rel, "species": species_name, "voc_type": voc_type}

        unique = list(seen.values())
        log.info("%d unique recordings from %d val pairs", len(unique), len(val_pairs))

        wavs_batch: list[np.ndarray] = []
        meta_batch: list[dict] = []
        all_embs: list[torch.Tensor] = []
        all_meta: list[dict] = []
        skipped = 0

        for item in unique:
            audio_rel = item["audio_rel"]
            wav = _load_file(root / audio_rel)
            if wav is None:
                skipped += 1
                continue

            recording_id = Path(audio_rel).stem
            csv_meta = meta_lookup.get(audio_rel, {})
            common_name = csv_meta.get("common_name") or item["species"]
            voc_type = item["voc_type"] or csv_meta.get("vocalization_type") or ""
            species_code = csv_meta.get("species_code") or ""
            sci_name = sci_lookup.get(common_name) or None

            meta_entry: dict[str, Any] = {
                "id": recording_id,
                "recording_id": recording_id,
                "title": f"{common_name} {voc_type}".strip(),
                "species": common_name,
                "scientific_name": sci_name,
                "vocalization_type": voc_type or None,
                "duration": csv_meta.get("duration") or None,
                "species_code": species_code or None,
                "audio_url": None,
                "image_url": None,
            }

            wavs_batch.append(wav)
            meta_batch.append(meta_entry)

            if len(wavs_batch) >= GALLERY_BATCH:
                all_embs.append(self._encode_audio(wavs_batch))
                all_meta.extend(meta_batch)
                wavs_batch, meta_batch = [], []

        if wavs_batch:
            all_embs.append(self._encode_audio(wavs_batch))
            all_meta.extend(meta_batch)

        if skipped:
            log.warning("Skipped %d recordings (missing or too short)", skipped)

        if not all_embs:
            raise RuntimeError(f"Gallery empty — no audio loaded from {audio_root!r}")

        return torch.cat(all_embs, dim=0), all_meta

    # ── encoding ────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def _encode_audio(self, wavs: list[np.ndarray]) -> torch.Tensor:
        inputs = self.processor(audio=wavs, return_tensors="pt", sampling_rate=TARGET_SR)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        feat = self.model.get_audio_features(
            input_features=inputs.get("input_features"),
            is_longer=inputs.get("is_longer"),
        )
        return F.normalize(feat.pooler_output, dim=-1).cpu()

    @torch.no_grad()
    def _encode_text(self, texts: list[str]) -> torch.Tensor:
        inputs = self.processor(text=texts, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        feat = self.model.get_text_features(
            input_ids=inputs.get("input_ids"),
            attention_mask=inputs.get("attention_mask"),
        )
        return F.normalize(feat.pooler_output, dim=-1).cpu()

    # ── retrieval ───────────────────────────────────────────────────────────────

    def _top_k(self, query_emb: torch.Tensor, top_k: int) -> list[dict[str, Any]]:
        sims = (query_emb @ self._embs.T).squeeze(0)
        k = min(top_k, len(self._meta))
        scores, indices = torch.topk(sims, k)
        return [{**self._meta[i], "score": round(float(s), 4)} for s, i in zip(scores.tolist(), indices.tolist())]

    # ── InferenceProvider interface ─────────────────────────────────────────────

    def search_text(self, query: str, top_k: int) -> list[dict[str, Any]]:
        emb = self._encode_text([query])
        return self._top_k(emb, top_k)

    def search_by_audio(self, file_bytes: bytes, filename: str, top_k: int) -> list[dict[str, Any]]:
        wav = _decode_bytes(file_bytes)
        if wav is None:
            from backend.errors import BackendError
            raise BackendError(422, "AUDIO_DECODE_FAILED", "Could not decode audio. Provide a WAV or MP3 file.")
        emb = self._encode_audio([wav])
        return self._top_k(emb, top_k)

    def classify_audio(self, file_bytes: bytes, filename: str, top_k: int) -> list[dict[str, Any]]:
        wav = _decode_bytes(file_bytes)
        if wav is None:
            from backend.errors import BackendError
            raise BackendError(422, "AUDIO_DECODE_FAILED", "Could not decode audio. Provide a WAV or MP3 file.")

        emb = self._encode_audio([wav]).squeeze(0)
        sims = emb @ self._embs.T

        # Per-species max similarity across all gallery recordings
        best: dict[str, float] = {}
        sci: dict[str, str | None] = {}
        for idx, score in enumerate(sims.tolist()):
            sp = self._meta[idx]["species"]
            if score > best.get(sp, -2.0):
                best[sp] = score
                sci[sp] = self._meta[idx].get("scientific_name")

        top = sorted(best.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [{"label": sp, "scientificName": sci.get(sp), "score": round(float(s), 4)} for sp, s in top]
