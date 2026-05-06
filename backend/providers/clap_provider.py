from __future__ import annotations

import io
import json
import logging
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from backend.providers.base import InferenceProvider

log = logging.getLogger(__name__)

TARGET_SR = 48_000
CLIP_DURATION_S = 10.0   # must match what the gallery was built with (10-s WAV sidecars)
MIN_DURATION_S = 0.5
GALLERY_BATCH = 8

# Bump this whenever the encoding strategy changes so stale caches are detected
# and automatically rebuilt rather than producing silent wrong results.
# v1 = centre-crop-10s (original)
# v2 = 10-s WAV sidecars loaded, query also centre-cropped to match
CACHE_VERSION = 2


def _torchaudio_load(src: str | io.BytesIO) -> tuple[np.ndarray, int] | None:
    """Load via torchaudio (handles MP3/WAV/FLAC without soundfile/audioread)."""
    try:
        import torchaudio
        wav, sr = torchaudio.load(src)  # type: ignore[arg-type]
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        return wav.squeeze(0).numpy(), sr
    except Exception:
        return None


def _crop_pad(y: np.ndarray, target_len: int, min_len: int) -> np.ndarray | None:
    """Centre-crop or zero-pad to exactly target_len samples."""
    if len(y) < min_len:
        return None
    if len(y) >= target_len:
        start = (len(y) - target_len) // 2
        y = y[start : start + target_len]
    else:
        y = np.pad(y, (0, target_len - len(y)))
    return y.astype(np.float32)


def _decode_bytes(data: bytes) -> np.ndarray | None:
    """Decode audio bytes → float32 mono 10-second centre-crop at 48 kHz.

    Matches the encoding used when the gallery was built (10-s WAV sidecars),
    so query and gallery embeddings live in the same space.
    """
    import librosa

    target_len = int(CLIP_DURATION_S * TARGET_SR)
    min_len = int(MIN_DURATION_S * TARGET_SR)

    result = _torchaudio_load(io.BytesIO(data))
    if result is not None:
        y, sr = result
        if sr != TARGET_SR:
            y = librosa.resample(y, orig_sr=sr, target_sr=TARGET_SR)
    else:
        try:
            y, _ = librosa.load(io.BytesIO(data), sr=TARGET_SR, mono=True)
        except Exception:
            return None

    return _crop_pad(y, target_len, min_len)


def _load_file(path: Path) -> np.ndarray | None:
    """Load audio from disk — WAV fast-path via soundfile, MP3 via torchaudio.

    Returns the full waveform (no cropping) so the CLAP processor can set
    ``is_longer=True`` and invoke the fusion path for long recordings.
    """
    import librosa
    import soundfile as sf

    min_len = int(MIN_DURATION_S * TARGET_SR)

    # WAV fast-path: soundfile is fast and lossless
    wav_path = path.with_suffix(".wav")
    if wav_path.is_file():
        try:
            y, sr = sf.read(str(wav_path), dtype="float32", always_2d=False)
            if y.ndim > 1:
                y = y.mean(axis=1)
            if sr != TARGET_SR:
                y = librosa.resample(y, orig_sr=sr, target_sr=TARGET_SR)
            if len(y) < min_len:
                return None
            return y.astype(np.float32)
        except Exception:
            pass

    # MP3 / anything else: torchaudio handles it without audioread
    if path.is_file():
        result = _torchaudio_load(str(path))
        if result is not None:
            y, sr = result
            if sr != TARGET_SR:
                y = librosa.resample(y, orig_sr=sr, target_sr=TARGET_SR)
            if len(y) < min_len:
                return None
            return y.astype(np.float32)

    return None


def _truncate_description(text: str, max_chars: int = 280) -> str:
    """Return first 1-2 sentences of text, capped at max_chars."""
    text = text.strip()
    sentences = re.split(r"(?<=[.!?])\s+", text)
    out = ""
    for s in sentences:
        candidate = f"{out} {s}".strip() if out else s
        if len(candidate) <= max_chars:
            out = candidate
        else:
            break
    if not out:
        out = text[:max_chars].rstrip()
    if len(text) > len(out):
        out = out.rstrip(".,;:") + "…"
    return out


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
        species_descriptions_path: str = "",
        clap_descriptions_path: str = "",
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
        _use_cache = False
        if cache_path.is_file():
            cached = torch.load(str(cache_path), map_location="cpu", weights_only=False)
            if cached.get("cache_version", 1) == CACHE_VERSION:
                log.info("Loading gallery cache v%d from %s", CACHE_VERSION, cache_path)
                self._embs: torch.Tensor = cached["embeddings"]
                self._meta: list[dict] = cached["items"]
                _use_cache = True
            else:
                old_ver = cached.get("cache_version", 1)
                log.warning(
                    "Gallery cache is v%d but current encoding is v%d — "
                    "discarding stale cache and rebuilding.  "
                    "This ensures full-duration fusion embeddings replace the old "
                    "centre-crop-10s embeddings.",
                    old_ver, CACHE_VERSION,
                )

        if not _use_cache:
            log.info("Building gallery (v%d) — this may take several minutes", CACHE_VERSION)
            self._embs, self._meta = self._build_gallery(
                audio_root=audio_root,
                val_pairs_path=val_pairs_path,
                metadata_path=metadata_path,
                taxonomy_path=taxonomy_path,
            )
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {"embeddings": self._embs, "items": self._meta, "cache_version": CACHE_VERSION},
                str(cache_path),
            )
            log.info("Gallery cached to %s (%d items)", cache_path, len(self._meta))

        log.info("Gallery ready: %d embeddings, dim=%d", self._embs.shape[0], self._embs.shape[1])

        # ── recording id → absolute audio path lookup ───────────────────────────
        self._audio_map: dict[str, Path] = self._build_audio_map(audio_root)
        log.info(
            "Audio path map: %d/%d recordings resolved to a file on disk",
            sum(1 for p in self._audio_map.values() if p.is_file()),
            len(self._meta),
        )

        # ── species description lookup (Tier 2) ────────────────────────────────
        self._desc_map: dict[str, str] = self._load_species_descriptions(species_descriptions_path)

        # ── species name → gallery index list (for text search routing) ──────
        # Keyed by the normalised name (lower-cased, stripped) so lookups are
        # case-insensitive even if desc_map and metadata CSV differ in case.
        self._species_to_indices: dict[str, list[int]] = {}
        for i, item in enumerate(self._meta):
            sp = (item.get("species") or "").strip()
            if sp:
                self._species_to_indices.setdefault(sp.lower(), []).append(i)
        log.info(
            "Species index: %d distinct species across %d gallery entries",
            len(self._species_to_indices),
            len(self._meta),
        )

        # ── acoustic text gallery for describe_audio (kept for Tier 3) ────────
        self._text_embs: torch.Tensor | None = None
        self._text_strings: list[str] = []
        self._build_text_gallery(clap_descriptions_path)

        # ── per-species text embeddings for text→text catalog search ──────────
        # Built unconditionally from desc_map (no dependency on clap_descriptions.json).
        # Cross-modal text→audio similarity is degraded after audio-only fine-tuning;
        # text→text (same encoder on both sides) is reliable.
        self._species_text_embs: torch.Tensor | None = None
        self._species_names_list: list[str] = []
        self._build_species_text_gallery()

    # ── audio path helpers ──────────────────────────────────────────────────────

    def _build_audio_map(self, audio_root: str) -> dict[str, Path]:
        """Build recording_id → Path for every gallery item.

        Strategy (in order of preference):
        1. Use ``audio_rel`` stored in each meta entry (new caches built after this change).
        2. Fall back to scanning ``audio_root`` by file stem — this makes *old* gallery
           caches work immediately without a rebuild.
        WAV is preferred over MP3 when both exist (lossless, faster decode).
        """
        root = Path(audio_root)
        audio_map: dict[str, Path] = {}

        # Pass 1: use audio_rel from meta (exact, no fs scan needed).
        # Prefer the original MP3 over the 10-second training WAV sidecar so
        # users hear the full recording, not the training clip.
        for item in self._meta:
            audio_rel = item.get("audio_rel", "")
            if not audio_rel:
                continue
            full = root / audio_rel
            mp3 = full.with_suffix(".mp3")
            wav = full.with_suffix(".wav")
            if mp3.is_file():
                audio_map[item["id"]] = mp3
            elif wav.is_file():
                audio_map[item["id"]] = wav
            elif full.is_file():
                audio_map[item["id"]] = full

        if audio_map:
            log.info("Audio map built from gallery meta audio_rel: %d entries", len(audio_map))
            return audio_map

        # Pass 2: no audio_rel in cache — scan the directory tree by stem
        if not root.is_dir():
            log.warning("audio_root %r is not a directory — audio serving disabled", audio_root)
            return {}

        log.info("Gallery meta has no audio_rel — scanning %s for audio files…", root)
        stem_map: dict[str, Path] = {}
        for path in root.rglob("*"):
            if path.suffix.lower() not in (".wav", ".mp3", ".ogg", ".flac"):
                continue
            stem = path.stem
            existing = stem_map.get(stem)
            # Prefer MP3 (full recording) over WAV (10-second training sidecar)
            if existing is None:
                stem_map[stem] = path
            elif path.suffix.lower() == ".mp3" and existing.suffix.lower() != ".mp3":
                stem_map[stem] = path

        # Only keep entries whose stem matches a gallery item id
        gallery_ids = {item["id"] for item in self._meta}
        for stem, path in stem_map.items():
            if stem in gallery_ids:
                audio_map[stem] = path

        log.info("Directory scan resolved %d/%d gallery recordings", len(audio_map), len(self._meta))
        return audio_map

    # ── description helpers ─────────────────────────────────────────────────────

    def _load_species_descriptions(self, path: str) -> dict[str, str]:
        """Load species_descriptions.json → {common_name: truncated_text}."""
        p = Path(path)
        if not p.is_file():
            log.warning("species_descriptions.json not found at %r — species descriptions unavailable", path)
            return {}
        with open(p, encoding="utf-8") as f:
            raw = json.load(f)
        result: dict[str, str] = {}
        for name, val in raw.items():
            text = val.get("text", "") if isinstance(val, dict) else str(val)
            text = text.strip()
            if text:
                result[name] = _truncate_description(text)
        log.info("Loaded species descriptions: %d entries", len(result))
        return result

    def _build_text_gallery(self, path: str) -> None:
        """Encode all CLAP training descriptions into a searchable text embedding matrix."""
        p = Path(path)
        if not p.is_file():
            log.warning(
                "clap_descriptions.json not found at %r — describe_audio will return empty", path
            )
            return
        with open(p, encoding="utf-8") as f:
            data = json.load(f)

        all_texts: list[str] = []
        for descs in data.values():
            if isinstance(descs, list):
                for d in descs:
                    if isinstance(d, str) and d.strip():
                        all_texts.append(d.strip())

        if not all_texts:
            log.warning("clap_descriptions.json contained no usable strings")
            return

        log.info("Building acoustic text gallery from %d descriptions…", len(all_texts))
        batch_size = 256
        chunks: list[torch.Tensor] = []
        for i in range(0, len(all_texts), batch_size):
            chunks.append(self._encode_text(all_texts[i : i + batch_size]))

        self._text_embs = torch.cat(chunks, dim=0)
        self._text_strings = all_texts
        log.info("Acoustic text gallery ready: %d embeddings", len(self._text_strings))

    def _build_species_text_gallery(self) -> None:
        """Encode every species as text for reliable text→text catalog search.

        Independent of clap_descriptions.json — uses species_descriptions.json
        (desc_map) which is always available.  Keyed by lower-cased name to
        match the normalised keys in _species_to_indices.
        """
        if not self._desc_map:
            log.warning("desc_map empty — species text gallery unavailable, text search will fall back to cross-modal")
            return

        sp_names: list[str] = []
        sp_texts: list[str] = []
        for name, desc in self._desc_map.items():
            sp_names.append(name)
            sp_texts.append(f"{name} — {desc}" if desc else name)

        log.info("Building species text gallery from %d species…", len(sp_names))
        batch_size = 256
        sp_chunks: list[torch.Tensor] = []
        for i in range(0, len(sp_texts), batch_size):
            sp_chunks.append(self._encode_text(sp_texts[i : i + batch_size]))

        self._species_text_embs = torch.cat(sp_chunks, dim=0)
        self._species_names_list = sp_names

        # Sanity-check: how many species in desc_map have recordings in the gallery?
        matched = sum(1 for n in sp_names if n.lower() in self._species_to_indices)
        log.info(
            "Species text gallery ready: %d species embeddings (%d/%d have gallery recordings)",
            len(sp_names), matched, len(sp_names),
        )

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

        # ── Primary source: metadata CSV (covers the full dataset) ───────────────
        # Val pairs only cover the eval split (~2k recordings); the metadata CSV
        # covers all ~27k recordings. We build the gallery from the CSV so that
        # every recording on disk is searchable, not just the eval subset.
        meta_rows: list[dict] = []
        if metadata_path and Path(metadata_path).is_file():
            df = pd.read_csv(metadata_path, encoding="utf-8")
            for _, row in df.iterrows():
                fp = str(row.get("filepath", "")).strip()
                if not fp:
                    continue
                meta_rows.append({
                    "audio_rel": fp,
                    "species_code": str(row.get("species_code", "") or ""),
                    "common_name": str(row.get("common_name", "") or ""),
                    "vocalization_type": str(row.get("vocalization_type", "") or ""),
                    "duration": str(row.get("duration", "") or ""),
                })
            log.info("Metadata CSV: %d rows -> building gallery from full dataset", len(meta_rows))
        else:
            # Fallback: derive candidate list from val pairs when CSV is absent
            log.warning("Metadata CSV not found at %r — falling back to val pairs", metadata_path)
            if not Path(val_pairs_path).is_file():
                raise FileNotFoundError(f"Neither metadata CSV nor val pairs found")
            with open(str(val_pairs_path), encoding="utf-8") as f:
                val_pairs = json.load(f)
            seen: dict[str, dict] = {}
            for pair in val_pairs:
                audio_rel = pair["audio"]
                if audio_rel not in seen:
                    combo = pair.get("combo", "")
                    species_name, voc_type = (combo.split("||", 1) if "||" in combo else (combo, ""))
                    seen[audio_rel] = {
                        "audio_rel": audio_rel,
                        "common_name": species_name,
                        "vocalization_type": voc_type,
                        "species_code": "",
                        "duration": "",
                    }
            meta_rows = list(seen.values())
            log.info("Val pairs fallback: %d unique recordings", len(meta_rows))

        # ── Taxonomy lookup: common_name → scientific name ────────────────────────
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

        log.info("Building gallery from %d candidate recordings under %s", len(meta_rows), root)

        wavs_batch: list[np.ndarray] = []
        meta_batch: list[dict] = []
        all_embs: list[torch.Tensor] = []
        all_meta: list[dict] = []
        skipped = 0

        for item in meta_rows:
            audio_rel = item["audio_rel"]
            wav = _load_file(root / audio_rel)
            if wav is None:
                skipped += 1
                continue

            recording_id = Path(audio_rel).stem
            common_name = item["common_name"]
            voc_type = item["vocalization_type"]
            species_code = item["species_code"]
            sci_name = sci_lookup.get(common_name) or None

            meta_entry: dict[str, Any] = {
                "id": recording_id,
                "recording_id": recording_id,
                "title": f"{common_name} {voc_type}".strip(),
                "species": common_name,
                "scientific_name": sci_name,
                "vocalization_type": voc_type or None,
                "duration": item["duration"] or None,
                "species_code": species_code or None,
                # audio_rel stored so _audio_map survives cache reloads
                "audio_rel": audio_rel,
                "audio_url": None,
                "image_url": None,
            }

            wavs_batch.append(wav)
            meta_batch.append(meta_entry)

            if len(wavs_batch) >= GALLERY_BATCH:
                all_embs.append(self._encode_audio(wavs_batch))
                all_meta.extend(meta_batch)
                wavs_batch, meta_batch = [], []
                log.info("Gallery progress: %d embedded so far", len(all_meta))

        if wavs_batch:
            all_embs.append(self._encode_audio(wavs_batch))
            all_meta.extend(meta_batch)

        if skipped:
            log.warning("Skipped %d recordings (missing or too short)", skipped)

        if not all_embs:
            raise RuntimeError(f"Gallery empty — no audio loaded from {audio_root!r}")

        log.info("Gallery built: %d embeddings from %d candidates", len(all_meta), len(meta_rows))
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

    def get_audio_path(self, recording_id: str) -> Path | None:
        path = self._audio_map.get(recording_id)
        return path if path and path.is_file() else None

    def _top_k(self, query_emb: torch.Tensor, top_k: int) -> list[dict[str, Any]]:
        sims = (query_emb @ self._embs.T).squeeze(0)
        k = min(top_k, len(self._meta))
        scores, indices = torch.topk(sims, k)
        results = []
        for s, i in zip(scores.tolist(), indices.tolist()):
            item: dict[str, Any] = {**self._meta[i], "score": round(float(s), 4)}
            # Set audio_url from map — None if file not on disk
            recording_id = item.get("id", "")
            if recording_id in self._audio_map and self._audio_map[recording_id].is_file():
                item["audio_url"] = f"/api/audio/{recording_id}"
            else:
                item["audio_url"] = None
            # Attach species description if not already baked into cached metadata
            if "species_description" not in item:
                item["species_description"] = self._desc_map.get(item.get("species", ""))
            results.append(item)
        return results

    # ── InferenceProvider interface ─────────────────────────────────────────────

    def search_text(self, query: str, top_k: int) -> list[dict[str, Any]]:
        """Catalog text search via text-to-text species matching.

        Fine-tuning on audio-only pairs degrades cross-modal text→audio alignment.
        Text→text (same encoder on both sides) is unaffected and returns reliable
        results.  Falls back to cross-modal only if the species gallery is absent.
        """
        query_emb = self._encode_text([query])  # (1, dim)

        if self._species_text_embs is not None and self._species_names_list:
            sims = (query_emb @ self._species_text_embs.T).squeeze(0)
            # Fetch more candidates than top_k so we can skip species with no recordings
            k_sp = min(len(self._species_names_list), max(top_k * 4, 40))
            sp_scores, sp_indices = torch.topk(sims, k_sp)

            results: list[dict[str, Any]] = []
            for score, sp_idx in zip(sp_scores.tolist(), sp_indices.tolist()):
                sp = self._species_names_list[sp_idx]
                # _species_to_indices is keyed by lower-cased name
                meta_indices = self._species_to_indices.get(sp.lower(), [])
                if not meta_indices:
                    log.debug("text search: no gallery recordings for species %r", sp)
                    continue
                # Return up to 2 recordings per matched species for variety
                for mi in meta_indices[:2]:
                    item: dict[str, Any] = {**self._meta[mi], "score": round(float(score), 4)}
                    rid = item.get("id", "")
                    item["audio_url"] = (
                        f"/api/audio/{rid}"
                        if rid in self._audio_map and self._audio_map[rid].is_file()
                        else None
                    )
                    item["species_description"] = self._desc_map.get(item.get("species", ""))
                    results.append(item)
                    if len(results) >= top_k:
                        break
                if len(results) >= top_k:
                    break

            if results:
                log.debug("text search returned %d results via text→text path", len(results))
                return results
            log.warning("text→text search produced no results (0 species matched in gallery) — falling back to cross-modal")

        # Fallback: original cross-modal text→audio (used when species gallery absent
        # or when no matched species have recordings in the gallery)
        return self._top_k(query_emb, top_k)

    def search_by_audio(self, file_bytes: bytes, filename: str, top_k: int) -> list[dict[str, Any]]:
        wav = _decode_bytes(file_bytes)
        if wav is None:
            from backend.errors import BackendError
            raise BackendError(422, "AUDIO_DECODE_FAILED", "Could not decode audio. Provide a WAV or MP3 file.")
        emb = self._encode_audio([wav])
        return self._top_k(emb, top_k)

    def describe_audio(self, file_bytes: bytes, filename: str, top_k: int = 4) -> list[str]:
        """Return acoustic descriptions for species nearest to the audio embedding.

        Uses the same audio gallery path as search_by_audio so the output is always
        consistent with similarity search results.  Cross-modal text similarity is
        unreliable after audio-only fine-tuning, so we route through _top_k instead.
        """
        wav = _decode_bytes(file_bytes)
        if wav is None:
            return []
        emb = self._encode_audio([wav])
        # Pull extra candidates so we can collect top_k distinct species
        hits = self._top_k(emb, top_k * 5)
        seen: set[str] = set()
        out: list[str] = []
        for item in hits:
            sp = item.get("species", "") or item.get("common_name", "")
            if not sp or sp in seen:
                continue
            seen.add(sp)
            desc = self._desc_map.get(sp)
            if desc:
                out.append(desc)
            if len(out) >= top_k:
                break
        return out

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
