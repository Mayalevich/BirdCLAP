from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class InferenceProvider(ABC):
    @abstractmethod
    def search_text(self, query: str, top_k: int) -> list[dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def search_by_audio(self, file_bytes: bytes, filename: str, top_k: int) -> list[dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def classify_audio(self, file_bytes: bytes, filename: str, top_k: int) -> list[dict[str, Any]]:
        raise NotImplementedError

    def get_audio_path(self, recording_id: str) -> Path | None:
        """Return the absolute path of a gallery recording file, or None if unavailable."""
        return None

    @abstractmethod
    def describe_audio(self, file_bytes: bytes, filename: str, top_k: int) -> list[str]:
        """Return top-k CLAP training descriptions nearest to the audio embedding.

        Descriptions are acoustic-only (species names scrubbed during training).
        Returns an empty list if the description gallery is unavailable.
        """
        raise NotImplementedError
