from __future__ import annotations

from abc import ABC, abstractmethod
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
