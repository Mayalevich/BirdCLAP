from __future__ import annotations

from backend.errors import BackendError
from backend.providers.base import InferenceProvider


class PlaceholderProvider(InferenceProvider):
    def _raise_not_ready(self) -> None:
        raise BackendError(
            status_code=501,
            code="MODEL_NOT_CONNECTED",
            message="Model provider is not connected yet. Configure MODEL_PROVIDER=clap when model integration is ready.",
        )

    def search_text(self, query: str, top_k: int) -> list[dict]:
        self._raise_not_ready()

    def search_by_audio(self, file_bytes: bytes, filename: str, top_k: int) -> list[dict]:
        self._raise_not_ready()

    def classify_audio(self, file_bytes: bytes, filename: str, top_k: int) -> list[dict]:
        self._raise_not_ready()
