from __future__ import annotations

from backend.config import Settings
from backend.providers.base import InferenceProvider
from backend.providers.placeholder import PlaceholderProvider


def build_provider(settings: Settings) -> InferenceProvider:
    if settings.model_provider == "placeholder":
        return PlaceholderProvider()
    if settings.model_provider == "clap":
        # Deliberately not implemented yet: reserved for upcoming real model integration.
        return PlaceholderProvider()
    return PlaceholderProvider()
