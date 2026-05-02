from __future__ import annotations

from backend.config import Settings
from backend.providers.base import InferenceProvider
from backend.providers.placeholder import PlaceholderProvider


def build_provider(settings: Settings) -> InferenceProvider:
    if settings.model_provider == "clap":
        from backend.providers.clap_provider import ClapProvider
        return ClapProvider(
            checkpoint_path=settings.checkpoint_path,
            audio_root=settings.audio_root,
            gallery_cache=settings.gallery_cache,
            base_model=settings.base_model,
            metadata_path=settings.metadata_path,
            taxonomy_path=settings.taxonomy_path,
            val_pairs_path=settings.val_pairs_path,
        )
    return PlaceholderProvider()
