from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    model_provider: str
    cors_origin: str
    # CLAP provider settings (used when MODEL_PROVIDER=clap)
    checkpoint_path: str
    audio_root: str
    gallery_cache: str
    base_model: str
    metadata_path: str
    taxonomy_path: str
    val_pairs_path: str


def load_settings() -> Settings:
    return Settings(
        model_provider=os.getenv("MODEL_PROVIDER", "placeholder").strip().lower(),
        cors_origin=os.getenv("CORS_ORIGIN", "http://localhost:5173").strip(),
        checkpoint_path=os.getenv("CHECKPOINT_PATH", "checkpoints/finetune11/best_r1.pt").strip(),
        audio_root=os.getenv("AUDIO_ROOT", "scripts/data/xc_audio").strip(),
        gallery_cache=os.getenv("GALLERY_CACHE", "data/gallery_embeddings.pt").strip(),
        base_model=os.getenv("BASE_MODEL", "laion/clap-htsat-fused").strip(),
        metadata_path=os.getenv("METADATA_PATH", "data/xc_metadata_unified.csv").strip(),
        taxonomy_path=os.getenv("TAXONOMY_PATH", "data/species_taxonomy.json").strip(),
        val_pairs_path=os.getenv("VAL_PAIRS_PATH", "data/clap_val_pairs.json").strip(),
    )
