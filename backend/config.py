from __future__ import annotations

import os
from dataclasses import dataclass


def load_cors_origins() -> tuple[str, ...]:
    """
    `http://127.0.0.1:<port>` vs `http://localhost:<port>` are different origins; Vite shows either URL.
    When only `CORS_ORIGIN` (singular) points at one of those, we add its host alias automatically.
    """
    multi_raw = os.getenv("CORS_ORIGINS", "").strip()
    single = os.getenv("CORS_ORIGIN", "").strip()
    defaults = (
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174",
    )

    if multi_raw:
        items = tuple(x.strip() for x in multi_raw.split(",") if x.strip())
        return items if items else defaults

    if single.startswith("http://localhost:"):
        sibling = single.replace("http://localhost:", "http://127.0.0.1:", 1)
        return tuple(dict.fromkeys((single, sibling)))
    if single.startswith("http://127.0.0.1:"):
        sibling = single.replace("http://127.0.0.1:", "http://localhost:", 1)
        return tuple(dict.fromkeys((single, sibling)))

    if single:
        return (single,)

    return defaults


@dataclass(frozen=True)
class Settings:
    model_provider: str
    cors_origins: tuple[str, ...]
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
        cors_origins=load_cors_origins(),
        checkpoint_path=os.getenv("CHECKPOINT_PATH", "checkpoints/finetune11/best_r1.pt").strip(),
        audio_root=os.getenv("AUDIO_ROOT", "scripts/data/xc_audio").strip(),
        gallery_cache=os.getenv("GALLERY_CACHE", "data/gallery_embeddings.pt").strip(),
        base_model=os.getenv("BASE_MODEL", "laion/clap-htsat-fused").strip(),
        metadata_path=os.getenv("METADATA_PATH", "data/xc_metadata_unified.csv").strip(),
        taxonomy_path=os.getenv("TAXONOMY_PATH", "data/species_taxonomy.json").strip(),
        val_pairs_path=os.getenv("VAL_PAIRS_PATH", "data/clap_val_pairs.json").strip(),
    )
