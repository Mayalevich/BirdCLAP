from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    model_provider: str
    cors_origin: str


def load_settings() -> Settings:
    return Settings(
        model_provider=os.getenv("MODEL_PROVIDER", "placeholder").strip().lower(),
        cors_origin=os.getenv("CORS_ORIGIN", "http://localhost:5173").strip(),
    )
