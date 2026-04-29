from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BackendError(Exception):
    status_code: int
    code: str
    message: str
