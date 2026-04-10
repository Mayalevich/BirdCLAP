#!/usr/bin/env python3
"""
Verify local setup after cloning. Run from repo root or from scripts/:

    python scripts/check_environment.py
    python scripts/check_environment.py --with-ml

Exit codes:
  0  All requested checks passed
  1  Core checks failed (fix these first)
  2  Core OK, --with-ml was set but ML stack incomplete
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path


def _ok(name: str, detail: str = "") -> None:
    extra = f" - {detail}" if detail else ""
    print(f"  [OK] {name}{extra}")


def _fail(name: str, detail: str = "") -> None:
    extra = f" - {detail}" if detail else ""
    print(f"  [!!] {name}{extra}")


def _can_import(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


def _repo_root() -> Path:
    here = Path(__file__).resolve().parent
    return here.parent


def check_core() -> bool:
    print("\n=== Core (CSV + notebooks + Xeno-Canto fetch) ===")
    ok = True

    if sys.version_info < (3, 10):
        _fail("Python 3.10+", f"found {sys.version_info.major}.{sys.version_info.minor}")
        ok = False
    else:
        _ok("Python 3.10+", f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")

    for mod in ("pandas", "requests", "tqdm", "dotenv", "jupyter_core"):
        if _can_import(mod):
            _ok(f"import {mod}")
        else:
            _fail(f"import {mod}", "pip install -r requirements.txt")
            ok = False

    root = _repo_root()
    env_paths = [root / ".env", root.parent / ".env", Path.cwd() / ".env"]
    try:
        from dotenv import load_dotenv

        for p in env_paths:
            if p.is_file():
                load_dotenv(p)
                break
    except ImportError:
        pass

    csv_primary = root / "scripts" / "xc_metadata_unified.csv"
    csv_alt = root / "xc_metadata_unified.csv"
    if csv_primary.is_file():
        _ok("metadata CSV", str(csv_primary))
    elif csv_alt.is_file():
        _ok("metadata CSV", str(csv_alt))
    else:
        _fail("metadata CSV", f"missing {csv_primary} (run get_xenocanto.ipynb or place CSV)")

    env_found = any(p.is_file() for p in env_paths)
    if os.getenv("XC_API_KEY"):
        _ok("XC_API_KEY", "set in environment")
    elif env_found:
        _ok(".env file", "present (XC_API_KEY loaded at runtime by notebooks/scripts)")
    else:
        _fail("XC_API_KEY / .env", "create .env with XC_API_KEY=... for API v3 downloads")

    try:
        import requests

        xc_key = os.getenv("XC_API_KEY")
        if not xc_key:
            _fail("Xeno-Canto API", "XC_API_KEY missing - cannot test API v3 (401 without key)")
            ok = False
        else:
            params = {"query": "cnt:canada", "page": 1, "per_page": 1, "key": xc_key}
            r = requests.get("https://xeno-canto.org/api/3/recordings", params=params, timeout=15)
            r.raise_for_status()
            body = r.json()
            if "recordings" in body:
                _ok("Xeno-Canto API reachable", "GET /api/3/recordings")
            else:
                _fail("Xeno-Canto API", "unexpected JSON shape")
                ok = False
    except Exception as e:
        hint = ""
        if "401" in str(e) or "Unauthorized" in str(e):
            hint = " (set XC_API_KEY in .env - API v3 requires key)"
        _fail("Xeno-Canto API", f"{e}{hint}")
        ok = False

    return ok


def check_ml() -> bool:
    print("\n=== ML (CLAP mini pipeline: transformers + torch + audio load) ===")
    ok = True

    if not _can_import("torch"):
        _fail("import torch", "pip install -r requirements-ml.txt")
        ok = False
    else:
        import torch

        _ok("import torch", f"cuda={'yes' if torch.cuda.is_available() else 'no'}")

    if not _can_import("transformers"):
        _fail("import transformers", "pip install -r requirements-ml.txt")
        ok = False
    else:
        _ok("import transformers")

    if not _can_import("librosa"):
        _fail("import librosa", "pip install -r requirements-ml.txt (MP3 needs ffmpeg on PATH for many setups)")
        ok = False
    else:
        _ok("import librosa")

    if ok:
        try:
            from transformers import ClapModel, ClapProcessor

            mid = "laion/clap-htsat-fused"
            ClapProcessor.from_pretrained(mid)
            _ok("Hugging Face CLAP processor", mid)
        except Exception as e:
            _fail("Hugging Face CLAP load", str(e))
            ok = False

    return ok


def main() -> int:
    p = argparse.ArgumentParser(description="Check lets-solve-it environment.")
    p.add_argument(
        "--with-ml",
        action="store_true",
        help="Also require torch, transformers, librosa, and a quick CLAP processor download.",
    )
    args = p.parse_args()

    print("Environment check (lets-solve-it)")
    print(f"Executable: {sys.executable}")

    core = check_core()
    if not core:
        print("\nResult: core setup incomplete (exit 1).")
        return 1

    if args.with_ml:
        ml = check_ml()
        if not ml:
            print("\nResult: core OK, ML stack incomplete (exit 2). pip install -r requirements-ml.txt")
            return 2

    print("\nResult: all requested checks passed (exit 0).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
