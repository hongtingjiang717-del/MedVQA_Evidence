"""Shared experiment artifact paths and serialization."""

import hashlib
import json
from pathlib import Path


SHARED_DIR = Path(__file__).resolve().parent
MODEL_ROOT = SHARED_DIR / "models"
RUNTIME = SHARED_DIR / "runtime"


def digest_image(image):
    """Hash RGB pixels and dimensions for clustering and split checks."""
    rgb = image.convert("RGB")
    digest = hashlib.sha256(f"{rgb.width}x{rgb.height}:".encode())
    digest.update(rgb.tobytes())
    return digest.hexdigest()


def file_digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)
