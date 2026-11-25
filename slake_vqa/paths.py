"""Default local paths for the SLAKE project."""

from pathlib import Path

from evidence.storage import MODEL_ROOT
from evidence.vqa.model import MODEL_ID

PROJECT_DIR = Path(__file__).resolve().parent
ARTIFACTS = PROJECT_DIR / "artifacts"
DATA = ARTIFACTS / "data" / "slake"
CHECKPOINTS = ARTIFACTS / "checkpoints"
RESULTS = ARTIFACTS / "results"
EXAMPLES = PROJECT_DIR / "examples"
FORMAL_ADAPTER = ARTIFACTS / "gpu-run" / "checkpoints" / "gpu" / "slake-lora-v1" / "adapter"


def default_model():
    local = MODEL_ROOT / "Qwen3-VL-2B-Instruct"
    return str(local) if (local / "config.json").is_file() else MODEL_ID
