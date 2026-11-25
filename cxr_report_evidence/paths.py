"""Default paths for local IU X-Ray runs."""

from pathlib import Path

from evidence.storage import MODEL_ROOT

ARTIFACTS = Path(__file__).resolve().parent / "artifacts"
DATA = ARTIFACTS / "data" / "cxr-openi"
MANIFEST = DATA / "manifest-formal.json"
IMAGES = DATA / "images"
LOCAL_RUN = ARTIFACTS / "results" / "local-test"
TEST_ARCHIVE = ARTIFACTS / "results" / "cxr-openi-test100-gpu"
LOCAL_TRAINING = ARTIFACTS / "checkpoints" / "local-lora"
FORMAL_ADAPTER = ARTIFACTS / "checkpoints" / "cxr-openi-formal.xlu8LK" / "adapter"
QWEN = MODEL_ROOT / "Qwen3-VL-2B-Instruct"
MAIRA = MODEL_ROOT / "maira-2"


def results_run():
    if not (LOCAL_RUN / "reports.json").is_file() and (TEST_ARCHIVE / "reports.json").is_file():
        return TEST_ARCHIVE
    return LOCAL_RUN
