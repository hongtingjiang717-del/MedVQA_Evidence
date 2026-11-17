"""Compatibility entry point. Prefer: python evaluate.py --limit 150."""
from evaluate import (
    MODES, aggregate, digest_image, main, make_row, percentile, prompt_hash,
    read_jsonl, report, safe_append, validate_record,
)

if __name__ == "__main__":
    main()
