"""Compatibility entry point. Prefer: python app.py [--check]."""
from app import (
    ROOT, SUMMARY, get_predictor, get_slake_examples, load_slake_example,
    load_summary, main, make_demo, predict, summary_md,
)

if __name__ == "__main__":
    main()
