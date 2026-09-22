"""Compatibility shim for the integrated preprocess transcript collector.

Prefer:
    python -m preprocess collect-transcripts METADATA --output-dir TRANSCRIPTS
"""
from __future__ import annotations

import sys

from preprocess.transcript_cleaner.workflow import main


if __name__ == "__main__":
    arguments = sys.argv[1:] or [
        "AIC2026_sample/media-info-aic25-b1/media-info",
        "--output-dir", "AIC2026_sample/transcripts",
    ]
    raise SystemExit(main(["collect", *arguments]))
