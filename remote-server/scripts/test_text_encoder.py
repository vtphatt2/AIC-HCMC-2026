"""
Test PECoreTextEncoder directly without FastAPI or Milvus.

Run from remote-server:
  python scripts/test_text_encoder.py --query "người đàn ông đang phát biểu"
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

SCRIPT_DIR = Path(__file__).resolve().parent
REMOTE_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REMOTE_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test PE-Core text encoder directly.")
    parser.add_argument("--query", required=True)
    return parser.parse_args()


def vector_norm(vector) -> float:
    import numpy as np

    return float(np.linalg.norm(vector))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args()
    load_dotenv(REMOTE_ROOT / ".env", override=True)

    from app.services.text_encoder import PECoreTextEncoder

    print(f"query: {args.query}")
    print("step: constructing PECoreTextEncoder")
    encoder = PECoreTextEncoder()
    print(f"device_config: {encoder.config.device}")

    try:
        import torch

        print(f"cuda_available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"cuda_device: {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("torch_import: failed")

    print("step: loading model")
    t0 = time.monotonic()
    encoder._ensure_loaded()
    model_load_ms = (time.monotonic() - t0) * 1000
    print(f"model_load time_ms: {model_load_ms:.3f}")

    print("step: encoding query")
    t1 = time.monotonic()
    vector = encoder.encode(args.query)
    encode_ms = (time.monotonic() - t1) * 1000
    print(f"encode time_ms: {encode_ms:.3f}")
    print(f"vector shape: {vector.shape}")
    print(f"vector dtype: {vector.dtype}")
    print(f"vector norm: {vector_norm(vector):.6f}")


if __name__ == "__main__":
    main()
