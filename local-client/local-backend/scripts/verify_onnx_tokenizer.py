"""
Verify the vendored torch-free SimpleTokenizer against open_clip's original,
and verify the ONNX backend's output against the reference vector documented
in docs/PE-Core-bigG-14-448-Text-Encoder.README.md.

Run from local-client/local-backend:
  python scripts/verify_onnx_tokenizer.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

TEST_QUERIES = [
    "a busy street with people",
    'nguoi dan ong dang phat bieu',
    '2 persons are in front of fruit stall in super market, with banner "Ban le khai thac thi truong cho muc tieu tang truong 2 con so" and the big 23',
    "",  # will be skipped (empty)
    "a",
    "!!! weird punctuation ??? and    extra   spaces",
]


def compare_tokenizers() -> bool:
    import open_clip

    from app.services.simple_tokenizer import SimpleTokenizer, _find_bpe_vocab

    vocab_path = _find_bpe_vocab()
    assert vocab_path is not None, "bpe_simple_vocab_16e6.txt.gz not found"

    reference_tokenizer = open_clip.get_tokenizer("hf-hub:timm/PE-Core-bigG-14-448")
    vendored_tokenizer = SimpleTokenizer(vocab_path, context_length=72)

    all_match = True
    for query in TEST_QUERIES:
        if not query.strip():
            continue
        ref_tokens = reference_tokenizer([query], context_length=72).numpy()
        vendored_tokens = vendored_tokenizer([query], context_length=72)
        match = np.array_equal(ref_tokens, vendored_tokens)
        all_match &= match
        status = "OK  " if match else "FAIL"
        print(f"[{status}] {query[:60]!r}")
        if not match:
            print(f"  reference: {ref_tokens.tolist()}")
            print(f"  vendored:  {vendored_tokens.tolist()}")
    return all_match


def compare_against_readme_reference() -> bool:
    """The README documents this exact query's first-3 embedding values."""
    import onnxruntime as ort

    from app.services.simple_tokenizer import SimpleTokenizer, _find_bpe_vocab
    from app.services.text_encoder import _find_onnx_model

    query = (
        '2 persons are in front of fruit stall in super market, with banner '
        '"Bán lẻ khai thác thị trường cho mục tiêu tăng trưởng 2 con số" and the big 23'
    )
    expected_first_3 = np.array([-0.01092283, -0.02374284, 0.0058420], dtype=np.float32)

    vocab_path = _find_bpe_vocab()
    model_path = _find_onnx_model()
    assert vocab_path is not None, "bpe_simple_vocab_16e6.txt.gz not found"
    assert model_path is not None, "text_model_int8.onnx not found"

    tokenizer = SimpleTokenizer(vocab_path, context_length=72)
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])

    tokens = tokenizer([query], context_length=72)
    outputs = session.run(None, {"input_tokens": tokens})
    vector = outputs[0][0].astype("float32")
    vector = vector / np.linalg.norm(vector)

    diff = np.abs(vector[:3] - expected_first_3)
    match = bool(np.all(diff < 1e-3))
    print(f"[{'OK  ' if match else 'FAIL'}] README reference vector")
    print(f"  expected: {expected_first_3.tolist()}")
    print(f"  actual:   {vector[:3].tolist()}")
    print(f"  max diff: {float(diff.max()):.6f}")
    return match


def main() -> None:
    print("=== 1. Tokenizer parity: vendored vs open_clip ===")
    tokenizer_ok = compare_tokenizers()

    print("\n=== 2. End-to-end vector vs README reference ===")
    vector_ok = compare_against_readme_reference()

    print("\n=== Result ===")
    if tokenizer_ok and vector_ok:
        print("PASS — vendored tokenizer matches open_clip and the ONNX pipeline matches the documented reference vector.")
        sys.exit(0)
    else:
        print("FAIL — see mismatches above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
