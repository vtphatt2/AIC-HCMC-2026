# PE-Core Text Encoder — ONNX INT8 (`PECORE_BACKEND=onnx`)

`text_model_int8.onnx` is the **text encoder only**, extracted from the full
multimodal `timm/PE-Core-bigG-14-448` model and dynamically quantized to INT8.
It produces the same 1280-dim semantic embedding used for search, without
loading the full CLIP model or requiring a GPU.

## Specs

| Property | Value |
|---|---|
| Source architecture | Text encoder of `timm/PE-Core-bigG-14-448` |
| Format | ONNX Runtime, INT8 dynamic quantization |
| File size | ~515 MB (vs. ~2.2 GB FP32 original) |
| Max context length | 72 tokens |
| Output embedding dim | 1280 |
| Accuracy loss vs. FP32 | Cosine similarity delta < 1% (measured: `0.2878` → `0.2946`) |
| Runtime deps | `onnxruntime`, `ftfy`, `regex`, `tokenizers` — no PyTorch |

## Where it's used in this repo

This is the `onnx` branch of `PECoreTextEncoder` in
[`app/services/text_encoder.py`](../local-client/local-backend/app/services/text_encoder.py)
(same implementation in `remote-server` and `local-client/local-backend`).
Set `PECORE_BACKEND=onnx` — see [setup.md → Environment Variable
Reference](setup.md#environment-variable-reference) for where that env var
goes and [launch_scripts.md](launch_scripts.md) for the `--backend onnx-cpu`
launcher flag.

It tokenizes with `SimpleTokenizer` (`app/services/simple_tokenizer.py`), not
`open_clip`'s tokenizer — the point of this backend is to avoid an
`open_clip`/PyTorch install entirely.

## Why use it over `PECORE_BACKEND=torch`

- No PyTorch or multi-GB weight download — CPU-only install is `onnxruntime` +
  a few lightweight packages.
- ~600 MB RAM at runtime instead of the full OpenCLIP model.
- Model load drops from tens of seconds to milliseconds.

Trade-off: CPU-only (no CUDA/MPS acceleration) and the INT8 encoder is the
only supported precision. Use `PECORE_BACKEND=torch` when you have a GPU and
want the full-precision model instead.
