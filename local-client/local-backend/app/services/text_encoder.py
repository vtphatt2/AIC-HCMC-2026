from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger(__name__)


class TextEncoderUnavailable(RuntimeError):
    """Raised when the PECore text encoder cannot be loaded."""


def _find_onnx_model() -> Path | None:
    """Search parent directories for onnx-models/text_model_int8.onnx.

    remote-server and local-backend nest this file at different depths, so we
    walk upward instead of hardcoding a fixed parents[] index.
    """
    override = os.getenv("PECORE_ONNX_MODEL_PATH")
    if override:
        return Path(override)

    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "challenge_resources" / "onnx-models" / "text_model_int8.onnx"
        if candidate.is_file():
            return candidate
    return None


@dataclass(frozen=True)
class TextEncoderConfig:
    cached_model: str | None = os.getenv("CACHED_MODEL")
    model_id: str = os.getenv("PECORE_MODEL_ID", "hf-hub:timm/PE-Core-bigG-14-448")
    device: str = os.getenv("PECORE_DEVICE", "cpu")
    precision: str = os.getenv("PECORE_PRECISION", "fp32")
    expected_dim: int = int(os.getenv("PECORE_TEXT_DIM", "1280"))
    # "torch" loads the full OpenCLIP model (GPU-capable, multi-GB weight download).
    # "onnx" loads the quantized INT8 text-encoder-only graph (CPU only, ~515MB,
    # no PE-Core weight download) — see docs/PE-Core-bigG-14-448-Text-Encoder.README.md
    backend: str = os.getenv("PECORE_BACKEND", "torch")
    context_length: int = int(os.getenv("PECORE_CONTEXT_LENGTH", "72"))

    def __post_init__(self) -> None:
        object.__setattr__(self, "device", self.device.strip().lower())
        object.__setattr__(self, "precision", self.precision.strip().lower())
        object.__setattr__(self, "backend", self.backend.strip().lower())


def _validate_device_config(torch, config: TextEncoderConfig) -> None:
    if config.device == "cuda" and not torch.cuda.is_available():
        raise TextEncoderUnavailable("PECORE_DEVICE=cuda but CUDA is not available.")
    if config.device == "mps":
        if not getattr(torch.backends, "mps", None) or not torch.backends.mps.is_available():
            raise TextEncoderUnavailable("PECORE_DEVICE=mps but PyTorch MPS is not available.")
        if config.precision != "fp32":
            raise TextEncoderUnavailable("PECORE_DEVICE=mps requires PECORE_PRECISION=fp32.")
    if config.device not in {"cpu", "cuda", "mps"}:
        raise TextEncoderUnavailable("PECORE_DEVICE must be one of: cpu, cuda, mps.")


class PECoreTextEncoder:
    """
    Lazy PECore text encoder for PECore.

    Two interchangeable backends behind the same encode() interface:
      torch — full OpenCLIP model via open_clip.create_model_and_transforms
      onnx  — quantized INT8 text-encoder-only ONNX graph (CPU, no big download)

    The model is loaded on the first semantic query so indexing and server
    startup do not require model weights to already be cached.
    """

    WARMUP_PASSES = 10

    def __init__(self, config: TextEncoderConfig | None = None):
        self.config = config or TextEncoderConfig()
        self._lock = threading.Lock()
        self._encode_lock = threading.Lock()
        self._loaded = False
        self._model = None
        self._session = None
        self._tokenizer = None
        self._torch = None

    def encode(self, text: str):
        timer_start = time.monotonic()
        text = text.strip()
        if not text:
            raise ValueError("Semantic query is empty")

        vector = self._cached_encode(text)
        logger.info(
            "[TIMER] text_encode %.3f ms backend=%s device=%s text_len=%s",
            (time.monotonic() - timer_start) * 1000,
            self.config.backend,
            self.config.device,
            len(text),
        )
        return vector

    @lru_cache(maxsize=128)
    def _cached_encode(self, text: str):
        return self._encode_uncached(text)

    def _encode_uncached(self, text: str):
        self._ensure_loaded()
        # One inference at a time, always — callers queue here rather than run
        # side by side. This is not a throughput sacrifice: the model is
        # memory-bandwidth bound (one encode drags 541 MB of weights across the
        # bus, and latency stops improving past 8 threads), so parallel encodes
        # cannot go faster than the bus. Measured on 20 cores, four encodes:
        # 412 ms one at a time, 373 ms across four thread-capped sessions (-9%),
        # and 2597 ms across four sessions left at the default thread count —
        # 6.3x WORSE, because ONNX Runtime hands every session every core and 80
        # threads thrash. Serialising costs ~9% in the best case and removes the
        # 6.3x cliff entirely.
        with self._encode_lock:
            if self.config.backend == "onnx":
                return self._encode_onnx(text)
            return self._encode_torch(text)

    def _encode_torch(self, text: str):
        torch = self._torch

        tokens = self._tokenizer(
            [text],
            context_length=self.config.context_length,
        ).to(self.config.device)

        with torch.inference_mode():
            features = self._model(tokens)
            features = torch.nn.functional.normalize(
                features,
                dim=-1,
            )

        vector = features.detach().float().cpu().numpy()[0]

        self._validate_dim(vector)

        return vector

    def _encode_onnx(self, text: str):
        import numpy as np

        tokens = self._tokenizer([text], context_length=self.config.context_length)
        outputs = self._session.run(None, {"input_tokens": tokens})
        vector = outputs[0][0].astype("float32")

        norm = np.linalg.norm(vector)
        if norm == 0:
            raise ValueError("PECore ONNX text encoder returned a zero-norm vector")
        vector = vector / norm
        self._validate_dim(vector)
        return vector

    def _validate_dim(self, vector) -> None:
        if vector.shape[0] != self.config.expected_dim:
            raise ValueError(
                f"PECore text encoder returned dim {vector.shape[0]}; "
                f"expected {self.config.expected_dim}"
            )

    def warmup(self, query: str = "warmup query", passes: int = WARMUP_PASSES) -> dict:
        load_start = time.monotonic()
        was_loaded = self._loaded
        self._ensure_loaded()
        model_load_ms = 0.0 if was_loaded else (time.monotonic() - load_start) * 1000

        encode_start = time.monotonic()
        vector = None
        passes = max(1, int(passes))
        for _ in range(passes):
            vector = self._encode_uncached(query)
        encode_ms = (time.monotonic() - encode_start) * 1000

        logger.info(
            "[TIMER] warmup_text_encoder model_load_ms=%.3f encode_ms=%.3f passes=%s "
            "backend=%s device=%s vector_shape=%s",
            model_load_ms,
            encode_ms,
            passes,
            self.config.backend,
            self.config.device,
            tuple(vector.shape),
        )
        return {
            "status": "ok",
            "model_load_ms": model_load_ms,
            "encode_ms": encode_ms,
            "passes": passes,
            "backend": self.config.backend,
            "device": self.config.device,
        }

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return

        with self._lock:
            if self._loaded:
                return

            timer_start = time.monotonic()
            if self.config.backend == "onnx":
                self._load_onnx()
            elif self.config.backend == "torch":
                self._load_torch()
            else:
                raise TextEncoderUnavailable(
                    f"PECORE_BACKEND must be 'torch' or 'onnx', got '{self.config.backend}'"
                )

            self._loaded = True
            logger.info(
                "[TIMER] model_load %.3f ms backend=%s model_id=%s device=%s",
                (time.monotonic() - timer_start) * 1000,
                self.config.backend,
                self.config.model_id,
                self.config.device,
            )

    def _load_torch(self) -> None:
        try:
            import json
            import torch

            from safetensors.torch import load_file
            from open_clip.model import _build_text_tower
            from open_clip.tokenizer import SimpleTokenizer
        except ImportError as exc:
            raise TextEncoderUnavailable(
                "PECore text encoder dependencies are missing. Install them with: "
                "python -m pip install open_clip_torch torch safetensors"
            ) from exc

        _validate_device_config(torch, self.config)

        cached_model = self.config.cached_model

        if not cached_model:
            raise TextEncoderUnavailable(
                "CACHED_MODEL is not set."
            )

        model_dir = Path(cached_model)

        config_path = model_dir / "config.json"
        weights_path = model_dir / "model.safetensors"
        bpe_path = model_dir / "bpe_simple_vocab_16e6.txt.gz"

        required_files = [
            config_path,
            weights_path,
            bpe_path,
        ]

        missing = [
            path
            for path in required_files
            if not path.is_file()
        ]

        if missing:
            raise TextEncoderUnavailable(
                "Cached PECore text encoder is incomplete. Missing: "
                + ", ".join(str(path) for path in missing)
            )

        try:
            # 1. architecture config
            with config_path.open("r", encoding="utf-8") as f:
                cfg = json.load(f)

            # 2. chỉ dựng text tower
            model = _build_text_tower(
                embed_dim=cfg["embed_dim"],
                text_cfg=cfg["text_cfg"],
            )

            # 3. load text weights
            weights = load_file(
                str(weights_path),
                device="cpu",
            )

            model.load_state_dict(
                weights,
                strict=False,
            )

            # 4. tokenizer local
            print("5. create tokenizer")
            tokenizer = SimpleTokenizer(
                bpe_path=str(bpe_path),
                context_length=cfg["text_cfg"]["context_length"],
            )

            # 5. device

            print("6. move model to device")
            model = model.to(self.config.device)
            model.eval()
            print("7. success")

        except Exception as exc:
            raise TextEncoderUnavailable(
                f"Could not load cached PECore text encoder from '{model_dir}'."
            ) from exc

        self._torch = torch
        self._model = model
        self._tokenizer = tokenizer

    def _load_onnx(self) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise TextEncoderUnavailable(
                "PECore ONNX text encoder dependencies are missing. Install them with: "
                "python -m pip install onnxruntime ftfy regex numpy"
            ) from exc

        from .simple_tokenizer import SimpleTokenizer, _find_bpe_vocab

        model_path = _find_onnx_model()
        if model_path is None:
            raise TextEncoderUnavailable(
                "Could not find onnx-models/text_model_int8.onnx. Set PECORE_ONNX_MODEL_PATH "
                "to its location, or place the file under <repo-root>/challenge_resources/onnx-models/."
            )

        vocab_path = _find_bpe_vocab()
        if vocab_path is None:
            raise TextEncoderUnavailable(
                "Could not find onnx-models/bpe_simple_vocab_16e6.txt.gz. Set "
                "PECORE_BPE_VOCAB_PATH to its location, or place the file under "
                "<repo-root>/challenge_resources/onnx-models/."
            )

        # Left at ONNX Runtime's default (every core), a single encode measured
        # 115 ms on 20 cores but 105 ms on 14 — past the point where memory
        # bandwidth saturates, extra threads only add contention. Pin it rather
        # than let the default scale with whatever machine this lands on.
        options = ort.SessionOptions()
        options.intra_op_num_threads = int(os.getenv("PECORE_ONNX_THREADS", "14"))

        try:
            session = ort.InferenceSession(
                str(model_path), sess_options=options, providers=["CPUExecutionProvider"]
            )
            tokenizer = SimpleTokenizer(vocab_path, context_length=self.config.context_length)
        except Exception as exc:
            raise TextEncoderUnavailable(
                f"Could not load PECore ONNX text encoder from '{model_path}'."
            ) from exc

        self._session = session
        self._tokenizer = tokenizer
