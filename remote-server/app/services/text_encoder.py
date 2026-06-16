from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass
from functools import lru_cache

logger = logging.getLogger(__name__)


class TextEncoderUnavailable(RuntimeError):
    """Raised when the PECore text encoder cannot be loaded."""


@dataclass(frozen=True)
class TextEncoderConfig:
    model_id: str = os.getenv("PECORE_MODEL_ID", "hf-hub:timm/PE-Core-bigG-14-448")
    device: str = os.getenv("PECORE_DEVICE", "cpu")
    precision: str = os.getenv("PECORE_PRECISION", "fp32")
    expected_dim: int = int(os.getenv("PECORE_TEXT_DIM", "1280"))

    def __post_init__(self) -> None:
        object.__setattr__(self, "device", self.device.strip().lower())
        object.__setattr__(self, "precision", self.precision.strip().lower())


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
    Lazy OpenCLIP text encoder for PECore.

    The model is loaded on the first semantic query so indexing and server
    startup do not require model weights to already be cached.
    """

    WARMUP_PASSES = 10

    def __init__(self, config: TextEncoderConfig | None = None):
        self.config = config or TextEncoderConfig()
        self._lock = threading.Lock()
        self._loaded = False
        self._model = None
        self._tokenizer = None
        self._torch = None

    def encode(self, text: str):
        timer_start = time.monotonic()
        text = text.strip()
        if not text:
            raise ValueError("Semantic query is empty")

        vector = self._cached_encode(text)
        logger.info(
            "[TIMER] text_encode %.3f ms device=%s text_len=%s",
            (time.monotonic() - timer_start) * 1000,
            self.config.device,
            len(text),
        )
        return vector

    @lru_cache(maxsize=128)
    def _cached_encode(self, text: str):
        return self._encode_uncached(text)

    def _encode_uncached(self, text: str):
        self._ensure_loaded()
        torch = self._torch

        tokens = self._tokenizer([text], context_length=self._model.context_length).to(self.config.device)
        with torch.inference_mode():
            features = self._model.encode_text(tokens, normalize=True)

        vector = features.detach().float().cpu().numpy()[0]
        if vector.shape[0] != self.config.expected_dim:
            raise ValueError(
                f"PECore text encoder returned dim {vector.shape[0]}; "
                f"expected {self.config.expected_dim}"
            )
        return vector

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
            "device=%s vector_shape=%s",
            model_load_ms,
            encode_ms,
            passes,
            self.config.device,
            tuple(vector.shape),
        )
        return {
            "status": "ok",
            "model_load_ms": model_load_ms,
            "encode_ms": encode_ms,
            "passes": passes,
            "device": self.config.device,
        }

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return

        with self._lock:
            if self._loaded:
                return

            timer_start = time.monotonic()
            try:
                import open_clip
                import torch
            except ImportError as exc:
                raise TextEncoderUnavailable(
                    "PECore text encoder dependencies are missing. Install them with: "
                    "python -m pip install open_clip_torch torch"
                ) from exc

            _validate_device_config(torch, self.config)

            try:
                model, _, _ = open_clip.create_model_and_transforms(
                    self.config.model_id,
                    precision=self.config.precision,
                    device=self.config.device,
                )
                tokenizer = open_clip.get_tokenizer(self.config.model_id)
                model.eval()
            except Exception as exc:
                raise TextEncoderUnavailable(
                    "Could not load PECore text encoder model "
                    f"'{self.config.model_id}'. Make sure the model is cached or "
                    "the server can download it from Hugging Face."
                ) from exc

            self._torch = torch
            self._model = model
            self._tokenizer = tokenizer
            self._loaded = True
            logger.info(
                "[TIMER] model_load %.3f ms model_id=%s device=%s",
                (time.monotonic() - timer_start) * 1000,
                self.config.model_id,
                self.config.device,
            )
