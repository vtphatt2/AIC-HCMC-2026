from __future__ import annotations

import os
import threading
from dataclasses import dataclass


class TextEncoderUnavailable(RuntimeError):
    """Raised when the PECore text encoder cannot be loaded in this environment."""


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

    The model is loaded only on first encode(), so backend startup and strategy
    discovery still work on machines that have not installed/downloaded PECore.
    """

    def __init__(self, config: TextEncoderConfig | None = None):
        self.config = config or TextEncoderConfig()
        self._lock = threading.Lock()
        self._loaded = False
        self._model = None
        self._tokenizer = None
        self._torch = None

    def encode(self, text: str):
        text = text.strip()
        if not text:
            raise ValueError("Semantic query is empty")

        self._ensure_loaded()
        torch = self._torch

        tokens = self._tokenizer([text], context_length=self._model.context_length).to(self.config.device)
        with torch.no_grad():
            features = self._model.encode_text(tokens, normalize=True)

        vector = features.detach().float().cpu().numpy()[0]
        if vector.shape[0] != self.config.expected_dim:
            raise ValueError(
                f"PECore text encoder returned dim {vector.shape[0]}; "
                f"expected {self.config.expected_dim}"
            )
        return vector

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return

        with self._lock:
            if self._loaded:
                return

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
                    f"'{self.config.model_id}'. This usually means the model is not downloaded "
                    "or Hugging Face access/network is unavailable. Install open_clip_torch/torch "
                    "and allow the model to download, or pre-cache the model locally."
                ) from exc

            self._torch = torch
            self._model = model
            self._tokenizer = tokenizer
            self._loaded = True
