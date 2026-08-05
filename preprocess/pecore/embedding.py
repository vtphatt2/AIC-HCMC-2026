"""Strategy-friendly PE-Core image embedding and feature persistence.

The module does not load PE-Core at import time.  This is intentional: the
model is large, and commands that only inspect or download a batch must still
work without downloading model weights.  ``OpenClipPECoreEncoder`` loads the
full image-capable OpenCLIP model on its first non-empty encode call.
"""
from __future__ import annotations

import os
import hashlib
import json
import tempfile
from abc import ABC, abstractmethod
from collections.abc import Mapping
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol, Sequence

import numpy as np

from preprocess.progress import ProgressConfig, ProgressReporter, TqdmProgressReporter
from preprocess.batch.provenance import atomic_json_write, sha256_file


class PECoreEmbeddingUnavailable(RuntimeError):
    """Raised when the PE-Core visual encoder cannot be initialized."""


def _validate_safe_directory_name(value: str, field_name: str) -> None:
    if not value or value in {".", ".."} or Path(value).name != value:
        raise ValueError(f"{field_name} must be one safe directory name")


@dataclass(frozen=True)
class EmbeddingDataLoaderConfig:
    """Torch DataLoader settings for image preprocessing workers.

    ``shuffle`` and ``drop_last`` are deliberately fixed by the pipeline:
    frame-to-vector filenames must remain in source order and no frame may be
    silently discarded.  ``batch_size`` remains on
    :class:`PECoreEmbeddingConfig` because it is also the encoder batch size.
    """

    num_workers: int = 0
    pin_memory: bool = False
    persistent_workers: bool = False
    prefetch_factor: int = 2

    def __post_init__(self) -> None:
        if self.num_workers < 0:
            raise ValueError("embedding.dataloader.num_workers must be >= 0")
        if self.prefetch_factor <= 0:
            raise ValueError("embedding.dataloader.prefetch_factor must be positive")
        if self.num_workers == 0 and self.persistent_workers:
            raise ValueError(
                "embedding.dataloader.persistent_workers requires num_workers > 0"
            )


@dataclass(frozen=True)
class AutocastConfig:
    """AMP inference settings applied around the image encoder call."""

    enabled: bool = False
    dtype: str = "bf16"
    cache_enabled: bool = True

    def __post_init__(self) -> None:
        dtype = self.dtype.strip().lower()
        aliases = {"float16": "fp16", "bfloat16": "bf16"}
        dtype = aliases.get(dtype, dtype)
        object.__setattr__(self, "dtype", dtype)
        if dtype not in {"fp16", "bf16"}:
            raise ValueError("embedding.autocast.dtype must be 'fp16' or 'bf16'")


@dataclass(frozen=True)
class TF32Config:
    """CUDA TF32 backend settings for eligible FP32 operations."""

    enabled: bool = False
    matmul: bool = True
    cudnn: bool = True

    def __post_init__(self) -> None:
        if self.enabled and not (self.matmul or self.cudnn):
            raise ValueError(
                "embedding.tf32 must enable matmul or cudnn when enabled"
            )


@dataclass(frozen=True)
class PECoreEmbeddingConfig:
    """Runtime settings for the full PE-Core image encoder.

    ``model_id`` follows the OpenCLIP identifier already used by the search
    service.  ``expected_dim`` is kept configurable so another visual model
    can be used through the same pipeline without changing its filesystem or
    batching code.
    """

    enabled: bool = False
    model_id: str = "hf-hub:timm/PE-Core-bigG-14-448"
    model_revision: str | None = None
    device: str = "auto"
    precision: str = "fp32"
    autocast: AutocastConfig = field(default_factory=AutocastConfig)
    tf32: TF32Config = field(default_factory=TF32Config)
    expected_dim: int = 1_280
    batch_size: int = 8
    dataloader: EmbeddingDataLoaderConfig = field(default_factory=EmbeddingDataLoaderConfig)
    overwrite: bool = False
    features_dir_name: str = "PECore-features"
    image_extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png")

    def __post_init__(self) -> None:
        model_id = self.model_id.strip()
        model_revision = self.model_revision.strip() if self.model_revision else None
        device = self.device.strip().lower()
        precision_aliases = {
            "float32": "fp32",
            "float16": "fp16",
            "bfloat16": "bf16",
        }
        precision = precision_aliases.get(self.precision.strip().lower(), self.precision.strip().lower())
        extensions = tuple(
            extension.lower() if extension.startswith(".") else f".{extension.lower()}"
            for extension in self.image_extensions
        )
        object.__setattr__(self, "model_id", model_id)
        object.__setattr__(self, "model_revision", model_revision)
        object.__setattr__(self, "device", device)
        object.__setattr__(self, "precision", precision)
        object.__setattr__(self, "image_extensions", extensions)
        if isinstance(self.autocast, Mapping):
            object.__setattr__(self, "autocast", AutocastConfig(**dict(self.autocast)))
        elif not isinstance(self.autocast, AutocastConfig):
            raise TypeError("autocast must be an AutocastConfig or mapping")
        if isinstance(self.tf32, Mapping):
            object.__setattr__(self, "tf32", TF32Config(**dict(self.tf32)))
        elif isinstance(self.tf32, bool):
            object.__setattr__(self, "tf32", TF32Config(enabled=self.tf32))
        elif not isinstance(self.tf32, TF32Config):
            raise TypeError("tf32 must be a TF32Config, mapping, or bool")
        if isinstance(self.dataloader, Mapping):
            object.__setattr__(
                self,
                "dataloader",
                EmbeddingDataLoaderConfig(**dict(self.dataloader)),
            )
        elif not isinstance(self.dataloader, EmbeddingDataLoaderConfig):
            raise TypeError("dataloader must be an EmbeddingDataLoaderConfig or mapping")

        if not model_id:
            raise ValueError("model_id must not be empty")
        if device not in {"auto", "cpu", "cuda", "mps"}:
            raise ValueError("device must be one of: auto, cpu, cuda, mps")
        if precision not in {"fp32", "fp16", "bf16"}:
            raise ValueError("precision must be one of: fp32, fp16, bf16")
        if self.expected_dim <= 0:
            raise ValueError("expected_dim must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not extensions:
            raise ValueError("image_extensions must not be empty")
        if any(not extension.startswith(".") for extension in extensions):
            raise ValueError("image_extensions must be file extensions")
        _validate_safe_directory_name(self.features_dir_name, "features_dir_name")


class VisualEmbeddingEncoder(ABC):
    """Replaceable image-to-vector strategy."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the number of values in one output vector."""
        raise NotImplementedError

    @abstractmethod
    def embed(self, image_paths: Sequence[Path]) -> np.ndarray:
        """Return one normalized vector for each image path."""
        raise NotImplementedError


class PreparedBatchEmbeddingEncoder(Protocol):
    """Optional protocol for encoders that can use the image DataLoader.

    Existing strategies only need ``VisualEmbeddingEncoder.embed`` and remain
    valid.  An encoder can implement this protocol to move image decoding and
    preprocessing into DataLoader workers while keeping model inference in the
    main process.
    """

    def image_transform(self) -> Any:
        """Return a serializable image transform used by worker processes."""

    def embed_prepared_batch(self, batch: Any, *, non_blocking: bool = False) -> np.ndarray:
        """Encode one already-transformed tensor batch."""


class _ImageTransformDataset:
    """Small torch-compatible dataset that keeps the model out of workers."""

    def __init__(self, image_paths: Sequence[Path], transform: Any) -> None:
        self.image_paths = tuple(image_paths)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> Any:
        try:
            from PIL import Image
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise PECoreEmbeddingUnavailable(
                "Pillow is required for PE-Core image embedding. "
                "Install preprocess/requirements.txt in the preprocess venv."
            ) from exc

        path = self.image_paths[index]
        try:
            with Image.open(path) as image:
                return self.transform(image.convert("RGB"))
        except Exception as exc:
            raise ValueError(f"Could not read keyframe image: {path}") from exc


class OpenClipPECoreEncoder(VisualEmbeddingEncoder):
    """Lazy full-image PE-Core encoder backed by ``open_clip_torch``."""

    def __init__(self, config: PECoreEmbeddingConfig | None = None) -> None:
        self.config = config or PECoreEmbeddingConfig(enabled=True)
        self._torch = None
        self._model = None
        self._preprocess = None
        self._resolved_device: str | None = None
        self._resolved_model_revision: str | None = None

    @property
    def dimension(self) -> int:
        return self.config.expected_dim

    @property
    def resolved_device(self) -> str | None:
        """Return the resolved device after model initialization, if loaded."""
        return self._resolved_device

    @property
    def cache_fingerprint(self) -> str:
        payload = {
            "encoder": type(self).__name__,
            "model_id": self.config.model_id,
            "model_revision": self.config.model_revision,
            "precision": self.config.precision,
            "autocast": asdict(self.config.autocast),
            "tf32": asdict(self.config.tf32),
            "expected_dim": self.config.expected_dim,
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    @property
    def provenance(self) -> dict[str, Any]:
        return {
            "model_id": self.config.model_id,
            "model_revision": self.config.model_revision,
            "precision": self.config.precision,
            "autocast": asdict(self.config.autocast),
            "tf32": asdict(self.config.tf32),
            "requested_device": self.config.device,
            "resolved_device": self._resolved_device,
            "resolved_model_revision": self._resolved_model_revision,
            "cache_fingerprint": self.cache_fingerprint,
        }

    def embed(self, image_paths: Sequence[Path]) -> np.ndarray:
        if not image_paths:
            return np.empty((0, self.dimension), dtype=np.float32)
        self._ensure_loaded()

        transform = self.image_transform()
        tensors = []
        try:
            from PIL import Image
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise PECoreEmbeddingUnavailable(
                "Pillow is required for PE-Core image embedding. "
                "Install preprocess/requirements.txt in the preprocess venv."
            ) from exc
        for path in image_paths:
            try:
                with Image.open(path) as image:
                    tensors.append(transform(image.convert("RGB")))
            except Exception as exc:
                raise ValueError(f"Could not read keyframe image: {path}") from exc
        return self.embed_prepared_batch(tensors)

    def image_transform(self) -> Any:
        """Return the OpenCLIP preprocessing callable for DataLoader workers."""
        self._ensure_loaded()
        return self._preprocess

    def embed_prepared_batch(self, batch: Any, *, non_blocking: bool = False) -> np.ndarray:
        """Encode tensors produced by :class:`_ImageTransformDataset`."""
        self._ensure_loaded()
        torch = self._torch
        if not hasattr(batch, "to"):
            batch = torch.stack(tuple(batch))
        expected_count = int(batch.shape[0])
        batch = batch.to(self._resolved_device, non_blocking=non_blocking)
        batch = self._input_dtype(batch)
        with torch.inference_mode(), self._tf32_context(), self._autocast_context():
            features = self._model.encode_image(batch, normalize=True)
        vectors = features.detach().float().cpu().numpy()
        return self._normalize_and_validate(vectors, expected_count)

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            import open_clip
            import torch
        except ImportError as exc:
            raise PECoreEmbeddingUnavailable(
                "PE-Core image embedding dependencies are missing. Install "
                "preprocess/requirements.txt, including open_clip_torch."
            ) from exc

        if os.environ.get("PREPROCESS_DETERMINISTIC") == "1":
            torch.use_deterministic_algorithms(True)
            seed = os.environ.get("PREPROCESS_SEED")
            if seed is not None:
                torch.manual_seed(int(seed))

        resolved_device = self._resolve_device(torch)
        self._validate_runtime_capabilities(torch, resolved_device)

        model_id = self._resolve_model_source()
        try:
            model, _, preprocess = open_clip.create_model_and_transforms(
                model_id,
                precision=self.config.precision,
                device=resolved_device,
            )
            model.eval()
        except Exception as exc:
            raise PECoreEmbeddingUnavailable(
                f"Could not load PE-Core image model '{self.config.model_id}'. "
                "Check the model cache/network access and Hugging Face credentials "
                "if the model is not cached."
            ) from exc

        self._torch = torch
        self._model = model
        self._preprocess = preprocess
        self._resolved_device = resolved_device

    def _validate_runtime_capabilities(self, torch, device: str) -> None:
        """Validate dtype features only after the actual device is resolved."""
        device_type = torch.device(device).type
        if device_type == "cuda" and self.config.precision == "bf16":
            if not self._cuda_supports_bfloat16(torch):
                raise PECoreEmbeddingUnavailable(
                    "PECore precision bf16 was requested, but the selected CUDA "
                    "device does not support bfloat16. Use fp16/fp32 or a supported GPU."
                )

        if self.config.tf32.enabled and device_type == "cuda":
            if not self._cuda_supports_tf32(torch):
                raise PECoreEmbeddingUnavailable(
                    "PECore tf32 was requested, but the selected CUDA device does not "
                    "support TF32 (requires compute capability 8.0 or newer). "
                    "Set embedding.tf32.enabled=false or use an Ampere-or-newer GPU."
                )
            cuda_backend = getattr(getattr(torch, "backends", None), "cuda", None)
            matmul_backend = getattr(cuda_backend, "matmul", None)
            cudnn_backend = getattr(getattr(torch, "backends", None), "cudnn", None)
            if self.config.tf32.matmul and not hasattr(matmul_backend, "allow_tf32"):
                raise PECoreEmbeddingUnavailable(
                    "PECore tf32.matmul was requested, but this PyTorch build does not "
                    "expose torch.backends.cuda.matmul.allow_tf32."
                )
            if self.config.tf32.cudnn and not hasattr(cudnn_backend, "allow_tf32"):
                raise PECoreEmbeddingUnavailable(
                    "PECore tf32.cudnn was requested, but this PyTorch build does not "
                    "expose torch.backends.cudnn.allow_tf32."
                )

        if not self.config.autocast.enabled:
            return
        if device_type == "cuda" and self.config.autocast.dtype == "bf16":
            if not self._cuda_supports_bfloat16(torch):
                raise PECoreEmbeddingUnavailable(
                    "PECore autocast dtype bf16 was requested, but the selected CUDA "
                    "device does not support bfloat16. Use autocast dtype fp16."
                )
        if device_type == "cpu" and self.config.autocast.dtype == "fp16":
            raise PECoreEmbeddingUnavailable(
                "CPU autocast supports bfloat16 in this pipeline; use autocast dtype bf16."
            )

    @staticmethod
    def _cuda_supports_bfloat16(torch) -> bool:
        checker = getattr(torch.cuda, "is_bf16_supported", None)
        if callable(checker):
            return bool(checker())
        try:
            major, _ = torch.cuda.get_device_capability()
        except (AttributeError, RuntimeError):
            return False
        return major >= 8

    @staticmethod
    def _cuda_supports_tf32(torch) -> bool:
        try:
            major, _ = torch.cuda.get_device_capability()
        except (AttributeError, RuntimeError):
            return False
        return major >= 8

    @contextmanager
    def _tf32_context(self):
        """Temporarily apply TF32 backend flags for one model inference call."""
        if not self.config.tf32.enabled:
            yield
            return

        torch = self._torch
        if torch is None or self._resolved_device is None or torch.device(self._resolved_device).type != "cuda":
            # TF32 is a CUDA-only optimization; keep CPU/MPS configurations portable.
            yield
            return

        cuda_backend = torch.backends.cuda
        matmul_backend = cuda_backend.matmul
        cudnn_backend = torch.backends.cudnn
        previous_matmul = matmul_backend.allow_tf32
        previous_cudnn = cudnn_backend.allow_tf32
        try:
            matmul_backend.allow_tf32 = self.config.tf32.matmul
            cudnn_backend.allow_tf32 = self.config.tf32.cudnn
            yield
        finally:
            matmul_backend.allow_tf32 = previous_matmul
            cudnn_backend.allow_tf32 = previous_cudnn

    def _autocast_context(self):
        """Return a device-aware AMP context; disabled mode is a no-op."""
        if not self.config.autocast.enabled:
            return nullcontext()
        torch = self._torch
        device_type = torch.device(self._resolved_device).type
        dtype = torch.float16 if self.config.autocast.dtype == "fp16" else torch.bfloat16
        return torch.autocast(
            device_type=device_type,
            dtype=dtype,
            enabled=True,
            cache_enabled=self.config.autocast.cache_enabled,
        )

    def _resolve_model_source(self) -> str:
        """Resolve an optional immutable Hugging Face revision to a local snapshot."""
        revision = self.config.model_revision
        if not revision:
            return self.config.model_id
        if not self.config.model_id.startswith("hf-hub:"):
            raise PECoreEmbeddingUnavailable(
                "embedding.model_revision is supported only for hf-hub model IDs"
            )
        try:
            from huggingface_hub import snapshot_download

            snapshot = Path(
                snapshot_download(
                    repo_id=self.config.model_id.removeprefix("hf-hub:"),
                    revision=revision,
                )
            )
        except Exception as exc:
            raise PECoreEmbeddingUnavailable(
                f"Could not resolve PE-Core revision {revision!r} for "
                f"{self.config.model_id}; check Hugging Face access/cache"
            ) from exc
        self._resolved_model_revision = snapshot.name
        return f"local-dir:{snapshot}"

    def _resolve_device(self, torch) -> str:
        requested = self.config.device
        if requested == "auto":
            if torch.cuda.is_available():
                return "cuda"
            mps = getattr(getattr(torch, "backends", None), "mps", None)
            if mps is not None and mps.is_available():
                return "mps"
            return "cpu"
        if requested == "cuda" and not torch.cuda.is_available():
            raise PECoreEmbeddingUnavailable("PECore device cuda was requested but CUDA is unavailable.")
        if requested == "mps":
            mps = getattr(getattr(torch, "backends", None), "mps", None)
            if mps is None or not mps.is_available():
                raise PECoreEmbeddingUnavailable("PECore device mps was requested but MPS is unavailable.")
            if self.config.precision != "fp32":
                raise PECoreEmbeddingUnavailable("PECore device mps requires precision fp32.")
        return requested

    def _input_dtype(self, batch):
        torch = self._torch
        try:
            model_dtype = next(self._model.parameters()).dtype
        except StopIteration:
            model_dtype = torch.float32
        if model_dtype != torch.float32:
            batch = batch.to(dtype=model_dtype)
        return batch

    def _normalize_and_validate(self, vectors: np.ndarray, expected_count: int) -> np.ndarray:
        if vectors.ndim != 2 or vectors.shape != (expected_count, self.dimension):
            raise ValueError(
                "PE-Core image encoder returned shape "
                f"{tuple(vectors.shape)}; expected {(expected_count, self.dimension)}"
            )
        if not np.isfinite(vectors).all():
            raise ValueError("PE-Core image encoder returned a non-finite vector")
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        if np.any(norms <= 0):
            raise ValueError("PE-Core image encoder returned a zero-norm vector")
        normalized = (vectors / norms).astype(np.float32, copy=False)
        return normalized


class NpyFeatureWriter:
    """Atomically write the sample-compatible one-vector-per-frame format."""

    def __init__(self, expected_dim: int) -> None:
        if expected_dim <= 0:
            raise ValueError("expected_dim must be positive")
        self.expected_dim = expected_dim

    def target_for(self, output_dir: Path, frame_id: str) -> Path:
        if not frame_id or frame_id in {".", ".."} or Path(frame_id).name != frame_id:
            raise ValueError(f"Unsafe frame_id: {frame_id!r}")
        return output_dir / f"{frame_id}.npy"

    @staticmethod
    def metadata_path(feature_path: Path) -> Path:
        return feature_path.with_suffix(".npy.meta.json")

    def can_skip(self, path: Path, *, metadata: Mapping[str, Any] | None = None) -> bool:
        """Return true only for an existing, valid sample-compatible vector."""
        if not path.is_file():
            return False
        self.validate_file(path)
        if metadata is None:
            return True
        metadata_path = self.metadata_path(path)
        if not metadata_path.is_file():
            return False
        try:
            actual = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        return actual == dict(metadata)

    def write(self, output_dir: Path, frame_id: str, vector: np.ndarray, *, overwrite: bool) -> tuple[Path, bool]:
        target = self.target_for(output_dir, frame_id)
        output_dir.mkdir(parents=True, exist_ok=True)
        if target.exists() and not overwrite:
            try:
                self.validate_file(target)
            except ValueError as exc:
                raise FileExistsError(
                    f"Existing feature is invalid: {target}; use embedding.overwrite=true "
                    "or --overwrite to replace it"
                ) from exc
            return target, False

        value = self.validate_array(vector)
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=output_dir,
                prefix=f".{target.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temporary_path = Path(handle.name)
                np.save(handle, value, allow_pickle=False)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_path, target)
            temporary_path = None
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
        return target, True

    def write_metadata(self, path: Path, metadata: Mapping[str, Any]) -> None:
        atomic_json_write(self.metadata_path(path), metadata)

    def validate_file(self, path: Path) -> np.ndarray:
        try:
            value = np.load(path, allow_pickle=False)
        except Exception as exc:
            raise ValueError(f"Could not load feature file: {path}") from exc
        return self.validate_array(value)

    def validate_array(self, value: np.ndarray) -> np.ndarray:
        array = np.asarray(value)
        if array.shape != (self.expected_dim,):
            raise ValueError(
                f"Feature shape {tuple(array.shape)} does not match {(self.expected_dim,)}"
            )
        if array.dtype != np.float32:
            raise ValueError(f"Feature dtype {array.dtype} does not match float32")
        if not np.isfinite(array).all():
            raise ValueError("Feature contains non-finite values")
        norm = float(np.linalg.norm(array))
        if norm <= 0:
            raise ValueError("Feature has zero L2 norm")
        if not np.isclose(norm, 1.0, rtol=1e-3, atol=1e-3):
            raise ValueError(f"Feature L2 norm {norm} is not close to 1.0")
        return array


@dataclass(frozen=True)
class EmbeddingVideoResult:
    video_id: str
    source_dir: Path
    output_dir: Path
    image_count: int
    embedded_count: int
    skipped_count: int
    dimension: int
    feature_files: tuple[Path, ...]
    cache_fingerprint: str | None = None

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["source_dir"] = str(self.source_dir)
        value["output_dir"] = str(self.output_dir)
        value["feature_files"] = [str(path) for path in self.feature_files]
        return value


@dataclass(frozen=True)
class EmbeddingBatchResult:
    videos: tuple[EmbeddingVideoResult, ...]
    dimension: int
    cache_fingerprint: str | None = None
    encoder_provenance: Mapping[str, Any] = field(default_factory=dict)

    @property
    def image_count(self) -> int:
        return sum(result.image_count for result in self.videos)

    @property
    def embedded_count(self) -> int:
        return sum(result.embedded_count for result in self.videos)

    @property
    def skipped_count(self) -> int:
        return sum(result.skipped_count for result in self.videos)

    def to_dict(self) -> dict[str, object]:
        return {
            "dimension": self.dimension,
            "image_count": self.image_count,
            "embedded_count": self.embedded_count,
            "skipped_count": self.skipped_count,
            "videos": [result.to_dict() for result in self.videos],
            "cache_fingerprint": self.cache_fingerprint,
            "encoder_provenance": dict(self.encoder_provenance),
        }


class PECoreEmbeddingPipeline:
    """Embed keyframe directories with an injectable visual encoder."""

    def __init__(
        self,
        encoder: VisualEmbeddingEncoder,
        *,
        batch_size: int = 8,
        image_extensions: Sequence[str] = (".jpg", ".jpeg", ".png"),
        dataloader: EmbeddingDataLoaderConfig | Mapping[str, Any] | None = None,
        overwrite: bool = False,
        writer: NpyFeatureWriter | None = None,
        progress: ProgressReporter | None = None,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        extensions = tuple(
            extension.lower() if extension.startswith(".") else f".{extension.lower()}"
            for extension in image_extensions
        )
        if not extensions:
            raise ValueError("image_extensions must not be empty")
        self.encoder = encoder
        self.batch_size = batch_size
        if dataloader is None:
            self.dataloader = EmbeddingDataLoaderConfig()
        elif isinstance(dataloader, Mapping):
            self.dataloader = EmbeddingDataLoaderConfig(**dict(dataloader))
        elif isinstance(dataloader, EmbeddingDataLoaderConfig):
            self.dataloader = dataloader
        else:
            raise TypeError("dataloader must be an EmbeddingDataLoaderConfig or mapping")
        self.image_extensions = frozenset(extensions)
        self.overwrite = overwrite
        self.writer = writer or NpyFeatureWriter(encoder.dimension)
        self.cache_fingerprint = str(
            getattr(
                encoder,
                "cache_fingerprint",
                f"{type(encoder).__module__}.{type(encoder).__qualname__}:{encoder.dimension}",
            )
        )
        self.progress = progress or TqdmProgressReporter(ProgressConfig())
        if self.writer.expected_dim != encoder.dimension:
            raise ValueError("Feature writer dimension must match encoder dimension")

    def embed_video(
        self,
        video_id: str,
        keyframe_dir: Path,
        output_root: Path,
    ) -> EmbeddingVideoResult:
        self._validate_id(video_id, "video_id")
        if not keyframe_dir.is_dir():
            raise FileNotFoundError(f"Keyframe directory not found: {keyframe_dir}")
        image_paths = self._discover_images(keyframe_dir)
        if not image_paths:
            raise ValueError(f"No keyframe images found in: {keyframe_dir}")

        output_dir = output_root / video_id
        output_dir.mkdir(parents=True, exist_ok=True)
        pending: list[tuple[Path, Path]] = []
        feature_files: list[Path] = []
        skipped_count = 0
        expected_feature_names = {image_path.stem for image_path in image_paths}
        for stale_path in output_dir.glob("*.npy"):
            if stale_path.stem not in expected_feature_names:
                stale_path.unlink(missing_ok=True)
                self.writer.metadata_path(stale_path).unlink(missing_ok=True)
        for stale_metadata in output_dir.glob("*.npy.meta.json"):
            frame_name = stale_metadata.name.removesuffix(".npy.meta.json")
            if frame_name not in expected_feature_names:
                stale_metadata.unlink(missing_ok=True)
        for image_path in image_paths:
            target = self.writer.target_for(output_dir, image_path.stem)
            expected_metadata = {
                "schema_version": 1,
                "image_sha256": sha256_file(image_path),
                "cache_fingerprint": self.cache_fingerprint,
            }
            if not self.overwrite and target.exists():
                try:
                    self.writer.validate_file(target)
                except ValueError as exc:
                    raise FileExistsError(
                        f"Existing feature is invalid: {target}; use overwrite to replace it"
                    ) from exc
                if self.writer.can_skip(target, metadata=expected_metadata):
                    skipped_count += 1
                    feature_files.append(target)
                else:
                    target.unlink()
                    self.writer.metadata_path(target).unlink(missing_ok=True)
                    pending.append((image_path, target))
            else:
                pending.append((image_path, target))

        embedded_count = 0
        if pending:
            for start, vectors in self._embed_pending(video_id, pending):
                chunk = pending[start : start + self.batch_size]
                if vectors.shape != (len(chunk), self.encoder.dimension):
                    raise ValueError(
                        "Embedding strategy returned shape "
                        f"{tuple(vectors.shape)}; expected {(len(chunk), self.encoder.dimension)}"
                    )
                for (image_path, _), vector in zip(chunk, vectors, strict=True):
                    target, written = self.writer.write(
                        output_dir,
                        image_path.stem,
                        vector,
                        overwrite=self.overwrite,
                    )
                    feature_files.append(target)
                    self.writer.write_metadata(
                        target,
                        {
                            "schema_version": 1,
                            "image_sha256": sha256_file(image_path),
                            "cache_fingerprint": self.cache_fingerprint,
                        },
                    )
                    if written:
                        embedded_count += 1

        return EmbeddingVideoResult(
            video_id=video_id,
            source_dir=keyframe_dir,
            output_dir=output_dir,
            image_count=len(image_paths),
            embedded_count=embedded_count,
            skipped_count=skipped_count,
            dimension=self.encoder.dimension,
            feature_files=tuple(sorted(feature_files)),
            cache_fingerprint=self.cache_fingerprint,
        )

    def _embed_pending(
        self,
        video_id: str,
        pending: Sequence[tuple[Path, Path]],
    ):
        if self._supports_prepared_batches():
            yield from self._embed_pending_with_dataloader(video_id, pending)
            return

        batch_starts = range(0, len(pending), self.batch_size)
        batch_count = (len(pending) + self.batch_size - 1) // self.batch_size
        for start in self.progress.iterate(
            batch_starts,
            total=batch_count,
            desc=f"{video_id}: embedding",
            unit="batch",
        ):
            chunk = pending[start : start + self.batch_size]
            yield start, np.asarray(self.encoder.embed([image for image, _ in chunk]))

    def _embed_pending_with_dataloader(
        self,
        video_id: str,
        pending: Sequence[tuple[Path, Path]],
    ):
        try:
            from torch.utils.data import DataLoader
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise PECoreEmbeddingUnavailable(
                "PyTorch is required for the configured embedding DataLoader. "
                "Install preprocess/requirements.txt in the preprocess venv."
            ) from exc

        image_paths = [image for image, _ in pending]
        transform = getattr(self.encoder, "image_transform")()
        dataset = _ImageTransformDataset(image_paths, transform)
        loader_kwargs: dict[str, Any] = {
            "batch_size": self.batch_size,
            "shuffle": False,
            "num_workers": self.dataloader.num_workers,
            "pin_memory": self.dataloader.pin_memory,
        }
        if self.dataloader.num_workers > 0:
            loader_kwargs["persistent_workers"] = self.dataloader.persistent_workers
            loader_kwargs["prefetch_factor"] = self.dataloader.prefetch_factor
        loader = DataLoader(dataset, **loader_kwargs)
        embed_batch = getattr(self.encoder, "embed_prepared_batch")
        for batch_index, prepared_batch in self.progress.iterate(
            enumerate(loader),
            total=len(loader),
            desc=f"{video_id}: embedding",
            unit="batch",
        ):
            start = batch_index * self.batch_size
            vectors = np.asarray(
                embed_batch(
                    prepared_batch,
                    non_blocking=self.dataloader.pin_memory,
                )
            )
            yield start, vectors

    def _supports_prepared_batches(self) -> bool:
        return callable(getattr(self.encoder, "image_transform", None)) and callable(
            getattr(self.encoder, "embed_prepared_batch", None)
        )

    def embed_all(
        self,
        keyframes_root: Path,
        output_root: Path,
        *,
        video_ids: Sequence[str] | None = None,
    ) -> EmbeddingBatchResult:
        if not keyframes_root.is_dir():
            raise FileNotFoundError(f"Keyframes root not found: {keyframes_root}")
        selected_ids = list(video_ids) if video_ids is not None else [
            path.name for path in sorted(keyframes_root.iterdir()) if path.is_dir()
        ]
        if not selected_ids:
            raise ValueError(f"No video directories found in: {keyframes_root}")
        if len(set(selected_ids)) != len(selected_ids):
            raise ValueError("video_ids must not contain duplicates")
        results_list: list[EmbeddingVideoResult] = []
        for video_id in self.progress.iterate(
            selected_ids,
            total=len(selected_ids),
            desc="PECore videos",
            unit="video",
        ):
            results_list.append(self.embed_video(video_id, keyframes_root / video_id, output_root))
        results = tuple(results_list)
        return EmbeddingBatchResult(
            videos=results,
            dimension=self.encoder.dimension,
            cache_fingerprint=self.cache_fingerprint,
            encoder_provenance=dict(getattr(self.encoder, "provenance", {})),
        )

    def _discover_images(self, keyframe_dir: Path) -> list[Path]:
        images = [
            path
            for path in sorted(keyframe_dir.iterdir())
            if path.is_file() and path.suffix.lower() in self.image_extensions
        ]
        stems = [path.stem for path in images]
        if len(set(stems)) != len(stems):
            raise ValueError(f"Duplicate frame stems in keyframe directory: {keyframe_dir}")
        if any(not stem or Path(stem).name != stem for stem in stems):
            raise ValueError(f"Invalid frame filename in keyframe directory: {keyframe_dir}")
        return images

    @staticmethod
    def _validate_id(value: str, field_name: str) -> None:
        if not value or value in {".", ".."} or Path(value).name != value:
            raise ValueError(f"Unsafe {field_name}: {value!r}")
