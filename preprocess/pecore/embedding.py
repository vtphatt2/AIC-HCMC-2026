"""Strategy-friendly PE-Core image embedding and feature persistence.

The module does not load PE-Core at import time.  This is intentional: the
model is large, and commands that only inspect or download a batch must still
work without downloading model weights.  ``OpenClipPECoreEncoder`` loads the
full image-capable OpenCLIP model on its first non-empty encode call.
"""
from __future__ import annotations

import os
import tempfile
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from preprocess.progress import ProgressConfig, ProgressReporter, TqdmProgressReporter


class PECoreEmbeddingUnavailable(RuntimeError):
    """Raised when the PE-Core visual encoder cannot be initialized."""


def _validate_safe_directory_name(value: str, field_name: str) -> None:
    if not value or value in {".", ".."} or Path(value).name != value:
        raise ValueError(f"{field_name} must be one safe directory name")


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
    device: str = "auto"
    precision: str = "fp32"
    expected_dim: int = 1_280
    batch_size: int = 8
    overwrite: bool = False
    features_dir_name: str = "PECore-features"
    image_extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png")

    def __post_init__(self) -> None:
        model_id = self.model_id.strip()
        device = self.device.strip().lower()
        precision = self.precision.strip().lower()
        extensions = tuple(
            extension.lower() if extension.startswith(".") else f".{extension.lower()}"
            for extension in self.image_extensions
        )
        object.__setattr__(self, "model_id", model_id)
        object.__setattr__(self, "device", device)
        object.__setattr__(self, "precision", precision)
        object.__setattr__(self, "image_extensions", extensions)

        if not model_id:
            raise ValueError("model_id must not be empty")
        if device not in {"auto", "cpu", "cuda", "mps"}:
            raise ValueError("device must be one of: auto, cpu, cuda, mps")
        if precision not in {"fp32", "fp16"}:
            raise ValueError("precision must be one of: fp32, fp16")
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


class OpenClipPECoreEncoder(VisualEmbeddingEncoder):
    """Lazy full-image PE-Core encoder backed by ``open_clip_torch``."""

    def __init__(self, config: PECoreEmbeddingConfig | None = None) -> None:
        self.config = config or PECoreEmbeddingConfig(enabled=True)
        self._torch = None
        self._model = None
        self._preprocess = None
        self._resolved_device: str | None = None

    @property
    def dimension(self) -> int:
        return self.config.expected_dim

    @property
    def resolved_device(self) -> str | None:
        """Return the resolved device after model initialization, if loaded."""
        return self._resolved_device

    def embed(self, image_paths: Sequence[Path]) -> np.ndarray:
        if not image_paths:
            return np.empty((0, self.dimension), dtype=np.float32)
        self._ensure_loaded()

        try:
            from PIL import Image
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise PECoreEmbeddingUnavailable(
                "Pillow is required for PE-Core image embedding. "
                "Install preprocess/requirements.txt in the preprocess venv."
            ) from exc

        tensors = []
        for path in image_paths:
            try:
                with Image.open(path) as image:
                    tensors.append(self._preprocess(image.convert("RGB")))
            except Exception as exc:
                raise ValueError(f"Could not read keyframe image: {path}") from exc

        torch = self._torch
        batch = torch.stack(tensors).to(self._resolved_device)
        batch = self._input_dtype(batch)
        with torch.inference_mode():
            features = self._model.encode_image(batch, normalize=True)
        vectors = features.detach().float().cpu().numpy()
        return self._normalize_and_validate(vectors, len(image_paths))

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

        resolved_device = self._resolve_device(torch)
        if resolved_device == "cpu" and self.config.precision == "fp16":
            raise PECoreEmbeddingUnavailable(
                "PECore precision fp16 requires CUDA; use --precision fp32 on CPU."
            )

        try:
            model, _, preprocess = open_clip.create_model_and_transforms(
                self.config.model_id,
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

    def can_skip(self, path: Path) -> bool:
        """Return true only for an existing, valid sample-compatible vector."""
        if not path.is_file():
            return False
        self.validate_file(path)
        return True

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
        }


class PECoreEmbeddingPipeline:
    """Embed keyframe directories with an injectable visual encoder."""

    def __init__(
        self,
        encoder: VisualEmbeddingEncoder,
        *,
        batch_size: int = 8,
        image_extensions: Sequence[str] = (".jpg", ".jpeg", ".png"),
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
        self.image_extensions = frozenset(extensions)
        self.overwrite = overwrite
        self.writer = writer or NpyFeatureWriter(encoder.dimension)
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
        for image_path in image_paths:
            target = self.writer.target_for(output_dir, image_path.stem)
            if not self.overwrite and target.exists():
                try:
                    valid_existing = self.writer.can_skip(target)
                except ValueError as exc:
                    raise FileExistsError(
                        f"Existing feature is invalid: {target}; use overwrite to replace it"
                    ) from exc
                if not valid_existing:
                    raise FileExistsError(
                        f"Existing feature is invalid: {target}; use overwrite to replace it"
                    )
                skipped_count += 1
                feature_files.append(target)
            else:
                pending.append((image_path, target))

        embedded_count = 0
        if pending:
            batch_starts = range(0, len(pending), self.batch_size)
            batch_count = (len(pending) + self.batch_size - 1) // self.batch_size
            for start in self.progress.iterate(
                batch_starts,
                total=batch_count,
                desc=f"{video_id}: embedding",
                unit="batch",
            ):
                chunk = pending[start : start + self.batch_size]
                vectors = np.asarray(self.encoder.embed([image for image, _ in chunk]))
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
        return EmbeddingBatchResult(videos=results, dimension=self.encoder.dimension)

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
