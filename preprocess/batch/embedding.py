"""Embedding strategies that connect visual encoders to a lot layout."""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Sequence

from preprocess.batch.layout import LotLayout
from preprocess.batch.models import ProcessingResult, VideoAsset
from preprocess.pecore.embedding import (
    EmbeddingBatchResult,
    OpenClipPECoreEncoder,
    PECoreEmbeddingConfig,
    PECoreEmbeddingPipeline,
    VisualEmbeddingEncoder,
)
from preprocess.progress import ProgressReporter


class BatchEmbeddingStrategy(ABC):
    """Replaceable boundary for validated keyframes to feature artifacts."""

    @abstractmethod
    def embed(
        self,
        layout: LotLayout,
        assets: Sequence[VideoAsset],
        results: Sequence[ProcessingResult],
    ) -> EmbeddingBatchResult:
        raise NotImplementedError


class PECoreEmbeddingStrategy(BatchEmbeddingStrategy):
    """Embed the rendered keyframes of a lot with full PE-Core."""

    def __init__(
        self,
        encoder: VisualEmbeddingEncoder,
        config: PECoreEmbeddingConfig,
        *,
        input_profile_id: str,
        progress: ProgressReporter | None = None,
    ) -> None:
        if not input_profile_id or input_profile_id in {".", ".."} or Path(input_profile_id).name != input_profile_id:
            raise ValueError(f"Unsafe input profile id: {input_profile_id!r}")
        self.config = config
        self.input_profile_id = input_profile_id
        self.pipeline = PECoreEmbeddingPipeline(
            encoder,
            batch_size=config.batch_size,
            image_extensions=config.image_extensions,
            overwrite=config.overwrite,
            progress=progress,
        )

    def embed(
        self,
        layout: LotLayout,
        assets: Sequence[VideoAsset],
        results: Sequence[ProcessingResult],
    ) -> EmbeddingBatchResult:
        if len(assets) != len(results):
            raise ValueError("Embedding requires one processing result per video asset")
        video_ids = [asset.video_id for asset in assets]
        return self.pipeline.embed_all(
            layout.dataset_dir / self.input_profile_id,
            layout.dataset_dir / self.config.features_dir_name,
            video_ids=video_ids,
        )


def default_pecore_embedding_strategy(
    config: PECoreEmbeddingConfig,
    *,
    input_profile_id: str,
    progress: ProgressReporter | None = None,
) -> PECoreEmbeddingStrategy:
    """Build the production PE-Core strategy without loading model weights."""
    return PECoreEmbeddingStrategy(
        OpenClipPECoreEncoder(config),
        config,
        input_profile_id=input_profile_id,
        progress=progress,
    )
