"""Strategy-driven video processing built on the existing keyframe pipeline."""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Mapping, Sequence

from preprocess.batch.config import LinearSelectionConfig, ProcessingConfig, ToolConfig
from preprocess.batch.layout import LotLayout
from preprocess.batch.models import ProcessingResult, VideoAsset
from preprocess.keyframes.contracts import (
    ExtractionOutput,
    KeyframeSelector,
    RenderProfile,
    SceneSegment,
    VideoSource,
)
from preprocess.keyframes.extractors.ffmpeg import FFmpegKeyframeExtractor
from preprocess.keyframes.manifest import write_selection_manifest
from preprocess.keyframes.selectors.linear_rulebase import LinearRuleBasedSelector
from preprocess.keyframes.selectors.scene_segments import SceneSegmentSelector
from preprocess.keyframes.selectors.uniform import UniformIntervalSelector
from preprocess.progress import ProgressReporter


class SelectionStrategy(ABC):
    """Factory boundary for interchangeable frame-selection policies."""

    name: str

    @abstractmethod
    def selector_for(self, video_id: str) -> KeyframeSelector:
        raise NotImplementedError


class UniformSelectionStrategy(SelectionStrategy):
    name = "uniform"

    def __init__(self, interval_ms: int) -> None:
        self.interval_ms = interval_ms

    def selector_for(self, video_id: str) -> KeyframeSelector:
        return UniformIntervalSelector(interval_ms=self.interval_ms)


def _parse_seconds_to_ms(value: object, *, path: Path) -> int:
    try:
        return round(float(value) * 1000)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid scene timestamp in {path}: {value!r}") from exc


def _load_scene_segments(path: Path) -> list[SceneSegment]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records: object = payload.get("segments") if isinstance(payload, dict) else payload
    if not isinstance(records, list):
        raise ValueError(f"Scene-segment manifest must contain a list: {path}")

    segments: list[SceneSegment] = []
    for item in records:
        if not isinstance(item, Mapping):
            raise ValueError(f"Scene segment must be an object: {path}")
        if "start_ms" in item and "end_ms" in item:
            start_ms = int(item["start_ms"])
            end_ms = int(item["end_ms"])
        elif "start_time" in item and "end_time" in item:
            # This is the JSON shape emitted by transnetv2-pytorch.
            start_ms = _parse_seconds_to_ms(item["start_time"], path=path)
            end_ms = _parse_seconds_to_ms(item["end_time"], path=path)
        else:
            raise ValueError(
                f"Scene segment needs start_ms/end_ms or start_time/end_time: {path}"
            )
        segments.append(SceneSegment(start_ms=start_ms, end_ms=end_ms))
    return segments


class SceneSegmentsSelectionStrategy(SelectionStrategy):
    name = "scene-segments"

    def __init__(self, segments_dir: Path) -> None:
        self.segments_dir = segments_dir

    def selector_for(self, video_id: str) -> KeyframeSelector:
        path = self.segments_dir / f"{video_id}.json"
        if not path.is_file():
            raise FileNotFoundError(f"Scene-segment manifest not found: {path}")
        return SceneSegmentSelector(_load_scene_segments(path))


class LinearRuleBasedSelectionStrategy(SelectionStrategy):
    """Use a configurable duration-to-count rule for detector-provided shots."""

    name = "linear-rulebase"

    def __init__(self, segments_dir: Path, rule: LinearSelectionConfig) -> None:
        self.segments_dir = segments_dir
        self.rule = rule

    def selector_for(self, video_id: str) -> KeyframeSelector:
        path = self.segments_dir / f"{video_id}.json"
        if not path.is_file():
            raise FileNotFoundError(f"Scene-segment manifest not found: {path}")
        return LinearRuleBasedSelector(
            _load_scene_segments(path),
            frame_count_for=self.rule.frame_count,
            rule_config=asdict(self.rule),
        )


class SelectionStrategyRegistry:
    """Named selection strategy registry; custom strategies can be registered."""

    def __init__(self) -> None:
        self._factories: dict[str, Callable[[ProcessingConfig], SelectionStrategy]] = {}

    def register(self, name: str, factory: Callable[[ProcessingConfig], SelectionStrategy]) -> None:
        if not name.strip():
            raise ValueError("selection strategy name must not be empty")
        self._factories[name] = factory

    def create(self, name: str, config: ProcessingConfig) -> SelectionStrategy:
        try:
            factory = self._factories[name]
        except KeyError as exc:
            raise ValueError(f"Unknown selection strategy: {name}") from exc
        return factory(config)


def default_selection_registry() -> SelectionStrategyRegistry:
    registry = SelectionStrategyRegistry()
    registry.register("uniform", lambda config: UniformSelectionStrategy(config.interval_ms))

    def scene_segments_factory(config: ProcessingConfig) -> SelectionStrategy:
        if config.scene_segments_dir is None:
            raise ValueError("processing.scene_segments_dir is required for scene-segments")
        return SceneSegmentsSelectionStrategy(config.scene_segments_dir)

    registry.register("scene-segments", scene_segments_factory)

    def linear_rulebase_factory(config: ProcessingConfig) -> SelectionStrategy:
        if config.scene_segments_dir is None:
            raise ValueError("processing.scene_segments_dir is required for linear-rulebase")
        return LinearRuleBasedSelectionStrategy(config.scene_segments_dir, config.linear_rule)

    registry.register("linear-rulebase", linear_rulebase_factory)
    registry.register("linear", linear_rulebase_factory)
    return registry


class VideoProcessingStrategy(ABC):
    """Replaceable boundary for video-to-artifact processing."""

    @abstractmethod
    def process(self, asset: VideoAsset, layout: LotLayout) -> ProcessingResult:
        raise NotImplementedError


class KeyframeProcessingStrategy(VideoProcessingStrategy):
    """Run a selector plus an injected FFmpeg/Pillow keyframe extractor."""

    def __init__(
        self,
        selector_strategy: SelectionStrategy,
        *,
        extractor: FFmpegKeyframeExtractor,
        profile: RenderProfile,
        overwrite: bool = False,
    ) -> None:
        self.selector_strategy = selector_strategy
        self.extractor = extractor
        self.profile = profile
        self.overwrite = overwrite

    def process(self, asset: VideoAsset, layout: LotLayout) -> ProcessingResult:
        source = VideoSource(video_id=asset.video_id, path=asset.path)
        video_info = self.extractor.probe(source)
        selector = self.selector_strategy.selector_for(asset.video_id)
        selected = selector.select(video_info, self.extractor.scan(source, video_info))

        selection_manifest = layout.dataset_dir / "selection-manifests" / f"{asset.video_id}.json"
        write_selection_manifest(
            selection_manifest,
            source,
            video_info,
            selector.name,
            selector.version,
            {"strategy": self.selector_strategy.name},
            selected,
        )
        result = self.extractor.materialize(
            source,
            video_info,
            selected,
            ExtractionOutput(
                rendered_root=layout.dataset_dir,
                selection_manifest_path=selection_manifest,
                profile=self.profile,
                overwrite=self.overwrite,
            ),
        )
        return ProcessingResult(
            asset=asset,
            video_info=asdict(video_info),
            selected_count=len(selected),
            selection_manifest_path=selection_manifest,
            rendered_manifest_path=result.rendered_manifest_path,
        )


class ProcessingStrategyRegistry:
    """Named registry for complete video-processing strategies."""

    def __init__(self) -> None:
        self._factories: dict[str, Callable[..., VideoProcessingStrategy]] = {}

    def register(self, name: str, factory: Callable[..., VideoProcessingStrategy]) -> None:
        if not name.strip():
            raise ValueError("processing strategy name must not be empty")
        self._factories[name] = factory

    def create(self, name: str, **dependencies: object) -> VideoProcessingStrategy:
        try:
            factory = self._factories[name]
        except KeyError as exc:
            raise ValueError(f"Unknown processing strategy: {name}") from exc
        return factory(**dependencies)


def default_processing_registry() -> ProcessingStrategyRegistry:
    registry = ProcessingStrategyRegistry()

    def keyframe_factory(
        *,
        selector_strategy: SelectionStrategy,
        tools: ToolConfig,
        processing_config: ProcessingConfig,
        progress: ProgressReporter | None = None,
    ) -> VideoProcessingStrategy:
        return KeyframeProcessingStrategy(
            selector_strategy,
            extractor=FFmpegKeyframeExtractor(tools.ffmpeg, tools.ffprobe, progress=progress),
            profile=processing_config.render_profile(),
            overwrite=processing_config.overwrite,
        )

    registry.register("keyframes", keyframe_factory)
    return registry


class BatchProcessor:
    """Apply one injected processing strategy to all discovered videos."""

    def __init__(self, strategy: VideoProcessingStrategy) -> None:
        self.strategy = strategy

    def process(self, assets: Sequence[VideoAsset], layout: LotLayout) -> list[ProcessingResult]:
        return [self.strategy.process(asset, layout) for asset in assets]
