"""FFmpeg-backed extractor with PTS-based frame inventory."""
from __future__ import annotations

import bisect
import json
import re
import shutil
import subprocess
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, Sequence

from PIL import Image

from preprocess.keyframes.contracts import (
    ExtractionOutput,
    ExtractionResult,
    ExtractedFrame,
    FrameCandidate,
    FrameRef,
    KeyframeExtractor,
    SelectedFrame,
    VideoInfo,
    VideoSource,
)
from preprocess.keyframes.manifest import write_rendered_manifest
from preprocess.keyframes.rendering import render_image
from preprocess.progress import ProgressConfig, ProgressReporter, TqdmProgressReporter


class FFmpegKeyframeExtractor(KeyframeExtractor):
    """Use installed ``ffprobe``/``ffmpeg`` binaries without Python decoder deps."""

    name = "ffmpeg"
    version = "1"
    _TIMESTAMP_MATCH_TOLERANCE_SECONDS = 1e-4
    _SHOWINFO_FRAME_RE = re.compile(
        r"Parsed_showinfo_\d+.*?\bn:\s*(?P<frame>\d+)"
        r"\s+pts:.*?\bpts_time:(?P<timestamp>\S+)"
    )

    def __init__(
        self,
        ffmpeg_bin: str = "ffmpeg",
        ffprobe_bin: str = "ffprobe",
        *,
        progress: ProgressReporter | None = None,
    ) -> None:
        self.ffmpeg_bin = ffmpeg_bin
        self.ffprobe_bin = ffprobe_bin
        self.progress = progress or TqdmProgressReporter(ProgressConfig())

    def _run(self, command: list[str]) -> subprocess.CompletedProcess[str]:
        try:
            return subprocess.run(command, check=True, text=True, capture_output=True)
        except FileNotFoundError as exc:
            raise RuntimeError(f"Required executable was not found: {command[0]}") from exc
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(exc.stderr.strip() or f"Command failed: {' '.join(command)}") from exc

    def probe(self, source: VideoSource) -> VideoInfo:
        if not source.path.is_file():
            raise FileNotFoundError(f"Video file not found: {source.path}")
        result = self._run([
            self.ffprobe_bin, "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=codec_name,width,height,avg_frame_rate,nb_frames:format=duration",
            "-of", "json", str(source.path),
        ])
        payload = json.loads(result.stdout)
        streams = payload.get("streams", [])
        if not streams:
            raise RuntimeError(f"No video stream found in {source.path}")
        stream = streams[0]
        numerator, _, denominator = str(stream.get("avg_frame_rate", "0/0")).partition("/")
        fps = float(numerator) / float(denominator) if denominator not in {"", "0"} else None
        frame_count = stream.get("nb_frames")
        return VideoInfo(
            video_id=source.video_id,
            duration_ms=round(float(payload["format"]["duration"]) * 1000),
            fps=fps,
            width=int(stream["width"]),
            height=int(stream["height"]),
            frame_count=int(frame_count) if frame_count and frame_count != "N/A" else None,
            codec=stream.get("codec_name"),
        )

    def scan(self, source: VideoSource, video: VideoInfo) -> Iterable[FrameCandidate]:
        """Yield the same decoder frame indexes used by FFmpeg materialization.

        Stream metadata such as ``nb_frames`` and ffprobe's frame inventory can
        disagree on VFR or damaged files.  ``showinfo`` gives both the filter
        ``n`` value and PTS from the decoder that later evaluates ``select``.
        """
        del video  # The authoritative FFmpeg timeline supplies its own count.
        timeline = self._authoritative_decoder_timeline(source)
        for frame_number, timestamp_seconds in self.progress.iterate(
            timeline,
            total=len(timeline),
            desc=f"{source.video_id}: scan",
            unit="frame",
        ):
            yield FrameCandidate(
                ref=FrameRef(
                    video_id=source.video_id,
                    source_frame_number=frame_number,
                    timestamp_ms=round(timestamp_seconds * 1000),
                    pts_time_seconds=timestamp_seconds,
                )
            )

    def _render_selected_frame_numbers(
        self,
        source: VideoSource,
        frame_numbers: Sequence[int],
        destination_dir: Path,
    ) -> list[Path]:
        """Decode selected FFmpeg frame indexes into a fresh temporary folder."""
        if len(set(frame_numbers)) != len(frame_numbers):
            raise ValueError("FFmpeg frame indexes must be unique")
        destination_dir.mkdir(parents=True, exist_ok=True)
        pattern = destination_dir / "decoded-%06d.png"
        expression = "+".join(f"eq(n\\,{number})" for number in frame_numbers)
        # ``-map`` makes this use exactly the v:0 stream inspected by ffprobe.
        # ``-vsync 0`` prevents the output muxer from duplicating selected frames.
        self._run([
            self.ffmpeg_bin,
            "-hide_banner",
            "-v",
            "error",
            "-i",
            str(source.path),
            "-map",
            "0:v:0",
            "-an",
            "-vf",
            f"select={expression}",
            "-vsync",
            "0",
            str(pattern),
        ])
        return sorted(destination_dir.glob("decoded-*.png"))

    def _authoritative_decoder_timeline(
        self,
        source: VideoSource,
    ) -> list[tuple[int, float]]:
        """Read frame indexes and PTS from the same FFmpeg decoder used to render.

        ``scan`` intentionally remains ffprobe-based because it is lightweight.
        For damaged/VFR inputs, however, ffprobe's frame ordinal can differ from
        FFmpeg's filter ``n``.  ``showinfo`` gives us the latter without changing
        the public frame references stored in manifests.
        """
        result = self._run([
            self.ffmpeg_bin,
            "-hide_banner",
            "-v",
            "info",
            "-i",
            str(source.path),
            "-map",
            "0:v:0",
            "-an",
            "-vf",
            "showinfo=checksum=0",
            "-vsync",
            "0",
            "-f",
            "null",
            "-",
        ])
        timeline: list[tuple[int, float]] = []
        for line in result.stderr.splitlines():
            match = self._SHOWINFO_FRAME_RE.search(line)
            if match is None:
                continue
            raw_timestamp = match.group("timestamp")
            if raw_timestamp == "N/A":
                continue
            try:
                timestamp_seconds = float(raw_timestamp)
            except ValueError:
                continue
            timeline.append((int(match.group("frame")), timestamp_seconds))
        return timeline

    def _map_selected_to_decoder_indexes(
        self,
        source: VideoSource,
        selected: Sequence[SelectedFrame],
    ) -> list[int]:
        """Map ffprobe-selected references to the authoritative FFmpeg indexes."""
        timeline = self._authoritative_decoder_timeline(source)
        if not timeline:
            raise RuntimeError(
                f"FFmpeg did not expose a decodable frame timeline for {source.video_id}"
            )

        timestamp_index: dict[float, list[tuple[int, float]]] = {}
        for frame_number, timestamp_seconds in timeline:
            timestamp_index.setdefault(round(timestamp_seconds, 6), []).append(
                (frame_number, timestamp_seconds)
            )
        # PTS is normally monotonic, but malformed streams can contain a
        # discontinuity.  Sort the lookup view before using bisect; decoder
        # frame numbers remain untouched in the mapped result.
        timeline_by_timestamp = sorted(timeline, key=lambda item: item[1])
        ordered_timestamps = [timestamp for _, timestamp in timeline_by_timestamp]
        mapped: list[int] = []
        used: set[int] = set()

        for selection in selected:
            target = selection.ref.pts_time_seconds
            candidates = [
                candidate
                for candidate in timestamp_index.get(round(target, 6), [])
                if candidate[0] not in used
            ]
            if not candidates:
                insertion = bisect.bisect_left(ordered_timestamps, target)
                nearby_indexes = range(
                    max(0, insertion - 2),
                    min(len(timeline_by_timestamp), insertion + 3),
                )
                candidates = [
                    timeline_by_timestamp[index]
                    for index in nearby_indexes
                    if timeline_by_timestamp[index][0] not in used
                ]

            best = min(
                candidates,
                key=lambda candidate: (
                    abs(candidate[1] - target),
                    abs(candidate[0] - selection.ref.source_frame_number),
                    candidate[0],
                ),
                default=None,
            )
            if best is None or abs(best[1] - target) > self._TIMESTAMP_MATCH_TOLERANCE_SECONDS:
                raise RuntimeError(
                    f"FFmpeg timeline cannot map selected frame "
                    f"{selection.ref.source_frame_number} at {target:.6f}s "
                    f"for {source.video_id}"
                )
            used.add(best[0])
            mapped.append(best[0])
        return mapped

    def materialize(
        self,
        source: VideoSource,
        video: VideoInfo,
        selected: Sequence[SelectedFrame],
        output: ExtractionOutput,
    ) -> ExtractionResult:
        """Decode selected frame indices once, then render profile images with Pillow.

        ``select`` preserves presentation order.  This is preferable to seeking
        each timestamp independently, which can silently return an adjacent frame
        for long-GOP H.264 sources.
        """
        selected = sorted(selected, key=lambda item: item.ref.source_frame_number)
        if not selected:
            raise ValueError("No selected frames to materialize")

        destination_dir = output.rendered_root / output.profile.profile_id / source.video_id
        rendered_manifest_path = destination_dir / "manifest.json"
        existing = [destination_dir / f"{item.ref.frame_id}{output.profile.file_extension}" for item in selected]
        if not output.overwrite and all(path.is_file() for path in existing) and rendered_manifest_path.is_file():
            return ExtractionResult(
                video_id=source.video_id,
                written=[],
                skipped=[item.ref for item in selected],
                rendered_manifest_path=rendered_manifest_path,
            )
        if destination_dir.is_dir() and any(destination_dir.iterdir()):
            # A killed FFmpeg/Pillow step can leave a partial profile.  The
            # directory is an output owned by this video/profile, so remove
            # exactly that incomplete directory before rebuilding it.  This
            # makes stage-level resume safe without touching source videos or
            # another video/profile.
            shutil.rmtree(destination_dir)

        destination_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="keyframe-decode-") as temp_dir_name:
            temp_dir = Path(temp_dir_name)
            fast_dir = temp_dir / "fast"
            decoded = self._render_selected_frame_numbers(
                source,
                [item.ref.source_frame_number for item in selected],
                fast_dir,
            )
            render_plan: list[tuple[SelectedFrame, Path]] = []
            if len(decoded) == len(selected):
                render_plan = list(zip(selected, decoded, strict=True))
            else:
                # ffprobe and FFmpeg normally expose the same frame ordinal.  A
                # damaged/VFR stream can violate that assumption.  Re-map by
                # PTS using FFmpeg's own decoder timeline, then render with the
                # authoritative filter indexes.  The public FrameRef remains
                # unchanged, so selection/render manifests retain provenance.
                authoritative_numbers = self._map_selected_to_decoder_indexes(source, selected)
                remapped_plan = sorted(
                    zip(selected, authoritative_numbers, strict=True),
                    key=lambda item: item[1],
                )
                fallback_dir = temp_dir / "fallback"
                decoded = self._render_selected_frame_numbers(
                    source,
                    [frame_number for _, frame_number in remapped_plan],
                    fallback_dir,
                )
                if len(decoded) != len(selected):
                    raise RuntimeError(
                        f"FFmpeg decoded {len(decoded)} frames after timeline remap, "
                        f"but selector requested {len(selected)} for {source.video_id}"
                    )
                render_plan = list(zip((item[0] for item in remapped_plan), decoded, strict=True))

            # FFmpeg emits frames in decoder order.  Normally this is already
            # selection order; sorting keeps manifests stable if the PTS remap
            # had to reorder a pathological stream.
            render_plan.sort(key=lambda item: item[0].ref.source_frame_number)

            written: list[ExtractedFrame] = []
            manifest_frames: list[dict[str, object]] = []
            for selection, decoded_path in self.progress.iterate(
                render_plan,
                total=len(selected),
                desc=f"{source.video_id}: render",
                unit="frame",
            ):
                destination = destination_dir / f"{selection.ref.frame_id}{output.profile.file_extension}"
                with Image.open(decoded_path) as image:
                    width, height = render_image(image, destination, output.profile)
                extracted = ExtractedFrame(selection.ref, destination, width, height)
                written.append(extracted)
                manifest_frames.append({
                    "frame": asdict(selection.ref),
                    "path": str(destination),
                    "width": width,
                    "height": height,
                    "selection": {
                        "score": selection.score,
                        "reasons": list(selection.reasons),
                        "rank": selection.rank,
                        "metadata": dict(selection.metadata),
                    },
                })

        write_rendered_manifest(
            rendered_manifest_path,
            source,
            output.profile,
            output.selection_manifest_path,
            manifest_frames,
        )
        return ExtractionResult(
            video_id=source.video_id,
            written=written,
            skipped=[],
            rendered_manifest_path=rendered_manifest_path,
        )
