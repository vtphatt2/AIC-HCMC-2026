"""FFmpeg-backed extractor with PTS-based frame inventory."""
from __future__ import annotations

import json
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
        """Yield each decoded presentation frame using ffprobe's best-effort PTS."""
        result = self._run([
            self.ffprobe_bin, "-v", "error", "-select_streams", "v:0",
            "-show_entries", "frame=best_effort_timestamp_time",
            "-of", "csv=p=0", str(source.path),
        ])
        lines = result.stdout.splitlines()
        for frame_number, raw_timestamp in enumerate(
            self.progress.iterate(
                lines,
                total=video.frame_count,
                desc=f"{source.video_id}: scan",
                unit="frame",
            )
        ):
            # ``-of csv=p=0`` may emit a trailing delimiter for a single field
            # on some FFmpeg builds, hence the explicit delimiter removal.
            raw_timestamp = raw_timestamp.strip().rstrip(",")
            if not raw_timestamp or raw_timestamp == "N/A":
                continue
            timestamp_seconds = float(raw_timestamp)
            yield FrameCandidate(
                ref=FrameRef(
                    video_id=source.video_id,
                    source_frame_number=frame_number,
                    timestamp_ms=round(timestamp_seconds * 1000),
                    pts_time_seconds=timestamp_seconds,
                )
            )

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
        if not output.overwrite and any(path.is_file() for path in existing):
            # A partial profile cannot be safely resumed without validating every
            # existing image and its provenance.  Failing explicitly prevents a
            # newly written manifest from claiming a complete render set.
            raise RuntimeError(
                f"Partial existing output found in {destination_dir}; "
                "use --overwrite after inspecting it or choose another profile ID."
            )

        destination_dir.mkdir(parents=True, exist_ok=True)
        expression = "+".join(f"eq(n\\,{item.ref.source_frame_number})" for item in selected)
        # The expression is passed as one argv element, so no shell escaping or
        # quoting is involved.  ``-vsync 0`` prevents FFmpeg from duplicating.
        with tempfile.TemporaryDirectory(prefix="keyframe-decode-") as temp_dir_name:
            temp_dir = Path(temp_dir_name)
            pattern = temp_dir / "decoded-%06d.png"
            self._run([
                self.ffmpeg_bin, "-v", "error", "-i", str(source.path),
                "-vf", f"select={expression}", "-vsync", "0", str(pattern),
            ])
            decoded = sorted(temp_dir.glob("decoded-*.png"))
            if len(decoded) != len(selected):
                raise RuntimeError(
                    f"FFmpeg decoded {len(decoded)} frames but selector requested {len(selected)}"
                )

            written: list[ExtractedFrame] = []
            manifest_frames: list[dict[str, object]] = []
            for selection, decoded_path in self.progress.iterate(
                zip(selected, decoded, strict=True),
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
