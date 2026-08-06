"""Build keyframe-to-transcript indexes directly from source transcript TXT files.

Run from the repository root:

    preprocess/.venv/bin/python -m \
        preprocess.transcript.build_keyframe_index --force
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from preprocess.transcript.frame_assignment import anchor_frame_id
from preprocess.transcript.sentences import Transcript, build_sentences
from preprocess.batch.provenance import atomic_json_write, digest_records, file_record

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Map source transcript sentences directly to sample keyframes."
    )
    parser.add_argument(
        "--video-id",
        action="append",
        help="Only process this video ID. Repeat for multiple videos.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rewrite output even when it is newer than the source transcript.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=PROJECT_ROOT / "data",
        help="Dataset root containing transcripts/, metadata/, keyframes/ and output directories.",
    )
    parser.add_argument(
        "--sample-root",
        dest="data_root",
        type=Path,
        help="Backward-compatible alias for --data-root.",
    )
    return parser.parse_args()


def frame_ids_for_video(video_id: str, features_dir: Path, keyframes_dir: Path) -> list[str]:
    """Use PE-Core feature names, falling back to all supported images."""
    frame_paths = sorted((features_dir / video_id).glob("*.npy"))
    if not frame_paths:
        frame_paths = sorted(
            path
            for path in (keyframes_dir / video_id).iterdir()
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ) if (keyframes_dir / video_id).is_dir() else []
    return [path.stem for path in frame_paths if path.stem.isdigit()]


def frame_paths_for_video(video_id: str, features_dir: Path, keyframes_dir: Path) -> list[Path]:
    """Return the exact frame artifacts used by the index."""
    feature_paths = sorted((features_dir / video_id).glob("*.npy"))
    if feature_paths:
        return feature_paths
    keyframe_dir = keyframes_dir / video_id
    if not keyframe_dir.is_dir():
        return []
    return sorted(
        path
        for path in keyframe_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )


def frame_timestamps_for_video(video_id: str, keyframes_dir: Path) -> dict[str, int]:
    """Read authoritative PTS timestamps from the rendered keyframe manifest."""
    manifest = keyframes_dir / video_id / "manifest.json"
    if not manifest.is_file():
        return {}
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return {}
    timestamps: dict[str, int] = {}
    for frame in payload.get("frames", []):
        if not isinstance(frame, dict):
            continue
        raw = frame.get("frame", {})
        frame_id = str(raw.get("frame_id") or raw.get("source_frame_number", ""))
        timestamp = raw.get("timestamp_ms")
        if frame_id.isdigit() and timestamp is not None:
            timestamps[frame_id] = int(timestamp)
    return timestamps


def optional_fps(metadata: object) -> float | None:
    """Return a usable positive FPS value, treating missing/invalid values as unavailable."""
    if not isinstance(metadata, dict):
        return None
    raw_fps = metadata.get("fps")
    if raw_fps in (None, ""):
        return None
    try:
        fps = float(raw_fps)
    except (TypeError, ValueError):
        return None
    return fps if fps > 0 else None


def main() -> int:
    args = parse_args()
    data_root = args.data_root.resolve()
    transcripts_dir = data_root / "transcripts"
    metadata_dir = data_root / "metadata"
    features_dir = data_root / "PECore-features"
    keyframes_dir = data_root / "keyframes"
    output_dir = data_root / "keyframe_transcript_index"
    output_dir.mkdir(parents=True, exist_ok=True)

    transcript_paths = sorted(transcripts_dir.glob("*_Transcript.txt"))
    if args.video_id:
        requested_ids = set(args.video_id)
        transcript_paths = [
            path for path in transcript_paths
            if path.stem.removesuffix("_Transcript") in requested_ids
        ]
    if not transcript_paths:
        print(f"No transcript TXT files found in {transcripts_dir}")
        return 1

    written = 0
    for transcript_path in transcript_paths:
        video_id = transcript_path.stem.removesuffix("_Transcript")
        output_path = output_dir / f"{video_id}.json"
        metadata_path = metadata_dir / f"{video_id}.json"
        if not metadata_path.is_file():
            print(f"SKIP {video_id}: metadata not found at {metadata_path}")
            continue
        frame_paths = frame_paths_for_video(video_id, features_dir, keyframes_dir)
        source_records = [
            file_record(transcript_path, relative_to=data_root),
            file_record(metadata_path, relative_to=data_root),
            *[file_record(path, relative_to=data_root) for path in frame_paths],
        ]
        rendered_manifest = keyframes_dir / video_id / "manifest.json"
        if rendered_manifest.is_file():
            source_records.append(file_record(rendered_manifest, relative_to=data_root))
        source_digest = digest_records(source_records)
        if (
            not args.force
            and output_path.is_file()
            and output_path.stat().st_mtime >= transcript_path.stat().st_mtime
        ):
            try:
                cached = json.loads(output_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                cached = {}
            if cached.get("source_digest") == source_digest:
                print(f"SKIP {video_id}: {output_path} is current")
                continue

        # Parse source directly: this preprocessing path never reads or writes
        # AIC2026_sample/transcripts_processed.
        transcript = Transcript.from_txt(transcript_path, video_id=video_id)
        sentences = build_sentences(transcript)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        fps = optional_fps(metadata)
        frame_ids = [path.stem for path in frame_paths if path.stem.isdigit()]
        frame_timestamps = frame_timestamps_for_video(video_id, keyframes_dir)
        if not frame_timestamps and fps is None:
            print(f"SKIP {video_id}: FPS is missing and rendered PTS manifest is unavailable")
            continue
        indexed_sentences = []
        for sentence_number, sentence in enumerate(sentences):
            indexed_sentences.append({
                "sentence_id": f"S{sentence_number:06d}",
                "start_ms": sentence.start_ms,
                "end_ms": sentence.end_ms,
                "text": sentence.text,
                "anchor_frame_id": anchor_frame_id(
                    frame_ids,
                    fps or 0.0,
                    sentence,
                    frame_timestamps=frame_timestamps,
                ),
            })
        payload = {
            "video_id": video_id,
            "fps": fps,
            "timing_source": "rendered_manifest_pts" if frame_timestamps else "metadata_fps",
            "source_digest": source_digest,
            "keyframe_count": len(frame_ids),
            "sentence_count": len(indexed_sentences),
            "anchored_sentence_count": sum(item["anchor_frame_id"] is not None for item in indexed_sentences),
            "sentences": indexed_sentences,
        }
        atomic_json_write(output_path, payload)
        print(
            f"OK {video_id}: {payload['anchored_sentence_count']}/{payload['sentence_count']} "
            f"sentences anchored -> {output_path}"
        )
        written += 1

    return 0 if written else 1


if __name__ == "__main__":
    raise SystemExit(main())
