"""Build keyframe-to-transcript indexes directly from source transcript TXT files.

Run from the repository root:

    preprocess/.venv/bin/python \
        preprocess/transcript/build_keyframe_index.py --force
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = PROJECT_ROOT / "local-client" / "local-backend"
for import_root in (PROJECT_ROOT, BACKEND_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from app.data_provider import sample_subdir
from app.services.transcript_index import Transcript
from preprocess.transcript.frame_assignment import anchor_frame_id
from preprocess.transcript.sentences import build_sentences


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
    return parser.parse_args()


def frame_ids_for_video(video_id: str, features_dir: Path, keyframes_dir: Path) -> list[str]:
    """Use PE-Core feature names as the keyframe inventory, falling back to JPGs."""
    frame_paths = sorted((features_dir / video_id).glob("*.npy"))
    if not frame_paths:
        frame_paths = sorted((keyframes_dir / video_id).glob("*.jpg"))
    return [path.stem for path in frame_paths if path.stem.isdigit()]


def main() -> int:
    args = parse_args()
    transcripts_dir = sample_subdir("transcripts")
    metadata_dir = sample_subdir("metadata")
    features_dir = sample_subdir("PECore-features")
    keyframes_dir = sample_subdir("keyframes")
    output_dir = sample_subdir("keyframe_transcript_index")
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
        if (
            not args.force
            and output_path.is_file()
            and output_path.stat().st_mtime >= transcript_path.stat().st_mtime
        ):
            print(f"SKIP {video_id}: {output_path} is current")
            continue

        metadata_path = metadata_dir / f"{video_id}.json"
        if not metadata_path.is_file():
            print(f"SKIP {video_id}: metadata not found at {metadata_path}")
            continue

        # Parse source directly: this preprocessing path never reads or writes
        # AIC2026_sample/transcripts_processed.
        transcript = Transcript.from_txt(transcript_path, video_id=video_id)
        sentences = build_sentences(transcript)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        fps = float(metadata.get("fps") or 25.0)
        frame_ids = frame_ids_for_video(video_id, features_dir, keyframes_dir)
        indexed_sentences = []
        for sentence_number, sentence in enumerate(sentences):
            indexed_sentences.append({
                "sentence_id": f"S{sentence_number:06d}",
                "start_ms": sentence.start_ms,
                "end_ms": sentence.end_ms,
                "text": sentence.text,
                "anchor_frame_id": anchor_frame_id(frame_ids, fps, sentence),
            })
        payload = {
            "video_id": video_id,
            "fps": fps,
            "keyframe_count": len(frame_ids),
            "sentence_count": len(indexed_sentences),
            "anchored_sentence_count": sum(item["anchor_frame_id"] is not None for item in indexed_sentences),
            "sentences": indexed_sentences,
        }
        output_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(
            f"OK {video_id}: {payload['anchored_sentence_count']}/{payload['sentence_count']} "
            f"sentences anchored -> {output_path}"
        )
        written += 1

    return 0 if written else 1


if __name__ == "__main__":
    raise SystemExit(main())
