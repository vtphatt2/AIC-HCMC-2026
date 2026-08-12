"""Whisper ASR for videos without YouTube auto-captions → JSONL transcript.
Usage:
  python scripts/whisper_transcribe.py /path/to/L28_V001.mp4
  python scripts/whisper_transcribe.py --video-dir /path/to/videos --failed-list failed_transcripts.txt
  python scripts/whisper_transcribe.py --video-dir /path/to/videos --video-ids L28_V001 L28_V002
"""
import argparse, json, sys, time, os
from pathlib import Path

import whisper

OUT_DIR = Path("AIC2026_sample/transcripts")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def transcribe_video(video_path: Path, model: whisper.Whisper, language: str = "vi") -> Path | None:
    video_id = video_path.stem
    out_path = OUT_DIR / f"{video_id}.jsonl"
    if out_path.exists():
        print(f"SKIP {video_id} — already exists")
        return out_path

    t0 = time.perf_counter()
    result = model.transcribe(str(video_path), language=language, word_timestamps=True)
    elapsed = time.perf_counter() - t0

    segments_jsonl = []
    for seg in result.get("segments", []):
        segments_jsonl.append({
            "start_time_ms": round(seg["start"] * 1000),
            "end_time_ms": round(seg["end"] * 1000),
            "text": seg["text"].strip(),
        })

    if not segments_jsonl:
        print(f"FAIL {video_id} — no segments")
        return None

    with open(out_path, "w", encoding="utf-8") as f:
        for seg in segments_jsonl:
            f.write(json.dumps(seg, ensure_ascii=False) + "\n")

    dur = result.get("segments", [{}])[-1].get("end", 0) if result.get("segments") else 0
    print(f"OK   {video_id} — {len(segments_jsonl)} segments | {dur:.0f}s audio | {elapsed:.1f}s")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Whisper ASR → JSONL transcript")
    parser.add_argument("video", nargs="?", help="Single video file path")
    parser.add_argument("--video-dir", type=Path, help="Directory with video files")
    parser.add_argument("--video-ids", nargs="+", help="Video IDs to transcribe")
    parser.add_argument("--failed-list", type=Path, help="Read video IDs from failed_transcripts.txt")
    parser.add_argument("--model", default="large-v3", help="Whisper model size (default: large-v3)")
    parser.add_argument("--language", default="vi", help="Audio language (default: vi)")
    parser.add_argument("--device", default="cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu")
    args = parser.parse_args()

    video_dir: Path | None = getattr(args, "video_dir", None)
    videos: list[Path] = []
    if args.video:
        videos = [Path(args.video)]
    elif video_dir:
        if args.failed_list:
            text = args.failed_list.read_text(encoding="utf-8")
            ids = [line.split()[0] for line in text.strip().split("\n")[2:] if line.strip()]
            videos = [video_dir / f"{vid}.mp4" for vid in ids]
        elif args.video_ids:
            videos = [video_dir / f"{vid}.mp4" for vid in args.video_ids]
        else:
            videos = sorted(video_dir.glob("*.mp4"))
    elif args.video_ids:
        for vid in args.video_ids:
            p = Path(f"{vid}.mp4")
            if p.exists():
                videos.append(p)

    videos = [v for v in videos if v.suffix.lower() in (".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v")]
    missing = [v for v in videos if not v.exists()]
    existing = [v for v in videos if (OUT_DIR / f"{v.stem}.jsonl").exists()]

    print(f"Model: {args.model} | Device: {args.device} | Language: {args.language}")
    print(f"Total: {len(videos)} | Missing: {len(missing)} | Done: {len(existing)}")
    if missing:
        for m in missing[:5]:
            print(f"  MISSING: {m}")
    print()

    model = whisper.load_model(args.model, device=args.device)
    ok = fail = skip = 0

    for v in videos:
        if not v.exists():
            print(f"FAIL {v.stem} — file not found")
            fail += 1
            continue
        if (OUT_DIR / f"{v.stem}.jsonl").exists():
            skip += 1
            continue
        result = transcribe_video(v, model, args.language)
        if result:
            ok += 1
        else:
            fail += 1

    print(f"\nDone. OK={ok} SKIP={skip} FAIL={fail}")


if __name__ == "__main__":
    main()
