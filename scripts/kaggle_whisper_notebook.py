"""Kaggle notebook: Whisper ASR → JSONL transcripts for videos without YouTube captions.

Copy each cell into a Kaggle notebook. Set GPU T4x2 accelerator.

Expected Kaggle datasets:
  - nguyennam/ai-challenge-videos-no-transcript   (32 video files)
  - nguyennam/keyframe-pipeline-global-v9-3       (pipeline source)

=== CELL 1: Install & imports ===
"""
# CELL 1 — Install & imports
# !pip install -q faster-whisper

import json, time, os
from pathlib import Path
from faster_whisper import WhisperModel

# === CELL 2: Config ===
# CELL 2 — Paths & settings
VIDEO_DIR = Path("/kaggle/input/ai-challenge-videos-no-transcript")  # adjust dataset name
OUT_DIR = Path("/kaggle/working/transcripts")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_SIZE = "large-v3"  # large-v3 best for Vietnamese
DEVICE = "cuda"
COMPUTE_TYPE = "float16"  # or "int8_float16" for less VRAM

# 32 videos to process (exclude L24 lion dances — no speech)
VIDEO_IDS = [
    "L26_V298", "L26_V313", "L26_V337", "L26_V381", "L26_V432", "L26_V491",
    "L28_V001", "L28_V002", "L28_V003", "L28_V004", "L28_V005", "L28_V006",
    "L28_V007", "L28_V008", "L28_V009", "L28_V010", "L28_V011", "L28_V012",
    "L28_V013", "L28_V014", "L28_V015", "L28_V016", "L28_V017", "L28_V018",
    "L28_V019", "L28_V020", "L28_V021", "L28_V022", "L28_V023", "L28_V024",
    "L30_V029", "L30_V039",
]

# === CELL 3: Load model ===
# CELL 3 — Load Whisper model
print(f"Loading {MODEL_SIZE} on {DEVICE} ({COMPUTE_TYPE})...")
model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)
print("Model loaded.")

# === CELL 4: Transcribe ===
# CELL 4 — Run transcription for all 32 videos
ok = skip = fail = 0

for vid in VIDEO_IDS:
    out_path = OUT_DIR / f"{vid}.jsonl"
    if out_path.exists():
        print(f"SKIP {vid}")
        skip += 1
        continue

    # Find video file (try .mp4, .mov, .mkv)
    video_path = None
    for ext in (".mp4", ".mov", ".mkv", ".avi", ".webm"):
        candidate = VIDEO_DIR / f"{vid}{ext}"
        if candidate.exists():
            video_path = candidate
            break

    if video_path is None:
        print(f"FAIL {vid} — file not found")
        fail += 1
        continue

    t0 = time.perf_counter()
    segments_out, info = model.transcribe(
        str(video_path),
        language="vi",
        beam_size=5,
        word_timestamps=True,
    )
    segs = list(segments_out)
    elapsed = time.perf_counter() - t0

    if not segs:
        print(f"FAIL {vid} — no segments")
        fail += 1
        continue

    with open(out_path, "w", encoding="utf-8") as f:
        for seg in segs:
            f.write(json.dumps({
                "start_time_ms": round(seg.start * 1000),
                "end_time_ms": round(seg.end * 1000),
                "text": seg.text.strip(),
            }, ensure_ascii=False) + "\n")

    dur = segs[-1].end if segs else 0
    print(f"OK   {vid} — {len(segs)} segments | {dur:.0f}s | {elapsed:.1f}s ({dur/elapsed:.1f}x)")
    ok += 1

print(f"\nDone. OK={ok} SKIP={skip} FAIL={fail}")

# === CELL 5: Pack results ===
# CELL 5 — Zip transcripts for download
# !cd /kaggle/working && zip -r transcripts_whisper.zip transcripts/
print(f"Output: {OUT_DIR} ({len(list(OUT_DIR.glob('*.jsonl')))} JSONL files)")
