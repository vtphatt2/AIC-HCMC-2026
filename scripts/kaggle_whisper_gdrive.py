"""Kaggle notebook: Download videos from Google Drive → Whisper ASR → JSONL transcripts.

=== CELL 1: Install ===
"""
# CELL 1 — Install
# !pip install -q faster-whisper gdown

import json, time, os, zipfile
from pathlib import Path
from faster_whisper import WhisperModel

# === CELL 2: Download videos from Google Drive ===
# CELL 2 — Download videos from Google Drive
# ⚠️ Before running: upload all 32 videos to a Google Drive folder, share it as "Anyone with link"

FOLDER_ID = "1abc123xyz..."  # <-- THAY BẰNG FOLDER ID CỦA BẠN

VIDEO_DIR = Path("/kaggle/working/videos")
VIDEO_DIR.mkdir(parents=True, exist_ok=True)

import gdown

# Option A: Download entire folder (if <50 files)
# gdown.download_folder(id=FOLDER_ID, output=str(VIDEO_DIR), quiet=False)

# Option B: Download individual files (more reliable)
FILE_IDS = {
    "L26_V298": "1xxx...",  # <-- thay bằng file ID
    "L26_V313": "1xxx...",
    # ... add all 32 videos with their file IDs
}

for vid, fid in FILE_IDS.items():
    out = VIDEO_DIR / f"{vid}.mp4"
    if out.exists():
        print(f"SKIP {vid} — already downloaded")
        continue
    gdown.download(id=fid, output=str(out), quiet=False)
    print(f"OK   {vid}")

print(f"\nDownloaded: {len(list(VIDEO_DIR.glob('*.mp4')))} videos")

# === CELL 3: Load Whisper ===
# CELL 3 — Load model
MODEL_SIZE = "large-v3"
print(f"Loading {MODEL_SIZE}...")
model = WhisperModel(MODEL_SIZE, device="cuda", compute_type="float16")
print("Loaded.")

# === CELL 4: Transcribe ===
# CELL 4 — Transcribe all videos
OUT_DIR = Path("/kaggle/working/transcripts")
OUT_DIR.mkdir(exist_ok=True)

VIDEO_IDS = [
    "L26_V298", "L26_V313", "L26_V337", "L26_V381", "L26_V432", "L26_V491",
    "L28_V001", "L28_V002", "L28_V003", "L28_V004", "L28_V005", "L28_V006",
    "L28_V007", "L28_V008", "L28_V009", "L28_V010", "L28_V011", "L28_V012",
    "L28_V013", "L28_V014", "L28_V015", "L28_V016", "L28_V017", "L28_V018",
    "L28_V019", "L28_V020", "L28_V021", "L28_V022", "L28_V023", "L28_V024",
    "L30_V029", "L30_V039",
]

ok = skip = fail = 0
for vid in VIDEO_IDS:
    out_path = OUT_DIR / f"{vid}.jsonl"
    if out_path.exists():
        print(f"SKIP {vid}")
        skip += 1
        continue

    video_path = None
    for ext in (".mp4", ".mov", ".mkv", ".avi", ".webm"):
        p = VIDEO_DIR / f"{vid}{ext}"
        if p.exists():
            video_path = p
            break

    if not video_path:
        print(f"FAIL {vid} — not found")
        fail += 1
        continue

    t0 = time.perf_counter()
    segs, info = model.transcribe(str(video_path), language="vi", beam_size=5, word_timestamps=True)
    segs = list(segs)
    elapsed = time.perf_counter() - t0

    if not segs:
        print(f"FAIL {vid} — empty")
        fail += 1
        continue

    with open(out_path, "w", encoding="utf-8") as f:
        for s in segs:
            f.write(json.dumps({
                "start_time_ms": round(s.start * 1000),
                "end_time_ms": round(s.end * 1000),
                "text": s.text.strip(),
            }, ensure_ascii=False) + "\n")

    dur = segs[-1].end
    print(f"OK   {vid} — {len(segs)} segs | {dur:.0f}s | {elapsed:.1f}s ({dur/elapsed:.1f}x)")
    ok += 1

print(f"\nDone. OK={ok} SKIP={skip} FAIL={fail}")

# === CELL 5: Download results ===
# CELL 5 — Zip for download
# !cd /kaggle/working && zip -r transcripts_whisper.zip transcripts/
# from IPython.display import FileLink
# FileLink("transcripts_whisper.zip")
