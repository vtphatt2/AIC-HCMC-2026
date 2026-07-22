import csv
import os
import re
import logging
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    TranscriptsDisabled,
    NoTranscriptFound,
    VideoUnavailable,
    InvalidVideoId,
    RequestBlocked,
)

# === CẤU HÌNH ===
CSV_FILE = "ytlink.csv"
OUTPUT_DIR = "transcripts"
LOG_FILE = "scrape_errors.log"
PREFERRED_LANGUAGES = ["vi", "en"]

os.makedirs(OUTPUT_DIR, exist_ok=True)

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.ERROR,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("yt_scraper")


def extract_yt_id(url: str) -> str | None:
    patterns = [
        r"(?:youtube\.com/watch\?.*v=)([a-zA-Z0-9_-]{11})",
        r"(?:youtu\.be/)([a-zA-Z0-9_-]{11})",
        r"(?:youtube\.com/embed/)([a-zA-Z0-9_-]{11})",
        r"(?:youtube\.com/shorts/)([a-zA-Z0-9_-]{11})",
    ]
    for pattern in patterns:
        match = re.search(pattern, url.strip())
        if match:
            return match.group(1)
    return None


def format_timestamp(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"[{h:02d}:{m:02d}:{s:02d}]"


def fetch_transcript(yt_id: str) -> str:
    ytt_api = YouTubeTranscriptApi()
    transcript = ytt_api.fetch(yt_id, languages=PREFERRED_LANGUAGES)
    lines = [f"{format_timestamp(s.start)} {s.text}" for s in transcript.snippets]
    return "\n".join(lines)


def read_csv(csv_path: str) -> list[dict]:
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for col in ["video_id", "yt_link"]:
            if col not in reader.fieldnames:
                raise ValueError(
                    f"Khong tim thay cot '{col}' trong CSV. "
                    f"Cac cot co san: {reader.fieldnames}"
                )
        for row in reader:
            if row["video_id"].strip():
                rows.append(row)
    return rows


def main():
    videos = read_csv(CSV_FILE)
    print(f"Tim thay {len(videos)} video trong file CSV.\n")

    success_count = 0
    fail_count = 0

    for i, row in enumerate(videos, 1):
        video_id = row["video_id"].strip()
        yt_link = row["yt_link"].strip()
        yt_id = extract_yt_id(yt_link)

        if not yt_id:
            msg = f"{video_id} | {yt_link} | URL khong hop le"
            print(f"  [{i}/{len(videos)}] FAIL {video_id} - URL khong hop le.")
            logger.error(msg)
            fail_count += 1
            continue

        output_path = os.path.join(OUTPUT_DIR, f"{video_id}_Transcript.txt")
        if os.path.exists(output_path):
            print(f"  [{i}/{len(videos)}] SKIP {video_id} - da ton tai.")
            success_count += 1
            continue

        try:
            transcript_text = fetch_transcript(yt_id)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(transcript_text)
            print(f"  [{i}/{len(videos)}] OK   {video_id} - luu thanh cong.")
            success_count += 1

        except (TranscriptsDisabled, NoTranscriptFound) as e:
            msg = f"{video_id} | {yt_link} | Transcript khong kha dung: {e}"
            print(f"  [{i}/{len(videos)}] FAIL {video_id} - khong co transcript.")
            logger.error(msg)
            fail_count += 1

        except VideoUnavailable as e:
            msg = f"{video_id} | {yt_link} | Video khong ton tai hoac bi xoa: {e}"
            print(f"  [{i}/{len(videos)}] FAIL {video_id} - video khong kha dung.")
            logger.error(msg)
            fail_count += 1

        except Exception as e:
            msg = f"{video_id} | {yt_link} | Loi: {type(e).__name__}: {e}"
            print(f"  [{i}/{len(videos)}] FAIL {video_id} - loi: {e}")
            logger.error(msg)
            fail_count += 1

    print(f"\n--- KET QUA ---")
    print(f"Thanh cong: {success_count}")
    print(f"That bai:   {fail_count}")
    print(f"Log loi:    {os.path.abspath(LOG_FILE)}")
    print(f"Transcript: {os.path.abspath(OUTPUT_DIR)}/")


if __name__ == "__main__":
    main()
