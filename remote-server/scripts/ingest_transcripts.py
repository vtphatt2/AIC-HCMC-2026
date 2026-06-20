"""
Ingest YouTube transcript .txt files into PostgreSQL transcripts table.

Each .txt file uses the format produced by notebooks/scrape_transcript.py:
    [HH:MM:SS] transcript text line
    [HH:MM:SS] next text line

Run from remote-server:
    python scripts/ingest_transcripts.py --transcripts-dir ../notebooks/transcripts
    python scripts/ingest_transcripts.py --transcripts-dir ../notebooks/transcripts --dry-run
    python scripts/ingest_transcripts.py --transcripts-dir ../notebooks/transcripts --clear-existing
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv

SCRIPT_DIR = Path(__file__).resolve().parent
REMOTE_ROOT = SCRIPT_DIR.parent
REPO_ROOT = REMOTE_ROOT.parent
sys.path.insert(0, str(REMOTE_ROOT))

logger = logging.getLogger("ingest_transcripts")

TIMESTAMP_RE = re.compile(r"\[(\d{2}):(\d{2}):(\d{2})\]\s+(.*)")
VIDEO_ID_RE = re.compile(r"^(L\d{2}_V\d{3})")  # e.g. L01_V001 from L01_V001_Transcript.txt
FALLBACK_DURATION_MS = 5000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ingest YouTube transcript .txt files into PostgreSQL transcripts table."
    )
    parser.add_argument(
        "--transcripts-dir",
        type=Path,
        required=True,
        help="Directory containing transcript .txt files (e.g. ../notebooks/transcripts).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and print records without inserting into the database.",
    )
    parser.add_argument(
        "--clear-existing",
        action="store_true",
        help="Delete existing transcript rows for each video before inserting new ones.",
    )
    return parser.parse_args()


def derive_video_id(filename: str) -> str | None:
    """Extract video_id from transcript filename.

    Examples:
        L01_V001.txt              -> L01_V001
        L01_V001_Transcript.txt   -> L01_V001
        vid_1_Transcript.txt      -> None (no match)
    """
    stem = Path(filename).stem
    # Try the standard sample pattern first
    match = VIDEO_ID_RE.match(stem)
    if match:
        return match.group(1)
    # Fallback: use the stem directly if it looks like a video ID
    if re.match(r"^[\w-]+$", stem):
        return stem
    return None


def parse_timestamp(line: str) -> tuple[int, int, int, str] | None:
    """Parse a [HH:MM:SS] text line. Returns (hours, mins, secs, text) or None."""
    match = TIMESTAMP_RE.match(line.strip())
    if not match:
        return None
    h, m, s = int(match.group(1)), int(match.group(2)), int(match.group(3))
    return h, m, s, match.group(4).strip()


def parse_transcript_file(filepath: Path) -> tuple[str, list[dict]]:
    """Parse one transcript .txt file into a list of segment dicts.

    Returns (video_id, segments) where each segment is:
        {start_time_ms, end_time_ms, text}
    Skips lines that don't match the [HH:MM:SS] format.
    """
    video_id = derive_video_id(filepath.name)
    if not video_id:
        raise ValueError(f"Cannot derive video_id from filename: {filepath.name}")

    segments: list[dict] = []
    raw_lines = filepath.read_text(encoding="utf-8").splitlines()
    skipped = 0

    # Parse all valid timestamp lines
    parsed = []
    for line in raw_lines:
        result = parse_timestamp(line)
        if result:
            parsed.append(result)
        elif line.strip():
            skipped += 1

    if not parsed:
        logger.warning("%s: no valid [HH:MM:SS] lines found (%d skipped)", filepath.name, skipped)
        return video_id, []

    for i, (h, m, s, text) in enumerate(parsed):
        start_ms = (h * 3600 + m * 60 + s) * 1000
        if i + 1 < len(parsed):
            nh, nm, ns, _ = parsed[i + 1]
            end_ms = (nh * 3600 + nm * 60 + ns) * 1000
        else:
            end_ms = start_ms + FALLBACK_DURATION_MS

        segments.append({
            "start_time_ms": start_ms,
            "end_time_ms": end_ms,
            "text": text,
        })

    logger.info("%s: parsed %d segments, skipped %d lines", filepath.name, len(segments), skipped)
    return video_id, segments


async def clear_existing(video_id: str) -> int:
    from app.db import postgres_client

    pool = await postgres_client.get_pool()
    async with pool.acquire() as conn:
        result = await conn.execute(
            "DELETE FROM transcripts WHERE video_id = $1", video_id
        )
    # asyncpg returns "DELETE N" as the command tag
    deleted = int(result.split()[-1]) if result else 0
    if deleted:
        logger.info("Cleared %d existing transcript rows for %s", deleted, video_id)
    return deleted


async def insert_transcripts(video_id: str, segments: list[dict]) -> int:
    from app.db import postgres_client

    await postgres_client.init_schema()
    pool = await postgres_client.get_pool()
    inserted = 0
    async with pool.acquire() as conn:
        for seg in segments:
            await conn.execute(
                """
                INSERT INTO transcripts (video_id, start_time_ms, end_time_ms, text)
                VALUES ($1, $2, $3, $4)
                """,
                video_id,
                seg["start_time_ms"],
                seg["end_time_ms"],
                seg["text"],
            )
            inserted += 1
    return inserted


async def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args()
    transcripts_dir = args.transcripts_dir.resolve()

    if not transcripts_dir.is_dir():
        logger.error("Transcripts directory not found: %s", transcripts_dir)
        sys.exit(1)

    # Always set ENV_MODE=SERVER so the postgres_client can connect.
    load_dotenv(REMOTE_ROOT / ".env", override=True)
    os.environ.setdefault("ENV_MODE", "SERVER")

    txt_files = sorted(transcripts_dir.glob("*.txt"))
    if not txt_files:
        logger.error("No .txt files found in %s", transcripts_dir)
        sys.exit(1)

    total_files = 0
    total_segments = 0
    total_skipped = 0
    total_inserted = 0
    failed_files = 0

    for filepath in txt_files:
        try:
            video_id, segments = parse_transcript_file(filepath)
        except ValueError as exc:
            logger.error("%s: %s", filepath.name, exc)
            failed_files += 1
            continue

        if not segments:
            continue

        total_files += 1
        total_segments += len(segments)

        if args.dry_run:
            print(f"\n{'='*60}")
            print(f"DRY RUN — {filepath.name} → video_id={video_id}")
            print(f"{'='*60}")
            for seg in segments:
                start_s = seg["start_time_ms"] / 1000
                end_s = seg["end_time_ms"] / 1000
                print(f"  [{start_s:7.2f}s → {end_s:7.2f}s] {seg['text'][:80]}")
            total_inserted += len(segments)
            continue

        if args.clear_existing:
            await clear_existing(video_id)

        try:
            inserted = await insert_transcripts(video_id, segments)
            total_inserted += inserted
            logger.info("%s: inserted %d rows", video_id, inserted)
        except Exception as exc:
            logger.error("%s: insert failed — %s", video_id, exc)
            failed_files += 1

    # Close the pool
    from app.db import postgres_client
    await postgres_client.close_pool()

    # Summary
    verb = "Would insert" if args.dry_run else "Inserted"
    print(f"\n{'='*60}")
    print(f"Ingest summary ({'DRY RUN' if args.dry_run else 'LIVE'})")
    print(f"{'='*60}")
    print(f"Files read:        {total_files}")
    print(f"Segments parsed:   {total_segments}")
    print(f"{verb}:          {total_inserted}")
    if failed_files:
        print(f"Failed files:      {failed_files}")
    print(f"Source directory:  {transcripts_dir}")


if __name__ == "__main__":
    asyncio.run(main())
