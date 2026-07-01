"""
Index transcript chunks into Milvus + PostgreSQL with topic-labelled embeddings.

Processes AIC2026_sample transcript .txt files:
  1. Parses [HH:MM:SS] format segments
  2. Groups into overlapping chunks (40-60s target, 50% overlap)
  3. Assigns topics via embedding-similarity (multilingual-e5-small)
  4. Inserts vectors into Milvus (transcript_chunks) and metadata into PostgreSQL

Run from remote-server:
    python scripts/index_transcripts.py
    python scripts/index_transcripts.py --sample-root ../AIC2026_sample --dry-run
    python scripts/index_transcripts.py --clear-existing
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import re
import sys
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
REMOTE_ROOT = SCRIPT_DIR.parent
REPO_ROOT = REMOTE_ROOT.parent
sys.path.insert(0, str(REMOTE_ROOT))

from scripts.sample_paths import default_sample_root, sample_subdir
from app.db import milvus_client, postgres_client
from app.db.milvus_client import TOPICS, TRANSCRIPT_VECTOR_DIM

logger = logging.getLogger("index_transcripts")
TIMESTAMP_RE = re.compile(r"\[(\d{2}):(\d{2}):(\d{2})\]\s+(.*)")
VIDEO_ID_RE = re.compile(r"^(L\d{2}_V\d{3})")
FALLBACK_DURATION_MS = 5000
CHUNK_MIN_S = 20
CHUNK_TARGET_MIN_S = 40
CHUNK_TARGET_MAX_S = 60
BATCH_SIZE = 128


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Index transcript chunks with topic-labelled embeddings."
    )
    parser.add_argument(
        "--sample-root",
        type=Path,
        default=default_sample_root(REPO_ROOT),
        help="Path to AIC2026_sample. Auto-detected if omitted.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and print chunks without inserting into databases.",
    )
    parser.add_argument(
        "--clear-existing",
        action="store_true",
        help="Drop and recreate the transcript_chunks collection + metadata table before indexing.",
    )
    parser.add_argument(
        "--model-device",
        default=os.getenv("TRANSCRIPT_MODEL_DEVICE", "cpu"),
        help="Device for the embedding model (cpu, cuda).",
    )
    return parser.parse_args()


def derive_video_id(filename: str) -> str | None:
    stem = Path(filename).stem
    match = VIDEO_ID_RE.match(stem)
    if match:
        return match.group(1)
    if re.match(r"^[\w-]+$", stem):
        return stem
    return None


def parse_transcript_file(filepath: Path) -> tuple[str, list[dict]] | None:
    video_id = derive_video_id(filepath.name)
    if not video_id:
        logger.warning("Cannot derive video_id from: %s", filepath.name)
        return None

    raw_lines = filepath.read_text(encoding="utf-8").splitlines()
    parsed = []
    for line in raw_lines:
        m = TIMESTAMP_RE.match(line.strip())
        if m:
            h, mm, s = int(m.group(1)), int(m.group(2)), int(m.group(3))
            text = m.group(4).strip()
            if text:
                parsed.append((h, mm, s, text))

    if not parsed:
        logger.warning("%s: no valid [HH:MM:SS] lines", filepath.name)
        return None

    segments = []
    for i, (h, mm, s, text) in enumerate(parsed):
        start_ms = (h * 3600 + mm * 60 + s) * 1000
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

    logger.info("%s: parsed %d segments", filepath.name, len(segments))
    return video_id, segments


def chunk_segments(segments: list[dict]) -> list[dict]:
    """
    Group segments into overlapping chunks.
    Each chunk targets 40-60 seconds with 50% segment overlap.
    Never splits a single transcript line.
    """
    if not segments:
        return []

    chunks = []
    i = 0
    while i < len(segments):
        j = i
        chunk_start_ms = segments[i]["start_time_ms"]
        chunk_end_ms = chunk_start_ms
        chunk_texts = []

        while j < len(segments):
            seg = segments[j]
            chunk_end_ms = seg["end_time_ms"]
            chunk_texts.append(seg["text"])
            duration_s = (chunk_end_ms - chunk_start_ms) / 1000.0

            if duration_s >= CHUNK_TARGET_MIN_S and len(chunk_texts) >= 3:
                j += 1
                break
            if duration_s >= CHUNK_TARGET_MAX_S:
                j += 1
                break
            j += 1

        duration_s = (chunk_end_ms - chunk_start_ms) / 1000.0
        if chunk_texts and (duration_s >= CHUNK_MIN_S or len(chunk_texts) >= 3):
            chunks.append({
                "start_time_ms": chunk_start_ms,
                "end_time_ms": chunk_end_ms,
                "text": " ".join(chunk_texts),
            })

        n_in_chunk = j - i
        step = max(1, n_in_chunk // 2)
        if n_in_chunk <= 1:
            i += 1
        else:
            i += step

    return chunks


async def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args()
    sample_root = args.sample_root.resolve()
    load_dotenv(REMOTE_ROOT / ".env", override=True)
    os.environ.setdefault("ENV_MODE", "SERVER")

    transcripts_dir = sample_subdir(sample_root, "transcripts")
    if not transcripts_dir.is_dir():
        logger.error("Transcripts directory not found: %s", transcripts_dir)
        sys.exit(1)

    txt_files = sorted(transcripts_dir.glob("*.txt"))
    if not txt_files:
        logger.error("No .txt files in %s", transcripts_dir)
        sys.exit(1)

    logger.info("Found %d transcript files in %s", len(txt_files), transcripts_dir)

    # ── Parse all transcripts into segments ────────────────────────────────────
    all_video_segments: dict[str, list[dict]] = {}
    for filepath in txt_files:
        result = parse_transcript_file(filepath)
        if result is None:
            continue
        video_id, segments = result
        all_video_segments[video_id] = segments

    logger.info("Parsed %d videos with segments", len(all_video_segments))

    # ── Load model ─────────────────────────────────────────────────────────────
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(
        os.getenv("TRANSCRIPT_MODEL_ID", "intfloat/multilingual-e5-small"),
        device=args.model_device,
    )
    actual_dim = model.get_sentence_embedding_dimension()
    logger.info("Loaded model dim=%s device=%s", actual_dim, args.model_device)
    if actual_dim != TRANSCRIPT_VECTOR_DIM:
        logger.error("Model dim %s != expected %s", actual_dim, TRANSCRIPT_VECTOR_DIM)
        sys.exit(1)

    # Pre-compute topic embeddings
    topic_vectors = model.encode(
        TOPICS,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype("float32")

    # ── Chunk + classify all videos ────────────────────────────────────────────
    all_chunks: list[dict] = []
    chunk_id_counter = 1

    for video_id, segments in sorted(all_video_segments.items()):
        chunks = chunk_segments(segments)
        if not chunks:
            logger.warning("%s: no chunks generated from %d segments", video_id, len(segments))
            continue

        chunk_texts = [c["text"] for c in chunks]
        prefixed_texts = [f"passage: {t}" for t in chunk_texts]
        embeddings = model.encode(
            prefixed_texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=BATCH_SIZE,
        ).astype("float32")

        for idx, (chunk, emb) in enumerate(zip(chunks, embeddings)):
            similarities = topic_vectors @ emb
            best_topic_idx = int(similarities.argmax())
            topic = TOPICS[best_topic_idx]

            all_chunks.append({
                "chunk_id":      chunk_id_counter,
                "video_id":      video_id,
                "topic":         topic,
                "start_time_ms": chunk["start_time_ms"],
                "end_time_ms":   chunk["end_time_ms"],
                "raw_text":      chunk["text"],
                "vector":        emb.tolist(),
            })
            chunk_id_counter += 1

        logger.info(
            "%s: %d chunks → %d topics assigned",
            video_id,
            len(chunks),
            len(set(c["topic"] for c in all_chunks[-len(chunks):])),
        )

    logger.info("Total chunks: %d across %d videos", len(all_chunks), len(all_video_segments))

    # ── Compute video_genre from dominant chunk topic ────────────────────────
    video_genre_map: dict[str, str] = {}
    for video_id in sorted(all_video_segments):
        topic_counts: dict[str, int] = {}
        for chunk in all_chunks:
            if chunk["video_id"] == video_id:
                topic_counts[chunk["topic"]] = topic_counts.get(chunk["topic"], 0) + 1
        if topic_counts:
            dominant = max(topic_counts, key=topic_counts.get)
            video_genre_map[video_id] = dominant
            logger.info("%s: genre=%s  (topic_dist=%s)", video_id, dominant, topic_counts)
        else:
            video_genre_map[video_id] = "Thời sự"  # default fallback

    if args.dry_run:
        print("\n" + "=" * 70)
        print("DRY RUN — first 10 chunks:")
        print("=" * 70)
        for chunk in all_chunks[:10]:
            start_s = chunk["start_time_ms"] / 1000
            end_s = chunk["end_time_ms"] / 1000
            preview = chunk["raw_text"][:100].replace("\n", " ")
            print(
                f"  [{chunk['chunk_id']:5d}] {chunk['video_id']} | {chunk['topic']:10s} "
                f"| {start_s:7.1f}s → {end_s:7.1f}s | {preview}"
            )
        print(f"\nVideo Genres (dominant topic per video):")
        for vid, genre in sorted(video_genre_map.items()):
            print(f"  {vid}: {genre}")
        print(f"\nWould index {len(all_chunks)} chunks and tag {len(video_genre_map)} video genres.")
        return

    # ── Connect to databases ───────────────────────────────────────────────────
    milvus_client.connect()
    await postgres_client.init_schema()

    if args.clear_existing:
        milvus_client.drop_transcript_collection_if_exists()
        pool = await postgres_client.get_pool()
        async with pool.acquire() as conn:
            await conn.execute("DELETE FROM transcript_chunks_metadata")
        logger.info("Cleared transcript_chunks_metadata table")

    collection = milvus_client.create_transcript_collection_if_missing()

    # ── Insert into PostgreSQL (batch) ─────────────────────────────────────────
    pg_records = [
        {k: v for k, v in c.items() if k != "vector"}
        for c in all_chunks
    ]
    inserted_pg = await postgres_client.upsert_transcript_chunks(pg_records)
    logger.info("PostgreSQL: upserted %d chunk metadata rows", inserted_pg)

    # ── Insert into Milvus (batch) ─────────────────────────────────────────────
    total_indexed = 0
    with tqdm(total=len(all_chunks), desc="Indexing Milvus vectors", unit="chunk") as progress:
        for start in range(0, len(all_chunks), BATCH_SIZE):
            batch = all_chunks[start : start + BATCH_SIZE]
            milvus_client.upsert_transcript_chunks(collection, batch)
            total_indexed += len(batch)
            progress.update(len(batch))
            progress.set_postfix(indexed=total_indexed)

    collection.flush()
    collection.load()
    logger.info("Milvus: indexed %d chunk vectors", total_indexed)

    # ── Update video genres in PostgreSQL ──────────────────────────────────────
    updated_genres = await postgres_client.update_video_genres(video_genre_map)
    logger.info("PostgreSQL: updated genre for %d videos", updated_genres)

    await postgres_client.close_pool()

    # ── Topic distribution report ──────────────────────────────────────────────
    topic_counts: dict[str, int] = {}
    for chunk in all_chunks:
        topic_counts[chunk["topic"]] = topic_counts.get(chunk["topic"], 0) + 1

    print("\n" + "=" * 70)
    print("INDEXING COMPLETE")
    print("=" * 70)
    print(f"Videos processed:  {len(all_video_segments)}")
    print(f"Chunks indexed:    {len(all_chunks)}")
    print(f"Milvus collection: {milvus_client.TRANSCRIPT_CHUNKS_COLLECTION}")
    print(f"PG table:          transcript_chunks_metadata")
    print(f"\nVideo Genres:")
    for vid, genre in sorted(video_genre_map.items()):
        print(f"  {vid}: {genre}")
    print("\nTopic distribution:")
    for topic in TOPICS:
        count = topic_counts.get(topic, 0)
        bar = "█" * (count * 50 // max(1, max(topic_counts.values(), default=1)))
        print(f"  {topic:12s} {count:4d}  {bar}")


if __name__ == "__main__":
    asyncio.run(main())
