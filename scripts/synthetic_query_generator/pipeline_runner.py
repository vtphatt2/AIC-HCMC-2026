import json
import logging
import sys
from datetime import datetime
from pathlib import Path

from . import config
from .context_extractor import extract_scene_contexts
from .query_generator import generate_queries

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

VIDEO_IDS = [
    "L01_V001", "L01_V002", "L01_V003", "L01_V004", "L01_V005",
    "L02_V001", "L02_V002", "L03_V001", "L03_V002",
]


def _load_metadata(video_id: str) -> dict:
    path = config.METADATA_ROOT / f"{video_id}.json"
    if path.exists():
        return json.loads(path.read_text())
    return {}


def run(video_ids: list[str] = None, dry_run: bool = False, limit_scenes: int = 0):
    if video_ids is None:
        video_ids = VIDEO_IDS

    if dry_run:
        client = None
    else:
        if not config.OPENAI_API_KEY:
            logger.error(
                "OPENAI_API_KEY not set.\n"
                "  Get a FREE key at https://aistudio.google.com/\n"
                "  Then copy .env.example → .env and fill in your key:\n"
                "    cp .env.example .env"
            )
            sys.exit(1)
        from openai import OpenAI
        client = OpenAI(api_key=config.OPENAI_API_KEY, base_url=config.OPENAI_BASE_URL)
    all_results = {}

    for video_id in video_ids:
        logger.info(f"Processing {video_id}...")
        contexts = extract_scene_contexts(video_id)

        if not contexts:
            logger.warning(f"No scene contexts extracted for {video_id}")
            continue

        if limit_scenes > 0:
            contexts = contexts[:limit_scenes]

        metadata = _load_metadata(video_id)
        video_results = []

        for i, ctx in enumerate(contexts):
            logger.info(f"  Scene {i+1}/{len(contexts)}: frames {ctx['scene_start_frame']}-{ctx['scene_end_frame']}")

            if dry_run:
                queries = [{"type": "mock", "query": f"DRY_RUN: scene {ctx['scene_start_frame']}-{ctx['scene_end_frame']}"}]
            else:
                queries = generate_queries(ctx, client)

            if queries:
                video_results.append({
                    "video_id": video_id,
                    "youtube_url": metadata.get("video_link", ""),
                    "fps": metadata.get("fps", config.FPS),
                    "scene_start_frame": ctx["scene_start_frame"],
                    "scene_end_frame": ctx["scene_end_frame"],
                    "timestamp_range": ctx["timestamp_range"],
                    "keyframe_ids": ctx["keyframe_ids"],
                    "generated_queries": queries,
                })

        if video_results:
            all_results[video_id] = video_results

    # Save output
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = config.OUTPUT_DIR / f"generated_queries_{timestamp}.json"
    out_path.write_text(json.dumps(all_results, ensure_ascii=False, indent=2))
    logger.info(f"Saved {sum(len(v) for v in all_results.values())} scenes → {out_path}")

    total_queries = sum(
        len(r["generated_queries"]) for v in all_results.values() for r in v
    )
    logger.info(f"Total queries generated: {total_queries}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, help="Single video ID, e.g. L01_V001")
    parser.add_argument("--dry-run", action="store_true", help="Extract contexts without calling LLM")
    parser.add_argument("--limit-scenes", type=int, default=0, help="Limit scenes per video for testing")
    args = parser.parse_args()

    video_ids = [args.video] if args.video else None
    run(video_ids=video_ids, dry_run=args.dry_run, limit_scenes=args.limit_scenes)
