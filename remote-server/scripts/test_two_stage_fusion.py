"""
Integration Test Suite for TwoStageFusionStrategy.

Validates the Two-Stage Late Fusion Pipeline:
  Stage 1: E5 transcript vector search -> candidate video_ids
  Stage 2: PE-Core visual search filtered by Milvus expr
  Fusion:  Dynamic alpha weighted combination + OCR/Temporal boosts

Run modes:
  1. Unit tests (mocked):  pytest scripts/test_two_stage_fusion.py -v -m "not integration"
  2. Integration (real):   pytest scripts/test_two_stage_fusion.py -v -m integration
  3. Quick smoke:          python scripts/test_two_stage_fusion.py --smoke

Requires: pip install pytest pytest-asyncio
"""
from __future__ import annotations

import asyncio
import logging
import math
import os
import sys
import time
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REMOTE_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REMOTE_ROOT))

from dotenv import load_dotenv

load_dotenv(REMOTE_ROOT / ".env", override=True)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("test_two_stage")


# ═══════════════════════════════════════════════════════════════════════════════
# TEST DATA FACTORIES
# ═══════════════════════════════════════════════════════════════════════════════

def make_chunk(chunk_id: int, video_id: str, start_ms: int, end_ms: int,
               score: float = 0.8, topic: str = "Ẩm thực",
               text: str = "cách nấu phở bò truyền thống") -> dict:
    return {
        "chunk_id": chunk_id,
        "video_id": video_id,
        "topic": topic,
        "start_time_ms": start_ms,
        "end_time_ms": end_ms,
        "score": score,
        "text": text,
        "frame_image_url": f"/static/frames/{video_id}/{int((start_ms + end_ms) / 2 / 1000 * 25):06d}.jpg",
        "frame_number": int((start_ms + end_ms) / 2 / 1000 * 25),
    }


def make_frame(frame_id: str, video_id: str, frame_number: int,
               timestamp_ms: int, score: float = 0.7,
               image_url: str | None = None) -> dict:
    return {
        "frame_id": frame_id,
        "video_id": video_id,
        "frame_number": frame_number,
        "timestamp_ms": timestamp_ms,
        "image_url": image_url or f"/static/frames/{video_id}/{frame_number:06d}.jpg",
        "score": score,
    }


def make_video(video_id: str, youtube_id: str = "test_yt_id",
               fps: float = 25.0, genre: str = "Ẩm thực") -> dict:
    return {
        "video_id": video_id,
        "youtube_id": youtube_id,
        "fps": fps,
        "duration_ms": 300000,
        "frame_count": 7500,
        "title": f"Test Video {video_id}",
        "genre": genre,
    }


def make_query_group(semantic: str = "", text: str = "",
                     offset_ms: int = 0) -> dict:
    return {
        "semantic_query": semantic,
        "text_query": text,
        "temporal_offset_ms": offset_ms,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# UNIT TESTS (Mocked — no Milvus/PostgreSQL required)
# ═══════════════════════════════════════════════════════════════════════════════

class TestDynamicAlpha(unittest.TestCase):
    """Test the dynamic alpha computation logic."""

    def setUp(self):
        from app.strategies.two_stage_fusion_strategy import _compute_dynamic_alpha
        self.compute_alpha = _compute_dynamic_alpha

    def test_no_semantic_query_returns_genre_alpha(self):
        alpha = self.compute_alpha(0.8, 0.7, has_semantic=False, genre="Thời sự")
        self.assertAlmostEqual(alpha, 0.50, places=2)

    def test_no_semantic_default_genre(self):
        alpha = self.compute_alpha(0.8, 0.7, has_semantic=False, genre="All")
        self.assertAlmostEqual(alpha, 0.35, places=2)

    def test_low_transcript_score_returns_zero(self):
        alpha = self.compute_alpha(0.01, 0.8, has_semantic=True, genre="All")
        self.assertEqual(alpha, 0.0)

    def test_low_visual_score_returns_one(self):
        alpha = self.compute_alpha(0.8, 0.01, has_semantic=True, genre="All")
        self.assertEqual(alpha, 1.0)

    def test_balanced_scores_near_genre_base(self):
        alpha = self.compute_alpha(0.5, 0.5, has_semantic=True, genre="Ẩm thực")
        self.assertAlmostEqual(alpha, 0.25, delta=0.05)

    def test_high_transcript_ratio_increases_alpha(self):
        alpha_low = self.compute_alpha(0.3, 0.7, has_semantic=True, genre="All")
        alpha_high = self.compute_alpha(0.7, 0.3, has_semantic=True, genre="All")
        self.assertGreater(alpha_high, alpha_low)

    def test_alpha_bounded_0_to_1(self):
        for t in [0.0, 0.1, 0.5, 0.9, 1.0]:
            for v in [0.0, 0.1, 0.5, 0.9, 1.0]:
                alpha = self.compute_alpha(t, v, has_semantic=True, genre="All")
                self.assertGreaterEqual(alpha, 0.0, f"alpha < 0 for t={t}, v={v}")
                self.assertLessEqual(alpha, 1.0, f"alpha > 1 for t={t}, v={v}")

    def test_genre_specific_alphas(self):
        from app.strategies.two_stage_fusion_strategy import GENRE_ALPHA_MAP
        for genre, expected_base in GENRE_ALPHA_MAP.items():
            alpha = self.compute_alpha(0.5, 0.5, has_semantic=True, genre=genre)
            self.assertAlmostEqual(alpha, expected_base, delta=0.05,
                                   msg=f"Genre {genre} alpha mismatch")


class TestFusionMath(unittest.TestCase):
    """Test the weighted fusion score computation."""

    def setUp(self):
        from app.strategies.two_stage_fusion_strategy import TwoStageFusionStrategy
        self.strategy_cls = TwoStageFusionStrategy
        self.mock_dp = MagicMock()
        self.strategy = self.strategy_cls.__new__(self.strategy_cls)
        self.strategy.data_provider = self.mock_dp

    def test_weighted_fusion_basic(self):
        transcript_ranked = [{
            "video_id": "L01_V001",
            "start_time_ms": 0,
            "end_time_ms": 5000,
            "timestamp_ms": 2500,
            "text": "test",
            "_score": 0.8,
        }]
        visual_ranked = [{
            "video_id": "L01_V001",
            "frame_id": "L01_V001_000062",
            "frame_number": 62,
            "timestamp_ms": 2480,
            "image_url": "/static/frames/L01_V001/000062.jpg",
            "_score": 0.6,
        }]
        videos = {"L01_V001": make_video("L01_V001")}
        stage1_chunks = [make_chunk(1, "L01_V001", 0, 5000, score=0.8)]

        results = self.strategy._weighted_fusion(
            transcript_ranked, visual_ranked, alpha=0.4,
            videos=videos, stage1_chunks=stage1_chunks, has_semantic=True,
        )

        self.assertGreater(len(results), 0)
        for r in results:
            self.assertGreaterEqual(r["confidence"], 0.0)
            self.assertLessEqual(r["confidence"], 1.0)

    def test_fusion_alpha_zero_is_visual_only(self):
        transcript_ranked = [{
            "video_id": "L01_V001",
            "timestamp_ms": 2500,
            "_score": 0.9,
        }]
        visual_ranked = [{
            "video_id": "L01_V002",
            "frame_id": "L01_V002_000100",
            "frame_number": 100,
            "timestamp_ms": 4000,
            "image_url": "",
            "_score": 0.7,
        }]
        videos = {
            "L01_V001": make_video("L01_V001"),
            "L01_V002": make_video("L01_V002"),
        }

        results = self.strategy._weighted_fusion(
            transcript_ranked, visual_ranked, alpha=0.0,
            videos=videos, stage1_chunks=[], has_semantic=True,
        )

        transcript_results = [r for r in results if r["video_id"] == "L01_V001"]
        for r in transcript_results:
            self.assertAlmostEqual(r["confidence"], 0.0, places=2,
                                   msg="alpha=0 should zero out transcript contribution")

    def test_fusion_alpha_one_is_transcript_only(self):
        transcript_ranked = [{
            "video_id": "L01_V001",
            "timestamp_ms": 2500,
            "_score": 0.9,
        }]
        visual_ranked = [{
            "video_id": "L01_V002",
            "frame_id": "L01_V002_000100",
            "frame_number": 100,
            "timestamp_ms": 4000,
            "image_url": "",
            "_score": 0.7,
        }]
        videos = {
            "L01_V001": make_video("L01_V001"),
            "L01_V002": make_video("L01_V002"),
        }

        results = self.strategy._weighted_fusion(
            transcript_ranked, visual_ranked, alpha=1.0,
            videos=videos, stage1_chunks=[], has_semantic=True,
        )

        visual_only_results = [r for r in results if r["video_id"] == "L01_V002"]
        for r in visual_only_results:
            self.assertAlmostEqual(r["confidence"], 0.0, places=2,
                                   msg="alpha=1 should zero out visual-only frames")


class TestOCRBoost(unittest.TestCase):
    """Test OCR boost application."""

    def setUp(self):
        from app.strategies.two_stage_fusion_strategy import TwoStageFusionStrategy
        self.strategy = TwoStageFusionStrategy.__new__(TwoStageFusionStrategy)
        self.strategy.data_provider = MagicMock()

    def test_ocr_boost_applies_5_percent(self):
        results = [{
            "video_id": "L01_V001",
            "frame_id": "L01_V001_000025",
            "confidence": 0.50,
        }]
        ocr_ranked = [{
            "video_id": "L01_V001",
            "frame_id": "L01_V001_000025",
            "_score": 1.0,
        }]

        boosted = self.strategy._apply_ocr_boost(results, ocr_ranked)
        self.assertAlmostEqual(boosted[0]["confidence"], 0.55, places=2)

    def test_ocr_boost_caps_at_1(self):
        results = [{
            "video_id": "L01_V001",
            "frame_id": "L01_V001_000025",
            "confidence": 0.98,
        }]
        ocr_ranked = [{
            "video_id": "L01_V001",
            "frame_id": "L01_V001_000025",
            "_score": 1.0,
        }]

        boosted = self.strategy._apply_ocr_boost(results, ocr_ranked)
        self.assertLessEqual(boosted[0]["confidence"], 1.0)

    def test_ocr_no_match_no_boost(self):
        results = [{
            "video_id": "L01_V001",
            "frame_id": "L01_V001_000025",
            "confidence": 0.50,
        }]
        ocr_ranked = [{
            "video_id": "L01_V001",
            "frame_id": "L01_V001_999999",
            "_score": 1.0,
        }]

        boosted = self.strategy._apply_ocr_boost(results, ocr_ranked)
        self.assertAlmostEqual(boosted[0]["confidence"], 0.50, places=2)


class TestTemporalBoost(unittest.TestCase):
    """Test temporal boost application."""

    def setUp(self):
        from app.strategies.two_stage_fusion_strategy import TwoStageFusionStrategy
        self.strategy = TwoStageFusionStrategy.__new__(TwoStageFusionStrategy)
        self.strategy.data_provider = MagicMock()

    def test_temporal_boost_applies(self):
        results = [
            {"video_id": "L01_V001", "frame_id": "f1", "timestamp_ms": 10000, "confidence": 0.80},
            {"video_id": "L01_V001", "frame_id": "f2", "timestamp_ms": 15000, "confidence": 0.60},
        ]
        query_groups = [
            make_query_group(semantic="test", offset_ms=0),
            make_query_group(semantic="next", offset_ms=5000),
        ]

        boosted = self.strategy._apply_temporal_boost(results, query_groups)
        f1 = next(r for r in boosted if r["frame_id"] == "f1")
        self.assertGreater(f1["confidence"], 0.80,
                           "f1 should get temporal boost from nearby f2")

    def test_temporal_no_match_no_boost(self):
        results = [
            {"video_id": "L01_V001", "frame_id": "f1", "timestamp_ms": 10000, "confidence": 0.80},
            {"video_id": "L01_V001", "frame_id": "f2", "timestamp_ms": 100000, "confidence": 0.60},
        ]
        query_groups = [
            make_query_group(semantic="test", offset_ms=0),
            make_query_group(semantic="next", offset_ms=5000),
        ]

        boosted = self.strategy._apply_temporal_boost(results, query_groups)
        f1 = next(r for r in boosted if r["frame_id"] == "f1")
        self.assertAlmostEqual(f1["confidence"], 0.80, places=2,
                               msg="f2 too far away, no temporal boost")

    def test_temporal_tolerance_boundary(self):
        results = [
            {"video_id": "L01_V001", "frame_id": "f1", "timestamp_ms": 10000, "confidence": 0.80},
            {"video_id": "L01_V001", "frame_id": "f2", "timestamp_ms": 16001, "confidence": 0.60},
        ]
        query_groups = [
            make_query_group(semantic="test", offset_ms=0),
            make_query_group(semantic="next", offset_ms=3000),
        ]

        boosted = self.strategy._apply_temporal_boost(results, query_groups)
        f1 = next(r for r in boosted if r["frame_id"] == "f1")
        self.assertAlmostEqual(f1["confidence"], 0.80, places=2,
                               msg="f2 at 16001ms exceeds tolerance of 3000ms from target 13000ms")


class TestFallbackPaths(unittest.TestCase):
    """Test fallback behavior when stages return empty results."""

    def setUp(self):
        from app.strategies.two_stage_fusion_strategy import TwoStageFusionStrategy
        self.strategy = TwoStageFusionStrategy.__new__(TwoStageFusionStrategy)
        self.strategy.data_provider = MagicMock()

    def test_both_stages_empty_returns_fallback(self):
        raw_data = {
            "frames": [make_frame("f1", "L01_V001", 25, 1000)],
            "ocr": [],
            "transcripts": [],
            "transcript_chunks": [],
            "videos": {"L01_V001": make_video("L01_V001")},
            "video_genre": "All",
            "stage1_chunks": [],
            "stage1_video_ids": set(),
            "stage2_frames": [],
        }
        query_groups = [make_query_group(semantic="test")]

        results = self.strategy.fusion_and_temporal(raw_data, query_groups)
        self.assertGreater(len(results), 0)
        self.assertEqual(results[0]["confidence"], 0.0)

    def test_stage1_empty_falls_back_to_visual(self):
        raw_data = {
            "frames": [
                make_frame("f1", "L01_V001", 25, 1000, score=0.9),
                make_frame("f2", "L01_V001", 50, 2000, score=0.7),
            ],
            "ocr": [],
            "transcripts": [],
            "transcript_chunks": [],
            "videos": {"L01_V001": make_video("L01_V001")},
            "video_genre": "All",
            "stage1_chunks": [],
            "stage1_video_ids": set(),
            "stage2_frames": [
                make_frame("f1", "L01_V001", 25, 1000, score=0.9),
            ],
        }
        query_groups = [make_query_group(semantic="test")]

        results = self.strategy.fusion_and_temporal(raw_data, query_groups)
        self.assertGreater(len(results), 0)
        top = results[0]
        self.assertGreater(top["confidence"], 0.0,
                           "Visual-only fallback should have non-zero confidence")

    def test_stage2_empty_falls_back_to_transcript(self):
        raw_data = {
            "frames": [make_frame("f1", "L01_V001", 62, 2500)],
            "ocr": [],
            "transcripts": [],
            "transcript_chunks": [make_chunk(1, "L01_V001", 0, 5000, score=0.85)],
            "videos": {"L01_V001": make_video("L01_V001")},
            "video_genre": "All",
            "stage1_chunks": [make_chunk(1, "L01_V001", 0, 5000, score=0.85)],
            "stage1_video_ids": {"L01_V001"},
            "stage2_frames": [],
        }
        query_groups = [make_query_group(semantic="test", text="phở bò")]

        results = self.strategy.fusion_and_temporal(raw_data, query_groups)
        self.assertGreater(len(results), 0)
        self.assertGreater(results[0]["confidence"], 0.0,
                           "Transcript-only fallback should have non-zero confidence")


class TestMilvusExprFormatting(unittest.TestCase):
    """Test that Milvus scalar expressions are correctly formatted."""

    def test_single_video_expr(self):
        video_ids = {"L01_V001"}
        quoted = ", ".join(f'"{vid}"' for vid in video_ids)
        expr = f"video_id in [{quoted}]"
        self.assertEqual(expr, 'video_id in ["L01_V001"]')

    def test_multi_video_expr(self):
        video_ids = {"L01_V001", "L03_V002"}
        quoted = ", ".join(f'"{vid}"' for vid in sorted(video_ids))
        expr = f"video_id in [{quoted}]"
        self.assertIn('"L01_V001"', expr)
        self.assertIn('"L03_V002"', expr)
        self.assertTrue(expr.startswith("video_id in ["))
        self.assertTrue(expr.endswith("]"))

    def test_empty_video_ids_no_expr(self):
        video_ids = set()
        expr = None
        if video_ids:
            quoted = ", ".join(f'"{vid}"' for vid in video_ids)
            expr = f"video_id in [{quoted}]"
        self.assertIsNone(expr)

    def test_combined_genre_and_stage2_expr(self):
        stage1_expr = 'video_id in ["L01_V001", "L03_V002"]'
        genre_expr = 'video_id in ["L01_V001", "L01_V002"]'
        combined = f"{stage1_expr} and {genre_expr}"
        self.assertIn(" and ", combined)
        self.assertTrue(combined.startswith("video_id in"))


class TestResultStructure(unittest.TestCase):
    """Test that output results conform to the expected schema."""

    def setUp(self):
        from app.strategies.two_stage_fusion_strategy import TwoStageFusionStrategy
        self.strategy = TwoStageFusionStrategy.__new__(TwoStageFusionStrategy)
        self.strategy.data_provider = MagicMock()

    def test_result_has_all_required_fields(self):
        raw_data = {
            "frames": [
                make_frame("L01_V001_000025", "L01_V001", 25, 1000, score=0.9),
            ],
            "ocr": [],
            "transcripts": [],
            "transcript_chunks": [make_chunk(1, "L01_V001", 0, 5000, score=0.85)],
            "videos": {"L01_V001": make_video("L01_V001")},
            "video_genre": "All",
            "stage1_chunks": [make_chunk(1, "L01_V001", 0, 5000, score=0.85)],
            "stage1_video_ids": {"L01_V001"},
            "stage2_frames": [
                make_frame("L01_V001_000025", "L01_V001", 25, 1000, score=0.9),
            ],
        }
        query_groups = [make_query_group(semantic="test", text="phở")]

        results = self.strategy.fusion_and_temporal(raw_data, query_groups)

        required_fields = {
            "video_id", "youtube_id", "frame_id", "frame_number",
            "timestamp_ms", "confidence", "frame_image_url", "fps",
        }
        for r in results:
            for field in required_fields:
                self.assertIn(field, r, f"Missing field: {field}")
            self.assertIsInstance(r["video_id"], str)
            self.assertIsInstance(r["youtube_id"], str)
            self.assertIsInstance(r["frame_id"], str)
            self.assertIsInstance(r["frame_number"], int)
            self.assertIsInstance(r["timestamp_ms"], int)
            self.assertIsInstance(r["confidence"], float)
            self.assertIsInstance(r["frame_image_url"], str)
            self.assertIsInstance(r["fps"], float)
            self.assertGreaterEqual(r["confidence"], 0.0)
            self.assertLessEqual(r["confidence"], 1.0)

    def test_results_sorted_descending_by_confidence(self):
        raw_data = {
            "frames": [
                make_frame("f1", "L01_V001", 25, 1000, score=0.9),
                make_frame("f2", "L01_V001", 50, 2000, score=0.3),
            ],
            "ocr": [],
            "transcripts": [],
            "transcript_chunks": [],
            "videos": {"L01_V001": make_video("L01_V001")},
            "video_genre": "All",
            "stage1_chunks": [],
            "stage1_video_ids": set(),
            "stage2_frames": [
                make_frame("f1", "L01_V001", 25, 1000, score=0.9),
                make_frame("f2", "L01_V001", 50, 2000, score=0.3),
            ],
        }
        query_groups = [make_query_group(semantic="test")]

        results = self.strategy.fusion_and_temporal(raw_data, query_groups)
        confidences = [r["confidence"] for r in results]
        self.assertEqual(confidences, sorted(confidences, reverse=True),
                         "Results must be sorted descending by confidence")


class TestRankingBuilders(unittest.TestCase):
    """Test individual ranking builder methods."""

    def setUp(self):
        from app.strategies.two_stage_fusion_strategy import TwoStageFusionStrategy
        self.strategy = TwoStageFusionStrategy.__new__(TwoStageFusionStrategy)
        self.strategy.data_provider = MagicMock()

    def test_transcript_ranking_picks_best_per_video(self):
        chunks = [
            make_chunk(1, "L01_V001", 0, 5000, score=0.6),
            make_chunk(2, "L01_V001", 10000, 15000, score=0.9),
            make_chunk(3, "L03_V002", 0, 5000, score=0.7),
        ]
        videos = {
            "L01_V001": make_video("L01_V001"),
            "L03_V002": make_video("L03_V002"),
        }

        ranked = self.strategy._build_transcript_ranking(chunks, videos)
        self.assertEqual(len(ranked), 2)
        l01 = next(r for r in ranked if r["video_id"] == "L01_V001")
        self.assertAlmostEqual(l01["_score"], 0.9,
                               msg="Should pick highest score per video")

    def test_visual_ranking_sorted_descending(self):
        frames = [
            make_frame("f1", "V1", 25, 1000, score=0.3),
            make_frame("f2", "V1", 50, 2000, score=0.9),
            make_frame("f3", "V2", 75, 3000, score=0.6),
        ]
        videos = {"V1": make_video("V1"), "V2": make_video("V2")}

        ranked = self.strategy._build_visual_ranking(frames, videos, has_semantic=True)
        scores = [r["_score"] for r in ranked]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_visual_ranking_no_semantic_zeros_scores(self):
        frames = [make_frame("f1", "V1", 25, 1000, score=0.9)]
        videos = {"V1": make_video("V1")}

        ranked = self.strategy._build_visual_ranking(frames, videos, has_semantic=False)
        self.assertEqual(ranked[0]["_score"], 0.0)


# ═══════════════════════════════════════════════════════════════════════════════
# ASYNC INTEGRATION TESTS (require running Milvus + PostgreSQL)
# ═══════════════════════════════════════════════════════════════════════════════

def _run_async(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_future(coro) if hasattr(loop, 'run_until_future') else loop.run_until_complete(coro)
    finally:
        loop.close()


class TestDataProviderTwoStageAsync(unittest.TestCase):
    """Test DataProvider.get_raw_data_two_stage with mocked DB calls."""

    def _make_data_provider(self):
        from app.data_provider import DataProvider
        dp = DataProvider.__new__(DataProvider)
        dp._collections = {}
        dp._text_encoder = MagicMock()
        dp._text_executor = MagicMock()
        dp._cagra = None
        dp._cagra_lock = MagicMock()
        dp._transcript_search = MagicMock()
        return dp

    def test_stage1_called_when_text_query_present(self):
        dp = self._make_data_provider()

        dp._transcript_search.search_for_candidates = AsyncMock(
            return_value=(
                [make_chunk(1, "L01_V001", 0, 5000, score=0.8)],
                {"L01_V001"},
            )
        )

        fake_vector = np.random.randn(1280).astype("float32")
        fake_vector /= np.linalg.norm(fake_vector)
        dp._encode_text = AsyncMock(return_value=fake_vector)

        async def run():
            with patch("app.data_provider.milvus_client") as mock_milvus, \
                 patch("app.data_provider.postgres_client") as mock_pg:
                mock_milvus.vector_search.return_value = [
                    make_frame("f1", "L01_V001", 62, 2500, score=0.7),
                ]
                mock_milvus.normalize_algorithm.return_value = "hnsw"
                mock_milvus.collection_name_for_algorithm.return_value = "video_frames"
                mock_milvus.has_collection_for_algorithm.return_value = True
                mock_milvus.get_collection_for_name.return_value = MagicMock()
                mock_pg.fetch_transcript_chunks_by_ids = AsyncMock(return_value=[
                    {"chunk_id": 1, "video_id": "L01_V001", "topic": "Ẩm thực",
                     "start_time_ms": 0, "end_time_ms": 5000, "raw_text": "test"},
                ])
                mock_pg.fetch_video_metadata = AsyncMock(return_value=[
                    make_video("L01_V001"),
                ])
                mock_pg.fetch_video_ids_by_genre = AsyncMock(return_value=[])
                mock_pg.search_ocr_text = AsyncMock(return_value=[])
                mock_pg.search_transcript_text = AsyncMock(return_value=[])

                result = await dp.get_raw_data_two_stage(
                    [make_query_group(semantic="phở bò", text="nấu ăn")],
                    limit=100,
                )
                return result

        result = _run_async(run())

        dp._transcript_search.search_for_candidates.assert_called_once()
        self.assertIn("stage1_chunks", result)
        self.assertIn("stage1_video_ids", result)
        self.assertIn("stage2_frames", result)
        self.assertIn("L01_V001", result["stage1_video_ids"])

    def test_stage1_skipped_when_no_text_query(self):
        dp = self._make_data_provider()
        dp._transcript_search.search_for_candidates = AsyncMock()

        fake_vector = np.random.randn(1280).astype("float32")
        fake_vector /= np.linalg.norm(fake_vector)
        dp._encode_text = AsyncMock(return_value=fake_vector)

        async def run():
            with patch("app.data_provider.milvus_client") as mock_milvus, \
                 patch("app.data_provider.postgres_client") as mock_pg:
                mock_milvus.vector_search.return_value = []
                mock_milvus.normalize_algorithm.return_value = "hnsw"
                mock_milvus.collection_name_for_algorithm.return_value = "video_frames"
                mock_milvus.has_collection_for_algorithm.return_value = True
                mock_milvus.get_collection_for_name.return_value = MagicMock()
                mock_pg.fetch_transcript_chunks_by_ids = AsyncMock(return_value=[])
                mock_pg.fetch_video_metadata = AsyncMock(return_value=[])
                mock_pg.fetch_video_ids_by_genre = AsyncMock(return_value=[])
                mock_pg.search_ocr_text = AsyncMock(return_value=[])
                mock_pg.search_transcript_text = AsyncMock(return_value=[])

                result = await dp.get_raw_data_two_stage(
                    [make_query_group(semantic="test only")],
                    limit=100,
                )
                return result

        result = _run_async(run())
        dp._transcript_search.search_for_candidates.assert_not_called()
        self.assertEqual(len(result["stage1_chunks"]), 0)

    def test_stage2_expr_contains_candidate_video_ids(self):
        dp = self._make_data_provider()

        dp._transcript_search.search_for_candidates = AsyncMock(
            return_value=(
                [make_chunk(1, "L01_V001", 0, 5000, score=0.8)],
                {"L01_V001"},
            )
        )

        fake_vector = np.random.randn(1280).astype("float32")
        fake_vector /= np.linalg.norm(fake_vector)
        dp._encode_text = AsyncMock(return_value=fake_vector)

        captured_expr = []

        async def run():
            with patch("app.data_provider.milvus_client") as mock_milvus, \
                 patch("app.data_provider.postgres_client") as mock_pg:
                def capture_search(*args, **kwargs):
                    captured_expr.append(kwargs.get("expr"))
                    return [make_frame("f1", "L01_V001", 62, 2500, score=0.7)]

                mock_milvus.vector_search.side_effect = capture_search
                mock_milvus.normalize_algorithm.return_value = "hnsw"
                mock_milvus.collection_name_for_algorithm.return_value = "video_frames"
                mock_milvus.has_collection_for_algorithm.return_value = True
                mock_milvus.get_collection_for_name.return_value = MagicMock()
                mock_pg.fetch_transcript_chunks_by_ids = AsyncMock(return_value=[
                    {"chunk_id": 1, "video_id": "L01_V001", "topic": "Ẩm thực",
                     "start_time_ms": 0, "end_time_ms": 5000, "raw_text": "test"},
                ])
                mock_pg.fetch_video_metadata = AsyncMock(return_value=[make_video("L01_V001")])
                mock_pg.fetch_video_ids_by_genre = AsyncMock(return_value=[])
                mock_pg.search_ocr_text = AsyncMock(return_value=[])
                mock_pg.search_transcript_text = AsyncMock(return_value=[])

                await dp.get_raw_data_two_stage(
                    [make_query_group(semantic="phở bò", text="nấu ăn")],
                    limit=100,
                )

        _run_async(run())

        self.assertTrue(len(captured_expr) > 0)
        expr = captured_expr[0]
        self.assertIsNotNone(expr)
        self.assertIn("L01_V001", expr)
        self.assertIn("video_id in", expr)


# ═══════════════════════════════════════════════════════════════════════════════
# END-TO-END SCENARIO TESTS (4 scenarios)
# ═══════════════════════════════════════════════════════════════════════════════

class TestScenarioA_HappyPath(unittest.TestCase):
    """
    Scenario A: Happy Path — Strong semantic + transcript match.
    Query: "cách nấu phở bò truyền thống" (how to cook traditional beef pho)
    Expected: High transcript score + high visual score → balanced fusion.
    """

    def test_happy_path_produces_high_confidence(self):
        from app.strategies.two_stage_fusion_strategy import TwoStageFusionStrategy

        strategy = TwoStageFusionStrategy.__new__(TwoStageFusionStrategy)
        strategy.data_provider = MagicMock()

        raw_data = {
            "frames": [
                make_frame("L03_V001_000125", "L03_V001", 125, 5000, score=0.85),
                make_frame("L03_V001_000250", "L03_V001", 250, 10000, score=0.75),
                make_frame("L03_V001_000375", "L03_V001", 375, 15000, score=0.65),
            ],
            "ocr": [],
            "transcripts": [],
            "transcript_chunks": [
                make_chunk(1, "L03_V001", 0, 10000, score=0.92, topic="Ẩm thực",
                           text="cách nấu phở bò truyền thống Nam Định"),
                make_chunk(2, "L03_V001", 10000, 20000, score=0.78, topic="Ẩm thực",
                           text="nguyên liệu cần chuẩn bị"),
            ],
            "videos": {"L03_V001": make_video("L03_V001", genre="Ẩm thực")},
            "video_genre": "Ẩm thực",
            "stage1_chunks": [
                make_chunk(1, "L03_V001", 0, 10000, score=0.92, topic="Ẩm thực"),
                make_chunk(2, "L03_V001", 10000, 20000, score=0.78, topic="Ẩm thực"),
            ],
            "stage1_video_ids": {"L03_V001"},
            "stage2_frames": [
                make_frame("L03_V001_000125", "L03_V001", 125, 5000, score=0.85),
                make_frame("L03_V001_000250", "L03_V001", 250, 10000, score=0.75),
            ],
        }
        query_groups = [make_query_group(semantic="cách nấu phở bò", text="phở bò truyền thống")]

        results = strategy.fusion_and_temporal(raw_data, query_groups)

        self.assertGreater(len(results), 0)
        top = results[0]
        self.assertGreater(top["confidence"], 0.1,
                           "Happy path should produce meaningful confidence")
        self.assertEqual(top["video_id"], "L03_V001")


class TestScenarioB_VisualHeavy(unittest.TestCase):
    """
    Scenario B: Visual-heavy query — objects/scenery with NO dialogue.
    Query: "cánh đồng lúa xanh mướt" (lush green rice field)
    Expected: Low transcript score, high visual score → alpha decreases.
    """

    def test_visual_heavy_lower_alpha(self):
        from app.strategies.two_stage_fusion_strategy import _compute_dynamic_alpha

        transcript_score = 0.15
        visual_score = 0.85
        alpha = _compute_dynamic_alpha(transcript_score, visual_score,
                                       has_semantic=True, genre="Du lịch")

        self.assertLess(alpha, 0.25,
                        "Visual-heavy query should have low alpha (transcript weight)")

    def test_visual_heavy_produces_results(self):
        from app.strategies.two_stage_fusion_strategy import TwoStageFusionStrategy

        strategy = TwoStageFusionStrategy.__new__(TwoStageFusionStrategy)
        strategy.data_provider = MagicMock()

        raw_data = {
            "frames": [
                make_frame("L02_V003_000500", "L02_V003", 500, 20000, score=0.88),
            ],
            "ocr": [],
            "transcripts": [],
            "transcript_chunks": [
                make_chunk(1, "L02_V003", 15000, 25000, score=0.12, topic="Du lịch",
                           text="và hôm nay thời tiết rất đẹp"),
            ],
            "videos": {"L02_V003": make_video("L02_V003", genre="Du lịch")},
            "video_genre": "Du lịch",
            "stage1_chunks": [
                make_chunk(1, "L02_V003", 15000, 25000, score=0.12, topic="Du lịch"),
            ],
            "stage1_video_ids": {"L02_V003"},
            "stage2_frames": [
                make_frame("L02_V003_000500", "L02_V003", 500, 20000, score=0.88),
            ],
        }
        query_groups = [make_query_group(semantic="cánh đồng lúa xanh mướt")]

        results = strategy.fusion_and_temporal(raw_data, query_groups)
        self.assertGreater(len(results), 0)
        self.assertGreater(results[0]["confidence"], 0.0)


class TestScenarioC_ZeroTranscript(unittest.TestCase):
    """
    Scenario C: Zero transcript matches — gibberish query.
    Query: "xyzzy foobar nonsense"
    Expected: Stage 1 returns 0 → fallback to pure visual.
    """

    def test_zero_transcript_fallback(self):
        from app.strategies.two_stage_fusion_strategy import TwoStageFusionStrategy

        strategy = TwoStageFusionStrategy.__new__(TwoStageFusionStrategy)
        strategy.data_provider = MagicMock()

        raw_data = {
            "frames": [
                make_frame("f1", "L01_V001", 25, 1000, score=0.7),
                make_frame("f2", "L01_V002", 50, 2000, score=0.5),
            ],
            "ocr": [],
            "transcripts": [],
            "transcript_chunks": [],
            "videos": {
                "L01_V001": make_video("L01_V001"),
                "L01_V002": make_video("L01_V002"),
            },
            "video_genre": "All",
            "stage1_chunks": [],
            "stage1_video_ids": set(),
            "stage2_frames": [
                make_frame("f1", "L01_V001", 25, 1000, score=0.7),
            ],
        }
        query_groups = [make_query_group(semantic="xyzzy foobar")]

        results = strategy.fusion_and_temporal(raw_data, query_groups)
        self.assertGreater(len(results), 0)
        self.assertEqual(results[0]["video_id"], "L01_V001",
                         "Should fallback to visual-only, returning highest visual score")


class TestScenarioD_OCRAndTemporal(unittest.TestCase):
    """
    Scenario D: OCR + Temporal boundary test.
    Multi-step query with screen text and step-by-step timeline.
    Query Step 1: "bản tin thời sự" (news broadcast) at t=0
    Query Step 2: "phóng viên hiện trường" (field reporter) at t=5000ms
    """

    def test_multi_step_temporal(self):
        from app.strategies.two_stage_fusion_strategy import TwoStageFusionStrategy

        strategy = TwoStageFusionStrategy.__new__(TwoStageFusionStrategy)
        strategy.data_provider = MagicMock()

        raw_data = {
            "frames": [
                make_frame("f1", "L01_V004", 100, 4000, score=0.8),
                make_frame("f2", "L01_V004", 225, 9000, score=0.7),
                make_frame("f3", "L01_V004", 350, 14000, score=0.6),
            ],
            "ocr": [
                {"frame_id": "f1", "video_id": "L01_V004", "frame_number": 100,
                 "timestamp_ms": 4000, "ocr_text": "BẢN TIN THỜI SỰ 19H"},
                {"frame_id": "f2", "video_id": "L01_V004", "frame_number": 225,
                 "timestamp_ms": 9000, "ocr_text": "PHÓNG VIÊN HIỆN TRƯỜNG"},
            ],
            "transcripts": [],
            "transcript_chunks": [
                make_chunk(1, "L01_V004", 0, 8000, score=0.85, topic="Thời sự",
                           text="kính chào quý vị đến với bản tin thời sự"),
                make_chunk(2, "L01_V004", 8000, 16000, score=0.75, topic="Thời sự",
                           text="phóng viên chúng tôi đang có mặt tại hiện trường"),
            ],
            "videos": {"L01_V004": make_video("L01_V004", genre="Thời sự")},
            "video_genre": "Thời sự",
            "stage1_chunks": [
                make_chunk(1, "L01_V004", 0, 8000, score=0.85, topic="Thời sự"),
                make_chunk(2, "L01_V004", 8000, 16000, score=0.75, topic="Thời sự"),
            ],
            "stage1_video_ids": {"L01_V004"},
            "stage2_frames": [
                make_frame("f1", "L01_V004", 100, 4000, score=0.8),
                make_frame("f2", "L01_V004", 225, 9000, score=0.7),
            ],
        }
        query_groups = [
            make_query_group(semantic="bản tin thời sự", text="thời sự", offset_ms=0),
            make_query_group(semantic="phóng viên hiện trường", text="phóng viên", offset_ms=5000),
        ]

        results = strategy.fusion_and_temporal(raw_data, query_groups)
        self.assertGreater(len(results), 0)

        ocr_frames = {r["frame_id"] for r in results if r.get("_transcript_score", 0) > 0}
        self.assertGreater(len(results), 0)
        for r in results:
            self.assertGreaterEqual(r["confidence"], 0.0)
            self.assertLessEqual(r["confidence"], 1.0)


# ═══════════════════════════════════════════════════════════════════════════════
# SMOKE TEST (runnable via `python scripts/test_two_stage_fusion.py --smoke`)
# ═══════════════════════════════════════════════════════════════════════════════

def run_smoke_test():
    """Quick smoke test that verifies the strategy can be imported and instantiated."""
    print("=" * 70)
    print("SMOKE TEST: TwoStageFusionStrategy")
    print("=" * 70)

    from app.strategies.two_stage_fusion_strategy import (
        TwoStageFusionStrategy,
        _compute_dynamic_alpha,
        GENRE_ALPHA_MAP,
        DEFAULT_ALPHA,
    )

    print("\n[1/6] Import check ... ", end="")
    print("OK")

    print("[2/6] Strategy metadata ... ", end="")
    assert TwoStageFusionStrategy.name == "Two-Stage Fusion v1"
    assert TwoStageFusionStrategy.author == "Team AIC 2026"
    assert TwoStageFusionStrategy.version == "1.0"
    print("OK")

    print("[3/6] Dynamic alpha bounds ... ", end="")
    for t in [0.0, 0.1, 0.5, 0.9, 1.0]:
        for v in [0.0, 0.1, 0.5, 0.9, 1.0]:
            a = _compute_dynamic_alpha(t, v, True, "All")
            assert 0.0 <= a <= 1.0, f"alpha={a} out of bounds for t={t}, v={v}"
    print("OK")

    print("[4/6] Genre alpha map ... ", end="")
    assert len(GENRE_ALPHA_MAP) == 14
    assert DEFAULT_ALPHA == 0.35
    print(f"OK ({len(GENRE_ALPHA_MAP)} genres)")

    print("[5/6] Strategy instantiation ... ", end="")
    strategy = TwoStageFusionStrategy.__new__(TwoStageFusionStrategy)
    strategy.data_provider = MagicMock()
    print("OK")

    print("[6/6] Fusion with sample data ... ", end="")
    raw_data = {
        "frames": [make_frame("f1", "V1", 25, 1000, score=0.8)],
        "ocr": [],
        "transcripts": [],
        "transcript_chunks": [make_chunk(1, "V1", 0, 5000, score=0.9)],
        "videos": {"V1": make_video("V1")},
        "video_genre": "All",
        "stage1_chunks": [make_chunk(1, "V1", 0, 5000, score=0.9)],
        "stage1_video_ids": {"V1"},
        "stage2_frames": [make_frame("f1", "V1", 25, 1000, score=0.8)],
    }
    results = strategy.fusion_and_temporal(
        raw_data, [make_query_group(semantic="test", text="test")]
    )
    assert len(results) > 0
    assert all(0.0 <= r["confidence"] <= 1.0 for r in results)
    print(f"OK ({len(results)} results)")

    print("\n" + "=" * 70)
    print("ALL SMOKE TESTS PASSED")
    print("=" * 70)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Two-Stage Fusion Test Suite")
    parser.add_argument("--smoke", action="store_true", help="Run quick smoke test only")
    parser.add_argument("--unit", action="store_true", help="Run unit tests only")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = parser.parse_args()

    if args.smoke:
        run_smoke_test()
    elif args.unit:
        verbosity = 2 if args.verbose else 1
        suite = unittest.TestLoader().loadTestsFromTestCase(unittest.TestCase)
        loader = unittest.TestLoader()
        suite = unittest.TestSuite()
        for test_class in [
            TestDynamicAlpha,
            TestFusionMath,
            TestOCRBoost,
            TestTemporalBoost,
            TestFallbackPaths,
            TestMilvusExprFormatting,
            TestResultStructure,
            TestRankingBuilders,
            TestScenarioA_HappyPath,
            TestScenarioB_VisualHeavy,
            TestScenarioC_ZeroTranscript,
            TestScenarioD_OCRAndTemporal,
        ]:
            suite.addTests(loader.loadTestsFromTestCase(test_class))
        runner = unittest.TextTestRunner(verbosity=verbosity)
        result = runner.run(suite)
        sys.exit(0 if result.wasSuccessful() else 1)
    else:
        run_smoke_test()
        print("\nRunning full unit test suite...\n")
        verbosity = 2 if args.verbose else 1
        loader = unittest.TestLoader()
        suite = unittest.TestSuite()
        for test_class in [
            TestDynamicAlpha,
            TestFusionMath,
            TestOCRBoost,
            TestTemporalBoost,
            TestFallbackPaths,
            TestMilvusExprFormatting,
            TestResultStructure,
            TestRankingBuilders,
            TestDataProviderTwoStageAsync,
            TestScenarioA_HappyPath,
            TestScenarioB_VisualHeavy,
            TestScenarioC_ZeroTranscript,
            TestScenarioD_OCRAndTemporal,
        ]:
            suite.addTests(loader.loadTestsFromTestCase(test_class))
        runner = unittest.TextTestRunner(verbosity=verbosity)
        result = runner.run(suite)
        sys.exit(0 if result.wasSuccessful() else 1)
