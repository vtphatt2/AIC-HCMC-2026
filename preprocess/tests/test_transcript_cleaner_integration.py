from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path

from preprocess.transcript_cleaner.models import Chunk, CleanedChunk
from preprocess.transcript_cleaner.pipeline import TranscriptPipeline
from preprocess.transcript_cleaner.state import StateStore
from preprocess.transcript_cleaner.verify import verify_output
from preprocess.transcript_cleaner.video_request import VideoRequestConfig


class FakeCleaner:
    async def clean(self, chunk: Chunk) -> CleanedChunk:
        return CleanedChunk(
            chunk_id=chunk.chunk_id,
            texts_by_index={segment.index: segment.text.capitalize() for segment in chunk.segments},
        )


class TranscriptCleanerIntegrationTest(unittest.TestCase):
    def test_preserves_timestamps_and_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp_name:
            root = Path(temp_name)
            source = root / "L30_V001.jsonl"
            rows = [
                {"start_time_ms": 0, "end_time_ms": 900, "text": "xin chao", "speaker": "A"},
                {"start_time_ms": 900, "end_time_ms": 1500, "text": "[âm nhạc]", "speaker": None},
            ]
            source.write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                encoding="utf-8",
            )
            pipeline = TranscriptPipeline(
                cleaner=FakeCleaner(),
                state_store=StateStore(root / "state"),
                output_dir=root / "clean",
                request_config=VideoRequestConfig(),
                model="fake",
                concurrency=2,
            )

            summary = asyncio.run(pipeline.run([source]))

            self.assertEqual(summary.completed_files, 1)
            output = root / "clean" / source.name
            report = verify_output(source, output)
            self.assertEqual(report.segment_count, 2)
            cleaned = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(cleaned[0]["start_time_ms"], 0)
            self.assertEqual(cleaned[0]["speaker"], "A")
            self.assertEqual(cleaned[0]["text"], "Xin chao")
            self.assertEqual(cleaned[1], rows[1])


if __name__ == "__main__":
    unittest.main()
