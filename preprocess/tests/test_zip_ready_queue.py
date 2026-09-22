from __future__ import annotations

import sys
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path


ENGINE_SRC = Path(__file__).resolve().parents[2] / "keyframe_pipeline_global_v9_3" / "src"
if str(ENGINE_SRC) not in sys.path:
    sys.path.insert(0, str(ENGINE_SRC))

from pipeline.ready_queue import ReadyEntryQueue


@dataclass(frozen=True)
class Entry:
    name: str


class ReadyEntryQueueTest(unittest.TestCase):
    def test_yields_whichever_video_finishes_first(self) -> None:
        with tempfile.TemporaryDirectory() as temp_name:
            root = Path(temp_name)
            entries = [Entry("first.mp4"), Entry("second.mp4")]
            queue = ReadyEntryQueue(entries, root, root / ".done")
            second = root / "second"
            second.mkdir()
            (second / "scenes.json").write_text("{}")
            (second / "keyframes.json").write_text("{}")

            self.assertEqual([entry.name for entry in queue.take_ready()], ["second.mp4"])
            self.assertTrue(queue)

    def test_reports_remaining_entries_after_producer_sentinel(self) -> None:
        with tempfile.TemporaryDirectory() as temp_name:
            root = Path(temp_name)
            sentinel = root / ".done"
            queue = ReadyEntryQueue([Entry("missing.mp4")], root, sentinel)
            self.assertFalse(queue.producer_finished)
            sentinel.touch()
            self.assertTrue(queue.producer_finished)
            self.assertEqual([entry.name for entry in queue.take_missing()], ["missing.mp4"])
            self.assertFalse(queue)


if __name__ == "__main__":
    unittest.main()
