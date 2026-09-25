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

    def test_rescans_after_yielding_an_earlier_ready_video(self) -> None:
        with tempfile.TemporaryDirectory() as temp_name:
            root = Path(temp_name)
            sentinel = root / ".done"
            entries = [Entry("first.mp4"), Entry("second.mp4")]
            queue = ReadyEntryQueue(entries, root, sentinel)

            first = root / "first"
            first.mkdir()
            (first / "scenes.json").write_text("{}")
            (first / "keyframes.json").write_text("{}")

            scheduled = iter(queue)
            self.assertEqual(next(scheduled).name, "first.mp4")

            # TransNet can publish more metadata while the caller spends a long
            # time embedding the first item, then publish its completion marker.
            second = root / "second"
            second.mkdir()
            (second / "scenes.json").write_text("{}")
            (second / "keyframes.json").write_text("{}")
            sentinel.touch()

            self.assertEqual(next(scheduled).name, "second.mp4")
            with self.assertRaises(StopIteration):
                next(scheduled)
            self.assertEqual(queue.take_missing(), [])


if __name__ == "__main__":
    unittest.main()

class FinalPublishRaceTest(unittest.TestCase):
    def test_final_publish_between_scan_and_sentinel_is_not_missing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            class RacingQueue(ReadyEntryQueue):
                @property
                def producer_finished(self):
                    video = root / 'last'
                    video.mkdir(exist_ok=True)
                    (video / 'scenes.json').write_text('{}')
                    (video / 'keyframes.json').write_text('{}')
                    return True
            queue = RacingQueue([Entry('last.mp4'), Entry('missing.mp4')], root, root / '.done')
            self.assertEqual([entry.name for entry in queue], ['last.mp4'])
            self.assertEqual([entry.name for entry in queue.take_missing()], ['missing.mp4'])
