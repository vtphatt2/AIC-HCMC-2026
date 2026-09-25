"""Diagnostic matching must not change indexed search identities."""
import unittest
from unittest.mock import AsyncMock, patch

import numpy as np

from app.data_provider import DataProvider
from scripts.build_n_frame_identity import match_selected_frames


class NFrameIdentityTests(unittest.TestCase):
    def test_checksum_mapping_handles_variable_fps_and_repeated_stills(self):
        table = np.array([
            [0, 2000, 11], [1, 2400, 22], [2, 2800, 33],
            [3, 3200, 22], [4, 6000, 44],
        ], dtype=np.int64)
        mapping, mismatches = match_selected_frames(
            [1, 4], [(22, .04), (44, .4)], table, 25.0, 10_000)
        self.assertEqual(mapping, {"1": [1, 40], "4": [4, 400]})
        self.assertEqual(mismatches, [])
        mapping, mismatches = match_selected_frames([1], [(999, .04)], table, 25.0, 10_000)
        self.assertEqual(mapping, {"1": [1, 40]})
        self.assertEqual(mismatches, [1])
        with self.assertRaisesRegex(ValueError, "No decoded source frame"):
            match_selected_frames([1], [(999, 1.0)], table, 25.0, 10_000)

class HydrationIdentityTests(unittest.IsolatedAsyncioTestCase):
    async def test_hydration_preserves_frame_timestamp_vector_identity_and_score(self):
        hits = [
            {"video_id": "N001-V001", "frame_id": "N001-V001_000005",
             "frame_number": 5, "timestamp_ms": 200, "image_url": "/static/stale-n-card.jpg", "score": .8},
            {"video_id": "M01_V001", "frame_id": "M01_V001_000005",
             "frame_number": 5, "timestamp_ms": 200, "image_url": "", "score": .7},
        ]
        with patch("app.data_provider.postgres_client.fetch_video_metadata",
                   new=AsyncMock(return_value=[])):
            await DataProvider._hydrate_frames(hits)
        self.assertEqual([hit["frame_id"] for hit in hits],
                         ["N001-V001_000005", "M01_V001_000005"])
        self.assertEqual([hit["score"] for hit in hits], [.8, .7])
        self.assertEqual((hits[0]["frame_number"], hits[0]["timestamp_ms"]), (5, 200))
        self.assertEqual(hits[0]["image_url"],
                         "/api/zip-frame/N001-V001/200?frame_number=5&v=7")
        self.assertEqual((hits[1]["frame_number"], hits[1]["timestamp_ms"]), (5, 200))


if __name__ == "__main__":
    unittest.main()
