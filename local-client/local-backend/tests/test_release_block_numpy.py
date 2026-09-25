"""Local exact search removes a bad video before applying top-k."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from app.db import numpy_vector_store as store


class ReleaseBlockNumpyTests(unittest.TestCase):
    def test_bad_video_cannot_win_search_or_direct_lookup(self):
        frame_id = np.array(['N031-V003_000001', 'M01_V001_000002', 'S01-V001_000003'])
        video_id = np.array(['N031-V003', 'M01_V001', 'S01-V001'])
        fake = SimpleNamespace(
            vectors=np.array([[1, 0], [0.9, 0.1], [0, 1]], dtype=np.float32),
            frame_id=frame_id, video_id=video_id,
            video_genre=np.array(['All'] * 3),
            frame_number=np.array([1, 2, 3]),
            timestamp_ms=np.array([10, 20, 30]),
            youtube_id=np.array([''] * 3),
            rows_for=lambda ids: np.array([list(frame_id).index(value) for value in ids], dtype=np.int64),
        )
        with (patch.object(store, 'get_store', return_value=fake),
              patch.object(store, 'release_blocked_video_ids',
                           return_value=frozenset({'N031-V003'}))):
            hits = store.vector_search([1, 0], top_k=1)
            self.assertEqual([hit['video_id'] for hit in hits], ['M01_V001'])
            self.assertEqual(store.frames_in_range('N031-V003', 0, 100), [])
            self.assertEqual(store.context_frames('N031-V003', 0, 100), ([], [], []))
            self.assertNotIn('N031-V003', store.video_ids())
            self.assertEqual(store.frame_vectors(['N031-V003_000001']), {})

    def test_hours_wide_context_is_bounded_and_keeps_span_endpoints(self):
        count = 5000
        frame_id = np.array([f'S01-V010_{i:06d}' for i in range(count)])
        fake = SimpleNamespace(
            video_id=np.array(['S01-V010'] * count), frame_id=frame_id,
            frame_number=np.arange(count), timestamp_ms=np.arange(count) * 10_000,
            youtube_id=np.array([''] * count),
        )
        with (patch.object(store, 'get_store', return_value=fake),
              patch.object(store, 'release_blocked_video_ids', return_value=frozenset())):
            before, middle, after = store.context_frames(
                'S01-V010', 0, int(fake.timestamp_ms[-1]), expand=20
            )
        self.assertEqual(before, [])
        self.assertEqual(after, [])
        self.assertLessEqual(len(middle), store.CONTEXT_FRAME_LIMIT)
        self.assertEqual(middle[0]['frame_id'], frame_id[0])
        self.assertEqual(middle[-1]['frame_id'], frame_id[-1])

    def test_negative_context_expansion_returns_no_side_frames(self):
        fake = SimpleNamespace(
            video_id=np.array(['M01_V001'] * 3),
            frame_id=np.array(['f0', 'f1', 'f2']),
            frame_number=np.arange(3), timestamp_ms=np.arange(3) * 1000,
            youtube_id=np.array([''] * 3),
        )
        with (patch.object(store, 'get_store', return_value=fake),
              patch.object(store, 'release_blocked_video_ids', return_value=frozenset())):
            before, middle, after = store.context_frames(
                'M01_V001', 1000, 1000, expand=-1
            )
        self.assertEqual(before, [])
        self.assertEqual([row['frame_id'] for row in middle], ['f1'])
        self.assertEqual(after, [])


if __name__ == '__main__':
    unittest.main()
