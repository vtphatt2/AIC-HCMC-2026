import unittest
from unittest.mock import patch
import numpy as np
from app.services.source_timeline import monotonic_entries, select_pictures, generation_signature
from app.services.readiness_policy import (SelectionPolicy, decode_provenance,
                                           exceptional_decode_provenance,
                                           source_map_decoder_threads, verified_embed_decoder_threads)


class SourceTimelineTests(unittest.TestCase):
    def test_vfr_grid_retains_valid_existing_and_original_ids(self):
        table = np.array([[0, 100, 1], [1, 150, 2], [2, 149, 3], [3, 150, 4],
                          [4, 300, 5], [5, 500, 6], [6, 600, 7]], dtype=np.int64)
        rows, omissions = select_pictures(table, 100, [{'start_frame': 0, 'end_frame': 6}],
                                         [{'frame_number': 1}, {'frame_number': 2}])
        self.assertEqual([r['frame_number'] for r in rows], [0, 1, 4, 5])
        self.assertEqual([r['timestamp_ms'] for r in rows], [1000, 1500, 3000, 5000])
        self.assertEqual([r['frame_number'] for r in omissions], [2, 3])
        self.assertEqual(rows[2]['source_checksum'], 5)

    def test_no_five_per_scene_cap_and_interval_is_configurable(self):
        table = np.array([[i, i * 25, i] for i in range(101)], dtype=np.int64)
        scenes = [{'start_frame': 0, 'end_frame': 100}]
        rows, _ = select_pictures(table, 100, scenes, [])
        more, _ = select_pictures(table, 100, scenes, [], SelectionPolicy(interval_seconds=1))
        self.assertEqual(len(rows), 13)
        self.assertEqual(len(more), 26)
        for interval in [0, -1, float('nan')]:
            with self.assertRaises(ValueError): SelectionPolicy(interval_seconds=interval)

    def test_sparse_map_rejected(self):
        with self.assertRaises(ValueError):
            monotonic_entries(np.array([[0, 0, 1], [2, 20, 2]], dtype=np.int64))

    def test_verified_profile_changes_only_its_generation(self):
        with patch('app.services.readiness_policy.decoder_profile',
                   side_effect=lambda video, index: {'threads': 1} if video == 'N099-V999' else None):
            self.assertEqual((source_map_decoder_threads('N099-V999', None),
                              verified_embed_decoder_threads('N099-V999', None)), (1, 1))
            self.assertEqual((source_map_decoder_threads('N032-V003', None),
                              verified_embed_decoder_threads('N032-V003', None)), (4, 4))
            self.assertIsNone(exceptional_decode_provenance('N032-V003', None))
            args = ({'size': 1}, SelectionPolicy(), {'scenes': []}, [])
            old = generation_signature(*args)
            self.assertEqual(old, generation_signature(*args,
                             exceptional_decode_provenance('N032-V003', None)))
            self.assertNotEqual(old, generation_signature(*args,
                                decode_provenance('N032-V003', None)))
            self.assertNotEqual(old, generation_signature(*args,
                                exceptional_decode_provenance('N099-V999', None)))
