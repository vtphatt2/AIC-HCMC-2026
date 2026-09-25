"""The readiness audit must separate export rounding from changed vectors."""
import unittest
import hashlib
import json

import numpy as np

from scripts.audit_readiness import (MAX_EXPORT_VECTOR_ABS_ERROR, export_vector_error,
                                     unexpected_export_ids,
                                     require_browser_playback,
                                     selected_timestamp_ms, verify_packaged_generation,
                                     verify_n_source_identity, verify_scene_association,
                                     verify_result_inventory)


class ExportVectorAuditTests(unittest.TestCase):
    def test_float32_renormalization_does_not_mark_a_vector_changed(self):
        packaged = np.array([0.6, 0.8, 0.0], dtype=np.float32)
        exported = packaged.copy()
        exported[0] = np.nextafter(exported[0], np.float32(1))
        self.assertLess(export_vector_error(packaged, exported), MAX_EXPORT_VECTOR_ABS_ERROR)

    def test_material_or_nonfinite_change_is_rejected(self):
        packaged = np.array([0.6, 0.8, 0.0], dtype=np.float32)
        changed = packaged.copy()
        changed[0] += 0.001
        self.assertGreater(export_vector_error(packaged, changed), MAX_EXPORT_VECTOR_ABS_ERROR)
        changed[0] = np.nan
        self.assertEqual(export_vector_error(packaged, changed), float('inf'))

    def test_numpy_export_must_not_keep_rows_from_excluded_generation(self):
        export_lookup = {'N001-V001_000001': 0, 'N001-V002_000001': 1}
        self.assertEqual(unexpected_export_ids(export_lookup, {'N001-V001_000001'}),
                         {'N001-V002_000001'})

    def test_versioned_n_timestamp_uses_source_pts_and_requires_identity(self):
        item = {'frame_number': 25, 'source_pts': 12345,
                'source_timebase': 10000, 'source_checksum': 123,
                'timestamp_ms': 1235}
        self.assertEqual(selected_timestamp_ms('N001-V001', item, 25.0, 2), 1235)
        self.assertEqual(selected_timestamp_ms('N001-V001', {'frame_number': 25}, 25.0, 1), 1000)
        with self.assertRaisesRegex(ValueError, 'source identity'):
            selected_timestamp_ms('N001-V001', {'frame_number': 25}, 25.0, 2)
        with self.assertRaisesRegex(ValueError, 'source PTS'):
            selected_timestamp_ms('N001-V001', {**item, 'timestamp_ms': 1000}, 25.0, 2)

    def test_versioned_result_uses_bundled_verified_digests(self):
        keyframes = {'version': 2, 'generation': 'a' * 64, 'num_keyframes': 1,
                     'keyframes': [{'frame_number': 0}]}
        payloads = {'keyframes': json.dumps(keyframes).encode(),
                    'scenes': b'{"num_frames":1}', 'embeddings': b'vector bytes'}
        marker = {'version': 1, 'generation': keyframes['generation'], 'rows': 1,
                  'source_checksums': 'exhaustive', 'source_time_base_verified': True,
                  'model': 'PE-Core-bigG', 'preprocess': 'scale=448', 'published': False,
                  **{f'{name}_sha256': hashlib.sha256(data).hexdigest()
                     for name, data in payloads.items()}}
        verify_packaged_generation('N001-V001', keyframes, marker, payloads)
        with self.assertRaisesRegex(ValueError, 'verification marker'):
            verify_packaged_generation('N001-V001', keyframes,
                                       {**marker, 'preprocess': ''}, payloads)
        with self.assertRaisesRegex(ValueError, 'digest'):
            verify_packaged_generation('N001-V001', keyframes, marker,
                                       {**payloads, 'embeddings': b'changed bytes'})

    def test_versioned_source_rows_must_match_map_and_omissions(self):
        table = np.array([[0, 1000, 10], [1, 2000, 20], [2, 1500, 30]], dtype=np.int64)
        selected = {'version': 2, 'source_identity': {'source': 'same'},
                    'source_time_base': {'num': 1, 'den': 1000},
                    'source_frame_count': 3,
                    'omitted_entries': [{'frame_number': 2, 'source_pts': 1500,
                                         'source_checksum': 30,
                                         'reason': 'non-increasing presentation timestamp'}],
                    'keyframes': [{'frame_number': 1, 'source_pts': 2000,
                                   'source_checksum': 20, 'source_timebase': 1000,
                                   'timestamp_ms': 2000}]}
        verify_n_source_identity('N001-V001', selected, table, 1000, {'source': 'same'})
        selected['keyframes'][0]['source_checksum'] = 99
        with self.assertRaisesRegex(ValueError, 'checksum'):
            verify_n_source_identity('N001-V001', selected, table, 1000, {'source': 'same'})
        selected['keyframes'][0]['source_checksum'] = 20
        selected['omitted_entries'][0]['source_pts'] = 9999
        with self.assertRaisesRegex(ValueError, 'omitted presentation'):
            verify_n_source_identity('N001-V001', selected, table, 1000, {'source': 'same'})

    def test_versioned_n_transition_gap_is_explicit_not_a_missing_frame(self):
        scenes = [{'start_frame': 0, 'end_frame': 2},
                  {'start_frame': 5, 'end_frame': 7}]
        selected = {'version': 2, 'keyframes': [
            {'frame_number': 1, 'scene_index': 0},
            {'frame_number': 3, 'scene_index': -1,
             'scene_association': 'transition gap in original TransNet output'},
            {'frame_number': 6, 'scene_index': 1},
        ]}
        verify_scene_association('N001-V001', selected, scenes)
        selected['keyframes'][1]['scene_index'] = 0
        with self.assertRaisesRegex(ValueError, 'scene association'):
            verify_scene_association('N001-V001', selected, scenes)
        with self.assertRaisesRegex(ValueError, 'transition gap'):
            verify_scene_association('M01_V001', {'version': 1,
                'keyframes': [{'frame_number': 3}]}, scenes)

    def test_n_archive_omissions_need_manifest_and_live_release_block(self):
        source = {'N001-V001': {'archive': 'Video_N001-N010.zip'},
                  'N001-V002': {'archive': 'Video_N001-N010.zip'}}
        packaged = {'N001-V001'}
        manifest = {'version': 2, 'videos': [{'video_id': 'N001-V001'}],
                    'release_blocked_excluded': ['N001-V002']}
        self.assertEqual(verify_result_inventory('N001-N010_results.zip', source,
                                                packaged, manifest, {'N001-V002'}),
                         {'N001-V002'})
        with self.assertRaisesRegex(ValueError, 'not release-blocked'):
            verify_result_inventory('N001-N010_results.zip', source,
                                    packaged, manifest, set())
        with self.assertRaisesRegex(ValueError, 'undocumented'):
            verify_result_inventory('N001-N010_results.zip', source,
                                    packaged, None, {'N001-V002'})

    def test_every_published_n_video_requires_a_validated_browser_copy(self):
        require_browser_playback('N001-V001', '/validated/video.mp4')
        with self.assertRaisesRegex(ValueError, 'browser playback copy'):
            require_browser_playback('N001-V001', None)
        require_browser_playback('M01_V001', None)


if __name__ == '__main__':
    unittest.main()
