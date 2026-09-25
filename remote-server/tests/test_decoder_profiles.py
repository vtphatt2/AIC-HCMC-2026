import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from app.services.decoder_profiles import decoder_profile, register_profile, source_identity
from app.services.exact_frame_pts import index_path
from app.services.readiness_policy import (source_map_decoder_threads, verified_embed_decoder_threads,
                                          playback_decoder_threads, exceptional_decode_provenance,
                                          decode_provenance, PlaybackPolicy)


class DecoderProfileTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(); self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.source = self.root / 'source.zip'; self.source.write_bytes(b'original')
        self.index = SimpleNamespace(zip_path=self.source, data_offset=0, video_size=8, timescale=1000)
        env = patch.dict(os.environ, {'SOURCE_DECODER_PROFILES_PATH': str(self.root / 'profiles.json')})
        env.start(); self.addCleanup(env.stop)
        self.video = 'N999-V999'
        self.map = index_path(self.video, self.index, self.root, decoder_threads=1)
        np.save(self.map, np.array([[0, 200, 11], [1, 300, 22]], dtype=np.int64))
        self.evidence = {'video_id': self.video, 'status': 'exact', 'source': source_identity(self.index),
                         'decoder_threads': 1, 'ffmpeg_exit': 0, 'source_map_rows': 2, 'decoded_rows': 2,
                         'pts_mismatch_count': 0, 'checksum_mismatch_count': 0,
                         'time_bases': [[1, 1000]], 'expected_time_base': [1, 1000],
                         'map_path': str(self.map), 'map_size': self.map.stat().st_size,
                         'map_mtime_ns': self.map.stat().st_mtime_ns}
        self.report = self.root / 'replay.json'; self.report.write_text(json.dumps(self.evidence))

    def register(self, evidence=None):
        with patch('app.services.exact_frame_pts.index_path', return_value=self.map):
            return register_profile(self.video, self.index, evidence or self.evidence, self.report)

    def test_any_verified_source_gets_one_profile_for_all_derivatives(self):
        ordinary = index_path(self.video, self.index, self.root)
        self.assertEqual(source_map_decoder_threads(self.video, self.index), 4)
        self.register()
        self.assertEqual(source_map_decoder_threads(self.video, self.index), 1)
        self.assertEqual(verified_embed_decoder_threads(self.video, self.index), 1)
        self.assertEqual(playback_decoder_threads(self.video, PlaybackPolicy(threads=3), self.index), 1)
        self.assertEqual(exceptional_decode_provenance(self.video, self.index),
                         {'source_map_threads': 1, 'verified_embed_threads': 1})
        self.assertEqual(index_path(self.video, self.index, self.root), self.map)
        self.assertNotEqual(ordinary, self.map)
        self.assertEqual(source_map_decoder_threads('N-other', self.index), 4)
        self.assertEqual(verified_embed_decoder_threads('N-other', self.index), 4)
        self.assertEqual(playback_decoder_threads(
            'N-other', PlaybackPolicy(threads=3), self.index), 4)
        self.assertEqual(decode_provenance('N-other', self.index),
                         {'source_map_threads': 4, 'verified_embed_threads': 4})

    def test_wrong_source_or_incomplete_replay_cannot_register(self):
        for change in ({'checksum_mismatch_count': 1}, {'decoded_rows': 1}, {'status': 'mismatch'},
                       {'video_id': 'N-other'}, {'source': {}}, {'time_bases': [[1, 500]]},
                       {'expected_time_base': None, 'time_bases': [None]},
                       {'expected_time_base': [1, 500], 'time_bases': [[1, 500]]}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.register({**self.evidence, **change})
        self.assertFalse((self.root / 'profiles.json').exists())

    def test_map_change_after_replay_and_source_replacement_are_rejected(self):
        self.map.write_bytes(self.map.read_bytes() + b'changed')
        with self.assertRaisesRegex(ValueError, 'changed after'):
            self.register()
        np.save(self.map, np.array([[0, 200, 11], [1, 300, 22]], dtype=np.int64))
        self.evidence.update(map_size=self.map.stat().st_size, map_mtime_ns=self.map.stat().st_mtime_ns)
        self.register()
        self.source.write_bytes(b'new source')
        with self.assertRaisesRegex(ValueError, 'source changed'):
            decoder_profile(self.video, self.index)

    def test_registered_profile_rejects_changed_map(self):
        self.register()
        decoder_profile(self.video, self.index)
        self.map.write_bytes(self.map.read_bytes() + b'changed')
        with self.assertRaisesRegex(ValueError, 'source map changed'):
            decoder_profile(self.video, self.index)

    def test_registered_profile_rejects_same_size_map_with_wrong_hash(self):
        self.register()
        # Verify the digest even when a file restores the recorded stat identity.
        contents = bytearray(self.map.read_bytes())
        contents[-1] ^= 1
        self.map.write_bytes(contents)
        os.utime(self.map, ns=(self.evidence['map_mtime_ns'], self.evidence['map_mtime_ns']))
        with self.assertRaisesRegex(ValueError, 'checksum changed'):
            decoder_profile(self.video, self.index)


if __name__ == '__main__':
    unittest.main()
