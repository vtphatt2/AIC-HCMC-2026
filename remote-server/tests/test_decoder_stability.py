import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from scripts import audit_decoder_stability as stability


class DecoderStabilityTests(unittest.TestCase):
    def test_current_source_map_and_setting_are_required(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / 'source.zip'
            archive.write_bytes(b'source')
            frame_map = root / 'source.npy'
            frame_map.write_bytes(b'map')
            index = SimpleNamespace(zip_path=archive, data_offset=0, video_size=6)
            with patch.object(stability, 'index_path', return_value=frame_map), \
                 patch.object(stability, 'source_map_decoder_threads', return_value=4):
                identity = stability.replay_identity('N001-V001', index, frame_map, 4)
                self.assertEqual(identity['map_sha256'], stability.source_map_sha256(frame_map))
                with self.assertRaisesRegex(ValueError, 'current source map'):
                    stability.replay_identity('N001-V001', index, frame_map, 1)
                with self.assertRaisesRegex(ValueError, 'current source map'):
                    stability.replay_identity('N001-V001', index, root / 'other.npy', 4)
                frame_map.write_bytes(b'new map')
                self.assertNotEqual(identity['map_sha256'], stability.replay_identity(
                    'N001-V001', index, frame_map, 4)['map_sha256'])


if __name__ == '__main__':
    unittest.main()
