"""A playback copy must encode the mapped source pictures before publication."""
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import zipfile

import numpy as np

from app.services.exact_frame_pts import build_exact_pts
from app.services.playback_copies import prepare_copy, serving_copy
from app.services.readiness_policy import PlaybackPolicy
from app.services import playback_copies


@unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'), 'FFmpeg required')
class PlaybackCopyAlignmentTests(unittest.TestCase):
    def test_full_source_picture_check_and_failed_copy_rollback(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / 'source.mp4'
            subprocess.run([
                'ffmpeg', '-hide_banner', '-loglevel', 'error', '-f', 'lavfi',
                '-i', 'testsrc=size=64x48:rate=5:duration=1.2', '-an', '-c:v', 'libx264',
                '-pix_fmt', 'yuv420p', '-threads', '1', '-video_track_timescale', '10000',
                '-y', str(source),
            ], check=True, timeout=30)
            archive = root / 'source.zip'
            with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_STORED) as file:
                file.write(source, 'video.mp4')
            with zipfile.ZipFile(archive) as file:
                entry = file.getinfo('video.mp4')
                offset = entry.header_offset + 30 + len(entry.filename.encode()) + len(entry.extra)
            index = SimpleNamespace(zip_path=archive, data_offset=offset,
                                    video_size=source.stat().st_size, timescale=10000)
            frame_map = build_exact_pts('N999-V001', index, [0, 5], 6,
                                        'ffmpeg', root / 'maps', store_full_timeline=True)
            table = np.load(frame_map)
            policy = PlaybackPolicy(height=48, threads=1)
            with patch.object(playback_copies, 'load_timeline', return_value=table):
                path, meta = prepare_copy('N999-V001', index, policy, root / 'valid')
                self.assertTrue(path.is_file())
                self.assertTrue(meta['picture_alignment_verified'])
                self.assertEqual(meta['validation']['frames'], 6)
                self.assertIsNotNone(serving_copy('N999-V001', index, policy, root / 'valid'))

                altered = table.copy()
                altered[3, 2] ^= 1
                with patch.object(playback_copies, 'load_timeline', return_value=altered):
                    with self.assertRaisesRegex(ValueError, 'source-picture alignment failed'):
                        prepare_copy('N999-V001', index, policy, root / 'bad')
                self.assertEqual(list((root / 'bad').glob('*.mp4')), [])
                self.assertEqual(list((root / 'bad').glob('*.json')), [])


if __name__ == '__main__':
    unittest.main()
