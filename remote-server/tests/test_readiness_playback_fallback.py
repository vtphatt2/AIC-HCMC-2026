import unittest
import hashlib
import json
import tempfile
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from app.services import local_zip_media as media
from app.services.readiness_policy import PlaybackPolicy, playback_decoder_threads
from app.services.source_timeline import source_fingerprint

class PlaybackFallbackTests(unittest.IsolatedAsyncioTestCase):
    @patch('app.services.readiness_policy.decoder_profile',
           side_effect=lambda video, index: {'threads': 1} if video == 'N031-V003' else None)
    def test_decoder_override_keeps_other_playback_settings(self, profiles):
        from app.services import playback_copies
        policy = PlaybackPolicy(threads=3)
        self.assertEqual(playback_decoder_threads('N031-V003', policy, None), 1)
        self.assertEqual(playback_decoder_threads('N032-V003', policy, None), 3)
        with tempfile.TemporaryDirectory() as scratch:
            source = Path(scratch) / 'source.zip'
            source.write_bytes(b'source')
            index = SimpleNamespace(zip_path=source, data_offset=0, video_size=6)
            old = hashlib.sha256(json.dumps({'source': source_fingerprint(index),
                        'policy': asdict(policy)}, sort_keys=True).encode()).hexdigest()[:20]
            ordinary, _ = playback_copies.copy_paths('N032-V003', index, policy, Path(scratch))
            exceptional, _ = playback_copies.copy_paths('N031-V003', index, policy, Path(scratch))
            self.assertEqual(ordinary.stem, f'N032-V003-{old}')
            self.assertNotEqual(exceptional.stem, f'N031-V003-{old}')

    async def test_only_browser_verified_n010_originals_fall_back_to_source(self):
        from app.services import playback_copies
        index = SimpleNamespace()
        with patch.object(media, '_get_index', new=AsyncMock(return_value=index)), \
             patch.object(playback_copies, 'serving_copy', return_value=None), \
             patch.object(media, 'lookup', new=AsyncMock(return_value={
                 'size': 8, 'data_offset': 0, 'zip_path': Path('/tmp/source.zip')})):
            headers, _ = await media.open_range('N010-V001', 'bytes=0-0')
            self.assertEqual(headers['Content-Range'], 'bytes 0-0/8')
            with self.assertRaisesRegex(media.LocalZipUnavailable, 'Validated playback'):
                await media.open_range('N031-V003', 'bytes=0-0')
