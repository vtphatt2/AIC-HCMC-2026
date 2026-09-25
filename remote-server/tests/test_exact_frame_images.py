"""Publish only verified source pictures, and never repeat full live decodes."""
import asyncio
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import numpy as np
from PIL import Image
from app.services import exact_frame_images as images
from app.services import local_zip_media as media
from scripts.prewarm_result_thumbnails import SequentialFallback


class ExactImageTests(unittest.TestCase):
    def setUp(self):
        profile = patch('app.services.readiness_policy.decoder_profile',
                        side_effect=lambda video, index: {'threads': 1} if video == 'N031-V003' else None)
        profile.start()
        self.addCleanup(profile.stop)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.map = self.root / 'N001-source.npy'
        np.save(self.map, np.array([[0, 2000, 11], [1, 2400, 22], [2, 2800, 33]], dtype=np.int64))
        self.index = SimpleNamespace(data_offset=0, video_size=10, fps=25,
                                     zip_path=self.root / 'source.zip')
        self.index.zip_path.write_bytes(b'source')
        self.patch = patch.object(images, 'index_path', return_value=self.map)
        self.patch.start()
        self.addCleanup(self.patch.stop)

    def decode(self, command, **kwargs):
        for number, pts, checksum in [(0, 2000, 11), (1, 2800, 33)]:
            path = Path(command[-1].replace('%08d', f'{number:08d}'))
            Image.new('RGB', (16, 16), (number, 20, 30)).save(path)
            kwargs['stderr'].write(f'[Parsed_showinfo_1 @ a] n: {number} pts: {pts} checksum:{checksum:08X}\n'.encode())
        return SimpleNamespace(returncode=0)

    def test_verified_batch_is_reused_and_keeps_original_frame_numbers(self):
        with patch.object(images.subprocess, 'run', side_effect=self.decode) as run:
            paths = images.build_exact_images('N001', self.index, [2, 0, 2], directory=self.root)
            self.assertEqual([p.name for p in paths], ['0.jpg', '2.jpg'])
            self.assertTrue(all(p.is_file() for p in paths))
            self.assertEqual(images.build_exact_images('N001', self.index, [0, 2], directory=self.root), paths)
            self.assertEqual(run.call_count, 1)

    def test_n031_uses_single_thread_source_decode(self):
        with patch.object(images.subprocess, 'run', side_effect=self.decode) as run:
            images.build_exact_images('N031-V003', self.index, [0, 2], directory=self.root)
        command = run.call_args.args[0]
        self.assertEqual(command[command.index('-threads') + 1], '1')

    def test_n031_card_cache_uses_new_map_generation(self):
        key = media._cache_key(self.index, 'N031-V003', 488187, 640, 'jpeg', 12165)
        self.assertIn('N031-V003-', key)
        other = media._cache_key(self.index, 'N032-V003', 488187, 640, 'jpeg', 12165)
        self.assertNotIn('N032-V003-', other)

    def test_checksum_failure_publishes_no_images(self):
        def wrong(command, **kwargs):
            result = self.decode(command, **kwargs)
            kwargs['stderr'].seek(0)
            data = kwargs['stderr'].read().replace(b'00000021', b'000000FF')
            kwargs['stderr'].seek(0)
            kwargs['stderr'].write(data)
            return result
        with patch.object(images.subprocess, 'run', side_effect=wrong):
            with self.assertRaisesRegex(RuntimeError, 'pixel verification failed'):
                images.build_exact_images('N001', self.index, [0, 2], directory=self.root)
        self.assertEqual(list(self.root.rglob('*.jpg')), [])

    def test_ambiguous_source_timestamp_uses_frame_identity(self):
        np.save(self.map, np.array([[0, 2000, 11], [1, 2000, 22]], dtype=np.int64))
        def decoded_frame(command, **kwargs):
            Image.new('RGB', (16, 16)).save(Path(command[-1].replace('%08d', '00000000')))
            kwargs['stderr'].write(b'[Parsed_showinfo_1 @ a] n: 0 pts: 2000 checksum:0000000B\n')
            return SimpleNamespace(returncode=0)
        with patch.object(images.subprocess, 'run', side_effect=decoded_frame) as run:
            paths = images.build_exact_images('N001', self.index, [0], directory=self.root)
        self.assertTrue(paths[0].is_file())
        self.assertIn(r'eq(n\,0)', run.call_args.args[0][run.call_args.args[0].index('-vf') + 1])

    def test_ambiguous_source_timestamp_still_rejects_wrong_picture(self):
        np.save(self.map, np.array([[0, 2000, 11], [1, 2000, 22]], dtype=np.int64))
        def wrong_picture(command, **kwargs):
            Image.new('RGB', (16, 16)).save(Path(command[-1].replace('%08d', '00000000')))
            kwargs['stderr'].write(b'[Parsed_showinfo_1 @ a] n: 0 pts: 2000 checksum:00000016\n')
            return SimpleNamespace(returncode=0)
        with patch.object(images.subprocess, 'run', side_effect=wrong_picture):
            with self.assertRaisesRegex(RuntimeError, 'pixel verification failed'):
                images.build_exact_images('N001', self.index, [0], directory=self.root)
        self.assertEqual(list(self.root.rglob('*.jpg')), [])


class SequentialCacheTests(unittest.IsolatedAsyncioTestCase):
    async def test_concurrent_failures_decode_once_per_video(self):
        recovery = SequentialFallback({'N001': [0, 2]})
        with patch('scripts.prewarm_result_thumbnails.build_exact_images', return_value=[]) as build, \
             patch.object(media, 'wait_for_live_card_idle', new=AsyncMock()):
            await asyncio.gather(*(recovery.prepare('N001', None) for _ in range(8)))
            build.assert_called_once_with('N001', None, [0, 2])

    async def test_bad_source_not_retried_by_every_card(self):
        recovery = SequentialFallback({'N001': [0, 2]})
        with patch('scripts.prewarm_result_thumbnails.build_exact_images', side_effect=ValueError('source')) as build, \
             patch.object(media, 'wait_for_live_card_idle', new=AsyncMock()):
            results = await asyncio.gather(*(recovery.prepare('N001', None) for _ in range(8)), return_exceptions=True)
            self.assertTrue(all(isinstance(result, ValueError) for result in results))
            self.assertEqual(build.call_count, 1)

    async def test_online_reads_exact_original_without_decoder(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / '2.jpg'
            Image.new('RGB', (16, 16)).save(path)
            with patch.object(media, '_get_index', new=AsyncMock(return_value=object())), \
                 patch.object(media, 'exact_image_path', return_value=path), \
                 patch.object(media, '_decode_seekable_jpeg', new=AsyncMock()) as decoder:
                result = await media._decode_frame_jpeg('N001', 80, frame_number=2)
                self.assertEqual(result, path.read_bytes())
                decoder.assert_not_called()


if __name__ == '__main__':
    unittest.main()
