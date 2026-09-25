import asyncio
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'keyframe_pipeline_global_v9_3/src'))
from pipeline.keyframe_selection import protect_verified_metadata, invalidate_mismatched_selection
from scripts.prewarm_result_thumbnails import warm_thumbnail, media


class WarmerTests(unittest.IsolatedAsyncioTestCase):
    async def test_idle_and_recovery_never_hold_cache_lock(self):
        lock = asyncio.Lock()
        events = []
        class Cache:
            async def get(self, key, producer):
                async with lock:
                    return await producer()
        async def idle():
            self.assertFalse(lock.locked())
            # A live client can acquire the same key while the warmer yields.
            async with lock:
                events.append('live')
        async def produce():
            self.assertTrue(lock.locked())
            if 'recovered' not in events:
                raise media.LocalZipUnavailable('sequential preparation needed')
            return b'image'
        async def recover():
            await idle()
            events.append('recovered')
        with patch.object(media, 'wait_for_live_card_idle', side_effect=idle):
            result = await asyncio.wait_for(warm_thumbnail(Cache(), 'same', produce, recover), 1)
        self.assertEqual(result, b'image')
        self.assertEqual(events.count('live'), 3)


class MetadataGuardTests(unittest.TestCase):
    def test_forced_replacement_and_invalidation_preserve_all_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, rows in [('A', []), ('N010-V002', [{'source_pts': 42}])]:
                video = root / name
                video.mkdir()
                (video / 'keyframes.json').write_text(json.dumps({'keyframes': rows}))
                (video / 'embeddings.npy').write_bytes(b'original')
            with self.assertRaisesRegex(ValueError, 'staging'):
                protect_verified_metadata(root / 'N010-V002')
            with self.assertRaisesRegex(ValueError, 'staging'):
                invalidate_mismatched_selection(root, strategy='linear', keyframes_per_second=.5,
                                               min_keyframes_per_scene=1, max_keyframes_per_scene=0)
            self.assertEqual(len(list(root.glob('*/keyframes.json'))), 2)
            self.assertTrue(all(p.read_bytes() == b'original' for p in root.glob('*/embeddings.npy')))
