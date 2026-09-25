"""Regression checks for encoder precision, duplicate filtering and JPEG caching."""
import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from app.services.text_encoder import PECoreTextEncoder, TextEncoderConfig
from app.services import local_zip_media as media
from app.strategies._similarity_filter import filter_similar_results


class DuplicateFilterTests(unittest.TestCase):
    def test_single_frame_matches_scalar_greedy_reference(self):
        import numpy as np
        rng = np.random.default_rng(2026)
        centers = rng.normal(size=(12, 1280)).astype('float32')
        vectors = [centers[i % 12] + rng.normal(scale=.08, size=1280) for i in range(120)]
        rows = [{'frame_id': str(i)} for i in range(len(vectors))]
        embeddings = {str(i): v for i, v in enumerate(vectors) if i % 19}
        embeddings['19'] = np.zeros(1280)
        normalized = {k: np.asarray(v, dtype='float32') / np.linalg.norm(np.asarray(v, dtype='float32'))
                      for k, v in embeddings.items() if np.linalg.norm(v)}
        for threshold in [0, .5, .98, .995, 1]:
            expected = []
            for row in rows:
                a = normalized.get(row['frame_id'])
                duplicate = any(a is not None and normalized.get(old['frame_id']) is not None
                                and 1.0 - (1.0 - float(np.dot(a, normalized[old['frame_id']]))) > threshold + 1e-6
                                for old in expected)
                if not duplicate:
                    expected.append(row)
            self.assertEqual(filter_similar_results(rows, embeddings, threshold=threshold, event_weights=[2]), expected)

    def test_threshold_boundary_uses_scalar_decision(self):
        import numpy as np
        a = np.array([1, 0], dtype='float32')
        for similarity in [.98 - 2e-6, .98, .98 + 1e-6, .98 + 2e-6]:
            b = np.array([similarity, (1-similarity**2)**.5], dtype='float32')
            b /= np.linalg.norm(b)
            rows = [{'frame_id': 'a'}, {'frame_id': 'b'}]
            expected = rows[:1] if 1-(1-float(np.dot(a,b))) > .98+1e-6 else rows
            self.assertEqual(filter_similar_results(rows, {'a':a,'b':b}, threshold=.98), expected)

    def test_temporal_weights_and_missing_steps_preserved(self):
        rows = [{'frame_id':'a', 'steps':[{'frame_id':'a'}, {'frame_id':'a'}]},
                {'frame_id':'b', 'steps':[{'frame_id':'b'}, {'frame_id':'a'}]},
                {'frame_id':'missing', 'steps':[{'frame_id':'a'}, {'frame_id':'missing'}]}]
        embeddings = {'a':[1.,0.], 'b':[.99, (1-.99**2)**.5]}
        self.assertEqual(filter_similar_results(rows, embeddings, threshold=.985, event_weights=[2,1]), rows)
        self.assertEqual(filter_similar_results(rows, embeddings, threshold=.985), [rows[0],rows[2]])


class PrecisionTests(unittest.TestCase):
    def test_invalid_precision_rejected(self):
        with self.assertRaises(ValueError):
            TextEncoderConfig(precision="fp61")

    def test_cuda_precision_is_applied_to_loaded_model(self):
        import torch
        for precision, dtype in [("fp16", torch.float16), ("fp32", torch.float32)]:
            with self.subTest(precision=precision), tempfile.TemporaryDirectory() as tmp:
                p = Path(tmp)
                (p / 'config.json').write_text(json.dumps({'embed_dim': 2, 'text_cfg': {'context_length': 72}}))
                (p / 'model.safetensors').touch()
                (p / 'bpe_simple_vocab_16e6.txt.gz').touch()
                model = MagicMock()
                model.to.return_value = model
                with (
                    patch('torch.cuda.is_available', return_value=True),
                    patch('open_clip.model._build_text_tower', return_value=model),
                    patch('open_clip.tokenizer.SimpleTokenizer'),
                    patch('safetensors.torch.load_file', return_value={}),
                ):
                    encoder = PECoreTextEncoder(TextEncoderConfig(cached_model=tmp, device='cuda', precision=precision))
                    encoder._ensure_loaded()
                model.to.assert_called_once_with(device='cuda', dtype=dtype)


class ThumbnailCacheTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.cache_budget = patch.object(media, 'JPEG_CACHE_MAX_BYTES', 128 * 1024 * 1024)
        self.cache_budget.start()
        self.addCleanup(self.cache_budget.stop)
        media._jpeg_cache.clear()
        media._jpeg_cache_bytes = 0
        media._jpeg_inflight.clear()
        media._jpeg_waiters.clear()

    async def test_simultaneous_viewers_decode_once_and_reuse_result(self):
        async def decode(*args):
            await asyncio.sleep(.01)
            return b'jpeg'
        with patch.object(media, '_decode_frame_jpeg', side_effect=decode) as worker:
            results = await asyncio.gather(*(media.get_frame_jpeg('V1', 100) for _ in range(12)))
            self.assertEqual(results, [b'jpeg'] * 12)
            self.assertEqual(await media.get_frame_jpeg('V1', 100), b'jpeg')
            self.assertEqual(worker.call_count, 1)

    async def test_failed_decode_can_be_retried(self):
        with patch.object(media, '_decode_frame_jpeg', new=AsyncMock(side_effect=[media.LocalZipUnavailable('bad'), b'good'])) as worker:
            with self.assertRaises(media.LocalZipUnavailable):
                await media.get_frame_jpeg('V1', 100)
            self.assertEqual(await media.get_frame_jpeg('V1', 100), b'good')
            self.assertEqual(worker.call_count, 2)

    async def test_byte_budget_evicts_oldest_and_oversize_is_not_cached(self):
        with patch.object(media, 'JPEG_CACHE_MAX_BYTES', 6), patch.object(media, '_decode_frame_jpeg', new=AsyncMock(return_value=b'1234')) as worker:
            await media.get_frame_jpeg('V1', 100)
            await media.get_frame_jpeg('V2', 100)
            self.assertLessEqual(media._jpeg_cache_bytes, 6)
            await media.get_frame_jpeg('V1', 100)
            self.assertEqual(worker.call_count, 3)
        media._jpeg_cache.clear()
        media._jpeg_cache_bytes = 0
        with patch.object(media, 'JPEG_CACHE_MAX_BYTES', 2), patch.object(media, '_decode_frame_jpeg', new=AsyncMock(return_value=b'1234')):
            await media.get_frame_jpeg('V1', 100)
            self.assertEqual(media._jpeg_cache_bytes, 0)

    async def test_cancelled_viewer_does_not_cancel_another_viewer(self):
        started, release = asyncio.Event(), asyncio.Event()
        async def decode(*args):
            started.set()
            await release.wait()
            return b'jpeg'
        with patch.object(media, '_decode_frame_jpeg', side_effect=decode) as worker:
            first = asyncio.create_task(media.get_frame_jpeg('V1', 100))
            await started.wait()
            second = asyncio.create_task(media.get_frame_jpeg('V1', 100))
            await asyncio.sleep(0)
            first.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await first
            release.set()
            self.assertEqual(await second, b'jpeg')
            self.assertEqual(worker.call_count, 1)
