"""A source-identity failure must not be served through a direct public URL."""
import unittest
from unittest.mock import patch

import httpx

import main


class ReleaseBlockHttpTests(unittest.IsolatedAsyncioTestCase):
    async def test_release_blocked_video_has_no_direct_media_or_metadata(self):
        paths = (
            '/api/video/N031-V003',
            '/api/video/N031-V003/frame-timeline?version=2',
            '/api/video/N031-V003/frame-offset/12165',
            '/api/video/N031-V003/context-frames?start_ms=0&end_ms=1000',
            '/api/zip-video/N031-V003',
            '/api/zip-frame/N031-V003/488187',
            '/api/transcript/N031-V003',
        )
        with patch.object(main, 'release_blocked_video_ids',
                          return_value=frozenset({'N031-V003'})):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=main.app),
                                         base_url='http://test') as client:
                for path in paths:
                    with self.subTest(path=path):
                        self.assertEqual((await client.get(path)).status_code, 404)
                self.assertEqual((await client.post('/api/frame-embeddings', json={
                    'frame_ids':['N031-V003_012165']})).status_code, 404)
                self.assertEqual((await client.post('/api/keyframes', json={
                    'video_id':'N031-V003','start_ms':0,'end_ms':1000})).status_code, 404)


if __name__ == '__main__':
    unittest.main()
