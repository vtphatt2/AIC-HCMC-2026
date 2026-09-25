"""The local/proxy client cannot expose a release-blocked source video."""
import os
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient
import httpx

import main


class ReleaseBlockHttpTests(unittest.TestCase):
    def test_direct_video_routes_reject_bad_source_before_proxy_or_decode(self):
        paths = (
            '/api/video/N031-V003',
            '/api/video/N031-V003/frame-timeline',
            '/api/video/N031-V003/frame-offset/12165',
            '/api/video/N031-V003/context-frames?start_ms=0&end_ms=1000',
            '/api/zip-video/N031-V003',
            '/api/zip-frame/N031-V003/488187',
            '/api/zip-frame-plan/N031-V003/488187',
            '/api/zip-bytes/N031-V003/0/100',
            '/api/transcript/N031-V003',
        )
        client = TestClient(main.app)
        with patch.object(main, 'release_blocked_video_ids',
                          return_value=frozenset({'N031-V003'})):
            for path in paths:
                with self.subTest(path=path):
                    self.assertEqual(client.get(path).status_code, 404)

    def test_video_alias_cannot_resolve_to_a_release_blocked_id(self):
        from app.db import numpy_vector_store

        with (patch.dict(os.environ, {'ENV_MODE': 'ZIP'}),
              patch.object(main, 'release_blocked_video_ids',
                           return_value=frozenset({'N031-V003'})),
              patch.object(numpy_vector_store, 'available', return_value=True),
              patch.object(numpy_vector_store, 'frames_in_range',
                           return_value=[{'video_id': 'N031-V003'}]),
              patch.object(main, 'search_video_catalog',
                           return_value=[{'video_id': 'N031-V003'}])):
            response = TestClient(main.app).get('/api/video/N031V003')
        self.assertEqual(response.status_code, 404)

    def test_proxy_forwards_verified_timing_and_playback_capabilities(self):
        metadata = {'video_id': 'N027-V003', 'fps': 25, 'submission_unit': 'milliseconds',
                    'verified_timing': True, 'timeline_version': 2,
                    'playback_available': True, 'playback_origin_seconds': 0.2,
                    'source_time_base': {'num': 1, 'den': 10000}}
        upstream = AsyncMock(return_value=httpx.Response(200, json=metadata))
        with (patch.dict(os.environ, {'ENV_MODE': 'LOCAL',
                                      'REMOTE_SERVER_URL': 'https://example.test/'}),
              patch.object(main, '_http_client', SimpleNamespace(get=upstream), create=True),
              patch.object(main, 'release_blocked_video_ids', return_value=frozenset())):
            response = TestClient(main.app).get('/api/video/N027-V003')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), metadata)
        upstream.assert_awaited_once_with(
            'https://example.test/api/video/N027-V003',
            headers={'ngrok-skip-browser-warning': '1'},
        )

    def test_proxy_forwards_explicit_ids_and_range_headers(self):
        timeline = {'version': 2, 'video_id': 'N027-V003', 'verified_timing': True,
                    'submission_unit': 'milliseconds', 'frame_ids': [0, 2],
                    'source_pts': [200, 1200], 'presentation_us': [0, 1000000],
                    'omitted_frame_ids': [1], 'time_base': {'num': 1, 'den': 1000},
                    'playback_origin_pts': 200}
        upstream_get = AsyncMock(return_value=httpx.Response(200, json=timeline))
        upstream_send = AsyncMock(return_value=httpx.Response(
            206, content=b'video-data', headers={
                'content-range': 'bytes 0-9/100', 'accept-ranges': 'bytes',
                'content-length': '10', 'content-type': 'video/mp4'}))
        with (patch.dict(os.environ, {'REMOTE_SERVER_URL': 'https://example.test/'}),
              patch.object(main, 'release_blocked_video_ids', return_value=frozenset()),
              patch.object(main._zip_upstream_client, 'get', upstream_get),
              patch.object(main._zip_upstream_client, 'send', upstream_send)):
            client = TestClient(main.app)
            timing_response = client.get('/api/video/N027-V003/frame-timeline?version=2')
            media_response = client.get('/api/zip-video/N027-V003', headers={'Range': 'bytes=0-9'})
        self.assertEqual(timing_response.status_code, 200)
        self.assertEqual(timing_response.json(), timeline)
        upstream_get.assert_awaited_once_with(
            'https://example.test/api/video/N027-V003/frame-timeline',
            params={'version': 2}, headers={'ngrok-skip-browser-warning': '1'},
        )
        self.assertEqual(media_response.status_code, 206)
        self.assertEqual(media_response.content, b'video-data')
        self.assertEqual(media_response.headers['content-range'], 'bytes 0-9/100')
        self.assertEqual(media_response.headers['content-type'], 'video/mp4')
        upstream_request = upstream_send.await_args.args[0]
        self.assertEqual(upstream_request.headers['range'], 'bytes=0-9')


if __name__ == '__main__':
    unittest.main()
