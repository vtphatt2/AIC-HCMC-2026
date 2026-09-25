"""LOCAL clients use the same verified media as the search server for every lot."""
import os
import unittest
from unittest.mock import AsyncMock, patch

import httpx
from fastapi.testclient import TestClient

import main


class ServerMediaProxyTests(unittest.TestCase):
    def test_query_video_lookup_preserves_titles_with_slashes(self):
        payload = {'video_id': 'M01_V001', 'youtube_id': 'aWlose7FZA8',
                   'frame_id': 'M01_V001_000001', 'frame_number': 1,
                   'timestamp_ms': 40, 'fps': 25}
        upstream = AsyncMock(return_value=httpx.Response(200, json=payload))
        title = 'News - 01/11/2025'
        with (patch.dict(os.environ, {'ENV_MODE': 'LOCAL',
                                      'REMOTE_SERVER_URL': 'https://server.test'}),
              patch.object(main._http_client, 'get', upstream),
              patch.object(main, 'release_blocked_video_ids', return_value=frozenset())):
            response = TestClient(main.app).get('/api/video', params={'lookup': title})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), payload)
        self.assertEqual(upstream.await_args.args[0], 'https://server.test/api/video')
        self.assertEqual(upstream.await_args.kwargs['params'], {'lookup': title})

    def test_local_search_runs_once_on_server(self):
        upstream = AsyncMock(return_value=httpx.Response(200, json={
            'results': [], 'strategy_id': 'raw_visual', 'total': 0,
            'execution_time_ms': 10,
        }))
        with (patch.dict(os.environ, {'ENV_MODE': 'LOCAL',
                                      'REMOTE_SERVER_URL': 'https://server.test'}),
              patch.object(main._http_client, 'post', upstream)):
            response = TestClient(main.app).post('/api/search', json={
                'strategy_id': 'raw_visual', 'query_groups': [{'query': 'street'}]
            })
        self.assertEqual(response.status_code, 200)
        self.assertEqual(upstream.await_count, 1)
        self.assertEqual(upstream.await_args.args[0], 'https://server.test/api/search')

    def test_local_context_scoring_runs_once_on_server(self):
        payload = {'fps': 25, 'before': [], 'middle': [], 'after': [], 'scores': {}}
        upstream = AsyncMock(return_value=httpx.Response(200, json=payload))
        with (patch.dict(os.environ, {'ENV_MODE': 'LOCAL',
                                      'REMOTE_SERVER_URL': 'https://server.test'}),
              patch.object(main._http_client, 'post', upstream),
              patch.object(main, 'release_blocked_video_ids', return_value=frozenset())):
            response = TestClient(main.app).post('/api/video/S01-V010/context-scores', json={
                'start_ms': 1000, 'end_ms': 1000, 'expand': 12,
                'frame_ids': ['S01-V010_000001'],
                'frame_numbers': {'S01-V010_000001': 1},
                'query_groups': [{'query': 'cycling'}],
            })
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), payload)
        self.assertEqual(upstream.await_count, 1)
        self.assertEqual(upstream.await_args.args[0],
                         'https://server.test/api/video/S01-V010/context-scores')

    def test_proxy_and_n_never_use_unverified_browser_decode_plans(self):
        for mode, video in (('LOCAL', 'M09_V028'), ('ZIP', 'N010-V002')):
            with (self.subTest(mode=mode), patch.dict(os.environ, {'ENV_MODE': mode}),
                  patch.object(main, 'release_blocked_video_ids', return_value=frozenset())):
                client = TestClient(main.app)
                for path in (f'/api/zip-frame-plan/{video}/1200', f'/api/zip-bytes/{video}/0/100'):
                    self.assertEqual(client.get(path).status_code, 404)

    def test_all_lots_forward_exact_picture_parameters_and_cache_policy(self):
        for video in ('L21_V008', 'M09_V028', 'S01-V005', 'N010-V002'):
            with self.subTest(video=video):
                upstream = AsyncMock(return_value=httpx.Response(
                    200, content=b'image', headers={'content-type': 'image/webp',
                                                   'cache-control': 'public, max-age=0, must-revalidate'}))
                with (patch.dict(os.environ, {'ENV_MODE': 'LOCAL', 'REMOTE_SERVER_URL': 'https://server.test/'}),
                      patch.object(main, 'release_blocked_video_ids', return_value=frozenset()),
                      patch.object(main._zip_upstream_client, 'get', upstream),
                      patch('app.services.zip_frame_source.get_frame_jpeg', AsyncMock(side_effect=AssertionError('organizer decode called')))):
                    response = TestClient(main.app).get(f'/api/zip-frame/{video}/1200?frame_number=30&width=640&format=webp&v=7')
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.content, b'image')
                self.assertEqual(response.headers['content-type'], 'image/webp')
                self.assertEqual(response.headers.get('cache-control'), 'public, max-age=0, must-revalidate')
                self.assertEqual(upstream.await_args.args[0],
                                 f'https://server.test/api/zip-frame/{video}/1200?frame_number=30&width=640&format=webp&v=7')

    def test_all_lots_stream_server_range_and_close_upstream(self):
        for video in ('L21_V008', 'M09_V028', 'S01-V005', 'N010-V002'):
            with self.subTest(video=video):
                upstream = httpx.Response(206, content=b'1234', headers={
                    'content-type': 'video/mp4', 'content-range': 'bytes 10-13/100',
                    'accept-ranges': 'bytes', 'content-length': '4'})
                send = AsyncMock(return_value=upstream)
                with (patch.dict(os.environ, {'ENV_MODE': 'LOCAL', 'REMOTE_SERVER_URL': 'https://server.test/'}),
                      patch.object(main, 'release_blocked_video_ids', return_value=frozenset()),
                      patch.object(main._zip_upstream_client, 'send', send),
                      patch.object(main._zip_video_proxy, 'open_range', AsyncMock(side_effect=AssertionError('organizer requested')))):
                    response = TestClient(main.app).get(f'/api/zip-video/{video}', headers={'Range': 'bytes=10-13'})
                self.assertEqual(response.status_code, 206)
                self.assertEqual(response.content, b'1234')
                self.assertEqual(response.headers['content-range'], 'bytes 10-13/100')
                self.assertEqual(str(send.await_args.args[0].url), f'https://server.test/api/zip-video/{video}')
                self.assertEqual(send.await_args.args[0].headers['range'], 'bytes=10-13')
                self.assertTrue(upstream.is_closed)

    def test_missing_proxy_url_fails_without_requesting_organizer(self):
        with patch.dict(os.environ, {'ENV_MODE': 'LOCAL', 'REMOTE_SERVER_URL': ''}):
            client = TestClient(main.app)
            for path in ('/api/zip-frame/M09_V028/1200', '/api/zip-video/M09_V028'):
                self.assertEqual(client.get(path).status_code, 503)

    def test_server_picture_error_is_not_retried_against_organizer(self):
        with (patch.dict(os.environ, {'ENV_MODE': 'LOCAL', 'REMOTE_SERVER_URL': 'https://server.test'}),
              patch.object(main._zip_upstream_client, 'get', AsyncMock(return_value=httpx.Response(404)))):
            self.assertEqual(TestClient(main.app).get('/api/zip-frame/S01-V005/1200').status_code, 404)


if __name__ == '__main__':
    unittest.main()
