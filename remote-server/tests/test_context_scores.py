import ast
import asyncio
from pathlib import Path
import unittest
from unittest.mock import AsyncMock, Mock

from pydantic import BaseModel, Field


class ContextScoresTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        # Load the actual endpoint and request models without starting models/DBs.
        tree = ast.parse((Path(__file__).resolve().parents[1] / 'main.py').read_text())
        names = {'QueryGroup', 'FrameScoresRequest', 'ContextScoresRequest', 'get_context_scores'}
        tree.body = [node for node in tree.body if getattr(node, 'name', '') in names]
        for node in tree.body:
            node.decorator_list = []
        self.context = {'fps': 25, 'before': [{'frame_id': 'before', 'frame_number': 10}],
                        'middle': [{'frame_id': 'hit', 'frame_number': 20}], 'after': []}
        self.ns = {'BaseModel': BaseModel, 'Field': Field, 'logger': Mock(),
                   '_require_released_video': lambda _: None,
                   'get_context_frames': AsyncMock(return_value=self.context),
                   'get_frame_scores': AsyncMock(return_value={'scores': {'hit': .8}})}
        exec(compile(tree, 'main.py', 'exec'), self.ns)
        self.req = self.ns['ContextScoresRequest'](
            start_ms=800, end_ms=800, expand=24,
            frame_ids=['hit', 'outside'], frame_numbers={'hit': 20, 'outside': 30},
            query_groups=[{'query': 'first'}, {'query': 'second'}],
            event_weights=[.3, .7], duplicate_threshold=.95,
        )

    async def test_union_preserves_matches_order_weights_and_threshold(self):
        result = await self.ns['get_context_scores']('video', self.req)
        self.assertEqual(result, {**self.context, 'scores': {'hit': .8}})
        self.ns['get_context_frames'].assert_awaited_once_with('video', 800, 800, 24)
        scored = self.ns['get_frame_scores'].await_args.args[0]
        self.assertEqual(scored.frame_ids, ['before', 'hit', 'outside'])
        self.assertEqual(scored.event_weights, [.3, .7])
        self.assertEqual(scored.duplicate_threshold, .95)
        self.assertEqual([q.query for q in scored.query_groups], ['first', 'second'])

    async def test_score_failure_retains_context(self):
        self.ns['get_frame_scores'].side_effect = RuntimeError('unavailable')
        result = await self.ns['get_context_scores']('video', self.req)
        self.assertEqual(result, {**self.context, 'scores': None})

    async def test_context_failure_does_not_return_false_empty_success(self):
        self.ns['get_context_frames'].side_effect = RuntimeError('unavailable')
        with self.assertRaises(RuntimeError):
            await self.ns['get_context_scores']('video', self.req)
        self.ns['get_frame_scores'].assert_not_awaited()

    async def test_cancellation_is_not_swallowed(self):
        self.ns['get_frame_scores'].side_effect = asyncio.CancelledError()
        with self.assertRaises(asyncio.CancelledError):
            await self.ns['get_context_scores']('video', self.req)
