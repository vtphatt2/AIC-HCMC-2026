import unittest

from main import _CONTEXT_FRAME_RESPONSE_LIMIT, _bounded_context


class ContextFrameBoundsTests(unittest.TestCase):
    def test_hours_wide_middle_is_sampled_and_keeps_endpoints(self):
        frames = [
            {'frame_id': f'f{i}', 'frame_number': i, 'timestamp_ms': i * 10_000}
            for i in range(5000)
        ]
        before, middle, after = _bounded_context(
            frames, frames[0]['timestamp_ms'], frames[-1]['timestamp_ms'], 20
        )
        self.assertEqual(before, [])
        self.assertEqual(after, [])
        self.assertLessEqual(len(middle), _CONTEXT_FRAME_RESPONSE_LIMIT)
        self.assertEqual(middle[0]['frame_id'], 'f0')
        self.assertEqual(middle[-1]['frame_id'], 'f4999')

    def test_nearby_sides_and_middle_share_one_fixed_budget(self):
        frames = [
            {'frame_id': f'f{i}', 'frame_number': i, 'timestamp_ms': i * 1000}
            for i in range(200)
        ]
        before, middle, after = _bounded_context(frames, 80_000, 120_000, 40)
        self.assertLessEqual(len(before) + len(middle) + len(after),
                             _CONTEXT_FRAME_RESPONSE_LIMIT)
        self.assertTrue(middle)
        self.assertLess(before[-1]['timestamp_ms'], 80_000)
        self.assertGreater(after[0]['timestamp_ms'], 120_000)


if __name__ == '__main__':
    unittest.main()
