import unittest

from scripts.audit_ms_semantics import selected_positions


class SemanticSamplingTests(unittest.TestCase):
    def test_includes_timeline_extremes_and_scene_boundaries(self):
        rows = [{'frame_number': frame} for frame in (2, 10, 18, 27, 39, 47, 59)]
        scenes = [{'start_frame': start, 'end_frame': start + 9}
                  for start in (0, 10, 20, 30, 40, 50)]
        positions = selected_positions(rows, scenes)
        sampled = [rows[position]['frame_number'] for position in positions]
        self.assertEqual(sampled[0], 2)
        self.assertEqual(sampled[-1], 59)
        self.assertIn(27, sampled)
        self.assertIn(39, sampled)
        self.assertEqual(positions, sorted(set(positions)))
        self.assertEqual(selected_positions(rows, scenes, exhaustive=True), list(range(len(rows))))

    def test_rejects_invalid_selection_order(self):
        with self.assertRaises(ValueError):
            selected_positions([{'frame_number': 2}, {'frame_number': 2}],
                               [{'start_frame': 0, 'end_frame': 3}])


if __name__ == '__main__':
    unittest.main()
