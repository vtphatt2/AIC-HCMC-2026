"""Playback preparation concurrency stays bounded and configurable."""
import os
import unittest
from unittest.mock import patch

from scripts.prepare_playback_copies import playback_job_settings


class PlaybackRunnerSettingsTests(unittest.TestCase):
    def test_environment_defaults_allow_four_low_priority_workers(self):
        with patch.dict(os.environ, {
            'PLAYBACK_PREPARE_WORKERS': '4',
            'PLAYBACK_PREPARE_NICE': '8',
        }, clear=False):
            self.assertEqual(playback_job_settings(None, None), (4, 8))

    def test_cli_values_override_environment(self):
        with patch.dict(os.environ, {
            'PLAYBACK_PREPARE_WORKERS': '4',
            'PLAYBACK_PREPARE_NICE': '8',
        }, clear=False):
            self.assertEqual(playback_job_settings(2, 3), (2, 3))

    def test_workers_and_nice_are_bounded(self):
        for workers, nice in ((0, 8), (5, 8), (4, -1), (4, 20)):
            with self.subTest(workers=workers, nice=nice), self.assertRaises(ValueError):
                playback_job_settings(workers, nice)

        with patch.dict(os.environ, {'PLAYBACK_PREPARE_WORKERS': 'bad'}, clear=False), \
             self.assertRaises(ValueError):
            playback_job_settings(None, 8)


if __name__ == '__main__':
    unittest.main()
