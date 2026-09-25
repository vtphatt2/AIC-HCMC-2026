"""Organizer metadata naming and aliases never change canonical video IDs."""
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from scripts.ingest_zip_pipeline_results import load_media_info


class MediaInfoAliasTests(unittest.TestCase):
    def test_both_archive_names_and_unambiguous_aliases(self):
        with tempfile.TemporaryDirectory() as scratch:
            root = Path(scratch)
            with zipfile.ZipFile(root / 'M01_results.zip', 'w') as result:
                result.writestr('phase1_transnet/videos__M01_V001/scenes.json', '{}')
                result.writestr('phase1_transnet/videos__S01-V001/scenes.json', '{}')
            with zipfile.ZipFile(root / 'media-info-m.zip', 'w') as archive:
                archive.writestr('media-info/M01-V001.json', json.dumps({
                    'title': 'Verified M title',
                    'watch_url': 'https://www.youtube.com/watch?v=dQw4w9WgXcQ'}))
            # The organizer's B2 archive uses this suffix name and mistakenly
            # spells S JSON members with underscores. Canonical result IDs use
            # the corrected hyphen form from the organizer clarification.
            with zipfile.ZipFile(root / 'aic26-b2-media-info.zip', 'w') as archive:
                archive.writestr('media-info/S01_V001.json', json.dumps({
                    'title': 'Verified S title',
                    'watch_url': 'https://www.youtube.com/watch?v=9bZkp7q19f0'}))
            info = load_media_info(root)
            self.assertEqual(info['M01_V001']['title'], 'Verified M title')
            self.assertEqual(info['M01_V001']['youtube_id'], 'dQw4w9WgXcQ')
            self.assertNotIn('M01-V001', info)
            self.assertEqual(info['S01-V001']['title'], 'Verified S title')
            self.assertEqual(info['S01-V001']['youtube_id'], '9bZkp7q19f0')
            self.assertNotIn('S01_V001', info)

    def test_ambiguous_normalized_alias_is_not_guessed(self):
        with tempfile.TemporaryDirectory() as scratch:
            root = Path(scratch)
            with zipfile.ZipFile(root / 'M01_results.zip', 'w') as result:
                for video in ('M01_V001', 'M01-V001'):
                    result.writestr(f'phase1_transnet/videos__{video}/scenes.json', '{}')
            with zipfile.ZipFile(root / 'media_info-m.zip', 'w') as archive:
                archive.writestr('media-info/m01-v001.json', json.dumps({
                    'title': 'Ambiguous',
                    'watch_url': 'https://www.youtube.com/watch?v=dQw4w9WgXcQ'}))
            self.assertEqual(load_media_info(root), {})

    def test_conflicting_organizer_records_are_rejected(self):
        with tempfile.TemporaryDirectory() as scratch:
            root = Path(scratch)
            with zipfile.ZipFile(root / 'M01_results.zip', 'w') as result:
                result.writestr('phase1_transnet/videos__M01_V001/scenes.json', '{}')
            for name, title in [('media-info-m.zip', 'First'), ('media_info-m.zip', 'Second')]:
                with zipfile.ZipFile(root / name, 'w') as archive:
                    archive.writestr('media-info/M01_V001.json', json.dumps({
                        'title': title,
                        'watch_url': 'https://www.youtube.com/watch?v=dQw4w9WgXcQ'}))
            with self.assertRaisesRegex(ValueError, 'Conflicting organizer metadata'):
                load_media_info(root)


if __name__ == '__main__':
    unittest.main()
