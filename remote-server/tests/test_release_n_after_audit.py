import tempfile
import unittest
from pathlib import Path

from scripts.audit_readiness import file_identity
from scripts.release_n_after_audit import validate_release_report


class ReleaseNAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.results = self.root / 'results'; self.exports = self.root / 'exports'
        self.results.mkdir(); self.exports.mkdir()
        archives = []
        for number in range(21):
            path = self.results / f'M{number:02d}_results.zip'; path.write_bytes(b'zip')
            archives.append({'name': path.name, 'videos': 1, 'crc_ok': True,
                             **file_identity(path)})
        vectors = self.exports / 'vectors.f32.npy'; vectors.write_bytes(b'vectors')
        metadata = self.exports / 'vectors.meta.npz'; metadata.write_bytes(b'metadata')
        n_videos = {f'N{number:03d}-V001': {
            'verified_generation': 'generation', 'playback_copy': True,
            'source_pts_rows': 1, 'selected': 1,
            'pts_map': {'sha256': 'map'},
            'numpy_export': {'missing': 0, 'differing': 0},
        } for number in range(1, 299)}
        self.report = {
            'version': 1, 'status': 'pass', 'issues': [],
            'checks': {'source_crc': True, 'live_indexes': True,
                       'source_picture_semantics': 'not assessed here'},
            'result_dir': str(self.results.resolve()),
            'export_dir': str(self.exports.resolve()),
            'counts': {'source_archives': 21, 'source_videos': 614,
                       'result_archives': 21, 'result_videos': 614,
                       'source_by_lot': {'N': 298}, 'result_rows': 298,
                       'release_blocked_missing_results': [],
                       'unexpected_numpy_rows': 0},
            'result_archives': archives,
            'numpy': {'mns_rows': 298, 'vectors_file': file_identity(vectors),
                      'metadata_file': file_identity(metadata)},
            'videos': n_videos,
            'postgres': {'videos': 614, 'missing': [], 'extra': []},
            'indexes': {
                'hnsw': {'videos_checked': 614, 'mismatches': 0},
                'flat': {'videos_checked': 614, 'mismatches': 0},
            },
        }
        map_path = self.root / 'map.npy'; map_path.write_bytes(b'map')
        source = {'archive': 'source.zip', 'entry': 'video.mov'}
        for row in n_videos.values():
            row['pts_map']['source'] = source
        image_rows = {video: {
            'status': 'pass', 'source_picture_identity': 'all_selected_pts_checksums',
            'generation': 'generation', 'source': source, 'cards': 1, 'webp_cards': 1,
            'map_path': str(map_path), 'map_size': map_path.stat().st_size,
            'map_mtime_ns': map_path.stat().st_mtime_ns,
        } for video in n_videos}
        self.image_reports = [{'version': 1, 'videos': image_rows,
                               'summary': {'pass': 298, 'error': 0}}]
        self.quarantine = {'N001-V001': {'reason': 'browser copy required'}}

    def validate(self):
        return validate_release_report(self.report, image_reports=self.image_reports,
                                       result_dir=self.results,
                                       export_dir=self.exports,
                                       quarantine_entries=self.quarantine)

    def test_accepts_current_complete_live_evidence(self):
        self.assertEqual(self.validate(), {'N001-V001': 'browser copy required'})

    def test_rejects_stale_artifact_or_incomplete_video(self):
        (self.results / 'M00_results.zip').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'changed after audit'):
            self.validate()
        self.report['result_archives'][0].update(file_identity(self.results / 'M00_results.zip'))
        self.report['videos']['N001-V001']['playback_copy'] = False
        with self.assertRaisesRegex(ValueError, 'release evidence'):
            self.validate()

    def test_rejects_active_source_identity_block(self):
        self.quarantine['N001-V001']['release_blocked'] = True
        with self.assertRaisesRegex(ValueError, 'still active'):
            self.validate()

    def test_rejects_missing_or_stale_selected_card_evidence(self):
        missing = self.image_reports[0]['videos'].pop('N001-V001')
        self.image_reports[0]['summary']['pass'] -= 1
        with self.assertRaisesRegex(ValueError, 'cover all'):
            self.validate()
        self.image_reports[0]['videos']['N001-V001'] = missing
        self.image_reports[0]['summary']['pass'] += 1
        missing['webp_cards'] = 0
        with self.assertRaisesRegex(ValueError, 'stale or incomplete'):
            self.validate()


if __name__ == '__main__':
    unittest.main()
