"""Publication candidates must match verified bytes and preserve prior stages."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import zipfile

import numpy as np

from app.services.staged_artifacts import artifact_digests
from scripts import stage_n_result_archives as stage


class StageResultArchiveTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.video = 'N001-V001'
        self.folder = self.root / 'stage'
        self.folder.mkdir()
        self.index = SimpleNamespace(timescale=1000, duration_ticks=1000)
        self.decoder = {'source_map_threads': 4, 'verified_embed_threads': 4}
        self.generation = 'a' * 64
        self.keyframes = {
            'version': 2, 'generation': self.generation, 'num_keyframes': 1,
            'source_identity': {'source': 'same'},
            'decode_provenance': self.decoder,
            'source_duration_ms': 1000,
            'omitted_entries': [],
            'keyframes': [{'frame_number': 0, 'source_pts': 500,
                           'source_checksum': 123, 'source_timebase': 1000,
                           'timestamp_ms': 500}],
        }
        self.scenes = {'num_frames': 1, 'scenes': [{'start_frame': 0, 'end_frame': 0}]}
        (self.folder / 'keyframes.json').write_text(json.dumps(self.keyframes))
        (self.folder / 'scenes.json').write_text(json.dumps(self.scenes))
        vector = np.zeros((1, 1280), dtype=np.float32)
        vector[0, 0] = 1
        np.save(self.folder / 'embeddings.npy', vector)
        self.verified = {
            'version': 1, 'generation': self.generation, 'rows': 1,
            'source_checksums': 'exhaustive', 'source_time_base_verified': True,
            'published': False, 'model': 'PE-Core', 'preprocess': 'same',
            'decode_provenance': self.decoder,
            **artifact_digests(self.folder),
        }
        (self.folder / 'verified.json').write_text(json.dumps(self.verified))
        self.timeline = np.array([[0, 500, 123]], dtype=np.int64)

    def validate(self):
        with patch.object(stage, 'source_fingerprint', return_value={'source': 'same'}), \
             patch.object(stage, 'decode_provenance', return_value=self.decoder), \
             patch.object(stage, 'load_timeline', return_value=self.timeline):
            return stage.validate_video(self.video, self.index, self.folder, self.generation)

    def test_rejects_stale_decoder_policy(self):
        self.verified['decode_provenance'] = {'source_map_threads': 2,
                                              'verified_embed_threads': 2}
        (self.folder / 'verified.json').write_text(json.dumps(self.verified))
        with self.assertRaisesRegex(ValueError, 'stale'):
            self.validate()

    def test_rejects_nominal_duration_instead_of_source_track(self):
        self.keyframes['source_duration_ms'] = 4000
        (self.folder / 'keyframes.json').write_text(json.dumps(self.keyframes))
        self.verified.update(artifact_digests(self.folder))
        (self.folder / 'verified.json').write_text(json.dumps(self.verified))
        with self.assertRaisesRegex(ValueError, 'source duration'):
            self.validate()

    def test_combines_distinct_recovery_stages_and_rejects_duplicates(self):
        first = self.root / 'first'
        second = self.root / 'second'
        for root, video in ((first, 'N001-V001'), (second, 'N002-V001')):
            root.mkdir()
            (root / 'manifest.json').write_text(json.dumps({'videos': [
                {'video_id': video, 'path': str(self.folder),
                 'generation': self.generation}]}))
        self.assertEqual(set(stage.stage_entries(first, second)),
                         {'N001-V001', 'N002-V001'})
        (second / 'manifest.json').write_text(json.dumps({'videos': [
            {'video_id': 'N001-V001', 'path': str(self.folder),
             'generation': self.generation}]}))
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            stage.stage_entries(first, second)

    def test_rejects_changed_normalized_vector_after_verification(self):
        vector = np.zeros((1, 1280), dtype=np.float32)
        vector[0, 1] = 1
        np.save(self.folder / 'embeddings.npy', vector)
        with self.assertRaisesRegex(ValueError, 'digest|changed'):
            self.validate()

    def test_rejects_wrong_presentation_milliseconds(self):
        self.keyframes['keyframes'][0]['timestamp_ms'] = 501
        (self.folder / 'keyframes.json').write_text(json.dumps(self.keyframes))
        self.verified.update(artifact_digests(self.folder))
        (self.folder / 'verified.json').write_text(json.dumps(self.verified))
        with self.assertRaisesRegex(ValueError, 'timestamp'):
            self.validate()

    def test_existing_candidate_is_not_replaced_by_new_content(self):
        source = self.root / 'N001-N010_results.zip'
        with zipfile.ZipFile(source, 'w') as archive:
            archive.writestr(f'phase1_transnet/videos__{self.video}/scenes.json', '{}')
        entries = {self.video: {'zip_path': str(self.root / 'Video_N001-N010.zip')}}
        stages = {self.video: (self.folder, self.generation)}
        output = self.root / 'publication'
        with patch.object(stage.media, '_build_index', return_value=self.index), \
             patch.object(stage, 'source_fingerprint', return_value={'source': 'same'}), \
             patch.object(stage, 'decode_provenance', return_value=self.decoder), \
             patch.object(stage, 'load_timeline', return_value=self.timeline):
            stage.build_archive(source, entries, stages, frozenset(), frozenset(), output)
            candidate = output / source.name
            before = candidate.read_bytes()
            stage.build_archive(source, entries, stages, frozenset(), frozenset(), output)
            self.assertEqual(candidate.read_bytes(), before)
            vector = np.zeros((1, 1280), dtype=np.float32)
            vector[0, 1] = 1
            np.save(self.folder / 'embeddings.npy', vector)
            self.verified.update(artifact_digests(self.folder))
            (self.folder / 'verified.json').write_text(json.dumps(self.verified))
            with self.assertRaises(FileExistsError):
                stage.build_archive(source, entries, stages, frozenset(), frozenset(), output)
            self.assertEqual(candidate.read_bytes(), before)


if __name__ == '__main__':
    unittest.main()
