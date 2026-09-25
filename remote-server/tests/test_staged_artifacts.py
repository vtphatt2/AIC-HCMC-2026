import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from app.services.staged_artifacts import metadata_generation, reusable_vectors, atomic_json


class StagedArtifactTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.payload = {'generation': 'a' * 64, 'num_keyframes': 1,
                        'keyframes': [{'frame_number': 0, 'source_pts': 200, 'source_checksum': 123}]}
        self.scenes = {'num_frames': 2, 'scenes': [{'start_frame': 0, 'end_frame': 1}]}
        self.folder, self.payload = metadata_generation(self.root, 'N001', self.payload, self.scenes)
        self.provenance = {'model': 'encoder', 'preprocess': 'same', 'decode_provenance': None}
        self.vector = np.zeros((1, 1280), dtype=np.float32)
        self.vector[0, 0] = 1
        np.save(self.folder / 'embeddings.npy', self.vector)

    def marker(self):
        return {'version': 1, 'generation': self.payload['generation'], 'rows': 1,
                'source_checksums': 'exhaustive', 'source_time_base_verified': True,
                **self.provenance}

    def test_interrupted_embedding_is_not_reused_until_complete(self):
        self.assertFalse(reusable_vectors(self.folder, self.payload, self.provenance))
        atomic_json(self.folder / 'verified.json', self.marker())
        self.assertTrue(reusable_vectors(self.folder, self.payload, self.provenance))
        marker = json.loads((self.folder / 'verified.json').read_text())
        self.assertIn('embeddings_sha256', marker)
        # A different normalized row is a real mismatch even with equal counts.
        self.vector[0, 0] = 0; self.vector[0, 1] = 1
        np.save(self.folder / 'embeddings.npy', self.vector)
        self.assertFalse(reusable_vectors(self.folder, self.payload, self.provenance))

    def test_repaired_source_map_does_not_overwrite_verified_metadata(self):
        atomic_json(self.folder / 'verified.json', self.marker())
        old_files = {p.name: p.read_bytes() for p in self.folder.iterdir()}
        replacement = {**self.payload, 'keyframes': [{'frame_number': 0, 'source_pts': 200, 'source_checksum': 456}]}
        new_folder, new_payload = metadata_generation(self.root, 'N001', replacement, self.scenes)
        self.assertNotEqual(new_folder, self.folder)
        self.assertNotEqual(new_payload['generation'], self.payload['generation'])
        self.assertEqual(old_files, {p.name: p.read_bytes() for p in self.folder.iterdir()})
        self.assertFalse(reusable_vectors(new_folder, new_payload, self.provenance))
        self.assertEqual(metadata_generation(self.root, 'N001', replacement, self.scenes)[0], new_folder)

    def test_unchanged_metadata_keeps_legacy_generation_and_encoder_changes_invalidate(self):
        old = (self.folder / 'keyframes.json').stat().st_mtime_ns
        self.assertEqual(metadata_generation(self.root, 'N001', self.payload, self.scenes)[0], self.folder)
        self.assertEqual((self.folder / 'keyframes.json').stat().st_mtime_ns, old)
        atomic_json(self.folder / 'verified.json', self.marker())
        self.assertFalse(reusable_vectors(self.folder, self.payload, {**self.provenance, 'preprocess': 'changed'}))
        marker = self.marker(); marker['generation'] = 'stale'
        atomic_json(self.folder / 'verified.json', marker)
        self.assertFalse(reusable_vectors(self.folder, self.payload, self.provenance))

    def test_interrupted_metadata_write_resumes_without_changing_existing_bytes(self):
        (self.folder / 'scenes.json').unlink()
        (self.folder / 'embeddings.npy').unlink()
        keyframes_before = (self.folder / 'keyframes.json').read_bytes()

        folder, payload = metadata_generation(self.root, 'N001', self.payload, self.scenes)

        self.assertEqual(folder, self.folder)
        self.assertEqual(payload, self.payload)
        self.assertEqual((folder / 'keyframes.json').read_bytes(), keyframes_before)
        self.assertEqual(json.loads((folder / 'scenes.json').read_text()), self.scenes)

    def test_missing_metadata_with_existing_vectors_is_not_reconstructed(self):
        (self.folder / 'scenes.json').unlink()
        folder, _ = metadata_generation(self.root, 'N001', self.payload, self.scenes)
        self.assertNotEqual(folder, self.folder)
        self.assertFalse((self.folder / 'scenes.json').exists())


if __name__ == '__main__':
    unittest.main()
