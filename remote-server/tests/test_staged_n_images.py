"""Resuming an image audit must recheck the exact source files."""
import unittest
from unittest.mock import patch

from scripts.audit_staged_n_images import reports_for_manifest, reusable_image_audit


class StagedImageResumeTests(unittest.TestCase):
    def test_resume_drops_rows_removed_from_new_manifest(self):
        previous = {'N001-V001': {'status': 'pass'}, 'N027-V003': {'status': 'error'}}
        self.assertEqual(
            reports_for_manifest(previous, {'N001-V001': {}}),
            {'N001-V001': {'status': 'pass'}},
        )

    def test_reuse_requires_current_verified_jpegs(self):
        identity = {'generation': 'g1', 'map_path': '/source/map.npy',
                    'source': {'zip': 'same'}}
        previous = {**identity, 'status': 'pass', 'cards': 2, 'webp_cards': 2}
        frames = [1, 3]
        with patch('scripts.audit_staged_n_images.verified_exact_image_set',
                   return_value=True) as verified:
            self.assertTrue(reusable_image_audit(previous, identity, 'N001', object(), frames))
            verified.assert_called_once()
        with patch('scripts.audit_staged_n_images.verified_exact_image_set',
                   return_value=False):
            self.assertFalse(reusable_image_audit(previous, identity, 'N001', object(), frames))
        self.assertFalse(reusable_image_audit({**previous, 'cards': 1}, identity,
                                               'N001', object(), frames))
        self.assertFalse(reusable_image_audit({k:v for k,v in previous.items()
                                               if k != 'webp_cards'}, identity,
                                              'N001', object(), frames))


if __name__ == '__main__':
    unittest.main()
