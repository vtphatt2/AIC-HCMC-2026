"""Resuming an image audit must recheck the exact source files."""
import unittest
from unittest.mock import patch

from scripts.audit_staged_n_images import reusable_image_audit


class StagedImageResumeTests(unittest.TestCase):
    def test_reuse_requires_current_verified_jpegs(self):
        identity = {'generation': 'g1', 'map_path': '/source/map.npy',
                    'source': {'zip': 'same'}}
        previous = {**identity, 'status': 'pass', 'cards': 2}
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


if __name__ == '__main__':
    unittest.main()
