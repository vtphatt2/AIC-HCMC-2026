"""The readiness audit must separate export rounding from changed vectors."""
import unittest

import numpy as np

from scripts.audit_readiness import MAX_EXPORT_VECTOR_ABS_ERROR, export_vector_error


class ExportVectorAuditTests(unittest.TestCase):
    def test_float32_renormalization_does_not_mark_a_vector_changed(self):
        packaged = np.array([0.6, 0.8, 0.0], dtype=np.float32)
        exported = packaged.copy()
        exported[0] = np.nextafter(exported[0], np.float32(1))
        self.assertLess(export_vector_error(packaged, exported), MAX_EXPORT_VECTOR_ABS_ERROR)

    def test_material_or_nonfinite_change_is_rejected(self):
        packaged = np.array([0.6, 0.8, 0.0], dtype=np.float32)
        changed = packaged.copy()
        changed[0] += 0.001
        self.assertGreater(export_vector_error(packaged, changed), MAX_EXPORT_VECTOR_ABS_ERROR)
        changed[0] = np.nan
        self.assertEqual(export_vector_error(packaged, changed), float('inf'))


if __name__ == '__main__':
    unittest.main()
