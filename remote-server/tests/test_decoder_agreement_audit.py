import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import audit_decoder_agreement as audit


class DecoderAgreementAuditTests(unittest.TestCase):
    def test_complete_mismatch_is_recorded_as_a_reversible_release_block(self):
        result = {'pts_mismatch_count': 0, 'checksum_mismatch_count': 3}
        report = (audit.ROOT / 'challenge_resources/data/zip_embeddings/verification'
                  / 'decoder_agreement_test.json')
        with patch.object(audit, 'block_release') as block:
            audit.block_mismatch('N099-V003', result, 1, report)
        block.assert_called_once()
        self.assertEqual(block.call_args.args, ('N099-V003',))
        self.assertEqual(block.call_args.kwargs['reason'],
                         'decoder_sensitive_source_picture')
        self.assertIn('pixel mismatches=3', block.call_args.kwargs['details'])
        self.assertEqual(block.call_args.kwargs['evidence'],
                         'zip_embeddings/verification/decoder_agreement_test.json')

    def test_external_report_path_remains_valid_evidence(self):
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(audit, 'block_release') as block:
            report = Path(directory) / 'report.json'
            audit.block_mismatch('N099-V003', {
                'pts_mismatch_count': 1, 'checksum_mismatch_count': 0,
            }, 1, report)
        self.assertEqual(block.call_args.kwargs['evidence'], str(report))


if __name__ == '__main__':
    unittest.main()
