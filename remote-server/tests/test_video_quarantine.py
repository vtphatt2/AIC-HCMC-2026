import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from app.services import video_quarantine as policy
from app.db import milvus_client
from app.services.video_catalog import search_video_catalog


class QuarantineTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.path=Path(self.tmp.name)/'quarantine.json'
        self.env=patch.dict('os.environ',{'VIDEO_QUARANTINE_PATH':str(self.path)})
        self.env.start();self.addCleanup(self.env.stop)

    def write(self, videos):
        self.path.write_text(json.dumps({'version':1,'videos':{v:{'reason':'source decode failure'} for v in videos}}))

    def test_missing_manifest_preserves_existing_expression(self):
        self.assertEqual(policy.excluded_video_ids(),frozenset())
        self.assertEqual(policy.milvus_expr('timestamp_ms > 0'),'timestamp_ms > 0')
        self.assertIsNone(policy.milvus_expr(None))

    def test_dynamic_restore_and_no_data_mutations(self):
        self.write(['N001-V001'])
        self.assertEqual(policy.excluded_video_ids(),frozenset({'N001-V001'}))
        self.assertEqual(policy.filter_rows([{'video_id':'N001-V001'},{'video_id':'M01_V001'}]),[{'video_id':'M01_V001'}])
        self.write([])
        self.assertEqual(policy.excluded_video_ids(),frozenset())

    def test_only_explicit_source_identity_blocks_offline_release_jobs(self):
        self.path.write_text(json.dumps({'version':1,'videos':{
            'N031-V003':{'reason':'unstable source picture','release_blocked':True},
            'N001-V001':{'reason':'browser playback needs conversion'},
        }}))
        self.assertEqual(policy.excluded_video_ids(),frozenset({'N031-V003','N001-V001'}))
        self.assertEqual(policy.release_blocked_video_ids(),frozenset({'N031-V003'}))

    def test_release_block_is_atomic_and_preserves_existing_context(self):
        self.path.write_text(json.dumps({'version':1,'videos':{
            'N015-V001':{'reason':'browser failure','custom':'keep'},
        }}))
        policy.block_release('N015-V001', reason='decoder mismatch',
                             details='complete replay differs', evidence='verification/report.json')
        data=json.loads(self.path.read_text())
        self.assertEqual(data['videos']['N015-V001']['custom'],'keep')
        self.assertTrue(data['videos']['N015-V001']['release_blocked'])
        self.assertEqual(policy.release_blocked_video_ids(),frozenset({'N015-V001'}))
        self.assertFalse(self.path.with_suffix('.json.partial').exists())

    def test_release_block_rejects_incomplete_or_non_n_evidence(self):
        with self.assertRaises(ValueError):
            policy.block_release('M01_V001', reason='x', details='y', evidence='z')
        with self.assertRaises(ValueError):
            policy.block_release('N001-V001', reason='', details='y', evidence='z')

    def test_clear_release_block_requires_the_expected_audited_reason(self):
        self.path.write_text(json.dumps({'version':1,'videos':{
            'N015-V001':{'reason':'decoder_sensitive_source_picture',
                         'details':'candidate','evidence':'old','release_blocked':True},
        }}))
        with self.assertRaises(ValueError):
            policy.clear_release_block('N015-V001', reason='stable', details='ok',
                                       evidence='new', expected_reason='different')
        policy.clear_release_block('N015-V001', reason='verified_decoder_setting_variation',
                                   details='four-thread replay exact', evidence='new',
                                   expected_reason='decoder_sensitive_source_picture')
        entry=json.loads(self.path.read_text())['videos']['N015-V001']
        self.assertNotIn('release_blocked',entry)
        self.assertEqual(entry['reason'],'verified_decoder_setting_variation')

    def test_milvus_filters_before_limit_without_changing_search_parameters(self):
        collection=MagicMock();collection.search.return_value=[[]]
        milvus_client.vector_search(collection,[1.,0.],top_k=100,algorithm='hnsw')
        baseline=collection.search.call_args.kwargs
        self.write(['N001-V001'])
        milvus_client.vector_search(collection,[1.,0.],top_k=100,algorithm='hnsw')
        changed=collection.search.call_args.kwargs
        self.assertEqual(changed['expr'],'video_id not in ["N001-V001"]')
        for key in ('data','limit','param','output_fields'):
            self.assertEqual(baseline[key],changed[key])
        self.assertEqual(policy.milvus_expr('video_id == "M01_V001"'),
                         '(video_id == "M01_V001") and (video_id not in ["N001-V001"])')
        collection.upsert.assert_not_called();collection.delete.assert_not_called()

    def test_catalog_exclusion_before_top_k(self):
        self.write(['N001-V001'])
        rows=[{'video_id':'N001-V001','title':'street'}, {'video_id':'M01_V001','title':'street'}]
        self.assertEqual(search_video_catalog(rows,'street',limit=1),[rows[1]])

    def test_invalid_manifest_is_not_silently_ignored(self):
        self.path.write_text('{"version":9,"videos":{}}')
        with self.assertRaises(ValueError):policy.excluded_video_ids()


if __name__=='__main__':unittest.main()
