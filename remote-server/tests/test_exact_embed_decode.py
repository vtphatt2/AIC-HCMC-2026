import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'keyframe_pipeline_global_v9_3/src'))
from pipeline.phase2_embed.decode import _decode_selected_frames
from pipeline.phase2_embed.preprocess import build_encoder_preprocess_plan


class ExactEmbedDecodeTests(unittest.TestCase):
    def setUp(self):
        self.args=SimpleNamespace(ffmpeg_bin='ffmpeg')
        self.plan=build_encoder_preprocess_plan({'input_size':(3,2,2)})
        self.expected={5:(200,11),9:(400,22)}

    def run_ffmpeg(self, command, stdout, stderr, timeout):
        self.assertNotIn('-ss',command)
        stdout.write(bytes(range(24)))
        # Local n resets; identity follows PTS/checksum and requested order.
        stderr.write(b'[showinfo @ a] n: 0 pts: 200 checksum:0000000B\n[showinfo @ b] n: 0 pts: 400 checksum:00000016\n')
        return SimpleNamespace(returncode=0)

    def test_exact_source_identity_keeps_index_order_and_uint8_preprocess(self):
        with patch('pipeline.phase2_embed.decode.subprocess.run',side_effect=self.run_ffmpeg):
            rows=list(_decode_selected_frames(self.args,'source',25,[5,9],None,self.plan,exact_pts=self.expected))
        self.assertEqual([frame for frame,_ in rows],[5,9])
        self.assertEqual(rows[0][1].shape,(3,2,2))
        self.assertEqual(rows[0][1][:,0,0].tolist(),[0,1,2])

    def test_mismatch_emits_no_partially_verified_vectors(self):
        bad=dict(self.expected);bad[9]=(400,99)
        with patch('pipeline.phase2_embed.decode.subprocess.run',side_effect=self.run_ffmpeg):
            stream=_decode_selected_frames(self.args,'source',25,[5,9],None,self.plan,exact_pts=bad)
            with self.assertRaisesRegex(RuntimeError,'verification failed'):next(stream)

    def test_partial_exact_metadata_is_rejected(self):
        with self.assertRaisesRegex(ValueError,'every selected frame'):
            list(_decode_selected_frames(self.args,'source',25,[5,9],None,self.plan,exact_pts={5:(200,11)}))

    def test_verified_decoder_override_is_used(self):
        self.args.verified_decoder_threads=1
        def check(command, stdout, stderr, timeout):
            self.assertEqual(command[command.index('-threads')+1], '1')
            return self.run_ffmpeg(command, stdout, stderr, timeout)
        with patch('pipeline.phase2_embed.decode.subprocess.run',side_effect=check):
            self.assertEqual(len(list(_decode_selected_frames(
                self.args,'source',25,[5,9],None,self.plan,exact_pts=self.expected))),2)

if __name__=='__main__':unittest.main()
