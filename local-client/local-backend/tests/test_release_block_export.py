"""A new standalone NumPy export cannot reintroduce a blocked video."""
import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import numpy as np

from scripts import export_vectors_npy


class ReleaseBlockExportTests(unittest.TestCase):
    def test_export_skips_bad_video_and_keeps_good_sibling(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / 'N031-N040_results.zip'
            with zipfile.ZipFile(archive, 'w') as zf:
                for video, frame in (('N031-V003', 12165), ('M01_V001', 12)):
                    folder = f'videos__{video}'
                    data = io.BytesIO()
                    np.save(data, np.ones((1, 1280), dtype=np.float32) / np.sqrt(1280))
                    zf.writestr(f'phase1_transnet/{folder}/scenes.json',
                                json.dumps({'fps':25.0,'num_frames':13000}))
                    zf.writestr(f'phase1_transnet/{folder}/keyframes.json',
                                json.dumps({'keyframes':[{'frame_number':frame}]}))
                    zf.writestr(f'phase2_embeddings/{folder}/embeddings.npy', data.getvalue())
            out = root / 'out'
            with patch.object(export_vectors_npy, 'release_blocked_video_ids',
                              return_value=frozenset({'N031-V003'})):
                self.assertEqual(export_vectors_npy.export(root, out), 0)
            with np.load(out / 'vectors.meta.npz', allow_pickle=False) as meta:
                self.assertEqual(meta['video_id'].tolist(), ['M01_V001'])
            self.assertEqual(np.load(out / 'vectors.f32.npy').shape, (1, 1280))


if __name__ == '__main__':
    unittest.main()
