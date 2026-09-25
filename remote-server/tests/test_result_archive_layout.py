"""Both organizer and pipeline result directory prefixes must be ingestible."""

from __future__ import annotations

import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import numpy as np

from scripts.export_video_fps import collect
from scripts.ingest_zip_pipeline_results import iter_video_records


class ResultArchiveLayoutTest(unittest.TestCase):
    def test_ingest_and_fps_support_video_and_videos_prefixes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for prefix, video_id in (("video__", "L21_V001"), ("videos__", "M01_V001")):
                archive_path = root / f"{video_id}_results.zip"
                folder = f"{prefix}{video_id}"
                array = io.BytesIO()
                np.save(array, np.ones((1, 1280), dtype=np.float32))
                with zipfile.ZipFile(archive_path, "w") as archive:
                    archive.writestr(
                        f"phase1_transnet/{folder}/scenes.json",
                        json.dumps({"fps": 30.0, "num_frames": 100}),
                    )
                    archive.writestr(
                        f"phase1_transnet/{folder}/keyframes.json",
                        json.dumps({"keyframes": [{"frame_number": 12}]}),
                    )
                    archive.writestr(
                        f"phase2_embeddings/{folder}/embeddings.npy", array.getvalue()
                    )

                records = list(iter_video_records(archive_path, {}))
                self.assertEqual([video["video_id"] for video, _ in records], [video_id])
                self.assertEqual(records[0][1][0]["frame_id"], f"{video_id}_000012")
                if video_id == "M01_V001":
                    with patch('app.services.video_quarantine.release_blocked_video_ids',
                               return_value=frozenset({video_id})):
                        self.assertEqual(list(iter_video_records(archive_path, {})), [])

            self.assertEqual(collect(root), {"L21_V001": 30.0, "M01_V001": 30.0})


if __name__ == "__main__":
    unittest.main()
