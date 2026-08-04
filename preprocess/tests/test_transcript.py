from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from preprocess.transcript.build_keyframe_index import main


class TranscriptPreprocessTests(unittest.TestCase):
    def test_index_uses_rendered_pts_when_metadata_has_no_fps(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "transcripts").mkdir()
            (root / "metadata").mkdir()
            keyframe_dir = root / "keyframes" / "L21_V030"
            keyframe_dir.mkdir(parents=True)
            (root / "transcripts" / "L21_V030_Transcript.txt").write_text(
                "[00:00:00] Xin chao.\n",
                encoding="utf-8",
            )
            (root / "metadata" / "L21_V030.json").write_text(
                json.dumps({"duration": 2.0}),
                encoding="utf-8",
            )
            (keyframe_dir / "000001.jpg").write_bytes(b"image")
            (keyframe_dir / "manifest.json").write_text(
                json.dumps(
                    {
                        "frames": [
                            {
                                "frame": {
                                    "frame_id": "000001",
                                    "source_frame_number": 1,
                                    "timestamp_ms": 1000,
                                },
                                "path": "000001.jpg",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            with patch.object(
                sys,
                "argv",
                ["build_keyframe_index.py", "--data-root", str(root)],
            ):
                self.assertEqual(main(), 0)

            payload = json.loads(
                (root / "keyframe_transcript_index" / "L21_V030.json").read_text(
                    encoding="utf-8"
                )
            )

        self.assertIsNone(payload["fps"])
        self.assertEqual(payload["timing_source"], "rendered_manifest_pts")
        self.assertEqual(payload["sentences"][0]["anchor_frame_id"], "000001")


if __name__ == "__main__":
    unittest.main()
