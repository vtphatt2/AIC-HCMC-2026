"""Exact timing must preserve decoded frame identity across edited MOVs."""

import io
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from app.services import exact_frame_pts as pts


class FakeFFmpeg:
    def __init__(self, lines):
        self.stderr = io.BytesIO(b"".join(lines))

    def wait(self):
        return 0

    def kill(self):
        pass


class ExactFramePtsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        source = self.root / "source.zip"
        source.write_bytes(b"source")
        self.index = SimpleNamespace(
            zip_path=source, data_offset=0, video_size=6,
            timescale=10_000, exact_frame_pts=None,
        )

    def test_full_timeline_uses_the_actual_decoded_count_and_exact_pts(self):
        lines = [b"[Parsed_showinfo_0 @ a] config in time_base: 1/10000\n"]
        for number, timestamp in enumerate((2000, 2400, 3200)):
            lines.append(
                f"[Parsed_showinfo_0 @ a] n: {number} pts: {timestamp} pts_time: 0 "
                f"checksum:{number + 1:08X}\n".encode()
            )
        with patch.object(pts.subprocess, "Popen", return_value=FakeFFmpeg(lines)) as popen:
            path = pts.build_exact_pts(
                "N001-V001", self.index, [2], 3, "ffmpeg", self.root,
                store_full_timeline=True,
            )
            self.assertNotIn("-frames:v", popen.call_args.args[0])
        table = np.load(path)
        self.assertEqual(table[:, 0].tolist(), [0, 1, 2])
        self.assertEqual(table[:, 1].tolist(), [2000, 2400, 3200])
        self.assertEqual(
            np.frombuffer(pts.read_full_timeline_us("N001-V001", self.index, self.root), "<u4").tolist(),
            [0, 40_000, 120_000],
        )
        self.assertEqual(pts.read_exact_pts("N001-V001", self.index, 2, self.root)[:2], (2000, 3200))

        with patch.object(pts.subprocess, "Popen") as popen:
            pts.build_exact_pts("N001-V001", self.index, [2], 3, "ffmpeg", self.root,
                                store_full_timeline=True)
            popen.assert_not_called()

    def test_n031_single_thread_map_has_new_identity_and_command(self):
        source = self.index.zip_path.stat()
        former = [str(self.index.zip_path.resolve()), source.st_size, source.st_mtime_ns,
                  self.index.data_offset, self.index.video_size, 'system-ffmpeg-global-frame-v2']
        old_hash = hashlib.sha256(json.dumps(former).encode()).hexdigest()[:16]
        self.assertNotEqual(pts.index_path('N031-V003', self.index, self.root, decoder_threads=1).name,
                            f'N031-V003-{old_hash}.npy')
        lines = [b'[Parsed_showinfo_0 @ a] config in time_base: 1/10000\n']
        lines += [f'[Parsed_showinfo_0 @ a] n: {i} pts: {i*400} checksum:{i+1:08X}\n'.encode()
                  for i in range(3)]
        with patch.object(pts.subprocess, 'Popen', return_value=FakeFFmpeg(lines)) as popen:
            pts.build_exact_pts('N031-V003', self.index, [2], 3, 'ffmpeg', self.root,
                                store_full_timeline=True, decoder_threads=1)
        command = popen.call_args.args[0]
        self.assertEqual(command[command.index('-threads') + 1], '1')

    def test_missing_searched_frame_rejects_map(self):
        lines = [b"[Parsed_showinfo_0 @ a] config in time_base: 1/10000\n"]
        lines += [f"[Parsed_showinfo_0 @ a] n: {i} pts: {i * 400} checksum:{i+1:08X}\n".encode()
                  for i in range(3)]
        with patch.object(pts.subprocess, "Popen", return_value=FakeFFmpeg(lines)):
            with self.assertRaisesRegex(RuntimeError, "selected missing \\[3\\]"):
                pts.build_exact_pts("N001-V001", self.index, [3], 4, "ffmpeg", self.root,
                                    store_full_timeline=True)

    def test_full_timeline_counts_frames_across_showinfo_reset(self):
        lines = [b"[Parsed_showinfo_0 @ a] config in time_base: 1/10000\n"]
        for local_number, timestamp in ((0, 2000), (1, 2400), (0, 2800), (1, 3200)):
            lines.append(
                f"[Parsed_showinfo_0 @ a] n: {local_number} pts: {timestamp} "
                f"checksum:{timestamp:08X}\n".encode()
            )
        with patch.object(pts.subprocess, "Popen", return_value=FakeFFmpeg(lines)):
            path = pts.build_exact_pts("N011-V003", self.index, [3], 4, "ffmpeg", self.root,
                                       store_full_timeline=True)
        table = np.load(path)
        self.assertEqual(table[:, 0].tolist(), [0, 1, 2, 3])
        self.assertEqual(table[:, 1].tolist(), [2000, 2400, 2800, 3200])

    def test_full_timeline_rejects_decoder_count_mismatch(self):
        lines = [b"[Parsed_showinfo_0 @ a] config in time_base: 1/10000\n"]
        lines += [f"[Parsed_showinfo_0 @ a] n: {i} pts: {i * 400} "
                  f"checksum:{i+1:08X}\n".encode() for i in range(3)]
        with patch.object(pts.subprocess, "Popen", return_value=FakeFFmpeg(lines)):
            with self.assertRaisesRegex(RuntimeError, "decoded 3/4"):
                pts.build_exact_pts("N001-V001", self.index, [2], 4, "ffmpeg", self.root,
                                    store_full_timeline=True)

    def test_parser_recovers_checksums_across_diagnostic_lines(self):
        diagnostics = (
            b"[Parsed_showinfo_0 @ a] n: 7 pts: 1234 pts_time:0.1\n"
            b"[h264 @ b] warning about an edited packet\n"
            b"checksum:0ABCDEF1\n"
            b"[Parsed_showinfo_0 @ a] n: 8 pts: 1274 pts_time:0.2 checksum:0ABCDEF2\n"
        )
        self.assertEqual(pts.showinfo_frames(diagnostics),
                         [(7, 1234, 0x0ABCDEF1), (8, 1274, 0x0ABCDEF2)])


if __name__ == "__main__":
    unittest.main()
