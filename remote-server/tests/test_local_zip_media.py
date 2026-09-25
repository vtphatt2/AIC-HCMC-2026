"""Frames served out of a local Videos_L*.zip must be the frame that was asked
for — checked against ffmpeg decoding the same MP4 unpacked, not against
ourselves.

Two ways this got it wrong before, both silent (a plausible-looking picture from
the wrong moment):

  * `select=eq(n,…)` counts frames in presentation order, but samples are stored
    in decode order — with B-frames those differ (was 3 frames off at 60 s).
  * the first presented frame's pts is `base_pts`, not 0, whenever there is
    reordering (was a further 1 frame off).

Skips when raw_zip_videos/ has no archive, so it is safe to run anywhere.
"""
from __future__ import annotations

import asyncio
import io
import shutil
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services import local_zip_media as media  # noqa: E402

def _ffmpeg() -> str | None:
    try:
        import imageio_ffmpeg
    except ImportError:
        return None
    return imageio_ffmpeg.get_ffmpeg_exe()


def _pick_video() -> str | None:
    index = asyncio.run(media._index())
    return sorted(index)[0] if index else None


class LocalZipMediaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ffmpeg = _ffmpeg()
        if cls.ffmpeg is None:
            raise unittest.SkipTest("imageio-ffmpeg is not installed")
        cls.video_id = _pick_video()
        if cls.video_id is None:
            raise unittest.SkipTest(f"no archives under {media.raw_zip_dir()}")

        cls.tmp = Path(tempfile.mkdtemp(prefix="zipmedia-"))
        entry = asyncio.run(media.lookup(cls.video_id))
        cls.mp4 = cls.tmp / f"{cls.video_id}.mp4"
        with zipfile.ZipFile(entry["zip_path"]) as archive:
            with archive.open(entry["entry_name"]) as src, cls.mp4.open("wb") as dst:
                shutil.copyfileobj(src, dst, length=1 << 20)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(getattr(cls, "tmp", ""), ignore_errors=True)

    def _raw(self, args: list[str]) -> bytes:
        result = subprocess.run(
            [self.ffmpeg, "-loglevel", "error", *args,
             "-pix_fmt", "yuvj420p", "-f", "rawvideo", "pipe:1"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, timeout=120,
        )
        return result.stdout

    def test_served_frame_is_the_frame_that_was_asked_for(self):
        index = asyncio.run(media._get_index(self.video_id))
        fps = index.fps
        self.assertGreater(fps, 0)

        for timestamp_ms in (0, 5_000, 60_000):
            expected = round(timestamp_ms / 1000 * fps)
            if expected >= len(index.sample_pts) - 8:
                continue  # video too short for this probe
            with self.subTest(timestamp_ms=timestamp_ms):
                jpeg = asyncio.run(media.get_frame_jpeg(self.video_id, timestamp_ms))
                self.assertEqual(jpeg[:2], b"\xff\xd8", "not a JPEG")

                ours = subprocess.run(
                    [self.ffmpeg, "-loglevel", "error", "-i", "pipe:0",
                     "-pix_fmt", "yuvj420p", "-f", "rawvideo", "pipe:1"],
                    input=jpeg, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    check=True, timeout=120,
                ).stdout

                # A window around the expected frame, so a near miss reports
                # *which* frame came back rather than just "different".
                low = max(0, expected - 4)
                window = self._raw([
                    "-i", str(self.mp4),
                    "-vf", f"select=between(n\\,{low}\\,{expected + 4})",
                    "-vsync", "0",
                ])
                frame_size = len(ours)
                count = len(window) // frame_size
                self.assertGreater(count, 0, "reference decode produced nothing")

                best, best_diff = None, None
                for i in range(count):
                    ref = window[i * frame_size:(i + 1) * frame_size]
                    diff = sum(abs(a - b) for a, b in zip(ref[::997], ours[::997]))
                    if best_diff is None or diff < best_diff:
                        best, best_diff = low + i, diff
                self.assertEqual(
                    best, expected,
                    f"ts={timestamp_ms} returned frame {best}, expected {expected}",
                )

    def test_fps_comes_from_the_ingest_not_the_container(self):
        """The value ingest used to build timestamp_ms is the one that inverts
        it correctly; a moov-derived average can differ on VFR sources."""
        fps = media.ingest_fps(self.video_id)
        if fps is None:
            self.skipTest(f"{self.video_id} has no ingest archive record")
        index = asyncio.run(media._get_index(self.video_id))
        self.assertAlmostEqual(index.fps, fps, places=9)

    def test_range_request_maps_into_the_archive(self):
        entry = asyncio.run(media.lookup(self.video_id))
        headers, body = asyncio.run(media.open_range(self.video_id, "bytes=0-63"))
        self.assertEqual(headers["Content-Range"], f"bytes 0-63/{entry['size']}")

        async def collect():
            return b"".join([chunk async for chunk in body])

        data = asyncio.run(collect())
        self.assertEqual(len(data), 64)
        # Byte 4 of an MP4 starts the ftyp box — proof the offset landed on the
        # file inside the archive rather than on the archive itself.
        self.assertEqual(data[4:8], b"ftyp")

    def test_unknown_video_is_reported_not_guessed(self):
        with self.assertRaises(media.LocalZipUnavailable):
            asyncio.run(media.lookup("ZZ_V999"))

    def test_new_archive_frames_match_full_decode(self):
        """Check actual indexed pictures for fast-start MP4, MOV HEVC and AV1."""
        for video_id, frame_number in (
            ("M08_V006", 18), ("N011-V001", 100),
            ("N031-V001", 1484), ("S01-V001", 25),
        ):
            with self.subTest(video_id=video_id):
                try:
                    entry = asyncio.run(media.lookup(video_id))
                except media.LocalZipUnavailable:
                    continue
                fps = media.ingest_fps(video_id)
                timestamp_ms = int(frame_number / fps * 1000)
                jpeg = asyncio.run(media.get_frame_jpeg(
                    video_id, timestamp_ms,
                    frame_number=frame_number if video_id.startswith(("N", "S")) else None,
                ))
                displayed = np.asarray(
                    Image.open(io.BytesIO(jpeg)).convert("RGB").resize((64, 36))
                ).astype(np.int16)
                source = (
                    f"subfile,,start,{entry['data_offset']},"
                    f"end,{entry['data_offset'] + entry['size']},,:{entry['zip_path']}"
                )
                reference = subprocess.run(
                    [self.ffmpeg, "-loglevel", "error", "-i", source,
                     "-vf", f"select=eq(n\\,{frame_number}),scale=64:36",
                     "-vsync", "0", "-frames:v", "1", "-pix_fmt", "rgb24",
                     "-f", "rawvideo", "pipe:1"],
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    check=True, timeout=60,
                ).stdout
                self.assertEqual(len(reference), 64 * 36 * 3)
                expected = np.frombuffer(reference, dtype=np.uint8).reshape(36, 64, 3)
                difference = np.abs(displayed - expected).mean()
                self.assertLess(difference, 10, f"wrong frame: MAE={difference:.2f}")

    def test_late_mov_search_frames_match_full_decode_exactly(self):
        """Catch seek-time PTS shifts that can show a nearby, wrong frame."""
        for video_id, frame_number in (
            ("N001-V001", 13184), ("N027-V003", 11025),
            ("N010-V001", 13725), ("N031-V001", 13357),
            ("S01-V012", 3021),
        ):
            with self.subTest(video_id=video_id):
                try:
                    entry = asyncio.run(media.lookup(video_id))
                except media.LocalZipUnavailable:
                    continue
                fps = media.ingest_fps(video_id)
                displayed = asyncio.run(media.get_frame_jpeg(
                    video_id, int(frame_number / fps * 1000), frame_number=frame_number
                ))
                source = (
                    f"subfile,,start,{entry['data_offset']},"
                    f"end,{entry['data_offset'] + entry['size']},,:{entry['zip_path']}"
                )
                reference = subprocess.run(
                    ["/usr/bin/ffmpeg" if video_id.startswith("N") else media._ffmpeg_path(),
                     "-loglevel", "error", "-threads", "2", "-i", source,
                     "-map", "0:v:0", "-vf", f"select=eq(n\\,{frame_number})",
                     "-vsync", "0", "-frames:v", "1", "-q:v", "4",
                     "-f", "image2pipe", "-vcodec", "mjpeg", "pipe:1"],
                    capture_output=True, check=True, timeout=90,
                ).stdout
                self.assertEqual(displayed, reference)


if __name__ == "__main__":
    unittest.main()
