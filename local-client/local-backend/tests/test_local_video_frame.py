import os
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from app.services.local_video_frame import get_local_frame_jpeg, local_video_path


class LocalVideoFrameTests(unittest.TestCase):
    def test_ffmpeg_reads_video_id_from_local_data_only(self):
        video = Path("D:/dataset/videos/L01_V001.mp4")
        completed = Mock(stdout=b"\xff\xd8jpeg\xff\xd9", stderr=b"")
        with (
            patch("app.services.local_video_frame.local_video_path", return_value=video),
            patch("app.services.local_video_frame._ffmpeg_path", return_value="ffmpeg"),
            patch("app.services.local_video_frame.subprocess.run", return_value=completed) as run,
        ):
            jpeg = get_local_frame_jpeg("L01_V001", 1500)

            command = run.call_args.args[0]
            self.assertEqual(command[command.index("-i") + 1], str(video))
            self.assertEqual(jpeg, completed.stdout)

    def test_video_id_cannot_escape_videos_directory(self):
        with patch.dict(os.environ, {"AIC_SAMPLE_ROOT": "D:/dataset"}):
            with self.assertRaisesRegex(ValueError, "Invalid video_id"):
                local_video_path("../secret")


if __name__ == "__main__":
    unittest.main()
