import os
import sys
from pathlib import Path

from dotenv import load_dotenv

BACKEND_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_ROOT))
load_dotenv(BACKEND_ROOT / ".env", override=True)

from app.services.local_video_frame import get_local_frame_jpeg


def main() -> None:
    data_root = Path(os.environ["AIC_SAMPLE_ROOT"])
    video_paths = sorted((data_root / "videos").glob("*.mp4"))
    assert video_paths
    jpeg = get_local_frame_jpeg(video_paths[0].stem, 1_000)
    assert jpeg.startswith(b"\xff\xd8") and jpeg.endswith(b"\xff\xd9")
    print(f"OK: found {len(video_paths)} local videos and decoded {len(jpeg)} JPEG bytes")


if __name__ == "__main__":
    main()
