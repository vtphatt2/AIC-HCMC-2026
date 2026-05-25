import os
import json
from pathlib import Path

import httpx
from dotenv import load_dotenv

load_dotenv()

ENV_MODE = os.getenv("ENV_MODE", "MOCK")
REMOTE_SERVER_URL = os.getenv("REMOTE_SERVER_URL", "").rstrip("/")

MOCK_DIR = Path(__file__).parent / "mock"


class DataProvider:
    """
    Switches data source based on ENV_MODE:

      MOCK  — returns data from app/mock/*.json (no server needed)
      LOCAL — proxies requests to the GPU server at REMOTE_SERVER_URL
    """

    def __init__(self):
        self.mode = ENV_MODE

        if self.mode == "LOCAL":
            if not REMOTE_SERVER_URL:
                raise RuntimeError(
                    "ENV_MODE=LOCAL requires REMOTE_SERVER_URL to be set in .env\n"
                    "Example: REMOTE_SERVER_URL=https://xxxx.ngrok.io"
                )
            print(f"DataProvider: LOCAL mode → {REMOTE_SERVER_URL}")

        elif self.mode == "MOCK":
            self._videos, self._frames, self._ocr, self._transcripts = self._load_mock()
            print(
                f"DataProvider: MOCK mode — "
                f"{len(self._videos)} videos, {len(self._frames)} frames loaded"
            )

        else:
            raise RuntimeError(
                f"Unknown ENV_MODE='{self.mode}'. Valid values: MOCK, LOCAL"
            )

    # ── Public interface ───────────────────────────────────────────────────────

    async def get_raw_data(self, query_groups: list[dict], limit: int = 1000) -> dict:
        """
        Returns a unified dict consumed by strategy.fusion_and_temporal():
          frames, ocr, transcripts (lists), videos (dict keyed by video_id)
        """
        if self.mode == "MOCK":
            return self._mock_raw_data(limit)
        return await self._fetch_from_remote(query_groups, limit)

    # ── MOCK ──────────────────────────────────────────────────────────────────

    def _load_mock(self):
        def load(name):
            return json.loads((MOCK_DIR / name).read_text(encoding="utf-8"))

        videos_list = load("mock_videos.json")
        frames = load("mock_frames.json")
        ocr = load("mock_ocr.json")
        transcripts = load("mock_transcripts.json")
        return videos_list, frames, ocr, transcripts

    def _mock_raw_data(self, limit: int) -> dict:
        videos_by_id = {v["video_id"]: v for v in self._videos}
        return {
            "frames":      self._frames[:limit],
            "ocr":         self._ocr[:limit],
            "transcripts": self._transcripts[:limit],
            "videos":      videos_by_id,
        }

    # ── LOCAL (proxy to remote server) ────────────────────────────────────────

    async def _fetch_from_remote(self, query_groups: list[dict], limit: int) -> dict:
        async with httpx.AsyncClient(base_url=REMOTE_SERVER_URL, timeout=15.0) as client:
            response = await client.post(
                "/api/raw-data",
                json={"query_groups": query_groups, "limit": limit},
            )
            response.raise_for_status()
            return response.json()
