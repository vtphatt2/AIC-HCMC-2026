"""Browser timing routes expose decoded presentation time without changing search data."""

import unittest
from unittest.mock import AsyncMock, patch

import httpx

import main


class FrameTimingApiTests(unittest.IsolatedAsyncioTestCase):
    async def test_n_timeline_and_exact_offset_contract(self):
        data = b"\x00\x00\x00\x00\x40\x9c\x00\x00"
        with (patch.object(main.local_zip_media, "full_frame_timeline_us",
                           AsyncMock(return_value=data)) as timeline,
              patch.object(main.local_zip_media, "exact_frame_offset_us",
                           AsyncMock(return_value=547_004_000)) as offset):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=main.app),
                                         base_url="http://test") as client:
                response = await client.get("/api/video/N027-V003/frame-timeline")
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.content, data)
                self.assertEqual(response.headers["content-type"], "application/octet-stream")
                response = await client.get("/api/video/N027-V003/frame-offset/11025")
                self.assertEqual(response.json(), {
                    "video_id": "N027-V003", "frame_number": 11025,
                    "offset_us": 547_004_000,
                })
                self.assertEqual((await client.get("/api/video/M03-V001/frame-timeline")).status_code, 404)
                self.assertEqual((await client.get("/api/video/N027-V003/frame-offset/-1")).status_code, 422)
        timeline.assert_awaited_once_with("N027-V003")
        offset.assert_awaited_once_with("N027-V003", 11025)

    async def test_unavailable_map_is_reported_without_a_wrong_timestamp(self):
        with patch.object(main.local_zip_media, "full_frame_timeline_us",
                          AsyncMock(side_effect=FileNotFoundError("map pending"))):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=main.app),
                                         base_url="http://test") as client:
                response = await client.get("/api/video/N027-V003/frame-timeline")
        self.assertEqual(response.status_code, 404)


if __name__ == "__main__":
    unittest.main()
