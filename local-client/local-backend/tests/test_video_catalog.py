import unittest

from app.services.video_catalog import search_video_catalog


VIDEOS = [
    {"video_id": "L01_V001", "title": "60 Giây Sáng - Tin Tức Mới Nhất"},
    {"video_id": "L01_V002", "title": "Khám phá khu chợ nổi"},
    {"video_id": "L02_V001", "title": "Bản tin thể thao 60 giây"},
    {"video_id": "L10_V001", "title": "Tin tức cuối ngày"},
]


class VideoCatalogSearchTests(unittest.TestCase):
    def test_exact_video_id_is_ranked_first_case_insensitively(self):
        results = search_video_catalog(VIDEOS, "l01_v002")
        self.assertEqual(results[0]["video_id"], "L01_V002")

    def test_compact_index_prefix_ignores_separators(self):
        results = search_video_catalog(VIDEOS, "L0_")
        self.assertEqual(
            [result["video_id"] for result in results],
            ["L01_V001", "L01_V002", "L02_V001"],
        )

    def test_title_search_is_case_and_accent_insensitive(self):
        results = search_video_catalog(VIDEOS, "tin tuc")
        self.assertEqual(
            [result["video_id"] for result in results],
            ["L10_V001", "L01_V001"],
        )


if __name__ == "__main__":
    unittest.main()
