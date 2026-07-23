import unittest
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

import main


class FrontendHttpContractTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(main.app)

    def test_local_frontend_origin_is_allowed(self):
        for origin in ("http://localhost:3000", "http://192.168.0.102:3000"):
            with self.subTest(origin=origin):
                response = self.client.get(
                    "/api/strategies",
                    headers={"Origin": origin},
                )

                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.headers.get("access-control-allow-origin"), origin)

    def test_warmup_contract_accepts_get_and_post(self):
        provider = AsyncMock()
        provider.warmup_text_encoder.return_value = {"status": "ok"}
        with patch.object(main, "_data_provider", provider):
            self.assertEqual(self.client.get("/api/warmup_text_encoder?passes=1").status_code, 200)
            self.assertEqual(self.client.post("/api/warmup_text_encoder?passes=1").status_code, 200)

        self.assertEqual(provider.warmup_text_encoder.await_count, 2)

    def test_translation_runs_in_sample_mode_without_remote_proxy(self):
        with patch.object(main._translation_service, "translate", return_value=["hello"]) as translate:
            response = self.client.post("/api/translate", json={"texts": ["xin chào"]})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"translations": ["hello"]})
        translate.assert_called_once_with(["xin chào"])


if __name__ == "__main__":
    unittest.main()
