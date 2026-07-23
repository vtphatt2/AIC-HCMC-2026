import shutil
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

from fastapi.testclient import TestClient

import main
from app.services.strategy_config import StrategyConfigStore


class StrategyConfigHttpTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).resolve().parents[3] / "data" / ".test-strategy-config-http"
        shutil.rmtree(self.root, ignore_errors=True)
        self.store = StrategyConfigStore(self.root)
        self.strategy = Mock()
        self.strategy.name = "Tunable"
        self.strategy.description = "test"
        self.strategy.author = "test"
        self.strategy.version = "2.0"
        self.strategy.config_schema = {
            "transcript.semantic": {
                "label": "Transcript",
                "default": 1.0,
                "min": 0.0,
                "max": 2.0,
                "step": 0.1,
            }
        }
        self.strategy.search = AsyncMock(return_value=[])
        self.client = TestClient(main.app)

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def test_save_list_and_search_with_named_config(self):
        with (
            patch.object(main, "_strategies", {"tunable": self.strategy}),
            patch.object(main, "_strategy_configs", self.store),
        ):
            saved = self.client.put(
                "/api/strategies/tunable/configs/transcript-heavy",
                json={"weights": {"transcript.semantic": 1.7}},
            )
            listed = self.client.get("/api/strategies/tunable/configs")
            searched = self.client.post(
                "/api/search",
                json={
                    "strategy_id": "tunable",
                    "config_id": "transcript-heavy",
                    "config_overrides": {"transcript.semantic": 0.4},
                    "query_groups": [{"query": "hello"}],
                },
            )

        self.assertEqual(saved.status_code, 200)
        self.assertEqual([row["id"] for row in listed.json()["configs"]], ["default", "transcript-heavy"])
        self.assertEqual(searched.json()["config_id"], "transcript-heavy")
        self.assertEqual(searched.json()["effective_config"], {"transcript.semantic": 0.4})
        self.strategy.search.assert_awaited_once()
        self.assertEqual(self.strategy.search.await_args.kwargs["options"], {"transcript.semantic": 0.4})
        self.assertEqual(
            self.store.get("tunable", "2.0", self.strategy.config_schema, "transcript-heavy")["weights"],
            {"transcript.semantic": 1.7},
        )

    def test_unknown_config_is_rejected(self):
        with (
            patch.object(main, "_strategies", {"tunable": self.strategy}),
            patch.object(main, "_strategy_configs", self.store),
        ):
            response = self.client.post(
                "/api/search",
                json={"strategy_id": "tunable", "config_id": "missing", "query_groups": []},
            )

        self.assertEqual(response.status_code, 404)


if __name__ == "__main__":
    unittest.main()
