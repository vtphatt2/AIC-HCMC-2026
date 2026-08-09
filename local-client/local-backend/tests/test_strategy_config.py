import shutil
import unittest
from pathlib import Path

from app.services.strategy_config import StrategyConfigStore


SCHEMA = {
    "raw.semantic": {"label": "Raw visual", "default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1},
    "transcript.semantic": {"label": "Transcript", "default": 0.7, "min": 0.0, "max": 2.0, "step": 0.1},
}


class StrategyConfigStoreTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).resolve().parents[3] / "challenge_resources" / "data" / ".test-strategy-configs"
        shutil.rmtree(self.root, ignore_errors=True)
        self.store = StrategyConfigStore(self.root)

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def test_default_and_saved_presets_are_resolved_from_schema(self):
        default = self.store.get("multi_source", "2.0", SCHEMA, "default")
        saved = self.store.save(
            "multi_source",
            "2.0",
            SCHEMA,
            "transcript-heavy",
            {"raw.semantic": 0.4, "transcript.semantic": 1.6},
        )

        self.assertEqual(default["weights"], {"raw.semantic": 1.0, "transcript.semantic": 0.7})
        self.assertEqual(saved["revision"], 1)
        self.assertEqual(
            [preset["id"] for preset in self.store.list("multi_source", "2.0", SCHEMA)],
            ["default", "transcript-heavy"],
        )
        self.assertEqual(
            self.store.get("multi_source", "2.0", SCHEMA, "transcript-heavy")["weights"],
            {"raw.semantic": 0.4, "transcript.semantic": 1.6},
        )

    def test_default_can_be_tuned_and_keeps_one_list_entry(self):
        saved = self.store.save(
            "multi_source", "2.0", SCHEMA, "default", {"raw.semantic": 1.8}
        )

        self.assertEqual(saved["revision"], 1)
        self.assertEqual(
            self.store.get("multi_source", "2.0", SCHEMA, "default")["weights"],
            {"raw.semantic": 1.8, "transcript.semantic": 0.7},
        )
        self.assertEqual(
            [preset["id"] for preset in self.store.list("multi_source", "2.0", SCHEMA)],
            ["default"],
        )

    def test_save_validates_ids_keys_and_numeric_ranges(self):
        invalid = [
            ("../escape", {"raw.semantic": 1.0}),
            ("bad-key", {"unknown": 1.0}),
            ("too-high", {"raw.semantic": 3.0}),
        ]
        for config_id, weights in invalid:
            with self.subTest(config_id=config_id), self.assertRaises(ValueError):
                self.store.save("multi_source", "2.0", SCHEMA, config_id, weights)

    def test_resaving_increments_revision_and_delete_preserves_default(self):
        self.store.save("multi_source", "2.0", SCHEMA, "balanced", {})
        updated = self.store.save(
            "multi_source", "2.0", SCHEMA, "balanced", {"raw.semantic": 0.5}
        )

        self.assertEqual(updated["revision"], 2)
        self.store.delete("multi_source", "balanced")
        with self.assertRaises(FileNotFoundError):
            self.store.get("multi_source", "2.0", SCHEMA, "balanced")
        with self.assertRaises(ValueError):
            self.store.delete("multi_source", "default")

    def test_number_list_field_supports_event_weights(self):
        schema = {
            "event_weights": {
                "type": "number_list",
                "label": "Event weights",
                "default": [1.0, 1.0, 1.0],
                "min": 0.0,
                "max": 2.0,
                "step": 0.1,
                "min_items": 1,
                "max_items": 20,
            }
        }

        saved = self.store.save(
            "temporal_visual", "2.0", schema, "last-event-heavy", {"event_weights": [1, 1, 1.8]}
        )

        self.assertEqual(saved["weights"]["event_weights"], [1.0, 1.0, 1.8])
        with self.assertRaises(ValueError):
            self.store.save(
                "temporal_visual", "2.0", schema, "invalid", {"event_weights": [1, 3]}
            )


if __name__ == "__main__":
    unittest.main()
