import time
import unittest
from threading import Event

from agent.state import SearchJobManager


def wait_until(predicate, timeout=2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("job did not reach expected state")


class SearchJobManagerTests(unittest.TestCase):
    def test_duplicate_query_is_shared_between_operators(self):
        release = Event()
        calls = []

        def run(query, *, top_k, cancel_event, on_status):
            calls.append(query)
            release.wait(1)
            return {"results": [{"frame_id": "frame-1"}], "total": 1}

        manager = SearchJobManager(run)
        first = manager.start("official query")
        second = manager.start("official query")
        self.assertEqual(first["query_id"], second["query_id"])
        release.set()
        wait_until(lambda: manager.snapshot()["status"] == "done")
        self.assertEqual(calls, ["official query"])

    def test_reset_discards_late_result_from_old_generation(self):
        started = Event()
        release = Event()

        def run(query, *, top_k, cancel_event, on_status):
            started.set()
            release.wait(1)
            return {"results": [{"frame_id": "stale-frame"}], "total": 1}

        manager = SearchJobManager(run)
        manager.start("old query")
        self.assertTrue(started.wait(1))
        reset = manager.reset()
        release.set()
        time.sleep(0.03)
        self.assertEqual(reset["status"], "idle")
        self.assertEqual(manager.snapshot()["results"], [])
        self.assertEqual(manager.snapshot()["query"], "")

    def test_new_query_replaces_old_without_old_result_publishing(self):
        old_started = Event()
        old_release = Event()

        def run(query, *, top_k, cancel_event, on_status):
            if query == "old":
                old_started.set()
                old_release.wait(1)
            return {"results": [{"frame_id": query}], "total": 1}

        manager = SearchJobManager(run)
        manager.start("old")
        self.assertTrue(old_started.wait(1))
        new_state = manager.start("new")
        old_release.set()
        wait_until(lambda: manager.snapshot()["status"] == "done")
        self.assertEqual(manager.snapshot()["query_id"], new_state["query_id"])
        self.assertEqual(manager.snapshot()["results"][0]["frame_id"], "new")

    def test_cancel_and_error_stay_in_agent_state(self):
        started = Event()
        release = Event()

        def run(query, *, top_k, cancel_event, on_status):
            started.set()
            release.wait(1)
            raise RuntimeError("Codex unavailable")

        manager = SearchJobManager(run)
        manager.start("query")
        self.assertTrue(started.wait(1))
        cancelled = manager.cancel()
        release.set()
        time.sleep(0.03)
        self.assertEqual(cancelled["status"], "cancelled")
        self.assertEqual(manager.snapshot()["status"], "cancelled")
        manager.start("another query")
        wait_until(lambda: manager.snapshot()["status"] == "error")
        self.assertIn("Codex unavailable", manager.snapshot()["error"])


if __name__ == "__main__":
    unittest.main()
