import time
import unittest
from threading import Event
from unittest.mock import patch

from fastapi.testclient import TestClient

from agent import server
from agent.state import SearchJobManager, VerifyJobManager


class AgentServerTests(unittest.TestCase):
    def test_two_clients_share_verify_queue_and_new_search_clears_old_verification(self):
        entered = Event()
        release = Event()
        calls = []

        def run_verify(query, candidate, cancel_event):
            calls.append(candidate["frame_id"])
            entered.set()
            release.wait(1)
            return {"checks": [{"requirement": "visual", "status": "UNKNOWN"}],
                    "overall": "INSPECT", "next_action": "operator_inspect"}

        def run_search(query, *, top_k, cancel_event, on_status):
            return {"results": [], "total": 0}

        verify = VerifyJobManager(run_verify)
        search = SearchJobManager(run_search)
        with patch.object(server, "verify_manager", verify), patch.object(server, "manager", search):
            with TestClient(server.app) as operator_a, TestClient(server.app) as operator_b:
                payload = {"query": "official query", "candidate": {"video_id": "L21_V011", "frame_id": "f1"}}
                first = operator_a.post("/api/agent/verify", json=payload)
                second = operator_b.post("/api/agent/verify", json=payload)
                self.assertEqual(first.status_code, 200)
                self.assertEqual(second.status_code, 200)
                self.assertTrue(entered.wait(1))
                self.assertEqual(calls, ["f1"])
                started = operator_b.post("/api/agent/search", json={"query": "new official query", "top_k": 3})
                self.assertEqual(started.status_code, 200)
                stale_request = operator_a.post("/api/agent/verify", json=payload)
                self.assertEqual(stale_request.status_code, 409)
                release.set()
                time.sleep(0.03)
                state = operator_a.get("/api/agent/verify").json()
                self.assertEqual(state["query"], "new official query")
                self.assertEqual(state["results"], {})

    def test_malformed_candidate_rejected_without_starting_job(self):
        verify = VerifyJobManager(lambda *args: {})
        with patch.object(server, "verify_manager", verify):
            with TestClient(server.app) as client:
                response = client.post("/api/agent/verify", json={
                    "query": "query", "candidate": {"video_id": "L21_V011", "steps": "bad"},
                })
                self.assertEqual(response.status_code, 400)
                self.assertEqual(client.get("/api/agent/verify").json()["status"], "idle")


if __name__ == "__main__":
    unittest.main()
