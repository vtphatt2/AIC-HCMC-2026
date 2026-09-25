import time
import unittest
from threading import Event

import httpx

from agent.state import VerifyJobManager
from agent.tools import VortaClient
from agent.verify_agent import VerifyAgent


class FakeVerifier:
    def __init__(self, checks):
        self.checks = checks

    def verify(self, query, evidence, cancel_event=None):
        return {"checks": self.checks}, "verify-thread"


class VerifyAgentTests(unittest.TestCase):
    def test_inspection_reads_only_candidate_video_and_small_context(self):
        calls = []

        def handler(request):
            calls.append(request)
            if request.url.path.endswith("/context-frames"):
                return httpx.Response(200, json={"before": [{"frame_id": "before"}], "middle": [], "after": []})
            return httpx.Response(200, json={"video_id": "L21_V011", "segments": [
                {"start_ms": 9000, "end_ms": 12000, "text": "The speaker says mangoes", "speaker": None},
                {"start_ms": 90000, "end_ms": 92000, "text": "unrelated", "speaker": None},
            ]})

        with httpx.Client(transport=httpx.MockTransport(handler), base_url="http://vorta.test") as http:
            evidence = VortaClient("http://vorta.test", http=http).inspect_candidate({
                "video_id": "L21_V011", "frame_id": "L21_V011_000250", "timestamp_ms": 10000,
            })

        self.assertEqual([call.url.path for call in calls], [
            "/api/video/L21_V011/context-frames", "/api/transcript/L21_V011",
        ])
        self.assertEqual(calls[0].url.params["expand"], "1")
        self.assertEqual(len(evidence["transcript_segments"]), 1)
        self.assertFalse(evidence["visual_content_available"])

    def test_visual_claims_and_unsupported_quotes_become_unknown(self):
        class FakeVorta:
            def inspect_candidate(self, candidate):
                return {"candidate": candidate, "matched_frames": [], "neighbor_frames": [],
                        "transcript_segments": [{"text": "The speaker says mangoes"}],
                        "visual_content_available": False, "errors": []}

        verifier = FakeVerifier([
            {"requirement": "red shirt visible", "modality": "visual", "status": "MATCH", "evidence_quote": None, "note": "red shirt"},
            {"requirement": "speaker says oranges", "modality": "transcript", "status": "MATCH", "evidence_quote": "oranges", "note": "spoken"},
        ])
        result = VerifyAgent(verifier, FakeVorta()).run("red shirt and oranges", {"video_id": "L21_V011"})
        self.assertEqual([check["status"] for check in result["checks"]], ["UNKNOWN", "UNKNOWN"])
        self.assertEqual(result["overall"], "INSPECT")

    def test_transcript_match_requires_quote_in_supplied_evidence(self):
        class FakeVorta:
            def inspect_candidate(self, candidate):
                return {"candidate": candidate, "matched_frames": [], "neighbor_frames": [],
                        "transcript_segments": [{"text": "The speaker says mangoes"}],
                        "visual_content_available": False, "errors": []}

        verifier = FakeVerifier([{
            "requirement": "speaker says mangoes", "modality": "transcript", "status": "MATCH",
            "evidence_quote": "says mangoes", "note": "audible in transcript",
        }])
        result = VerifyAgent(verifier, FakeVorta()).run("says mangoes", {"video_id": "L21_V011"})
        self.assertEqual(result["checks"][0]["status"], "MATCH")
        self.assertEqual(result["overall"], "PROMISING")

    def test_missing_transcript_cannot_become_match(self):
        def handler(request):
            return httpx.Response(404, json={"detail": "missing"})

        with httpx.Client(transport=httpx.MockTransport(handler), base_url="http://vorta.test") as http:
            vorta = VortaClient("http://vorta.test", http=http)
            result = VerifyAgent(FakeVerifier([{
                "requirement": "speaker says mangoes", "modality": "transcript", "status": "MATCH",
                "evidence_quote": "mangoes", "note": "claimed",
            }]), vorta).run("speaker says mangoes", {"video_id": "L21_V011", "timestamp_ms": 10000})
        self.assertEqual(result["checks"][0]["status"], "UNKNOWN")
        self.assertEqual(len(result["evidence"]["errors"]), 2)

    def test_malformed_verifier_output_fails_without_inventing_checks(self):
        class FakeVorta:
            def inspect_candidate(self, candidate):
                return {"matched_frames": [], "neighbor_frames": [], "transcript_segments": [], "errors": []}

        with self.assertRaisesRegex(ValueError, "invalid checks"):
            VerifyAgent(FakeVerifier([]), FakeVorta()).run("query", {"video_id": "L21_V011"})

    def test_duplicate_queue_and_new_query_discard_old_work(self):
        started = Event()
        release = Event()

        def run(query, candidate, cancel_event):
            if query == "old":
                started.set()
                release.wait(1)
            return {"checks": [], "overall": "INSPECT", "next_action": "operator_inspect"}

        manager = VerifyJobManager(run)
        candidate = {"video_id": "L21_V011", "frame_id": "f1"}
        first = manager.start("old", candidate)
        duplicate = manager.start("old", candidate)
        first_id = (first["current_candidate"] or first["queue"][0])["candidate_id"]
        duplicate_id = (duplicate["current_candidate"] or duplicate["queue"][0])["candidate_id"]
        self.assertEqual(first_id, duplicate_id)
        self.assertTrue(started.wait(1))
        manager.start("new", candidate)
        release.set()
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline and manager.snapshot()["status"] != "done":
            time.sleep(0.01)
        state = manager.snapshot()
        self.assertEqual(state["query"], "new")
        self.assertEqual(len(state["results"]), 1)
        self.assertEqual(next(iter(state["results"].values()))["query"], "new")

    def test_cancel_discards_late_verification(self):
        entered = Event()
        release = Event()

        def run(query, candidate, cancel_event):
            entered.set()
            release.wait(1)
            return {"checks": [], "overall": "INSPECT"}

        manager = VerifyJobManager(run)
        manager.start("query", {"video_id": "L21_V011", "frame_id": "f1"})
        self.assertTrue(entered.wait(1))
        self.assertEqual(manager.cancel()["status"], "cancelled")
        release.set()
        time.sleep(0.03)
        self.assertEqual(manager.snapshot()["results"], {})

    def test_two_distinct_candidates_run_serially(self):
        entered = Event()
        release = Event()
        calls = []

        def run(query, candidate, cancel_event):
            calls.append(candidate["frame_id"])
            if candidate["frame_id"] == "f1":
                entered.set()
                release.wait(1)
            return {"checks": [], "overall": "INSPECT"}

        manager = VerifyJobManager(run)
        manager.start("query", {"video_id": "L21_V011", "frame_id": "f1"})
        self.assertTrue(entered.wait(1))
        queued = manager.start("query", {"video_id": "L21_V011", "frame_id": "f2"})
        self.assertEqual(len(queued["queue"]), 1)
        self.assertEqual(calls, ["f1"])
        release.set()
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline and len(manager.snapshot()["results"]) < 2:
            time.sleep(0.01)
        self.assertEqual(calls, ["f1", "f2"])
        self.assertEqual(len(manager.snapshot()["results"]), 2)


if __name__ == "__main__":
    unittest.main()
