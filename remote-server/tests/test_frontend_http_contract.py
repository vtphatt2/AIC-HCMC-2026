import unittest
from unittest.mock import patch

from app.services import transcript_jsonl_reader


class FrontendHttpContractTests(unittest.TestCase):
    def test_transcript_matches_frontend_segment_contract(self):
        chunks = [{"start_time_ms": 1000, "end_time_ms": 2000, "text": "hello"}]
        with patch.object(
            transcript_jsonl_reader, "read_transcript_jsonl_as_api", return_value=chunks
        ):
            response = transcript_jsonl_reader.transcript_response("L01_V001")

        self.assertEqual(response, {
            "video_id": "L01_V001",
            "segments": [{
                "start_ms": 1000,
                "end_ms": 2000,
                "text": "hello",
                "speaker": None,
            }],
        })

    def test_missing_transcript_is_404(self):
        with patch.object(
            transcript_jsonl_reader, "read_transcript_jsonl_as_api", return_value=[]
        ):
            response = transcript_jsonl_reader.transcript_response("L01_V999")

        self.assertIsNone(response)

    def test_reader_falls_back_to_current_txt_layout(self):
        parsed = [
            {"start_time_ms": 1000, "end_time_ms": 3000, "text": "hello"},
            {"start_time_ms": 3000, "end_time_ms": 3000, "text": "world"},
        ]
        with (
            patch.object(transcript_jsonl_reader, "_find_jsonl", return_value=None),
            patch.object(transcript_jsonl_reader, "_find_txt", return_value="transcript.txt"),
            patch.object(transcript_jsonl_reader, "parse_transcript_file", return_value=(None, parsed)),
        ):
            segments = transcript_jsonl_reader.read_transcript_jsonl("L01_V001")

        self.assertEqual(segments[0]["start_time_ms"], 1000)
        self.assertEqual(segments[0]["end_time_ms"], 3000)


if __name__ == "__main__":
    unittest.main()
