from __future__ import annotations

import json
import tempfile
import unittest
import zipfile
from dataclasses import dataclass
from pathlib import Path

from preprocess.transcript_cleaner.collect import (
    CollectionConfig,
    collect_transcripts,
    extract_youtube_id,
)
from preprocess.transcript_cleaner.normalize import normalize_txt_directory


@dataclass(frozen=True)
class Snippet:
    start: float
    text: str


class TranscriptCollectionTest(unittest.TestCase):
    def test_extracts_supported_youtube_urls(self) -> None:
        expected = "Rzpw5WR7nAY"
        self.assertEqual(extract_youtube_id(f"https://youtube.com/watch?v={expected}"), expected)
        self.assertEqual(extract_youtube_id(f"https://youtu.be/{expected}"), expected)
        self.assertEqual(extract_youtube_id(f"https://youtube.com/shorts/{expected}"), expected)

    def test_reads_metadata_directly_from_zip_and_writes_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as temp_name:
            root = Path(temp_name)
            metadata_zip = root / "media-info.zip"
            with zipfile.ZipFile(metadata_zip, "w") as archive:
                archive.writestr(
                    "media-info/L30_V001.json",
                    json.dumps({"watch_url": "https://youtube.com/watch?v=Rzpw5WR7nAY"}),
                )
                archive.writestr(
                    "media-info/L29_V001.json",
                    json.dumps({"watch_url": "https://youtube.com/watch?v=abcdefghijk"}),
                )
                archive.writestr("media-info/L30_V002.json", json.dumps({"watch_url": ""}))

            calls: list[tuple[str, tuple[str, ...]]] = []

            def fetch(video_id: str, languages: tuple[str, ...]):
                calls.append((video_id, languages))
                return [Snippet(0.0, "xin chao"), Snippet(1.25, "cac ban")]

            output = root / "transcripts"
            summary = collect_transcripts(
                CollectionConfig(
                    metadata_path=metadata_zip,
                    output_dir=output,
                    concurrency=2,
                    video_id_regex=r"^L30_",
                ),
                fetcher=fetch,
            )

            self.assertEqual(summary.completed, 1)
            self.assertEqual(summary.failed, 1)
            self.assertEqual(calls, [("Rzpw5WR7nAY", ("vi", "en"))])
            self.assertEqual(
                (output / "missing_transcript_ids.txt").read_text(encoding="utf-8"),
                "L30_V002\n",
            )
            rows = [json.loads(line) for line in (output / "L30_V001.jsonl").read_text().splitlines()]
            self.assertEqual(rows, [
                {"start_time_ms": 0, "end_time_ms": 1250, "text": "xin chao"},
                {"start_time_ms": 1250, "end_time_ms": 6250, "text": "cac ban"},
            ])

    def test_normalizes_legacy_txt_before_cleaning(self) -> None:
        with tempfile.TemporaryDirectory() as temp_name:
            root = Path(temp_name)
            source = root / "raw"
            output = root / "jsonl"
            source.mkdir()
            (source / "L30_V001_Transcript.txt").write_text(
                "[00:00:01] xin chao\n[00:00:03.500] cac ban\n",
                encoding="utf-8",
            )

            converted = normalize_txt_directory(source, output)

            self.assertEqual(converted, 1)
            rows = [json.loads(line) for line in (output / "L30_V001.jsonl").read_text().splitlines()]
            self.assertEqual(rows[0]["start_time_ms"], 1000)
            self.assertEqual(rows[0]["end_time_ms"], 3500)
            self.assertEqual(rows[1]["end_time_ms"], 8500)

    def test_records_youtube_caption_failures_as_retryable_video_ids(self) -> None:
        with tempfile.TemporaryDirectory() as temp_name:
            root = Path(temp_name)
            metadata = root / "L30_V001.json"
            metadata.write_text(
                json.dumps({"watch_url": "https://youtube.com/watch?v=Rzpw5WR7nAY"}),
                encoding="utf-8",
            )

            class NoTranscriptFound(Exception):
                pass

            summary = collect_transcripts(
                CollectionConfig(metadata_path=metadata, output_dir=root / "raw"),
                fetcher=lambda _video_id, _languages: (_ for _ in ()).throw(NoTranscriptFound("none")),
            )

            self.assertEqual(summary.completed, 0)
            self.assertEqual(summary.failed, 1)
            self.assertEqual(
                (root / "raw" / "missing_transcript_ids.txt").read_text(encoding="utf-8"),
                "L30_V001\n",
            )


if __name__ == "__main__":
    unittest.main()
