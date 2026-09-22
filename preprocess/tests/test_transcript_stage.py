from __future__ import annotations

import io
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch
from pathlib import Path

from preprocess.transcript_stage import (
    CollectionCleaningConfig,
    TranscriptStageConfig,
    build_collection_cleaning_command,
    build_transcript_command,
    lot_video_id_regex,
    run_commands_concurrently,
)
from preprocess.__main__ import build_parser
from preprocess.transcript_cleaner.collect import CollectionSummary
from preprocess.transcript_cleaner.workflow import main as transcript_workflow_main


class TranscriptStageConfigTest(unittest.TestCase):
    def test_builds_vendored_cleaner_command(self) -> None:
        config = TranscriptStageConfig(
            input_path=Path("transcripts"),
            output_dir=Path("work/clean_transcript"),
            state_dir=Path("work/state"),
            concurrency=6,
            rpm=30,
            tpm=100_000,
            overwrite=True,
        )

        command = build_transcript_command(config)

        self.assertEqual(command[:3], [sys.executable, "-m", "preprocess.transcript_cleaner.cli"])
        self.assertEqual(command[3], "transcripts")
        self.assertEqual(command[command.index("--concurrency") + 1], "6")
        self.assertIn("--overwrite", command)

    def test_rejects_invalid_concurrency(self) -> None:
        with self.assertRaisesRegex(ValueError, "concurrency"):
            TranscriptStageConfig(input_path=Path("transcripts"), concurrency=0)

    def test_builds_collection_then_cleaning_workflow(self) -> None:
        command = build_collection_cleaning_command(CollectionCleaningConfig(
            metadata_path=Path("media-info.zip"),
            raw_dir=Path("work/raw"),
            output_dir=Path("work/clean"),
            state_dir=Path("work/state"),
        ))
        self.assertEqual(command[1:4], ["-m", "preprocess.transcript_cleaner.workflow", "collect-clean"])
        self.assertEqual(command[4], "media-info.zip")

    def test_maps_split_archive_to_its_metadata_block(self) -> None:
        self.assertEqual(lot_video_id_regex("L26_a"), r"^L26_V0\d{2}$")
        self.assertEqual(lot_video_id_regex("L26_c"), r"^L26_V2\d{2}$")

    def test_maps_new_numeric_range_to_exact_video_ids(self) -> None:
        expression = lot_video_id_regex("N001-N010")
        self.assertIsNotNone(expression)
        self.assertRegex("N001", expression)
        self.assertRegex("N010", expression)
        self.assertNotRegex("N011", expression)
        self.assertNotRegex("L30_V001", expression)


class ConcurrentRunnerTest(unittest.TestCase):
    def test_starts_every_stage_before_waiting(self) -> None:
        events: list[str] = []

        class Process:
            def __init__(self, name: str) -> None:
                self.name = name
                self.returncode = None

            def wait(self) -> int:
                events.append(f"wait:{self.name}")
                self.returncode = 0
                return 0

            def terminate(self) -> None:
                events.append(f"terminate:{self.name}")

        def popen(command: list[str]) -> Process:
            events.append(f"start:{command[0]}")
            return Process(command[0])

        result = run_commands_concurrently(
            {"video": ["video"], "transcript": ["transcript"]},
            popen=popen,
        )

        self.assertEqual(events[:2], ["start:video", "start:transcript"])
        self.assertEqual(result, {"video": 0, "transcript": 0})

    def test_collection_cleaning_starts_watcher_before_collection(self) -> None:
        events: list[str] = []

        class Process:
            returncode = None

            def wait(self) -> int:
                events.append("wait-cleaner")
                self.returncode = 0
                return 0

        def start_cleaner(command):
            self.assertIn("--watch-sentinel", command)
            events.append("start-cleaner")
            return Process()

        def collect(_config):
            events.append("collect")
            return CollectionSummary(1, 1, 0, 0, None)

        with tempfile.TemporaryDirectory() as temp_name:
            root = Path(temp_name)
            with (
                patch("preprocess.transcript_cleaner.workflow.subprocess.Popen", side_effect=start_cleaner),
                patch("preprocess.transcript_cleaner.workflow.collect_transcripts", side_effect=collect),
            ):
                with redirect_stdout(io.StringIO()):
                    code = transcript_workflow_main([
                        "collect-clean", str(root / "metadata.zip"),
                        "--raw-dir", str(root / "raw"),
                        "--output-dir", str(root / "clean"),
                        "--state-dir", str(root / "state"),
                    ])

        self.assertEqual(code, 0)
        self.assertEqual(events, ["start-cleaner", "collect", "wait-cleaner"])


class MainPipelineParserTest(unittest.TestCase):
    def test_zip_run_accepts_metadata_collection_stage(self) -> None:
        args = build_parser().parse_args([
            "run", "--zip", "Videos_L30_a.zip",
            "--transcript-metadata", "media-info.zip",
            "--transcript-concurrency", "4",
        ])
        self.assertEqual(args.transcript_metadata, Path("media-info.zip"))
        self.assertEqual(args.transcript_concurrency, 4)


if __name__ == "__main__":
    unittest.main()
