import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SAMPLE_ROOT = ROOT / "AIC2026_sample"

QUERIES_DIR = ROOT / "queries_2"
QUERIES_DIR_1 = ROOT / "queries_1"

OUTPUT_DIR = ROOT / "generated_queries"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FPS = 25
TRANSCRIPT_WINDOW_SEC = 10.0  # +- seconds around scene for transcript extraction
MAX_KEYFRAMES_PER_SCENE = 3   # images sent to LLM per scene

USED_KEYFRAMES_ROOT = SAMPLE_ROOT / "keyframes" / "keyframes"
SCENES_ROOT = SAMPLE_ROOT / "transnetv2_outputs" / "transnetv2_outputs"
TRANSCRIPTS_ROOT = SAMPLE_ROOT / "transcripts" / "transcripts"
METADATA_ROOT = SAMPLE_ROOT / "metadata" / "metadata"
SELECTED_KEYFRAMES_ROOT = USED_KEYFRAMES_ROOT / "selected_keyframes"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")

MAX_RETRIES = 3
BACKOFF_SEC = 2.0
