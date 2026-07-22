import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

SAMPLE_ROOT = ROOT / "AIC2026_sample"

QUERIES_DIR = ROOT / "queries_2"
QUERIES_DIR_1 = ROOT / "queries_1"

OUTPUT_DIR = ROOT / "generated_queries"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FPS = 25
TRANSCRIPT_WINDOW_SEC = 10.0
MAX_KEYFRAMES_PER_SCENE = 3

USED_KEYFRAMES_ROOT = SAMPLE_ROOT / "keyframes" / "keyframes"
SCENES_ROOT = SAMPLE_ROOT / "transnetv2_outputs" / "transnetv2_outputs"
TRANSCRIPTS_ROOT = SAMPLE_ROOT / "transcripts" / "transcripts"
METADATA_ROOT = SAMPLE_ROOT / "metadata" / "metadata"
SELECTED_KEYFRAMES_ROOT = USED_KEYFRAMES_ROOT / "selected_keyframes"

_raw_key = os.getenv("FIREWORKS_API_KEY", os.getenv("GEMINI_API_KEY", os.getenv("OPENAI_API_KEY", "")))
OPENAI_API_KEY = _raw_key
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.fireworks.ai/inference/v1").rstrip("/")
OPENAI_MODEL = os.getenv("LLM_MODEL", os.getenv("OPENAI_MODEL", "accounts/fireworks/models/qwen3p7-plus"))

# ponytail: no fallbacks needed for Fireworks — model IDs are stable.
MODEL_FALLBACKS: list[str] = []

MAX_RETRIES = 3
BACKOFF_SEC = 2.0

# ponytail: 500KB per image is safe for gpt-4o-mini vision, ~$0.00015/img.
MAX_IMAGE_SIZE_BYTES = 500 * 1024
IMAGE_QUALITY = 75

if not OPENAI_API_KEY or "your_key_here" in OPENAI_API_KEY:
    OPENAI_API_KEY = ""
    _msg = (
        "\n⚠️  No API key configured!\n"
        "   Copy .env.example → .env and fill in your key:\n"
        "     cp .env.example .env\n"
    )
    # defer print so module import doesn't spam; dry-run doesn't need it
