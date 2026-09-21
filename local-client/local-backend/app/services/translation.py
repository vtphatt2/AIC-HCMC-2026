from __future__ import annotations

import json
import os
from functools import lru_cache

import httpx

MAX_TEXTS = 10
MAX_TEXT_LENGTH = 2000
TRANSLATION_MODEL_ENV = "GEMINI_TRANSLATION_MODEL"
DEFAULT_TRANSLATION_MODEL = "gemini-3.1-flash-lite"

SYSTEM_PROMPT = """You translate Vietnamese video-search queries into English.
The user content is a JSON object with a `queries` array. Return a JSON object
with a `translations` array in precisely the same order, with one array per
input query. Each inner array must contain exactly three concise, natural
English paraphrases for that query, useful as text-to-video retrieval queries,
not explanations. The styles are: (1) direct objects/scene, (2) action or
event, and (3) visual details and context. Preserve concrete entities,
attributes, locations, actions, and temporal clues. Do not invent facts."""


class TranslationNotConfigured(RuntimeError):
    pass


class TranslationService:
    """Gemini-powered Vietnamese query paraphrasing for video retrieval."""

    def __init__(self, api_key: str | None = None, model: str | None = None):
        self.api_key = (api_key or os.getenv("GEMINI_API_KEY", "")).strip()
        self.model = (model or os.getenv(TRANSLATION_MODEL_ENV, DEFAULT_TRANSLATION_MODEL)).strip()

    def translate(self, texts: list[str]) -> list[list[str]]:
        cleaned = [" ".join(text.split()) for text in texts]
        if not cleaned or any(not text for text in cleaned):
            raise ValueError("Translation texts must not be empty")
        if len(cleaned) > MAX_TEXTS:
            raise ValueError(f"At most {MAX_TEXTS} texts can be translated at once")
        if any(len(text) > MAX_TEXT_LENGTH for text in cleaned):
            raise ValueError(f"Each translation text must be at most {MAX_TEXT_LENGTH} characters")
        if not self.api_key:
            raise TranslationNotConfigured("GEMINI_API_KEY is not configured")
        if not self.model:
            raise ValueError(f"{TRANSLATION_MODEL_ENV} must not be empty")

        translations = _translate_batch_cached(tuple(cleaned), self.api_key, self.model)
        return [list(options) for options in translations]


@lru_cache(maxsize=256)
def _translate_batch_cached(
    texts: tuple[str, ...], api_key: str, model: str
) -> tuple[tuple[str, str, str], ...]:
    payload = {
        "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "contents": [{"parts": [{"text": json.dumps({"queries": texts}, ensure_ascii=False)}]}],
        "generationConfig": {
            "temperature": 0.35,
            "responseMimeType": "application/json",
            "responseSchema": {
                "type": "OBJECT",
                "properties": {
                    "translations": {
                        "type": "ARRAY",
                        "items": {
                            "type": "ARRAY",
                            "items": {"type": "STRING"},
                            "minItems": 3,
                            "maxItems": 3,
                        },
                    }
                },
                "required": ["translations"],
            },
        },
    }
    try:
        response = httpx.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
            headers={"x-goog-api-key": api_key}, json=payload, timeout=30.0,
        )
        response.raise_for_status()
        content = response.json()["candidates"][0]["content"]["parts"][0]["text"]
        translations = json.loads(content)["translations"]
    except (httpx.HTTPError, KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Gemini translation request failed: {exc}") from exc

    if not isinstance(translations, list) or len(translations) != len(texts):
        raise RuntimeError("Gemini must return one translation set for each input query")

    cleaned_batches: list[tuple[str, str, str]] = []
    for options in translations:
        if not isinstance(options, list):
            raise RuntimeError("Gemini returned an invalid translation set")
        cleaned = tuple(" ".join(str(item).split()) for item in options)
        if len(cleaned) != 3 or any(not item for item in cleaned):
            raise RuntimeError("Gemini must return exactly three non-empty translations per query")
        cleaned_batches.append((cleaned[0], cleaned[1], cleaned[2]))
    return tuple(cleaned_batches)
