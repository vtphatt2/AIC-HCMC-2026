from __future__ import annotations

import json
import os
import threading
from html import unescape
from functools import lru_cache

import httpx

MAX_TEXTS = 10
MAX_TEXT_LENGTH = 2000


class TranslationService:
    def __init__(self) -> None:
        self._nmt_client = None
        self._gemini_client = None
        self._lock = threading.Lock()

    def translate(self, texts: list[str], provider: str) -> list[str]:
        cleaned = [" ".join(text.split()) for text in texts]
        if not cleaned or any(not text for text in cleaned):
            raise ValueError("Translation texts must not be empty")
        if len(cleaned) > MAX_TEXTS:
            raise ValueError(f"At most {MAX_TEXTS} texts can be translated at once")
        if any(len(text) > MAX_TEXT_LENGTH for text in cleaned):
            raise ValueError(f"Each translation text must be at most {MAX_TEXT_LENGTH} characters")

        provider = provider.strip().lower()
        if provider not in {"nmt", "gemini"}:
            raise ValueError("Translation provider must be 'nmt' or 'gemini'")

        return list(self._translate_cached(provider, tuple(cleaned)))

    @lru_cache(maxsize=256)
    def _translate_cached(self, provider: str, texts: tuple[str, ...]) -> tuple[str, ...]:
        if provider == "nmt":
            return self._translate_nmt(texts)
        return self._translate_gemini(texts)

    def _translate_nmt(self, texts: tuple[str, ...]) -> tuple[str, ...]:
        project_id = os.getenv("GOOGLE_CLOUD_PROJECT", "").strip()
        if not project_id:
            raise RuntimeError("GOOGLE_CLOUD_PROJECT is not configured")

        if self._nmt_client is None:
            from google.cloud import translate_v3

            self._nmt_client = translate_v3.TranslationServiceClient()

        response = self._nmt_client.translate_text(
            request={
                "parent": f"projects/{project_id}/locations/global",
                "contents": list(texts),
                "mime_type": "text/plain",
                "target_language_code": "en",
                "model": f"projects/{project_id}/locations/global/models/general/nmt",
            }
        )
        return tuple(unescape(item.translated_text).strip() for item in response.translations)

    def _translate_gemini(self, texts: tuple[str, ...]) -> tuple[str, ...]:
        api_key = os.getenv("GEMINI_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not configured")

        if self._gemini_client is None:
            self._gemini_client = httpx.Client(timeout=30.0)

        model = os.getenv("GEMINI_TRANSLATION_MODEL", "gemini-3.1-flash-lite")
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        payload = {
            "systemInstruction": {
                "parts": [{
                    "text": (
                        "Translate each input from Vietnamese to concise, literal English for "
                        "visual video retrieval. Preserve every person, object, action, color, "
                        "number, name, location, negation, and temporal relationship. Do not "
                        "explain, summarize, add details, or omit details. Return JSON with one "
                        "key named translations containing strings in the same order. If an "
                        "input is already English, return it unchanged."
                    )
                }]
            },
            "contents": [{
                "parts": [{
                    "text": json.dumps({"texts": list(texts)}, ensure_ascii=False)
                }]
            }],
            "generationConfig": {
                "maxOutputTokens": 1024,
                "responseMimeType": "application/json",
                "thinkingConfig": {"thinkingLevel": "minimal"},
            },
        }
        with self._lock:
            response = self._gemini_client.post(
                url,
                headers={"x-goog-api-key": api_key},
                json=payload,
            )
        response.raise_for_status()
        body = response.json()
        output = body["candidates"][0]["content"]["parts"][0]["text"]
        translations = tuple(
            text.strip()
            for text in json.loads(output)["translations"]
        )
        if len(translations) != len(texts) or any(not text for text in translations):
            raise RuntimeError("Gemini returned an invalid number of translations")
        return translations
