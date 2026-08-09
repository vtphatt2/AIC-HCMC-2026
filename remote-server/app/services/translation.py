from __future__ import annotations

from functools import lru_cache

MAX_TEXTS = 10
MAX_TEXT_LENGTH = 2000


class TranslationService:
    """Vietnamese -> English via Google Translate's free web endpoint
    (deep-translator), not the paid Cloud Translation API — no API key, no
    billing. Traded off deliberately: unofficial/free means it can start
    rate-limiting or breaking without notice, acceptable at the ~1-2 req/s
    this app actually sees. Swap back to the local CTranslate2 model (see
    git history) if that ever becomes a real problem."""

    def translate(self, texts: list[str]) -> list[str]:
        cleaned = [" ".join(text.split()) for text in texts]
        if not cleaned or any(not text for text in cleaned):
            raise ValueError("Translation texts must not be empty")
        if len(cleaned) > MAX_TEXTS:
            raise ValueError(f"At most {MAX_TEXTS} texts can be translated at once")
        if any(len(text) > MAX_TEXT_LENGTH for text in cleaned):
            raise ValueError(f"Each translation text must be at most {MAX_TEXT_LENGTH} characters")

        return [_translate_one_cached(text) for text in cleaned]


@lru_cache(maxsize=256)
def _translate_one_cached(text: str) -> str:
    from deep_translator import GoogleTranslator
    from deep_translator.exceptions import RequestError, TooManyRequests, TranslationNotFound

    try:
        result = GoogleTranslator(source="vi", target="en").translate(text)
    except (RequestError, TooManyRequests, TranslationNotFound) as exc:
        raise RuntimeError(f"Google Translate request failed: {exc}") from exc

    if not result:
        raise RuntimeError("Google Translate returned an empty result")
    return result
