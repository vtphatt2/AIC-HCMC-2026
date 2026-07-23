from __future__ import annotations

import os
import threading
from functools import lru_cache
from pathlib import Path

MAX_TEXTS = 10
MAX_TEXT_LENGTH = 2000
DEFAULT_MODEL_ID = "dekthedev/opus-mt-vi-en-ct2-int8"
DEFAULT_MODEL_REVISION = "14a921f3c4b7238b2b49d247e53810f0f7c78236"


class TranslationService:
    def __init__(self) -> None:
        self._source_tokenizer = None
        self._target_tokenizer = None
        self._translator = None
        self._lock = threading.Lock()

    def translate(self, texts: list[str]) -> list[str]:
        cleaned = [" ".join(text.split()) for text in texts]
        if not cleaned or any(not text for text in cleaned):
            raise ValueError("Translation texts must not be empty")
        if len(cleaned) > MAX_TEXTS:
            raise ValueError(f"At most {MAX_TEXTS} texts can be translated at once")
        if any(len(text) > MAX_TEXT_LENGTH for text in cleaned):
            raise ValueError(f"Each translation text must be at most {MAX_TEXT_LENGTH} characters")

        return list(self._translate_cached(tuple(cleaned)))

    @lru_cache(maxsize=256)
    def _translate_cached(self, texts: tuple[str, ...]) -> tuple[str, ...]:
        # ponytail: one translation at a time; remove the lock only if concurrent UI traffic matters.
        with self._lock:
            self._load_model()
            source_tokens = [
                self._source_tokenizer.encode(text, out_type=str) + ["</s>"]
                for text in texts
            ]
            results = self._translator.translate_batch(
                source_tokens,
                beam_size=1,
                max_decoding_length=512,
            )
            translations = tuple(
                self._target_tokenizer.decode(result.hypotheses[0]).strip()
                for result in results
            )

        if len(translations) != len(texts) or any(not text for text in translations):
            raise RuntimeError("Local translation model returned invalid output")
        return translations

    def _load_model(self) -> None:
        if self._translator is not None:
            return

        import ctranslate2
        import sentencepiece as spm
        from huggingface_hub import snapshot_download
        from huggingface_hub.errors import LocalEntryNotFoundError

        model_path = os.getenv("TRANSLATION_MODEL_PATH", "").strip()
        if not model_path:
            model_id = os.getenv("TRANSLATION_MODEL_ID", DEFAULT_MODEL_ID).strip()
            revision = os.getenv("TRANSLATION_MODEL_REVISION", DEFAULT_MODEL_REVISION).strip()
            try:
                model_path = snapshot_download(
                    repo_id=model_id,
                    revision=revision,
                    local_files_only=True,
                )
            except LocalEntryNotFoundError:
                model_path = snapshot_download(repo_id=model_id, revision=revision)
        threads = max(1, int(os.getenv("TRANSLATION_CPU_THREADS", "4")))
        model_path = Path(model_path)
        self._source_tokenizer = spm.SentencePieceProcessor(
            model_file=str(model_path / "source.spm")
        )
        self._target_tokenizer = spm.SentencePieceProcessor(
            model_file=str(model_path / "target.spm")
        )
        self._translator = ctranslate2.Translator(
            str(model_path),
            device="cpu",
            compute_type="int8",
            inter_threads=1,
            intra_threads=threads,
        )
