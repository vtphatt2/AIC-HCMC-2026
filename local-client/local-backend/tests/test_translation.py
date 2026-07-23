import unittest
from types import SimpleNamespace

from app.services.translation import TranslationService


class _Tokenizer:
    def encode(self, text, **kwargs):
        return ["▁xin", "</s>"]

    def decode(self, tokens):
        return " hello "


class _Translator:

    def __init__(self):
        self.calls = 0

    def translate_batch(self, source_tokens, **kwargs):
        self.calls += 1
        return [SimpleNamespace(hypotheses=[["▁hello", "</s>"]])]


class TranslationServiceTest(unittest.TestCase):
    def test_translates_locally_and_caches_normalized_text(self):
        service = TranslationService()
        service._source_tokenizer = _Tokenizer()
        service._target_tokenizer = _Tokenizer()
        service._translator = _Translator()

        self.assertEqual(service.translate(["  xin   chào  "]), ["hello"])
        self.assertEqual(service.translate(["xin chào"]), ["hello"])
        self.assertEqual(service._translator.calls, 1)

    def test_rejects_empty_text(self):
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            TranslationService().translate(["  "])


if __name__ == "__main__":
    unittest.main()
