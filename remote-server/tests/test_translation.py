import unittest
from unittest.mock import patch

from app.services.translation import TranslationService, _translate_one_cached


class TranslationServiceTest(unittest.TestCase):
    def test_translates_and_caches_normalized_text(self):
        service = TranslationService()
        _translate_one_cached.cache_clear()
        self.addCleanup(_translate_one_cached.cache_clear)
        with patch('deep_translator.GoogleTranslator') as translator:
            translator.return_value.translate.return_value = 'hello'
            self.assertEqual(service.translate(["  xin   chào  "]), ["hello"])
            self.assertEqual(service.translate(["xin chào"]), ["hello"])
            translator.return_value.translate.assert_called_once_with('xin chào')

    def test_rejects_empty_text(self):
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            TranslationService().translate(["  "])


if __name__ == "__main__":
    unittest.main()
