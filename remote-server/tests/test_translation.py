import json
import unittest
from unittest.mock import Mock, patch

from app.services.translation import TranslationNotConfigured, TranslationService, _translate_batch_cached


class TranslationServiceTest(unittest.TestCase):
    def test_translates_all_queries_in_one_cached_gemini_request(self):
        _translate_batch_cached.cache_clear()
        self.addCleanup(_translate_batch_cached.cache_clear)
        response = Mock()
        response.json.return_value = {"candidates": [{"content": {"parts": [{"text": json.dumps({"translations": [["a red car", "a car drives", "red vehicle on a street"], ["a person", "a person runs", "runner in a park"]]})}]}}]}
        with patch("app.services.translation.httpx.post", return_value=response) as post:
            service = TranslationService(api_key="test-key", model="gemini-test")
            expected = [["a red car", "a car drives", "red vehicle on a street"], ["a person", "a person runs", "runner in a park"]]
            self.assertEqual(service.translate(["  xe   hơi đỏ  ", "người chạy"]), expected)
            self.assertEqual(service.translate(["xe hơi đỏ", "người chạy"]), expected)
        post.assert_called_once()
        self.assertEqual(post.call_args.kwargs["headers"], {"x-goog-api-key": "test-key"})
        request_text = post.call_args.kwargs["json"]["contents"][0]["parts"][0]["text"]
        self.assertEqual(json.loads(request_text), {"queries": ["xe hơi đỏ", "người chạy"]})

    def test_requires_gemini_key(self):
        with self.assertRaises(TranslationNotConfigured):
            TranslationService(api_key="").translate(["xin chào"])

    def test_rejects_empty_text(self):
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            TranslationService(api_key="test-key").translate(["  "])


if __name__ == "__main__":
    unittest.main()
