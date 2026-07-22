import unittest
from unittest.mock import AsyncMock, Mock, patch

from pydantic import BaseModel

from app.services.query_parser import QueryParser


class Plan(BaseModel):
    visual: str


class QueryParserTests(unittest.IsolatedAsyncioTestCase):
    async def test_validates_gemini_json_against_strategy_model(self):
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "candidates": [{"content": {"parts": [{"text": '{"visual":"red bus"}'}]}}]
        }
        client = AsyncMock()
        client.__aenter__.return_value.post.return_value = response

        with patch("app.services.query_parser.httpx.AsyncClient", return_value=client):
            plan = await QueryParser("test-key").parse_json(
                system_prompt="Return a visual query",
                user_input="find a bus",
                response_model=Plan,
            )

        self.assertEqual(plan.visual, "red bus")
        payload = client.__aenter__.return_value.post.await_args.kwargs["json"]
        self.assertEqual(payload["generationConfig"]["responseSchema"]["required"], ["visual"])


if __name__ == "__main__":
    unittest.main()
