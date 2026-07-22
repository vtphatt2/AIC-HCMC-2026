import json
import os

import httpx


class QueryParser:
    """Optional Gemini JSON parser used only when a strategy asks for it."""

    def __init__(self, api_key: str | None = None):
        self.api_key = (api_key or os.getenv("GEMINI_API_KEY", "")).strip()
        if not self.api_key:
            raise RuntimeError("GEMINI_API_KEY is not configured")

    async def parse_json(self, *, system_prompt, user_input, response_model):
        model = os.getenv("GEMINI_QUERY_MODEL", "gemini-3.1-flash-lite")
        payload = {
            "systemInstruction": {"parts": [{"text": system_prompt}]},
            "contents": [{"parts": [{"text": user_input}]}],
            "generationConfig": {
                "responseMimeType": "application/json",
                "responseSchema": response_model.model_json_schema(),
            },
        }
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
                headers={"x-goog-api-key": self.api_key},
                json=payload,
            )
        response.raise_for_status()
        text = response.json()["candidates"][0]["content"]["parts"][0]["text"]
        return response_model.model_validate(json.loads(text))
