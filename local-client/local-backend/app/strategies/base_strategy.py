from abc import ABC, abstractmethod
import asyncio

# Hard limits applied to every strategy — cannot be overridden per-strategy.
FETCH_CAP = 1000          # Max records pulled from the data provider per query
EXECUTION_TIMEOUT_SEC = 2.0  # Max seconds allowed for fusion_and_temporal()


class BaseStrategy(ABC):
    """
    Base class for all retrieval strategies.

    To create a new strategy:
      1. Create a new .py file inside strategies/ (e.g. yourname_idea_v1.py)
      2. Subclass BaseStrategy
      3. Fill in name, description, author, and version
      4. Implement fusion_and_temporal()
      5. Restart the backend — it will appear in the strategy dropdown automatically

    Guardrails (applied automatically by search()):
      - Fetch cap: at most FETCH_CAP records are pulled from the data provider
      - Execution timeout: fusion_and_temporal() is cancelled after EXECUTION_TIMEOUT_SEC seconds
        Note: the timeout cancels the HTTP response; a background thread may finish later.

    This file is kept identical to remote-server/app/strategies/base_strategy.py
    so strategies can be copied to the server without modification.
    """

    # --- Required metadata — subclasses MUST override all four ---
    name: str = ""
    description: str = ""
    author: str = ""
    version: str = "1.0"

    def __init__(self, data_provider):
        self.data_provider = data_provider
        missing = [f for f in ("name", "description", "author") if not getattr(self, f)]
        if missing:
            raise ValueError(
                f"{self.__class__.__name__} must define class attributes: {', '.join(missing)}"
            )

    async def search(self, query_groups: list[dict]) -> list[dict]:
        """Full pipeline: pre-process → fetch → execute (with timeout) → post-filter."""
        processed = self.pre_process(query_groups)
        raw_data = await self.data_provider.get_raw_data(processed, limit=FETCH_CAP)

        try:
            results = await asyncio.wait_for(
                asyncio.to_thread(self.fusion_and_temporal, raw_data, processed),
                timeout=EXECUTION_TIMEOUT_SEC,
            )
        except asyncio.TimeoutError:
            raise TimeoutError(
                f"[{self.name}] fusion_and_temporal() exceeded {EXECUTION_TIMEOUT_SEC}s. "
                "Check for infinite loops or very expensive operations."
            )

        return self.post_filter(results)

    def pre_process(self, query_groups: list[dict]) -> list[dict]:
        """Optional: transform/validate query groups before data is fetched. Default is pass-through."""
        return query_groups

    @abstractmethod
    def fusion_and_temporal(self, raw_data: dict, query_groups: list[dict]) -> list[dict]:
        """
        Core algorithm. Merge multi-modal data and apply temporal logic.
        Runs synchronously in a worker thread (safe to use plain Python, no async).

        Args:
            raw_data: dict with keys:
                "frames"      — list of frame dicts (frame_id, video_id, frame_number, timestamp_ms, image_url)
                "ocr"         — list of OCR dicts   (frame_id, video_id, frame_number, timestamp_ms, ocr_text)
                "transcripts" — list of transcript dicts (video_id, start_time_ms, end_time_ms, text)
                "videos"      — dict keyed by video_id  (fps, duration_ms, title, youtube_id, ...)
            query_groups: list of query group dicts after pre_process()
                Each group has: semantic_query (str), text_query (str), temporal_offset_ms (int)

        Returns:
            list of result dicts, each with:
                video_id        (str)
                frame_id        (str)
                frame_number    (int)
                timestamp_ms    (int)
                confidence      (float, 0.0–1.0)
                frame_image_url (str)
        """
        ...

    def post_filter(self, results: list[dict]) -> list[dict]:
        """Optional: re-rank or remove results after fusion. Default is pass-through."""
        return results
