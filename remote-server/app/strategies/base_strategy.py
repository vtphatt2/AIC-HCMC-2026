from abc import ABC, abstractmethod
import asyncio
import logging
import time

logger = logging.getLogger(__name__)

# Hard limits applied to every strategy — cannot be overridden per-strategy.
FETCH_CAP = 1000          # Max records pulled from DB per query
MULTI_STEP_FETCH_MIN = 300
EXECUTION_TIMEOUT_SEC = 2.0  # Max seconds allowed for fusion_and_temporal()


class BaseStrategy(ABC):
    """
    Base class for all retrieval strategies.

    To create a new strategy:
      1. Subclass BaseStrategy in a new .py file inside strategies/
      2. Set name, description, author, and version class attributes
      3. Implement fusion_and_temporal()
      4. Optionally override pre_process() and post_filter()

    The engine auto-discovers this file and exposes it in the strategy dropdown.

    Guardrails (applied automatically by search()):
      - Fetch cap: at most FETCH_CAP records are pulled from the data provider
      - Execution timeout: fusion_and_temporal() is cancelled after EXECUTION_TIMEOUT_SEC seconds
        Note: the timeout cancels the HTTP response, but background threads may continue briefly.
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

    async def search(self, query_groups: list[dict], limit: int = 100, video_genre: str = "All") -> list[dict]:
        """Full pipeline: pre-process → fetch → execute (with timeout) → post-filter."""
        processed = self.pre_process(query_groups)
        fetch_limit = min(max(int(limit), 1), FETCH_CAP)
        if len(processed) > 1:
            fetch_limit = max(fetch_limit, MULTI_STEP_FETCH_MIN)
        raw_data = await self.data_provider.get_raw_data(
            processed, limit=fetch_limit, video_genre=video_genre
        )

        try:
            timer_start = time.monotonic()
            results = await asyncio.wait_for(
                asyncio.to_thread(self.fusion_and_temporal, raw_data, processed),
                timeout=EXECUTION_TIMEOUT_SEC,
            )
            logger.info(
                "[TIMER] fusion %.3f ms strategy=%s results=%s",
                (time.monotonic() - timer_start) * 1000,
                self.__class__.__name__,
                len(results),
            )
        except asyncio.TimeoutError:
            logger.info(
                "[TIMER] fusion %.3f ms strategy=%s timeout=true",
                (time.monotonic() - timer_start) * 1000,
                self.__class__.__name__,
            )
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
                "frames"      — list of frame records (frame_id, video_id, frame_number, timestamp_ms, image_url)
                "ocr"         — list of OCR records   (frame_id, video_id, frame_number, timestamp_ms, ocr_text)
                "transcripts" — list of transcript intervals (video_id, start_time_ms, end_time_ms, text)
                "videos"      — dict keyed by video_id (fps, duration_ms, title, ...)
            query_groups: the processed query groups from pre_process()

        Returns:
            list of result dicts, each containing:
                video_id        (str)
                youtube_id      (str, YouTube video ID for playback)
                frame_id        (str)
                frame_number    (int)
                timestamp_ms    (int)
                confidence      (float, 0.0–1.0)
                frame_image_url (str)
                fps             (float)
        """
        ...

    def post_filter(self, results: list[dict]) -> list[dict]:
        """Optional: re-rank or remove results after fusion. Default is pass-through."""
        return results
