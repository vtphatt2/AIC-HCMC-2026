from abc import ABC, abstractmethod
import asyncio
import logging
import time

from ._similarity_filter import filter_similar_results, result_frame_ids

logger = logging.getLogger(__name__)

FETCH_CAP = 1000
EXECUTION_TIMEOUT_SEC = 30.0
# When duplicate filtering is on, fetch this many times top_k on the first
# page so filtering usually clears top_k in one round instead of needing a
# second retrieve+frame_embeddings+filter round trip. Measured against a real
# Milvus Lite collection: larger factors cost more than expected (payload
# size/filter cost scale with candidate count, ef isn't the only driver), so
# keep this modest rather than assuming HNSW search is "free" up to ef.
OVERSAMPLE_FACTOR = 1.5


class SearchContext:
    """Request-scoped API exposed to strategies."""

    def __init__(
        self,
        query_groups: list[dict],
        top_k: int,
        video_genre: str,
        data_provider,
        parser=None,
        vector_search_algorithm: str | None = None,
        options: dict | None = None,
        config_id: str = "default",
        config_revision: int = 0,
        duplicate_threshold: float | None = None,
    ):
        self.query_groups = query_groups
        self.top_k = top_k
        self.video_genre = video_genre
        self._data_provider = data_provider
        self._parser = parser
        self._vector_search_algorithm = vector_search_algorithm
        self.options = dict(options or {})
        self.config_id = config_id
        self.config_revision = config_revision
        self.duplicate_threshold = duplicate_threshold
        self._retrieval_round = 0
        self._retrieval_cache = {}
        self._embedding_cache = {}
        self._embedding_requested = set()

    def option(self, key: str, default=None):
        return self.options.get(key, default)

    async def retrieve(self, channel: str, query: str, *, top_k: int | None = None):
        requested = self.top_k if top_k is None else min(max(int(top_k), 1), FETCH_CAP)
        kwargs = {
            "top_k": requested,
            "video_genre": self.video_genre,
        }
        if self._vector_search_algorithm:
            kwargs["vector_search_algorithm"] = self._vector_search_algorithm
        key = (channel, query, requested)
        state = self._retrieval_cache.setdefault(key, {"hits": [], "exhausted": False})
        while len(state.setdefault("pages", [])) <= self._retrieval_round and not state["exhausted"]:
            remaining = FETCH_CAP - len(state["hits"])
            if remaining <= 0 or channel not in {"raw.semantic", "subtitled.semantic"} and state["hits"]:
                state["exhausted"] = True
                break
            page_size = min(requested, remaining)
            if self.duplicate_threshold is not None and not state["pages"]:
                page_size = min(int(requested * OVERSAMPLE_FACTOR), remaining)
            excluded = [str(hit["frame_id"]) for hit in state["hits"] if hit.get("frame_id")]
            if excluded:
                kwargs["exclude_frame_ids"] = excluded
            t_page = time.monotonic()
            page = await self._data_provider.retrieve(channel, query, **{**kwargs, "top_k": page_size})
            logger.info(
                "[TIMER] retrieve_page %.3f ms channel=%s round=%s page_size=%s excluded=%s hits=%s",
                (time.monotonic() - t_page) * 1000, channel, self._retrieval_round, page_size,
                len(excluded), len(page),
            )
            # Vector search may piggyback each hit's embedding (private "_vector"
            # key) so the duplicate filter doesn't need a separate
            # frame_embeddings round trip later — capture it here, before the
            # hit goes anywhere else.
            for hit in page:
                vector = hit.pop("_vector", None)
                frame_id = hit.get("frame_id")
                if vector is not None and frame_id:
                    frame_id = str(frame_id)
                    self._embedding_cache[frame_id] = vector
                    self._embedding_requested.add(frame_id)
            seen = {hit.get("frame_id") or hit.get("item_id") or hit.get("chunk_id") for hit in state["hits"]}
            unique = [
                hit for hit in page
                if (hit.get("frame_id") or hit.get("item_id") or hit.get("chunk_id")) not in seen
            ]
            state["hits"].extend(unique)
            state["pages"].append(len(unique))
            state["exhausted"] = len(page) < page_size or not unique or len(state["hits"]) >= FETCH_CAP
        return list(state["hits"])

    def request_more_results(self) -> bool:
        if not any(not state["exhausted"] for state in self._retrieval_cache.values()):
            return False
        self._retrieval_round += 1
        return True

    async def frame_embeddings(self, frame_ids: list[str]):
        missing = [frame_id for frame_id in frame_ids if frame_id not in self._embedding_requested]
        if missing:
            self._embedding_requested.update(missing)
            self._embedding_cache.update(await self._data_provider.frame_embeddings(missing))
        return {
            frame_id: self._embedding_cache[frame_id]
            for frame_id in frame_ids
            if frame_id in self._embedding_cache
        }

    async def parse_json(self, *, system_prompt: str, user_input: str, response_model):
        if self._parser is None:
            raise RuntimeError("Query parser is not configured")
        return await self._parser.parse_json(
            system_prompt=system_prompt,
            user_input=user_input,
            response_model=response_model,
        )

    async def keyframes(
        self,
        video_id: str,
        start_ms: int,
        end_ms: int,
        *,
        limit: int = 20,
    ):
        return await self._data_provider.keyframes(
            video_id,
            start_ms,
            end_ms,
            limit=limit,
        )

    def results(self, hits: list[dict]) -> list[dict]:
        return self._data_provider.results(hits)


class BaseStrategy(ABC):
    name = ""
    description = ""
    author = ""
    version = "2.0"
    config_schema = {}

    def __init__(self, data_provider, parser=None):
        self.data_provider = data_provider
        self.parser = parser
        missing = [field for field in ("name", "description", "author") if not getattr(self, field)]
        if missing:
            raise ValueError(
                f"{self.__class__.__name__} must define class attributes: {', '.join(missing)}"
            )

    async def search(
        self,
        query_groups: list[dict],
        limit: int = 100,
        video_genre: str = "All",
        vector_search_algorithm: str | None = None,
        options: dict | None = None,
        config_id: str = "default",
        config_revision: int = 0,
        duplicate_threshold: float | None = None,
    ) -> list[dict]:
        top_k = min(max(int(limit), 1), FETCH_CAP)
        context = SearchContext(
            query_groups=query_groups,
            top_k=top_k,
            video_genre=video_genre,
            data_provider=self.data_provider,
            parser=self.parser,
            vector_search_algorithm=vector_search_algorithm,
            options=options,
            config_id=config_id,
            config_revision=config_revision,
            duplicate_threshold=duplicate_threshold,
        )
        try:
            results = await asyncio.wait_for(
                self._run_with_filter(context), timeout=EXECUTION_TIMEOUT_SEC
            )
        except asyncio.TimeoutError as exc:
            raise TimeoutError(
                f"[{self.name}] run() exceeded {EXECUTION_TIMEOUT_SEC}s"
            ) from exc
        if not isinstance(results, list):
            raise TypeError(f"[{self.name}] run() must return a list")
        return results[:top_k]

    async def _run_with_filter(self, context: SearchContext) -> list[dict]:
        while True:
            t_run = time.monotonic()
            results = await self.run(context)
            logger.info(
                "[TIMER] strategy_run %.3f ms strategy=%s round=%s results=%s",
                (time.monotonic() - t_run) * 1000, self.name, context._retrieval_round, len(results)
                if isinstance(results, list) else -1,
            )
            if not isinstance(results, list):
                return results
            if context.duplicate_threshold is None:
                return results
            frame_ids = list(dict.fromkeys(
                frame_id for result in results for frame_id in result_frame_ids(result)
            ))
            t_embed = time.monotonic()
            embeddings = await context.frame_embeddings(frame_ids)
            t_filter = time.monotonic()
            filtered = filter_similar_results(
                results,
                embeddings,
                threshold=context.duplicate_threshold,
                event_weights=context.option("event_weights", None),
            )
            logger.info(
                "[TIMER] dedup_round %.3f ms round=%s frame_embeddings_ms=%.3f filter_ms=%.3f "
                "candidates=%s kept=%s",
                (time.monotonic() - t_run) * 1000, context._retrieval_round,
                (t_filter - t_embed) * 1000, (time.monotonic() - t_filter) * 1000,
                len(results), len(filtered),
            )
            if len(filtered) >= context.top_k or not context.request_more_results():
                return filtered

    @abstractmethod
    async def run(self, context: SearchContext) -> list[dict]:
        """Execute this strategy using only the request-scoped context API."""
        ...
