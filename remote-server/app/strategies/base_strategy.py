from abc import ABC, abstractmethod
import asyncio


FETCH_CAP = 1000
EXECUTION_TIMEOUT_SEC = 30.0


class SearchContext:
    def __init__(
        self,
        query_groups,
        top_k,
        video_genre,
        data_provider,
        parser=None,
        vector_search_algorithm=None,
        options=None,
        config_id="default",
        config_revision=0,
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

    def option(self, key, default=None):
        return self.options.get(key, default)

    async def retrieve(self, channel, query, *, top_k=None):
        kwargs = {
            "top_k": self.top_k if top_k is None else top_k,
            "video_genre": self.video_genre,
        }
        if self._vector_search_algorithm:
            kwargs["vector_search_algorithm"] = self._vector_search_algorithm
        return await self._data_provider.retrieve(channel, query, **kwargs)

    async def parse_json(self, *, system_prompt, user_input, response_model):
        if self._parser is None:
            raise RuntimeError("Query parser is not configured")
        return await self._parser.parse_json(
            system_prompt=system_prompt,
            user_input=user_input,
            response_model=response_model,
        )

    async def keyframes(self, video_id, start_ms, end_ms, *, limit=20):
        return await self._data_provider.keyframes(video_id, start_ms, end_ms, limit=limit)

    def results(self, hits):
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
        query_groups,
        limit=100,
        video_genre="All",
        vector_search_algorithm=None,
        options=None,
        config_id="default",
        config_revision=0,
    ):
        top_k = min(max(int(limit), 1), FETCH_CAP)
        context = SearchContext(
            query_groups,
            top_k,
            video_genre,
            self.data_provider,
            self.parser,
            vector_search_algorithm,
            options,
            config_id,
            config_revision,
        )
        try:
            results = await asyncio.wait_for(self.run(context), timeout=EXECUTION_TIMEOUT_SEC)
        except asyncio.TimeoutError as exc:
            raise TimeoutError(f"[{self.name}] run() exceeded {EXECUTION_TIMEOUT_SEC}s") from exc
        if not isinstance(results, list):
            raise TypeError(f"[{self.name}] run() must return a list")
        return results[:top_k]

    @abstractmethod
    async def run(self, context):
        ...
