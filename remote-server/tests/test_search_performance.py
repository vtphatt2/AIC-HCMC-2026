from concurrent.futures import ThreadPoolExecutor
import threading
import unittest
from unittest.mock import AsyncMock, Mock

import numpy as np
from pymilvus import DataType
from pymilvus.client import entity_helper
from pymilvus.grpc_gen import schema_pb2

from app.db import milvus_compat as compat
from app.db import milvus_client
from app.data_provider import DataProvider, DEFAULT_VECTOR_SEARCH_ALGORITHM
from app.strategies.duy_multi_detail_search import DuyMultiDetailSearch
from app.strategies.raw_visual import RawVisual
from app.strategies._event_scoring import event_weighted_scores
from app.strategies._similarity_filter import filter_similar_results


class HnswSearchParameterTests(unittest.TestCase):
    def test_deep_unfiltered_search_bypasses_cached_entry_point(self):
        for top_k in [50, 100, 1000, 3000]:
            with self.subTest(top_k=top_k):
                collection = Mock(search=Mock(return_value=[[]]))
                milvus_client.vector_search(collection, [1., 0.], top_k=top_k, algorithm="hnsw")
                kwargs = collection.search.call_args.kwargs
                self.assertEqual(kwargs["param"]["metric_type"], "COSINE")
                self.assertEqual(kwargs["param"]["params"], {
                    "ef": max(milvus_client.DEEP_SEARCH_EF, top_k), "for_tuning": True,
                })
                self.assertEqual(kwargs["limit"], top_k)
                self.assertEqual(kwargs["data"], [[1., 0.]])

    def test_shallow_filtered_and_other_indexes_keep_existing_parameters(self):
        for algorithm, top_k, expr, expected in [
            ("hnsw", 20, None, {"ef": max(milvus_client.SEARCH_PARAMS["ef"], 20)}),
            ("hnsw", 1000, 'video_id == "video"', {"ef": max(milvus_client.DEEP_SEARCH_EF, 1000)}),
            ("flat", 1000, None, {}),
            ("scann", 1000, None, {"nprobe": 32, "reorder_k": 5000}),
        ]:
            with self.subTest(algorithm=algorithm, top_k=top_k, expr=expr):
                collection = Mock(search=Mock(return_value=[[]]))
                milvus_client.vector_search(collection, [1., 0.], top_k=top_k,
                                            algorithm=algorithm, expr=expr)
                kwargs = collection.search.call_args.kwargs
                self.assertEqual(kwargs["param"]["params"], expected)
                self.assertEqual(kwargs["expr"], expr)


class VectorDecodingTests(unittest.TestCase):
    def setUp(self):
        ids = schema_pb2.FieldData(type=DataType.VARCHAR, field_name="frame_id")
        ids.scalars.string_data.data.extend(["a", "b"])
        vector = schema_pb2.FieldData(type=DataType.FLOAT_VECTOR, field_name="vector")
        vector.vectors.dim = 4
        vector.vectors.float_vector.data.extend([0., -0., 1e-38, -.999999, .25, -1., 1e30, 1.])
        self.fields = [ids, vector]

    def test_values_types_order_and_signed_zero_match_sdk(self):
        expected = [compat._original_extract(self.fields, i) for i in range(2)]
        collection = Mock()
        collection.query.side_effect = lambda **kw: [
            entity_helper.extract_row_data_from_fields_data(self.fields, i) for i in range(2)
        ]
        actual = compat.query_frame_vector_rows(collection, expr='frame_id in ["a", "b"]')
        self.assertEqual([r["frame_id"] for r in actual], ["a", "b"])
        for before, after in zip(expected, actual):
            self.assertTrue(all(type(v) is np.float32 for v in after["vector"]))
            self.assertEqual(np.asarray(before["vector"]).tobytes(), np.asarray(after["vector"]).tobytes())
        self.assertFalse(compat._bulk_decode.get())

    def test_exception_resets_scope(self):
        collection = Mock()
        collection.query.side_effect = RuntimeError("query failed")
        with self.assertRaisesRegex(RuntimeError, "query failed"):
            compat.query_frame_vector_rows(collection)
        self.assertFalse(compat._bulk_decode.get())

    def test_array_query_keeps_exact_float32_values_and_sdk_default(self):
        collection = Mock()
        collection.query.side_effect = lambda **kw: [
            entity_helper.extract_row_data_from_fields_data(self.fields, i) for i in range(2)
        ]
        actual = milvus_client.query_frame_vectors(collection, ["a", "b", "a", ""])
        self.assertEqual(list(actual), ["a", "b"])
        self.assertEqual(collection.query.call_args.kwargs["limit"], 2)
        for i, key in enumerate(actual):
            self.assertIsInstance(actual[key], np.ndarray)
            self.assertEqual(actual[key].dtype, np.float32)
            original = compat._original_extract(self.fields, i)["vector"]
            self.assertEqual(actual[key].tobytes(), np.asarray(original).tobytes())
        self.assertFalse(compat._bulk_decode.get())
        self.assertIsInstance(entity_helper.extract_row_data_from_fields_data(self.fields, 0)["vector"], list)

    def test_array_scope_restored_after_nested_query_and_failure(self):
        token = compat._bulk_decode.set(True)
        try:
            with self.assertRaisesRegex(RuntimeError, "query failed"):
                compat.query_frame_vector_rows(Mock(query=Mock(side_effect=RuntimeError("query failed"))), as_array=True)
            self.assertIs(compat._bulk_decode.get(), True)
        finally:
            compat._bulk_decode.reset(token)

    def test_array_query_accepts_sdk_fallback_lists_and_empty_input(self):
        collection = Mock(query=Mock(return_value=[{"frame_id": "a", "vector": [0., -0., .25]}]))
        self.assertEqual(milvus_client.query_frame_vectors(collection, []), {})
        collection.query.assert_not_called()
        actual = milvus_client.query_frame_vectors(collection, ["a", "missing"])
        self.assertEqual(list(actual), ["a"])
        self.assertEqual(actual["a"].tobytes(), np.array([0., -0., .25], dtype=np.float32).tobytes())

    def test_arrays_preserve_weighted_scores_and_duplicate_decisions_without_mutation(self):
        rng = np.random.default_rng(2026)
        arrays = {str(i): rng.standard_normal(1280).astype(np.float32) for i in range(30)}
        arrays["30"] = arrays["0"].copy()
        arrays["31"] = np.zeros(1280, dtype=np.float32)
        arrays["32"] = arrays["0"] + np.float32(1e-6)
        original = {key: value.tobytes() for key, value in arrays.items()}
        lists = {key: list(value) for key, value in arrays.items()}
        queries = [list(arrays[str(i)]) for i in range(3)]
        for weights in [None, [.2, .3, .5]]:
            self.assertEqual(event_weighted_scores(queries, arrays, weights=weights),
                             event_weighted_scores(queries, lists, weights=weights))
        singles = [{"frame_id": key} for key in [*arrays, "missing"]]
        temporal = [{"steps": [singles[i], singles[i+1]]} for i in range(len(singles)-1)]
        for results in [singles, temporal]:
            for threshold in [0., .7, .98, 1.]:
                self.assertEqual(filter_similar_results(results, arrays, threshold=threshold, event_weights=[.3,.7]),
                                 filter_similar_results(results, lists, threshold=threshold, event_weights=[.3,.7]))
        self.assertEqual(original, {key: value.tobytes() for key, value in arrays.items()})

    def test_scope_does_not_affect_another_thread(self):
        entered, release = threading.Event(), threading.Event()
        def query(**kw):
            entered.set()
            release.wait(2)
            return compat._bulk_decode.get()
        with ThreadPoolExecutor(max_workers=1) as pool:
            task = pool.submit(compat.query_frame_vector_rows, Mock(query=query))
            try:
                self.assertTrue(entered.wait(2))
                self.assertFalse(compat._bulk_decode.get())
            finally:
                release.set()
            self.assertTrue(task.result())

    def test_other_response_shapes_use_original_decoder(self):
        token = compat._bulk_decode.set(True)
        try:
            self.assertEqual(compat._extract_frame_vectors(self.fields[:1], 0), {"frame_id": "a"})
            self.assertEqual(compat._extract_frame_vectors([], 0), {})
        finally:
            compat._bulk_decode.reset(token)


class DeferredVectorTests(unittest.IsolatedAsyncioTestCase):
    async def test_alternate_collection_keeps_its_own_vectors(self):
        p = DataProvider.__new__(DataProvider)
        p._encode_text = AsyncMock(return_value=np.array([1., 0.]))
        p._genre_expr = AsyncMock(return_value=None)
        p._get_collection = Mock()
        p._run_db = AsyncMock(return_value=[{"frame_id": "a"}])
        p._hydrate_frames = AsyncMock()
        alternate = "flat" if DEFAULT_VECTOR_SEARCH_ALGORITHM != "flat" else "hnsw"
        await p.retrieve("raw.semantic", "query", vector_search_algorithm=alternate,
                         include_vector=False)
        self.assertTrue(p._run_db.await_args.kwargs["include_vector"])

    def provider(self):
        p = Mock()
        self.calls = []
        async def retrieve(channel, query, **kwargs):
            self.calls.append((channel, query, kwargs))
            hits = [{"frame_id": "a", "score": .9}, {"frame_id": "b", "score": .8}]
            if kwargs.get("include_vector", True):
                for hit in hits:
                    hit["_vector"] = [1., 0.]
            return hits
        p.retrieve = retrieve
        p.results = lambda hits: hits
        p.frame_embeddings = AsyncMock(return_value={"a": [1., 0.], "b": [1., 0.]})
        return p

    async def test_small_multi_detail_preserves_ranking_weights_and_dedup(self):
        for threshold in [.9, .98, 1.]:
            for weights in [[1., 1.], [0., 2.], [0., 0.]]:
                p = self.provider()
                output = await DuyMultiDetailSearch(p).search(
                    [{"query": "one"}, {"query": "two"}], limit=100,
                    duplicate_threshold=threshold, options={"event_weights": weights},
                )
                self.assertTrue(all(call[2]["include_vector"] is False for call in self.calls))
                self.assertTrue(all(call[2]["top_k"] == 1000 for call in self.calls))
                expected = [] if not any(weights) else (["a", "b"] if threshold == 1 else ["a"])
                self.assertEqual([row["frame_id"] for row in output], expected)

    async def test_large_filtered_and_other_strategies_keep_piggyback_vectors(self):
        for cls, limit, genre in [(DuyMultiDetailSearch, 101, "All"),
                                  (DuyMultiDetailSearch, 100, "Ẩm thực"),
                                  (RawVisual, 100, "All")]:
            p = self.provider()
            await cls(p).search([{"query": "one"}], limit=limit, video_genre=genre,
                                duplicate_threshold=.98)
            self.assertTrue(all("include_vector" not in call[2] for call in self.calls))
            p.frame_embeddings.assert_not_awaited()

    async def test_failure_is_not_returned_as_empty_success(self):
        p = self.provider()
        p.frame_embeddings.side_effect = RuntimeError("query failed")
        with self.assertRaisesRegex(RuntimeError, "query failed"):
            await DuyMultiDetailSearch(p).search([{"query": "one"}], duplicate_threshold=.98)
