"""Scoped float-vector decoding backport for the pinned PyMilvus 2.3.7."""
from contextvars import ContextVar

import numpy as np
import pymilvus
from pymilvus.client import entity_helper


_bulk_decode = ContextVar("bulk_frame_vector_decode", default=False)
_original_extract = getattr(entity_helper, "extract_row_data_from_fields_data", None)


def _extract_frame_vectors(fields_data, index, dynamic_output_fields=None):
    # Leave every other SDK operation and unsupported response shape alone.
    if (
        not _bulk_decode.get()
        or index < 0
        or len(fields_data) != 2
        or {field.field_name for field in fields_data} != {"frame_id", "vector"}
    ):
        return _original_extract(fields_data, index, dynamic_output_fields)
    vector = next(field for field in fields_data if field.field_name == "vector")
    dim = vector.vectors.dim
    if (
        vector.type != pymilvus.DataType.FLOAT_VECTOR
        or dim <= 0
        or len(vector.vectors.float_vector.data) < (index + 1) * dim
    ):
        return _original_extract(fields_data, index, dynamic_output_fields)
    scalars = [field for field in fields_data if field.field_name != "vector"]
    row = _original_extract(scalars, index, dynamic_output_fields)
    # Same values AND numpy.float32 element types as the SDK's scalar loop.
    # PyMilvus 2.4.15 uses this bulk conversion in entity_helper.py.
    values = np.asarray(
        vector.vectors.float_vector.data[index * dim:(index + 1) * dim],
        dtype=np.float32,
    )
    row["vector"] = values if _bulk_decode.get() == "array" else list(values)
    return row


if pymilvus.__version__ == "2.3.7" and _original_extract is not None:
    entity_helper.extract_row_data_from_fields_data = _extract_frame_vectors


def query_frame_vector_rows(collection, *, as_array=False, **kwargs):
    token = _bulk_decode.set("array" if as_array else True)
    try:
        return collection.query(**kwargs)
    finally:
        _bulk_decode.reset(token)
