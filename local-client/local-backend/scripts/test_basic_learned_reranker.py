from app.strategies.basic_learned_reranker import _rrf_confidence, _raw_transcript_window


def main() -> None:
    assert _rrf_confidence(1, 1, num_lists=2) == 1.0
    assert _rrf_confidence(1, None, num_lists=1) == 1.0
    assert _rrf_confidence(10, 1, num_lists=2) > _rrf_confidence(10, 10, num_lists=2)
    # a frame with no text evidence must be normalized against the same ceiling
    # as frames that do have a learned score, not judged on an easier curve
    assert _rrf_confidence(5, None, num_lists=2) < _rrf_confidence(5, 1, num_lists=2)

    rows = [
        {"start_time_ms": 0, "end_time_ms": 5_000, "text": "đầu"},
        {"start_time_ms": 20_000, "end_time_ms": 25_000, "text": "giữa"},
        {"start_time_ms": 50_000, "end_time_ms": 55_000, "text": "cuối"},
    ]
    assert _raw_transcript_window(rows, 20_000) == "giữa"
    print("basic learned reranker checks passed")


if __name__ == "__main__":
    main()
