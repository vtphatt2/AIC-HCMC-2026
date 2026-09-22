# Transcript collection and cleaning

The maintained implementation is now under `preprocess/transcript_cleaner/`.
The previous notebook and standalone scripts are historical references; new runs
should use `python -m preprocess`.

## Current flow

```text
media-info directory/ZIP
  → filter metadata to the current lot
  → YouTube captions (prefer vi, then en)
  → atomic raw JSONL {start_time_ms, end_time_ms, text}
  → one resumable Gemini request per video
  → contract validation
  → atomic clean JSONL (only text may change)
```

Metadata ZIPs are read directly with `zipfile`; they are not extracted. Collection
uses bounded thread concurrency, request pacing, retry/backoff, per-video failures,
and atomic output. Cleaning preserves row count, order, timestamp, field set, and
all non-text metadata. Checkpoints allow a restarted run to avoid repeated Gemini
requests.

When passed to `python -m preprocess run --transcript-metadata ...`, this whole
branch runs concurrently with ZIP video decoding/TransNet/PE-Core. Collection and
cleaning are themselves connected by a filesystem-backed producer/consumer queue:
Gemini may clean the first atomically published JSONL while captions for later
videos are still downloading. Only the same video's `collect → clean` dependency
remains ordered.

See [`preprocess/README.md`](../preprocess/README.md#thu-thập-và-làm-sạch-transcript)
for commands and output locations.

## Historical dataset notes

- YouTube captions were unavailable for 60/835 videos.
- Whisper recovered 32 videos; 28 speechless L24 videos were intentionally left
  without transcripts to avoid hallucinated ASR.
- Existing Whisper/Kaggle scripts remain under `scripts/` for that explicit fallback;
  they are not silently invoked by the caption collector.
