from __future__ import annotations

import asyncio
import os
from pathlib import Path

from app.strategies.base_strategy import BaseStrategy, FETCH_CAP


MODEL_ID = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
MODEL_FILE = "onnx/model_quint8_avx2.onnx"


class BasicLearnedReranker(BaseStrategy):
    """PECore retrieval followed by a small multilingual CPU text reranker."""

    name = "Basic Learned Reranker (CPU)"
    description = "PECore top-50 + multilingual MiniLM ONNX INT8 over nearby transcript text."
    author = "Team AIC 2026"
    version = "0.1"

    def __init__(self, data_provider):
        super().__init__(data_provider)
        self._reranker: _OnnxReranker | None = None

    async def search(
        self,
        query_groups: list[dict],
        limit: int = 100,
        video_genre: str = "All",
    ) -> list[dict]:
        processed = self.pre_process(query_groups)
        fetch_limit = min(FETCH_CAP, max(50, int(limit)))
        raw_data = await self.data_provider.get_raw_data(
            processed, limit=fetch_limit, video_genre=video_genre
        )
        # ponytail: generous for cold CPU/model load; lower after measuring target hardware.
        timeout = float(os.getenv("LEARNED_RERANKER_TIMEOUT_SEC", "60"))
        return await asyncio.wait_for(
            asyncio.to_thread(self.fusion_and_temporal, raw_data, processed),
            timeout=timeout,
        )

    def pre_process(self, query_groups: list[dict]) -> list[dict]:
        if len(query_groups) != 1:
            raise ValueError("Basic Learned Reranker currently supports one query step.")
        semantic = str(query_groups[0].get("semantic_query", "")).strip()
        if not semantic:
            raise ValueError("Basic Learned Reranker requires a semantic query.")
        return [{**query_groups[0], "semantic_query": semantic}]

    def fusion_and_temporal(self, raw_data: dict, query_groups: list[dict]) -> list[dict]:
        frames = sorted(
            raw_data.get("frames", []),
            key=lambda frame: float(frame.get("score", 0.0)),
            reverse=True,
        )
        if not frames:
            return []

        query = " ".join(
            part
            for part in (
                query_groups[0].get("semantic_query", ""),
                query_groups[0].get("text_query", ""),
            )
            if str(part).strip()
        )
        passages = _candidate_passages(frames, raw_data)
        eligible = [(frame, passages.get(frame["frame_id"], "")) for frame in frames]
        eligible = [(frame, passage) for frame, passage in eligible if passage]

        learned_scores: dict[str, float] = {}
        if eligible:
            if self._reranker is None:
                self._reranker = _OnnxReranker()
            scores = self._reranker.predict(query, [passage for _, passage in eligible])
            learned_scores = {
                frame["frame_id"]: float(score)
                for (frame, _), score in zip(eligible, scores)
            }

        visual_rank = {frame["frame_id"]: rank for rank, frame in enumerate(frames, 1)}
        learned_rank = {
            frame_id: rank
            for rank, (frame_id, _) in enumerate(
                sorted(learned_scores.items(), key=lambda item: item[1], reverse=True), 1
            )
        }
        num_lists = 2 if learned_scores else 1
        videos = raw_data.get("videos", {})
        results = []
        for frame in frames:
            frame_id = frame["frame_id"]
            confidence = _rrf_confidence(visual_rank[frame_id], learned_rank.get(frame_id), num_lists)
            video = videos.get(frame["video_id"], {})
            results.append(
                {
                    "video_id": frame["video_id"],
                    "youtube_id": str(video.get("youtube_id") or ""),
                    "frame_id": frame_id,
                    "frame_number": int(frame.get("frame_number", 0)),
                    "timestamp_ms": int(frame.get("timestamp_ms", 0)),
                    "confidence": round(confidence, 4),
                    "visual_score": round(float(frame.get("score", 0.0)), 4),
                    "reranker_score": learned_scores.get(frame_id),
                    "frame_image_url": str(frame.get("image_url") or ""),
                    "fps": float(video.get("fps", 25.0)),
                }
            )
        results.sort(key=lambda row: row["confidence"], reverse=True)
        return results


# ponytail: text cross-encoder is weaker and domain-mismatched vs. the visual
# encoder (trained on query<->passage text, not scene<->narration), so its RRF
# term is downweighted. Re-measure this on labeled query/frame data before trusting it.
LEARNED_RRF_WEIGHT = float(os.getenv("LEARNED_RERANKER_RRF_WEIGHT", "0.5"))


def _rrf_confidence(
    visual_rank: int, learned_rank: int | None, num_lists: int, k: int = 60
) -> float:
    score = 1.0 / (k + visual_rank)
    max_possible = 1.0 / (k + 1)
    if num_lists > 1:
        max_possible += LEARNED_RRF_WEIGHT / (k + 1)
        if learned_rank is not None:
            score += LEARNED_RRF_WEIGHT / (k + learned_rank)
    return round(min(1.0, score / max_possible), 4)


def _candidate_passages(frames: list[dict], raw_data: dict) -> dict[str, str]:
    ocr_by_frame = {
        row.get("frame_id"): str(row.get("ocr_text", "")).strip()
        for row in raw_data.get("ocr", [])
    }
    transcripts_by_video: dict[str, list[dict]] = {}
    for row in raw_data.get("transcripts", []):
        transcripts_by_video.setdefault(row.get("video_id", ""), []).append(row)

    local_transcripts: dict[str, object | None] = {}
    passages = {}
    for frame in frames:
        frame_id = frame["frame_id"]
        video_id = frame["video_id"]
        timestamp_ms = int(frame.get("timestamp_ms", 0))
        transcript_text = _raw_transcript_window(
            transcripts_by_video.get(video_id, []), timestamp_ms
        )
        if not transcript_text:
            if video_id not in local_transcripts:
                try:
                    from app.services.transcript_index import load_or_build_transcript

                    local_transcripts[video_id] = load_or_build_transcript(video_id)
                except (ImportError, OSError, ValueError):
                    local_transcripts[video_id] = None
            transcript = local_transcripts[video_id]
            if transcript is not None:
                transcript_text = transcript.text_window(timestamp_ms, radius_s=12)

        parts = []
        if ocr_by_frame.get(frame_id):
            parts.append(f"OCR: {ocr_by_frame[frame_id]}")
        if transcript_text:
            parts.append(f"Transcript: {transcript_text}")
        passages[frame_id] = " ".join(parts)
    return passages


def _raw_transcript_window(rows: list[dict], timestamp_ms: int, radius_ms: int = 12_000) -> str:
    return " ".join(
        str(row.get("text", "")).strip()
        for row in rows
        if int(row.get("end_time_ms", 0)) >= timestamp_ms - radius_ms
        and int(row.get("start_time_ms", 0)) <= timestamp_ms + radius_ms
        and str(row.get("text", "")).strip()
    )


class _OnnxReranker:
    def __init__(self):
        import onnxruntime as ort
        from huggingface_hub import snapshot_download
        from tokenizers import Tokenizer

        model_dir = Path(
            os.getenv(
                "LEARNED_RERANKER_DIR",
                Path(__file__).resolve().parents[2] / "models" / "mmarco-mMiniLMv2-L12-H384-v1",
            )
        )
        model_dir = Path(
            snapshot_download(
                MODEL_ID,
                local_dir=model_dir,
                allow_patterns=[
                    MODEL_FILE,
                    "config.json",
                    "tokenizer.json",
                    "tokenizer_config.json",
                    "special_tokens_map.json",
                    "sentencepiece.bpe.model",
                ],
            )
        )
        self.tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        self.tokenizer.enable_truncation(max_length=256)
        self.tokenizer.enable_padding(pad_id=1, pad_token="<pad>")
        self.session = ort.InferenceSession(
            str(model_dir / MODEL_FILE), providers=["CPUExecutionProvider"]
        )
        self.input_names = {item.name for item in self.session.get_inputs()}

    def predict(self, query: str, passages: list[str], batch_size: int = 8) -> list[float]:
        scores = []
        for start in range(0, len(passages), batch_size):
            batch = passages[start : start + batch_size]
            encoded = self.tokenizer.encode_batch([(query, passage) for passage in batch])
            inputs = {
                "input_ids": _as_int64([item.ids for item in encoded]),
                "attention_mask": _as_int64([item.attention_mask for item in encoded]),
            }
            inputs = {key: value for key, value in inputs.items() if key in self.input_names}
            logits = self.session.run(None, inputs)[0]
            scores.extend(float(value) for value in logits.reshape(-1))
        return scores


def _as_int64(rows: list[list[int]]):
    import numpy as np

    return np.asarray(rows, dtype="int64")
