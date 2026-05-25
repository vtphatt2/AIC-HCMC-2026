"""
Nam Defensive Fusion v1 — adaptive multimodal retrieval with duplicate suppression.

Fuses visual (Milvus cosine), OCR, and transcript signals with adaptive weighting.
Defensive against missing fields, empty queries, and dense frame clusters.
"""

from app.strategies.base_strategy import BaseStrategy


class NamDefensiveFusionV1(BaseStrategy):

    name = "Nam Defensive Fusion v1"
    description = (
        "Adaptive 3-modal fusion (visual + OCR + transcript) with duplicate "
        "suppression and defensive fallbacks. Weights adapt to query type."
    )
    author = "Nam"
    version = "1.0"

    DUPLICATE_MIN_GAP_MS = 500
    TEMPORAL_TOLERANCE_MS = 3000

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fusion_and_temporal(self, raw_data: dict, query_groups: list[dict]) -> list[dict]:
        frames = raw_data.get("frames", [])
        ocr = raw_data.get("ocr", [])
        transcripts = raw_data.get("transcripts", [])
        videos = raw_data.get("videos", {})

        if not frames:
            return []

        if len(query_groups) == 1:
            results = self._single_step(frames, videos, query_groups[0], ocr, transcripts)
        else:
            results = self._multi_step(frames, videos, query_groups, ocr, transcripts)

        results.sort(key=lambda x: x["confidence"], reverse=True)
        return self._suppress_duplicates(results)

    # ------------------------------------------------------------------
    # Single-step scoring
    # ------------------------------------------------------------------

    def _single_step(self, frames, videos, query_group, ocr, transcripts):
        ocr_index = self._build_ocr_index(ocr)
        transcript_index = self._build_transcript_index(transcripts)
        results = []
        for frame in frames:
            confidence = self._score_frame(frame, query_group, ocr_index, transcript_index)
            results.append(self._to_result(frame, videos, confidence))
        return results

    # ------------------------------------------------------------------
    # Multi-step temporal
    # ------------------------------------------------------------------

    def _multi_step(self, frames, videos, query_groups, ocr, transcripts):
        ocr_index = self._build_ocr_index(ocr)
        transcript_index = self._build_transcript_index(transcripts)

        # Score every frame against step 1
        scored = []
        for frame in frames:
            conf = self._score_frame(frame, query_groups[0], ocr_index, transcript_index)
            scored.append((frame, conf))
        scored.sort(key=lambda x: x[1], reverse=True)

        results = []
        for step1_frame, step1_conf in scored[:20]:
            results.append(self._to_result(step1_frame, videos, step1_conf))

            for i in range(1, len(query_groups)):
                offset_ms = query_groups[i].get("temporal_offset_ms", 0)
                target_ts = step1_frame["timestamp_ms"] + offset_ms

                for frame in frames:
                    if frame["video_id"] != step1_frame["video_id"]:
                        continue
                    if abs(frame["timestamp_ms"] - target_ts) <= self.TEMPORAL_TOLERANCE_MS:
                        step_conf = self._score_frame(frame, query_groups[i], ocr_index, transcript_index)
                        combined = (step1_conf + step_conf) / 2.0
                        results.append(self._to_result(frame, videos, combined))

        return results

    # ------------------------------------------------------------------
    # Frame scoring
    # ------------------------------------------------------------------

    def _score_frame(self, frame, query_group, ocr_index, transcript_index):
        semantic_q = query_group.get("semantic_query", "").strip().lower()
        text_q = query_group.get("text_query", "").strip().lower()

        has_semantic = len(semantic_q) > 0
        has_text = len(text_q) > 0

        raw_score = float(frame.get("score", 0.5))
        visual_conf = (raw_score + 1.0) / 2.0

        ocr_conf = 0.0
        trans_conf = 0.0

        if has_text:
            ocr_conf = self._ocr_match_score(frame, text_q, ocr_index)
            trans_conf = self._transcript_match_score(frame, text_q, transcript_index)

        if has_semantic and has_text:
            w_visual, w_ocr, w_trans = 0.35, 0.40, 0.25
        elif has_semantic:
            w_visual, w_ocr, w_trans = 1.0, 0.0, 0.0
        elif has_text:
            w_visual, w_ocr, w_trans = 0.0, 0.55, 0.45
        else:
            w_visual, w_ocr, w_trans = 1.0, 0.0, 0.0

        confidence = w_visual * visual_conf + w_ocr * ocr_conf + w_trans * trans_conf
        return min(max(confidence, 0.0), 1.0)

    @staticmethod
    def _ocr_match_score(frame, text_q, ocr_index):
        frame_ocrs = ocr_index.get(frame.get("frame_id", ""), [])
        for ocr_text in frame_ocrs:
            if text_q in ocr_text:
                return 1.0
        return 0.0

    @staticmethod
    def _transcript_match_score(frame, text_q, transcript_index):
        video_transcripts = transcript_index.get(frame.get("video_id", ""), [])
        if not video_transcripts:
            return 0.0
        frame_ts = frame.get("timestamp_ms", 0)
        best = 0.0
        for t_start, t_end, t_text in video_transcripts:
            if text_q in t_text:
                if t_start <= frame_ts <= t_end:
                    return 1.0
                best = max(best, 0.3)
        return best

    # ------------------------------------------------------------------
    # Index builders (O(N) once, O(1) per lookup)
    # ------------------------------------------------------------------

    @staticmethod
    def _build_ocr_index(ocr_list):
        index = {}
        for r in ocr_list:
            fid = r.get("frame_id")
            text = r.get("ocr_text")
            if fid and text:
                index.setdefault(fid, []).append(text.lower())
        return index

    @staticmethod
    def _build_transcript_index(transcript_list):
        index = {}
        for t in transcript_list:
            vid = t.get("video_id")
            if vid is None:
                continue
            index.setdefault(vid, []).append((
                t.get("start_time_ms", 0),
                t.get("end_time_ms", 0),
                t.get("text", "").lower(),
            ))
        return index

    # ------------------------------------------------------------------
    # Result builder
    # ------------------------------------------------------------------

    @staticmethod
    def _to_result(frame, videos, confidence):
        video = videos.get(frame.get("video_id", ""), {})
        return {
            "video_id":        frame.get("video_id", ""),
            "frame_id":        frame.get("frame_id", ""),
            "frame_number":    int(frame.get("frame_number", 0)),
            "timestamp_ms":    int(frame.get("timestamp_ms", 0)),
            "confidence":      round(min(max(confidence, 0.0), 1.0), 4),
            "frame_image_url": frame.get("image_url", ""),
            "fps":             float(video.get("fps", 25.0)),
        }

    # ------------------------------------------------------------------
    # Duplicate suppression
    # ------------------------------------------------------------------

    def _suppress_duplicates(self, results):
        by_video = {}
        for r in results:
            by_video.setdefault(r["video_id"], []).append(r)

        filtered = []
        for frames in by_video.values():
            frames.sort(key=lambda x: x["timestamp_ms"])
            last_ts = -999999
            for r in frames:
                if r["timestamp_ms"] - last_ts >= self.DUPLICATE_MIN_GAP_MS:
                    filtered.append(r)
                    last_ts = r["timestamp_ms"]

        filtered.sort(key=lambda x: x["confidence"], reverse=True)
        return filtered
