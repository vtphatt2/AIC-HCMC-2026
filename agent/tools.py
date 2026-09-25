import re
from urllib.parse import quote, urljoin

import httpx

from .models import SearchAttempt, VortaCapabilities


class VortaToolError(RuntimeError):
    pass


class VortaClient:
    def __init__(self, base_url: str, *, timeout_seconds: float = 35.0, http=None):
        self.base_url = base_url.rstrip("/") + "/"
        self._owns_http = http is None
        self.http = http or httpx.Client(base_url=self.base_url, timeout=timeout_seconds)
        self._strategy_ids: set[str] | None = None
        self._transcript_algorithms: set[str] | None = None

    def close(self):
        if self._owns_http:
            self.http.close()

    def capabilities(self) -> VortaCapabilities:
        health = self._get_json("/api/health")
        strategies = self._get_json("/api/strategies")
        transcript = self._get_json("/api/transcript-search-algorithms")
        self._strategy_ids = {item["id"] for item in strategies}
        self._transcript_algorithms = {
            item["id"] for item in transcript.get("algorithms", []) if item.get("available")
        }
        return VortaCapabilities(
            env_mode=str(health.get("env_mode", "unknown")).upper(),
            strategies=strategies,
            transcript_default=transcript.get("default", "fuzzy"),
            transcript_algorithms=transcript.get("algorithms", []),
        )

    def search_vorta(self, attempt: SearchAttempt, *, top_k: int) -> dict:
        top_k = min(max(int(top_k), 1), 100)
        if self._strategy_ids is None or self._transcript_algorithms is None:
            self.capabilities()

        if attempt.mode == "transcript":
            if attempt.strategy not in self._transcript_algorithms:
                raise VortaToolError(f"Transcript algorithm is unavailable: {attempt.strategy}")
            event = attempt.events[0]
            query = event.transcript or event.visual
            return self._post_json("/api/search/transcript", {
                "query": query,
                "top_k": top_k,
                "algorithm": attempt.strategy,
            })

        if attempt.strategy not in self._strategy_ids:
            raise VortaToolError(f"Frame strategy is unavailable: {attempt.strategy}")
        query_groups = []
        for event in attempt.events:
            query = " ".join(part for part in (event.visual, event.transcript) if part).strip()
            query_groups.append({"query": query, "temporal_offset_ms": event.min_offset_ms})
        return self._post_json("/api/search", {
            "strategy_id": attempt.strategy,
            "query_groups": query_groups,
            "top_k": top_k,
        })

    def inspect_candidate(self, candidate: dict) -> dict:
        """Small metadata/transcript bundle. No image bytes are sent to Codex yet."""
        video_id = str(candidate.get("video_id", ""))
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", video_id):
            raise ValueError("invalid candidate video_id")
        raw_steps = candidate.get("steps") or [candidate]
        if not isinstance(raw_steps, list):
            raise ValueError("candidate steps must be a list")
        matched = []
        for step in raw_steps[:4]:
            if not isinstance(step, dict) or step.get("video_id", video_id) != video_id:
                raise ValueError("candidate steps must belong to one video")
            timestamp = next((step.get(key) for key in
                              ("timestamp_ms", "nearest_timestamp_ms", "start_time_ms")
                              if step.get(key) is not None), None)
            if timestamp is None:
                continue
            timestamp = int(timestamp)
            if timestamp < 0:
                raise ValueError("candidate timestamp must be nonnegative")
            matched.append({
                "frame_id": str(step.get("frame_id") or ""),
                "frame_number": step.get("frame_number"),
                "timestamp_ms": timestamp,
            })

        path_id = quote(video_id, safe="")
        neighbors = []
        errors = []
        context_targets = matched[:1] + (matched[-1:] if len(matched) > 1 else [])
        for frame in context_targets:
            try:
                context = self._request_json("GET", f"/api/video/{path_id}/context-frames",
                                             params={"start_ms": frame["timestamp_ms"],
                                                     "end_ms": frame["timestamp_ms"], "expand": 1})
                for group in ("before", "after"):
                    neighbors.extend(context.get(group, [])[:1])
            except VortaToolError as exc:
                errors.append(str(exc)[:160])
                break

        transcript_segments = []
        try:
            transcript = self._request_json("GET", f"/api/transcript/{path_id}")
            times = [frame["timestamp_ms"] for frame in matched]
            start_ms = candidate.get("start_time_ms")
            if start_ms is not None:
                times.append(int(start_ms))
            for segment in transcript.get("segments", []):
                if not times or any(
                    int(segment.get("start_ms", 0)) <= t + 15000
                    and int(segment.get("end_ms", 0)) >= t - 15000 for t in times
                ):
                    transcript_segments.append({
                        "start_ms": segment.get("start_ms"),
                        "end_ms": segment.get("end_ms"),
                        "text": str(segment.get("text", ""))[:350],
                        "speaker": segment.get("speaker"),
                    })
                if len(transcript_segments) >= 8:
                    break
        except VortaToolError as exc:
            errors.append(str(exc)[:160])

        return {
            "candidate": {"video_id": video_id, "frame_id": candidate.get("frame_id"),
                          "chunk_id": candidate.get("chunk_id")},
            "matched_frames": matched,
            "neighbor_frames": neighbors[:8],
            "transcript_segments": transcript_segments,
            "visual_content_available": False,
            "errors": errors,
        }

    def _get_json(self, path: str):
        return self._request_json("GET", path)

    def _post_json(self, path: str, payload: dict):
        return self._request_json("POST", path, json=payload)

    def _request_json(self, method: str, path: str, **kwargs):
        try:
            response = self.http.request(method, urljoin(self.base_url, path.lstrip("/")), **kwargs)
            response.raise_for_status()
            return response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise VortaToolError(f"VORTA {method} {path} failed: {exc}") from exc


def search_vorta(client: VortaClient, attempt: SearchAttempt, *, top_k: int) -> dict:
    """Named tool entrypoint used by the Search Agent controller."""
    return client.search_vorta(attempt, top_k=top_k)
