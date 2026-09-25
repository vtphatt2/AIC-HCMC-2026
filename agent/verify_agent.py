"""Transcript/metadata-only verification until real image transport is proven."""

import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
from threading import Event

from .codex import _stop_process, _thread_id_from_events


VERIFY_POLICY = """You are the Verify Agent for VORTA. You receive ONE human-selected candidate.
Do not search the collection. Never submit an answer.
Decompose the official query into at most eight important requirements.
Only transcript text is content evidence. Frame IDs, timestamps, neighbors and scores are metadata,
not visual evidence. No image content is supplied in this version.
For every visual requirement use UNKNOWN. Do not claim to see an object, action or scene.
For transcript requirements use MATCH or MISMATCH only with an EXACT short quote from a supplied
transcript segment. Missing evidence is UNKNOWN, never automatically MISMATCH.
Keep requirement labels and notes short. Return only the requested structured JSON."""


class CodexVerifier:
    def __init__(self, *, timeout_seconds: float = 30.0, codex_bin: str = "codex"):
        self.timeout_seconds = timeout_seconds
        self.codex_bin = codex_bin
        self.schema_path = Path(__file__).with_name("verify_result.schema.json").resolve()

    def verify(self, query: str, evidence: dict, cancel_event: Event | None = None):
        prompt = (f"{VERIFY_POLICY}\n\nOfficial query:\n{query.strip()}\n\n"
                  f"Candidate evidence (no image pixels):\n{json.dumps(evidence, ensure_ascii=False)}")
        output_path = None
        try:
            with tempfile.NamedTemporaryFile(prefix="vorta-verify-", suffix=".json", delete=False) as output:
                output_path = Path(output.name)
            command = [self.codex_bin, "exec", "--ephemeral", "--skip-git-repo-check",
                       "--sandbox", "read-only", "--output-schema", str(self.schema_path),
                       "--output-last-message", str(output_path), "--json", "-C", "/tmp", "-"]
            try:
                process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                           stderr=subprocess.PIPE, text=True, start_new_session=True)
            except FileNotFoundError as exc:
                raise RuntimeError(f"Codex executable not found: {self.codex_bin}") from exc
            deadline = time.monotonic() + self.timeout_seconds
            pending_input = prompt
            while True:
                if cancel_event is not None and cancel_event.is_set():
                    _stop_process(process)
                    raise InterruptedError("Verify Agent cancelled")
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    _stop_process(process)
                    raise TimeoutError("Verify Agent exceeded its time budget")
                try:
                    stdout, stderr = process.communicate(input=pending_input, timeout=min(0.25, remaining))
                    break
                except subprocess.TimeoutExpired:
                    pending_input = None
            if process.returncode != 0:
                raise RuntimeError(f"Verify Agent failed: {(stderr or stdout or 'unknown error').strip()[-500:]}")
            try:
                output = json.loads(output_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError("Verify Agent returned malformed JSON") from exc
            return output, _thread_id_from_events(stdout)
        finally:
            if output_path is not None:
                output_path.unlink(missing_ok=True)


class VerifyAgent:
    def __init__(self, verifier, vorta):
        self.verifier = verifier
        self.vorta = vorta

    def run(self, query: str, candidate: dict, *, cancel_event: Event | None = None) -> dict:
        started = time.monotonic()
        if cancel_event is not None and cancel_event.is_set():
            raise InterruptedError("Verify Agent cancelled")
        evidence = self.vorta.inspect_candidate(candidate)
        if cancel_event is not None and cancel_event.is_set():
            raise InterruptedError("Verify Agent cancelled")
        raw, thread_id = self.verifier.verify(query, evidence, cancel_event=cancel_event)
        checks = raw.get("checks") if isinstance(raw, dict) else None
        if not isinstance(checks, list) or not 1 <= len(checks) <= 8:
            raise ValueError("Verify Agent returned invalid checks")
        transcript = "\n".join(str(item.get("text", "")) for item in evidence["transcript_segments"])
        normalized_transcript = " ".join(transcript.casefold().split())
        safe_checks = []
        for check in checks:
            if not isinstance(check, dict) or not all(key in check for key in
                ("requirement", "modality", "status", "evidence_quote", "note")):
                raise ValueError("Verify Agent returned a malformed check")
            modality = check["modality"]
            status = check["status"]
            if modality not in {"visual", "transcript", "unknown"} or status not in {"MATCH", "MISMATCH", "UNKNOWN"}:
                raise ValueError("Verify Agent returned an invalid status")
            quote = check["evidence_quote"] if isinstance(check["evidence_quote"], str) else ""
            normalized_quote = " ".join(quote.casefold().split())
            if modality != "transcript" or not normalized_quote or normalized_quote not in normalized_transcript:
                status = "UNKNOWN"
                quote = ""
            safe_checks.append({
                "requirement": str(check["requirement"])[:160],
                "modality": modality,
                "status": status,
                "evidence_quote": quote[:180] or None,
                "note": str(check["note"])[:180] if status != "UNKNOWN" else "Not established by supplied evidence",
            })
        statuses = [item["status"] for item in safe_checks]
        overall = "UNLIKELY" if "MISMATCH" in statuses else "PROMISING" if all(s == "MATCH" for s in statuses) else "INSPECT"
        return {
            "checks": safe_checks,
            "overall": overall,
            "next_action": "operator_inspect",
            "thread_id": thread_id,
            "evidence": {"matched_frames": evidence["matched_frames"],
                         "neighbor_frames": evidence["neighbor_frames"],
                         "transcript_segments": evidence["transcript_segments"],
                         "visual_content_available": False, "errors": evidence["errors"]},
            "timing_ms": int((time.monotonic() - started) * 1000),
        }
