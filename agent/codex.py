import json
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import time
from threading import Event

from .models import SearchPlan, VortaCapabilities
from .prompts import build_search_prompt


class CodexPlannerError(RuntimeError):
    pass


class CodexPlanner:
    def __init__(self, *, timeout_seconds: float = 30.0, codex_bin: str = "codex"):
        self.timeout_seconds = timeout_seconds
        self.codex_bin = codex_bin
        self.schema_path = Path(__file__).with_name("search_plan.schema.json").resolve()

    def plan(self, query: str, capabilities: VortaCapabilities,
             cancel_event: Event | None = None):
        prompt = build_search_prompt(query, capabilities)
        output_path = None
        try:
            with tempfile.NamedTemporaryFile(prefix="vorta-search-plan-", suffix=".json", delete=False) as output:
                output_path = Path(output.name)
            command = [
                self.codex_bin,
                "exec",
                "--ephemeral",
                "--skip-git-repo-check",
                "--sandbox",
                "read-only",
                "--output-schema",
                str(self.schema_path),
                "--output-last-message",
                str(output_path),
                "--json",
                "-C",
                "/tmp",
                "-",
            ]
            try:
                process = subprocess.Popen(
                    command,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    start_new_session=True,
                )
            except FileNotFoundError as exc:
                raise CodexPlannerError(f"Codex executable not found: {self.codex_bin}") from exc
            deadline = time.monotonic() + self.timeout_seconds
            pending_input = prompt
            while True:
                if cancel_event is not None and cancel_event.is_set():
                    _stop_process(process)
                    raise InterruptedError("Search Agent cancelled")
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    _stop_process(process)
                    raise CodexPlannerError(f"Codex planning exceeded {self.timeout_seconds:.0f}s")
                try:
                    stdout, stderr = process.communicate(input=pending_input, timeout=min(0.25, remaining))
                    break
                except subprocess.TimeoutExpired:
                    pending_input = None

            if process.returncode != 0:
                detail = (stderr or stdout or "unknown error").strip()[-1200:]
                raise CodexPlannerError(f"Codex planning failed: {detail}")
            try:
                raw_plan = json.loads(output_path.read_text(encoding="utf-8"))
                plan = SearchPlan.model_validate(raw_plan)
            except (OSError, json.JSONDecodeError, ValueError) as exc:
                raise CodexPlannerError(f"Codex returned an invalid search plan: {exc}") from exc
            return plan, _thread_id_from_events(stdout)
        finally:
            if output_path is not None:
                try:
                    os.unlink(output_path)
                except FileNotFoundError:
                    pass


def _stop_process(process: subprocess.Popen):
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except (OSError, AttributeError):
        process.terminate()
    try:
        process.communicate(timeout=1)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except (OSError, AttributeError):
            process.kill()
        process.communicate()


def _thread_id_from_events(stdout: str) -> str | None:
    for line in stdout.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("type") == "thread.started":
            return event.get("thread_id") or event.get("thread", {}).get("id")
    return None
