"""Injectable progress reporting and lightweight SSH resource telemetry."""
from __future__ import annotations

import csv
import shutil
import subprocess
import sys
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence, TypeVar


T = TypeVar("T")


def _format_bytes(value: int | None) -> str:
    if value is None:
        return "--"
    size = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if size < 1024 or unit == "TiB":
            return f"{size:.1f}{unit}"
        size /= 1024
    return f"{size:.1f}TiB"


@dataclass(frozen=True)
class ProgressConfig:
    """Configuration for interactive progress bars and resource telemetry."""

    enabled: bool = True
    leave: bool = False
    min_interval_seconds: float = 0.2
    bar_width: int = 24
    show_system_metrics: bool = True
    metrics_interval_seconds: float = 1.0

    def __post_init__(self) -> None:
        if self.min_interval_seconds <= 0:
            raise ValueError("min_interval_seconds must be positive")
        if self.bar_width < 8:
            raise ValueError("bar_width must be at least 8")
        if self.metrics_interval_seconds <= 0:
            raise ValueError("metrics_interval_seconds must be positive")


@dataclass(frozen=True)
class ResourceSnapshot:
    """Point-in-time process host metrics formatted for a terminal postfix."""

    cpu_percent: float | None
    device: str
    gpu_util_percent: float | None
    gpu_memory_used_bytes: int | None
    gpu_memory_total_bytes: int | None
    disk_used_bytes: int | None
    disk_total_bytes: int | None
    disk_free_bytes: int | None

    def format_compact(self) -> str:
        cpu = f"cpu={self.cpu_percent:.0f}%" if self.cpu_percent is not None else "cpu=--"
        if self.gpu_util_percent is None:
            gpu = f"gpu={self.device}"
        else:
            gpu = f"gpu={self.device} {self.gpu_util_percent:.0f}%"
        if self.gpu_memory_total_bytes:
            gpu += (
                f" mem={_format_bytes(self.gpu_memory_used_bytes)}"
                f"/{_format_bytes(self.gpu_memory_total_bytes)}"
            )

        if self.disk_total_bytes:
            used_percent = 100.0 * (self.disk_used_bytes or 0) / self.disk_total_bytes
            disk = (
                f"disk={_format_bytes(self.disk_used_bytes)}"
                f"/{_format_bytes(self.disk_total_bytes)} ({used_percent:.1f}%)"
                f" free={_format_bytes(self.disk_free_bytes)}"
            )
        else:
            disk = "disk=--"
        return f"{cpu} {gpu} {disk}"

    def format_lines(self) -> tuple[str, str]:
        """Split compact telemetry into CPU/GPU and disk rows."""
        compact = self.format_compact()
        if " disk=" not in compact:
            return compact, "disk=--"
        cpu_gpu, disk = compact.rsplit(" disk=", 1)
        return cpu_gpu, f"disk={disk}"


class SystemResourceMonitor:
    """Read CPU, CUDA/NVIDIA and filesystem metrics without hard dependencies."""

    def __init__(
        self,
        disk_root: Path,
        *,
        device_hint: str = "auto",
        min_interval_seconds: float = 1.0,
    ) -> None:
        self.disk_root = disk_root
        self.device_hint = device_hint
        self.min_interval_seconds = min_interval_seconds
        self._last_snapshot: ResourceSnapshot | None = None
        self._last_snapshot_at = 0.0
        self._previous_cpu_ticks = self._read_cpu_ticks()
        self._torch_module: object | None = None
        self._torch_loaded = False

    def snapshot(self, *, force: bool = False) -> ResourceSnapshot:
        now = time.monotonic()
        if (
            not force
            and self._last_snapshot is not None
            and now - self._last_snapshot_at < self.min_interval_seconds
        ):
            return self._last_snapshot

        snapshot = ResourceSnapshot(
            cpu_percent=self._cpu_percent(),
            **self._gpu_metrics(),
            **self._disk_metrics(),
        )
        self._last_snapshot = snapshot
        self._last_snapshot_at = now
        return snapshot

    @staticmethod
    def _read_cpu_ticks() -> tuple[int, int] | None:
        try:
            first_line = Path("/proc/stat").read_text(encoding="utf-8").splitlines()[0]
            fields = first_line.split()
            if fields[0] != "cpu":
                return None
            values = [int(value) for value in fields[1:]]
            idle = values[3] + (values[4] if len(values) > 4 else 0)
            return sum(values), idle
        except (OSError, IndexError, ValueError):
            return None

    def _cpu_percent(self) -> float | None:
        current = self._read_cpu_ticks()
        previous = self._previous_cpu_ticks
        self._previous_cpu_ticks = current
        if current is None or previous is None:
            return None
        total_delta = current[0] - previous[0]
        idle_delta = current[1] - previous[1]
        if total_delta <= 0:
            return None
        return max(0.0, min(100.0, 100.0 * (1.0 - idle_delta / total_delta)))

    def _disk_metrics(self) -> dict[str, int | None]:
        try:
            usage = shutil.disk_usage(self.disk_root)
        except OSError:
            return {
                "disk_used_bytes": None,
                "disk_total_bytes": None,
                "disk_free_bytes": None,
            }
        return {
            "disk_used_bytes": usage.used,
            "disk_total_bytes": usage.total,
            "disk_free_bytes": usage.free,
        }

    def _load_torch(self) -> object | None:
        if self._torch_loaded:
            return self._torch_module
        self._torch_loaded = True
        try:
            import torch

            self._torch_module = torch
        except ImportError:
            self._torch_module = None
        return self._torch_module

    def _gpu_metrics(self) -> dict[str, object]:
        torch = self._load_torch()
        cuda_available = False
        mps_available = False
        device_name: str | None = None
        memory_used: int | None = None
        memory_total: int | None = None

        if torch is not None:
            cuda = getattr(torch, "cuda", None)
            if cuda is not None:
                try:
                    cuda_available = bool(cuda.is_available())
                except (RuntimeError, OSError):
                    cuda_available = False
            backends = getattr(torch, "backends", None)
            mps = getattr(backends, "mps", None) if backends is not None else None
            if mps is not None:
                try:
                    mps_available = bool(mps.is_available())
                except (RuntimeError, OSError):
                    mps_available = False

            if cuda_available and cuda is not None:
                try:
                    index = cuda.current_device()
                    device_name = str(cuda.get_device_name(index))
                    memory_used = int(cuda.memory_reserved(index))
                    memory_total = int(cuda.get_device_properties(index).total_memory)
                except (RuntimeError, OSError, AttributeError):
                    pass

        nvidia = self._nvidia_smi()
        gpu_util = nvidia.get("utilization")
        if nvidia.get("name"):
            device_name = str(nvidia["name"])
        if nvidia.get("memory_used") is not None:
            memory_used = int(nvidia["memory_used"])
        if nvidia.get("memory_total") is not None:
            memory_total = int(nvidia["memory_total"])

        hint = self.device_hint.lower()
        if hint.startswith("cuda"):
            device = f"CUDA:{device_name}" if cuda_available else "CUDA unavailable"
        elif hint == "mps":
            device = "MPS" if mps_available else "MPS unavailable"
        elif hint == "cpu":
            device = "CPU"
        elif cuda_available:
            device = f"CUDA:{device_name}"
        elif mps_available:
            # TransNetV2's auto mode intentionally falls back to CPU on MPS.
            device = "CPU (MPS auto-disabled)"
        elif nvidia:
            device = "CUDA unavailable"
        else:
            device = "CPU"

        return {
            "device": device,
            "gpu_util_percent": gpu_util,
            "gpu_memory_used_bytes": memory_used,
            "gpu_memory_total_bytes": memory_total,
        }

    @staticmethod
    def _nvidia_smi() -> dict[str, object]:
        executable = shutil.which("nvidia-smi")
        if executable is None:
            return {}
        command = [
            executable,
            "--query-gpu=index,name,utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ]
        try:
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=0.75,
            )
        except (OSError, subprocess.SubprocessError):
            return {}
        if completed.returncode != 0 or not completed.stdout.strip():
            return {}
        try:
            row = next(csv.reader([completed.stdout.splitlines()[0]]))
            return {
                "name": row[1].strip(),
                "utilization": float(row[2].strip()),
                "memory_used": int(float(row[3].strip()) * 1024 * 1024),
                "memory_total": int(float(row[4].strip()) * 1024 * 1024),
            }
        except (IndexError, ValueError):
            return {}

    def format_compact(self) -> str:
        return self.snapshot().format_compact()

    def format_lines(self) -> tuple[str, str]:
        return self.snapshot().format_lines()


class ProgressReporter(ABC):
    """Replaceable progress boundary for CLI, logs or another UI."""

    @abstractmethod
    def iterate(
        self,
        iterable: Iterable[T],
        *,
        total: int | None,
        desc: str,
        unit: str,
    ) -> Iterable[T]:
        raise NotImplementedError

    # Lifecycle hooks intentionally have no-op defaults so injected reporters
    # written before the full-pipeline bar was introduced remain compatible.
    def start_pipeline(self, *, total_units: int, total_lots: int, stage_names: Sequence[str]) -> None:
        return None

    def set_lot_context(self, *, lot_index: int, total_lots: int, lot_id: str) -> None:
        return None

    def start_stage(self, *, name: str, lot_id: str) -> None:
        return None

    def complete_stage(self, *, name: str, lot_id: str) -> None:
        return None

    def finish_pipeline(self) -> None:
        return None


@dataclass
class _DetailState:
    """Mutable state used while one reusable detail bar is nested."""

    desc: str
    total: int | None
    current: int
    unit: str
    started_at: float


class TqdmProgressReporter(ProgressReporter):
    """Render three metric rows and two reusable progress bars.

    Interactive terminals always keep the same five rows: lot/stage context,
    CPU/GPU metrics, disk metrics, full-pipeline progress, and the currently
    active operation. Nested iterators reuse the operation row and restore
    their parent afterwards.
    """

    _DESCRIPTION_WIDTH = 24
    _MAX_STATUS_WIDTH = 120

    def __init__(
        self,
        config: ProgressConfig | None = None,
        *,
        disk_root: Path | None = None,
        device_hint: str = "auto",
        resource_monitor: SystemResourceMonitor | None = None,
    ) -> None:
        self.config = config or ProgressConfig()
        self._detail_stack: list[_DetailState] = []
        self._plain_detail_stack: list[_DetailState] = []
        self._status_bars: list[object] = []
        self._pipeline_bar: object | None = None
        self._detail_bar: object | None = None
        self._pipeline_completed = 0
        self._lot_context = "lot=--"
        self._stage_context = "stage=idle"
        self._tqdm = None
        self._tqdm_loaded = False
        # ``2>&1 | tee`` makes stderr non-interactive even inside tmux.  Keep
        # one carriage-returned line there; direct SSH/tmux gets five fixed
        # rows (three metrics and two progress bars).
        self._plain_mode = not sys.stderr.isatty()
        self._plain_line_active = False
        self._plain_last_render_at = 0.0
        self._plain_last_line_length = 0
        self._plain_stage_desc = "idle"
        self._plain_stage_current = 0
        self._plain_stage_total: int | None = None
        self._plain_stage_started_at: float | None = None
        self._plain_pipeline_total = 0
        self._plain_pipeline_started_at: float | None = None
        self._plain_pipeline_started = False
        if self.config.show_system_metrics:
            self.resource_monitor = resource_monitor or SystemResourceMonitor(
                disk_root or Path.cwd(),
                device_hint=device_hint,
                min_interval_seconds=self.config.metrics_interval_seconds,
            )
        else:
            self.resource_monitor = None

    def _load_tqdm(self):
        if self._tqdm_loaded:
            return self._tqdm
        self._tqdm_loaded = True
        try:
            from tqdm import tqdm
        except ImportError:  # pragma: no cover - requirements includes tqdm
            return None
        self._tqdm = tqdm
        return tqdm

    @staticmethod
    def _compact_text(value: str, max_length: int) -> str:
        value = " ".join(value.split())
        if len(value) <= max_length:
            return value
        if max_length <= 3:
            return value[:max_length]
        return value[: max_length - 3] + "..."

    def _status_lines(self) -> tuple[str, str, str]:
        if self.resource_monitor is None:
            cpu_gpu, disk = "cpu/gpu=disabled", "disk=disabled"
        else:
            cpu_gpu, disk = self.resource_monitor.format_lines()
        terminal_width = shutil.get_terminal_size(fallback=(120, 24)).columns
        max_width = max(32, min(self._MAX_STATUS_WIDTH, terminal_width - 1))
        lines = (
            f"{self._lot_context} | {self._stage_context}",
            cpu_gpu,
            disk,
        )
        return (
            self._compact_text(lines[0], max_width),
            self._compact_text(lines[1], max_width),
            self._compact_text(lines[2], max_width),
        )

    def _bar_format(self) -> str:
        return (
            f"{{desc:<{self._DESCRIPTION_WIDTH}}} "
            f"|{{bar:{self.config.bar_width}}}| "
            "{n_fmt}/{total_fmt} {unit} eta={remaining}"
        )

    def _refresh_status(self) -> None:
        if not self._status_bars:
            return
        for bar, line in zip(self._status_bars, self._status_lines(), strict=True):
            bar.set_description_str(line, refresh=False)
            bar.refresh()

    def _refresh_bars(self) -> None:
        self._refresh_status()
        if self._pipeline_bar is not None:
            self._pipeline_bar.refresh()
        if self._detail_bar is not None:
            self._detail_bar.refresh()

    def _refresh_current_operation(self) -> None:
        """Refresh status and detail rows without redrawing unchanged pipeline state."""
        self._refresh_status()
        if self._detail_bar is not None:
            self._detail_bar.refresh()

    def _ensure_status_bars(self, tqdm) -> None:
        if self._status_bars:
            return
        for position, line in enumerate(self._status_lines()):
            self._status_bars.append(
                tqdm(
                    total=0,
                    desc=line,
                    bar_format="{desc}",
                    position=position,
                    leave=False,
                    mininterval=self.config.min_interval_seconds,
                    dynamic_ncols=True,
                )
            )

    def _ensure_detail_bar(self, tqdm, state: _DetailState) -> None:
        self._ensure_status_bars(tqdm)
        position = 4 if self._pipeline_bar is not None else 3
        if self._detail_bar is None:
            self._detail_bar = tqdm(
                total=state.total,
                initial=state.current,
                desc=self._compact_text(state.desc, self._DESCRIPTION_WIDTH),
                unit=state.unit,
                position=position,
                leave=self.config.leave,
                mininterval=self.config.min_interval_seconds,
                dynamic_ncols=True,
                bar_format=self._bar_format(),
            )
            return

        self._detail_bar.reset(total=state.total)
        self._detail_bar.n = state.current
        self._detail_bar.last_print_n = state.current
        self._detail_bar.unit = state.unit
        self._detail_bar.set_description_str(
            self._compact_text(state.desc, self._DESCRIPTION_WIDTH),
            refresh=False,
        )
        self._detail_bar.refresh()

    @staticmethod
    def _ascii_bar(current: int, total: int | None, width: int) -> str:
        if total is None or total <= 0:
            return "[" + "." * width + "]"
        fraction = max(0.0, min(1.0, current / total))
        filled = round(width * fraction)
        return "[" + "#" * filled + "." * (width - filled) + "]"

    @staticmethod
    def _format_duration(seconds: float) -> str:
        rounded = max(0, round(seconds))
        hours, remainder = divmod(rounded, 3600)
        minutes, seconds_value = divmod(remainder, 60)
        if hours:
            return f"{hours}:{minutes:02d}:{seconds_value:02d}"
        return f"{minutes:02d}:{seconds_value:02d}"

    @classmethod
    def _format_eta(
        cls,
        current: int,
        total: int | None,
        started_at: float | None,
    ) -> str:
        if total is None or total <= 0 or started_at is None or current <= 0:
            return "--"
        if current >= total:
            return "00:00"
        elapsed = max(0.0, time.monotonic() - started_at)
        remaining = elapsed * (total - current) / current
        return cls._format_duration(remaining)

    def _render_plain(self, *, force: bool = False) -> None:
        if not self.config.enabled or not self._plain_mode:
            return
        now = time.monotonic()
        if not force and now - self._plain_last_render_at < self.config.min_interval_seconds:
            return
        self._plain_last_render_at = now
        parts: list[str] = []
        if self.resource_monitor is not None:
            parts.append(self.resource_monitor.format_compact())
        parts.append(self._lot_context)
        if self._plain_pipeline_started:
            pipeline_eta = self._format_eta(
                self._pipeline_completed,
                self._plain_pipeline_total,
                self._plain_pipeline_started_at,
            )
            parts.append(
                f"pipeline {self._ascii_bar(self._pipeline_completed, self._plain_pipeline_total, self.config.bar_width)} "
                f"{self._pipeline_completed}/{self._plain_pipeline_total} eta={pipeline_eta}"
            )
        if self._plain_stage_total is None:
            stage_eta = "--"
            stage = (
                f"{self._plain_stage_desc} "
                f"{self._ascii_bar(0, None, self.config.bar_width)} "
                f"eta={stage_eta}"
            )
        else:
            stage_eta = self._format_eta(
                self._plain_stage_current,
                self._plain_stage_total,
                self._plain_stage_started_at,
            )
            stage = (
                f"{self._plain_stage_desc} "
                f"{self._ascii_bar(self._plain_stage_current, self._plain_stage_total, self.config.bar_width)} "
                f"{self._plain_stage_current}/{self._plain_stage_total} eta={stage_eta}"
            )
        parts.append(stage)
        line = " | ".join(parts)
        if len(line) < self._plain_last_line_length:
            line = line.ljust(self._plain_last_line_length)
        self._plain_last_line_length = len(line)
        sys.stderr.write("\r" + line)
        sys.stderr.flush()
        self._plain_line_active = True

    def start_pipeline(self, *, total_units: int, total_lots: int, stage_names: Sequence[str]) -> None:
        del stage_names
        if not self.config.enabled:
            return
        self._pipeline_completed = 0
        self._plain_pipeline_total = total_units
        self._plain_pipeline_started_at = time.monotonic()
        self._plain_pipeline_started = True
        self._lot_context = f"lots=0/{total_lots}"
        self._stage_context = "stage=starting"
        if self._plain_mode:
            self._render_plain(force=True)
            return
        tqdm = self._load_tqdm()
        if tqdm is None:
            return
        self._ensure_status_bars(tqdm)
        self._pipeline_bar = tqdm(
            total=total_units,
            desc="pipeline",
            unit="stage",
            position=3,
            leave=self.config.leave,
            mininterval=self.config.min_interval_seconds,
            dynamic_ncols=True,
            bar_format=self._bar_format(),
        )
        self._refresh_bars()

    def set_lot_context(self, *, lot_index: int, total_lots: int, lot_id: str) -> None:
        self._lot_context = f"lot={lot_index}/{total_lots}:{lot_id}"
        if self._plain_mode:
            self._render_plain(force=True)
            return
        self._refresh_bars()

    def start_stage(self, *, name: str, lot_id: str) -> None:
        self._stage_context = f"stage={name}({lot_id})"
        state = _DetailState(f"stage={name}({lot_id})", None, 0, "item", time.monotonic())
        self._plain_stage_desc = state.desc
        self._plain_stage_current = state.current
        self._plain_stage_total = state.total
        self._plain_stage_started_at = state.started_at
        if self._plain_mode:
            self._render_plain(force=True)
            return
        tqdm = self._load_tqdm()
        if tqdm is not None:
            self._ensure_detail_bar(tqdm, state)
            self._refresh_bars()

    def complete_stage(self, *, name: str, lot_id: str) -> None:
        if self._pipeline_bar is not None:
            self._pipeline_bar.update(1)
            self._pipeline_completed += 1
        elif self._plain_mode:
            self._pipeline_completed += 1
        self._stage_context = f"stage={name}:done({lot_id})"
        self._plain_stage_desc = f"stage={name}:done({lot_id})"
        if self._plain_mode:
            self._render_plain(force=True)
            return
        self._refresh_bars()

    def finish_pipeline(self) -> None:
        if self._detail_bar is not None:
            self._detail_bar.close()
            self._detail_bar = None
        if self._pipeline_bar is not None:
            self._pipeline_bar.close()
            self._pipeline_bar = None
        for bar in reversed(self._status_bars):
            bar.close()
        self._status_bars.clear()
        self._detail_stack.clear()
        if self._plain_line_active:
            sys.stderr.write("\n")
            sys.stderr.flush()
            self._plain_line_active = False
        self._plain_last_line_length = 0
        self._plain_pipeline_started_at = None
        self._plain_pipeline_started = False

    def iterate(
        self,
        iterable: Iterable[T],
        *,
        total: int | None,
        desc: str,
        unit: str,
    ) -> Iterable[T]:
        if not self.config.enabled:
            for item in iterable:
                yield item
            return

        tqdm = self._load_tqdm()
        if tqdm is None:
            for item in iterable:
                yield item
            return

        if self._plain_mode:
            state = _DetailState(desc, total, 0, unit, time.monotonic())
            self._plain_detail_stack.append(state)
            self._plain_stage_desc = desc
            self._plain_stage_current = 0
            self._plain_stage_total = total
            self._plain_stage_started_at = state.started_at
            self._render_plain(force=True)
            try:
                for item in iterable:
                    yield item
                    state.current += 1
                    self._plain_stage_current = state.current
                    self._render_plain()
            finally:
                self._plain_detail_stack.pop()
                if self._plain_detail_stack:
                    parent = self._plain_detail_stack[-1]
                    self._plain_stage_desc = parent.desc
                    self._plain_stage_current = parent.current
                    self._plain_stage_total = parent.total
                    self._plain_stage_started_at = parent.started_at
                else:
                    self._plain_stage_current = state.current
                    self._plain_stage_total = state.total
                    self._plain_stage_started_at = state.started_at
                self._render_plain(force=True)
                if not self._plain_detail_stack and not self._plain_pipeline_started:
                    sys.stderr.write("\n")
                    sys.stderr.flush()
                    self._plain_line_active = False
            return

        state = _DetailState(desc, total, 0, unit, time.monotonic())
        self._detail_stack.append(state)
        self._ensure_detail_bar(tqdm, state)
        self._refresh_bars()
        try:
            for item in iterable:
                yield item
                state.current += 1
                if self._detail_bar is not None:
                    self._detail_bar.update(1)
                self._refresh_current_operation()
        finally:
            self._detail_stack.pop()
            if self._detail_stack:
                self._ensure_detail_bar(tqdm, self._detail_stack[-1])
            self._refresh_current_operation()
