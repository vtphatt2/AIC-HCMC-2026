"""Injectable progress reporting for interactive and SSH/tmux runs."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterable, TypeVar


T = TypeVar("T")


@dataclass(frozen=True)
class ProgressConfig:
    """Configuration for terminal progress bars."""

    enabled: bool = True
    leave: bool = True
    min_interval_seconds: float = 0.2

    def __post_init__(self) -> None:
        if self.min_interval_seconds <= 0:
            raise ValueError("min_interval_seconds must be positive")


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


class TqdmProgressReporter(ProgressReporter):
    """Render progress bars through the isolated ``tqdm`` dependency."""

    def __init__(self, config: ProgressConfig | None = None) -> None:
        self.config = config or ProgressConfig()

    def iterate(
        self,
        iterable: Iterable[T],
        *,
        total: int | None,
        desc: str,
        unit: str,
    ) -> Iterable[T]:
        try:
            from tqdm import tqdm
        except ImportError:  # pragma: no cover - requirements includes tqdm
            return iterable
        return tqdm(
            iterable,
            total=total,
            desc=desc,
            unit=unit,
            leave=self.config.leave,
            disable=not self.config.enabled,
            mininterval=self.config.min_interval_seconds,
        )
