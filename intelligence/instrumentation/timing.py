"""Record stage timings/counts/errors without retaining requirement or chunk text."""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter


@dataclass(frozen=True)
class StageMetric:
    duration_ms: float
    item_count: int
    error: str | None = None


@dataclass
class PipelineMetrics:
    stages: dict[str, StageMetric] = field(default_factory=dict)

    def record(self, stage: str, started: float, item_count: int, error: Exception | None = None) -> None:
        self.stages[stage] = StageMetric(
            duration_ms=(perf_counter() - started) * 1000,
            item_count=item_count,
            error=str(error) if error else None,
        )
