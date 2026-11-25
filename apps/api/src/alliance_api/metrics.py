"""Métriques techniques pour l'administration (L3.7, étape C).

Deux sources, sans changer le contrat C :
- les journaux d'audit (persistants) : volumes et taux d'erreur par action et par
  connecteur, codes d'erreur, sur les N derniers jours (heure de Paris) ;
- `MetricsRecorder` (dans le processus, remis à zéro au redémarrage) : durée de chaque
  génération mesurée par l'API, et latence et tokens déclarés par le générateur (contrat B).

Durées de collecte : non mesurées. Le journal du lot 1 ne note que l'heure de fin de chaque
connecteur, pas son début ; il faudrait un champ de durée dans le contrat C (PR
`contract-change`) ou une mesure côté lot 1.
"""

from __future__ import annotations

import math
from collections import Counter, deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from datetime import date as Date
from zoneinfo import ZoneInfo

from pydantic import BaseModel

from contracts.models import AuditEvent, Brief

PARIS = ZoneInfo("Europe/Paris")


@dataclass(frozen=True)
class GenerationSample:
    ts: datetime
    duration_ms: int  # mesurée par l'API autour du générateur
    ok: bool
    latency_ms: int | None = None  # déclarée par le générateur (brief.stats)
    tokens_in: int = 0
    tokens_out: int = 0


class MetricsRecorder:
    """Les dernières générations du processus (comme les briefs en cache, propre au worker)."""

    def __init__(
        self, keep: int = 10_000, clock: Callable[[], datetime] = lambda: datetime.now(UTC)
    ) -> None:
        self.clock = clock
        self.since = clock()
        self._samples: deque[GenerationSample] = deque(maxlen=keep)

    def generation(self, duration_ms: int, brief: Brief | None) -> None:
        """Une génération terminée : `brief` None = échec."""
        stats = brief.stats if brief else None
        self._samples.append(
            GenerationSample(
                ts=self.clock(),
                duration_ms=duration_ms,
                ok=brief is not None,
                latency_ms=stats.latency_ms if stats else None,
                tokens_in=stats.tokens_in if stats else 0,
                tokens_out=stats.tokens_out if stats else 0,
            )
        )

    def samples(self) -> list[GenerationSample]:
        return list(self._samples)


class Rate(BaseModel):
    total: int
    errors: int
    error_rate: float | None  # None quand il n'y a aucun événement


class ConnectorRate(Rate):
    connector: str
    items: int
    last_run: datetime | None


class Durations(BaseModel):
    count: int
    p50_ms: int | None
    p95_ms: int | None
    max_ms: int | None


class GenerationMetrics(BaseModel):
    since: datetime  # démarrage du processus
    total: int
    failures: int
    measured: Durations  # durée vue par l'API
    declared: Durations  # latence annoncée par le générateur
    tokens_in: int
    tokens_out: int


class Metrics(BaseModel):
    start: Date
    end: Date
    actions: dict[str, Rate]
    connectors: list[ConnectorRate]
    error_codes: dict[str, int]
    generation: GenerationMetrics


def compute(
    events: Iterable[AuditEvent], recorder: MetricsRecorder, end: Date, days: int
) -> Metrics:
    start = end - timedelta(days=days - 1)
    window = [e for e in events if start <= e.ts.astimezone(PARIS).date() <= end]
    actions = {
        action: _rate([e for e in window if e.action == action])
        for action in ("collect", "generate", "view")
    }
    codes = Counter(e.error_code for e in window if e.error_code)
    return Metrics(
        start=start,
        end=end,
        actions=actions,
        connectors=_connectors([e for e in window if e.action == "collect"]),
        error_codes=dict(codes.most_common()),
        generation=_generation(recorder.since, recorder.samples()),
    )


def _rate(events: list[AuditEvent]) -> Rate:
    errors = sum(e.status == "error" for e in events)
    return Rate(total=len(events), errors=errors, error_rate=_ratio(errors, len(events)))


def _ratio(part: int, whole: int) -> float | None:
    return round(part / whole, 4) if whole else None


def _connectors(events: list[AuditEvent]) -> list[ConnectorRate]:
    by_connector: dict[str, list[AuditEvent]] = {}
    for event in events:
        by_connector.setdefault(event.actor.removeprefix("connector:"), []).append(event)
    return [
        ConnectorRate(
            connector=name,
            items=sum(e.count for e in found),
            last_run=max(e.ts for e in found),
            **_rate(found).model_dump(),
        )
        for name, found in sorted(by_connector.items())
    ]


def _generation(since: datetime, samples: list[GenerationSample]) -> GenerationMetrics:
    ok = [s for s in samples if s.ok]
    return GenerationMetrics(
        since=since,
        total=len(samples),
        failures=len(samples) - len(ok),
        measured=_durations([s.duration_ms for s in samples]),
        declared=_durations([s.latency_ms for s in ok if s.latency_ms is not None]),
        tokens_in=sum(s.tokens_in for s in ok),
        tokens_out=sum(s.tokens_out for s in ok),
    )


def _durations(values: list[int]) -> Durations:
    ordered = sorted(values)
    return Durations(
        count=len(ordered),
        p50_ms=_percentile(ordered, 0.50),
        p95_ms=_percentile(ordered, 0.95),
        max_ms=ordered[-1] if ordered else None,
    )


def _percentile(ordered: list[int], q: float) -> int | None:
    """Rang le plus proche : la plus petite valeur dont au moins q des mesures sont ≤."""
    if not ordered:
        return None
    return ordered[max(1, math.ceil(len(ordered) * q)) - 1]
