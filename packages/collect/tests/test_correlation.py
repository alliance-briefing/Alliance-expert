"""L1.6 : une invitation reçue par mail et l'événement du calendrier ont le même correlation_id."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import pytest

from packages.collect.core.correlation import (
    graph_event_correlation_id,
    graph_invitation_correlation_id,
    meeting_correlation_id,
)


def _dt(value: str) -> dict:
    return {"dateTime": value, "timeZone": "UTC"}


EVENT = {"subject": "Comité de suivi — Groupe Altaïr", "start": _dt("2026-10-21T07:00:00.0000000")}
INVITATION = {
    "subject": "Comité de suivi — Groupe Altaïr",
    "meetingMessageType": "meetingRequest",
    "startDateTime": _dt("2026-10-21T07:00:00.0000000"),
}


def test_invitation_and_event_share_the_same_id() -> None:
    assert graph_invitation_correlation_id(INVITATION) == graph_event_correlation_id(EVENT)
    assert (
        graph_event_correlation_id(EVENT) == "meet:comite-de-suivi-groupe-altair:2026-10-21T07:00Z"
    )


def test_prefixes_and_case_are_ignored() -> None:
    cancelled = EVENT | {"subject": "Annulé : COMITÉ de suivi — Groupe Altaïr"}
    reply = INVITATION | {"subject": "RE: Comité de suivi — Groupe Altaïr"}
    assert graph_event_correlation_id(cancelled) == graph_event_correlation_id(EVENT)
    assert graph_invitation_correlation_id(reply) == graph_event_correlation_id(EVENT)


def test_different_start_means_different_meeting() -> None:
    later = EVENT | {"start": _dt("2026-10-21T08:00:00.0000000")}
    assert graph_event_correlation_id(later) != graph_event_correlation_id(EVENT)


def test_same_instant_in_paris_or_utc() -> None:
    """Changement d'heure du 25/10 : on compare des instants, pas des heures affichées."""
    paris_winter = datetime(2026, 10, 26, 9, 0, tzinfo=timezone(timedelta(hours=1)))
    utc = datetime(2026, 10, 26, 8, 0, tzinfo=UTC)
    assert meeting_correlation_id("Point flash", paris_winter) == meeting_correlation_id(
        "Point flash", utc
    )


def test_ordinary_mail_has_no_correlation() -> None:
    assert graph_invitation_correlation_id({"subject": "Devis", "isRead": False}) is None


def test_naive_or_non_utc_dates_are_refused() -> None:
    with pytest.raises(ValueError):
        meeting_correlation_id("Point", datetime(2026, 10, 21, 9, 0))  # noqa: DTZ001
    with pytest.raises(ValueError):
        graph_event_correlation_id(
            EVENT
            | {"start": {"dateTime": "2026-10-21T09:00:00", "timeZone": "Romance Standard Time"}}
        )


GRAPH_SYNTH = Path(
    os.environ.get("GRAPH_SYNTH_DIR", "packages/collect/fixtures/alliance-graph-synth/data")
)


@pytest.mark.skipif(not (GRAPH_SYNTH / "mail").is_dir(), reason="données de David absentes")
def test_every_invitation_finds_its_event_in_davids_data() -> None:
    events = {
        graph_event_correlation_id(e)
        for f in (GRAPH_SYNTH / "calendar").glob("*.json")
        for e in json.loads(f.read_text("utf-8"))["value"]
    }
    invitations = [
        graph_invitation_correlation_id(m)
        for f in (GRAPH_SYNTH / "mail").glob("*.json")
        for m in json.loads(f.read_text("utf-8"))["value"]
    ]
    invitations = [i for i in invitations if i is not None]
    assert invitations and all(i in events for i in invitations)
