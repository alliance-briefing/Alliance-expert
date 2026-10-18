from datetime import date

from packages.collect.connectors.notes_demo import NotesDemoConnector
from packages.collect.core.connector import collection_window
from packages.collect.core.registry import discover
from packages.collect.core.runner import run

USER = "u_demo_001"


def window(day: str):
    return collection_window(date.fromisoformat(day))  # la veille 8h30 → le jour 8h30 (Paris)


def test_registry_finds_the_new_source() -> None:
    assert "notes_demo" in discover()


def test_only_notes_inside_the_window_are_collected() -> None:
    items = list(NotesDemoConnector().collect(USER, window("2026-10-21")))
    assert [item.title for item in items] == ["Point équipe du lundi"]  # écrite le 20/10 à 9h
    assert all(item.source == "file" and item.owner_user_id == USER for item in items)
    later = list(NotesDemoConnector().collect(USER, window("2026-10-22")))
    assert [item.title for item in later] == ["Idées pour le reporting mensuel"]


def test_unknown_user_gets_nothing() -> None:
    """Isolation (L1.7) : un utilisateur hors démo ne reçoit aucune note."""
    assert list(NotesDemoConnector().collect("u_inconnu", window("2026-10-21"))) == []


def test_runs_alongside_the_demo_source() -> None:
    result = run(USER, window("2026-10-21"))
    names = {item.connector.name for item in result.items}
    assert {"demo", "notes_demo"} <= names
    assert result.warnings == []


def test_health() -> None:
    assert NotesDemoConnector().health().ok
