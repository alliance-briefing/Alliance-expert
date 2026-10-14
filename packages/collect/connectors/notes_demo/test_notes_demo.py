from datetime import UTC, datetime, timedelta

from packages.collect.connectors.notes_demo import NotesDemoConnector
from packages.collect.core.connector import Window
from packages.collect.core.registry import discover
from packages.collect.core.runner import run


def window(day: str) -> Window:
    start = datetime.fromisoformat(day).replace(tzinfo=UTC)
    return Window(start, start + timedelta(days=1))


def test_registry_finds_the_new_source() -> None:
    assert "notes_demo" in discover()


def test_only_recent_notes_are_collected() -> None:
    items = list(NotesDemoConnector().collect("u_demo_001", window("2026-10-21")))
    assert [item.title for item in items] == [
        "Point équipe du lundi",
        "Idées pour le reporting mensuel",
    ]
    assert all(item.source == "file" and item.owner_user_id == "u_demo_001" for item in items)


def test_runs_alongside_the_demo_source() -> None:
    result = run("u_demo_001", window("2026-10-21"))
    names = {item.connector.name for item in result.items}
    assert {"demo", "notes_demo"} <= names
    assert result.warnings == []


def test_health() -> None:
    assert NotesDemoConnector().health().ok
