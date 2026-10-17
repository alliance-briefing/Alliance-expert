"""Tests de L1.1 (interface, registre, lanceur) et du pipeline demo → contrat A."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from contracts.demo_data import available_days
from contracts.models import Item
from packages.collect.cli import main
from packages.collect.connectors.demo_fixtures import DemoFixturesConnector
from packages.collect.core.connector import Connector, Health, Window, collection_window
from packages.collect.core.registry import discover, enabled_connectors
from packages.collect.core.runner import run

USER = "u_demo_001"


def _window(day: str) -> Window:
    return collection_window(date.fromisoformat(day))


class Crashing(Connector):
    name, version = "en_panne", "0"

    def required_scopes(self) -> list[str]:
        return []

    def collect(self, user: str, window: Window) -> Iterator[Item]:
        raise ConnectionError("jeton secret-123 refusé pour claire@exemple.test")

    def health(self) -> Health:
        return Health(False)


def test_registry_finds_the_demo_connector() -> None:
    assert discover()["demo"] is DemoFixturesConnector


def test_a_connector_can_be_disabled_by_config(tmp_path: Path) -> None:
    config = tmp_path / "collect.toml"
    config.write_text('[connectors]\ndisabled = ["demo"]\n', encoding="utf-8")
    assert "demo" not in [c.name for c in enabled_connectors(config)]
    assert "demo" in [c.name for c in enabled_connectors(tmp_path / "absent.toml")]


def test_pipeline_demo_gives_valid_items_of_the_user() -> None:
    result = run(USER, _window(available_days()[0]))
    assert result.items and not result.warnings
    assert all(isinstance(i, Item) and i.owner_user_id == USER for i in result.items)


def test_a_crashing_connector_does_not_stop_the_others(caplog: pytest.LogCaptureFixture) -> None:
    result = run(USER, _window(available_days()[0]), [Crashing(), DemoFixturesConnector()])
    assert result.items  # le connecteur demo a tourné quand même
    assert result.warnings == ["en_panne: échec (ConnectionError)"]
    assert "secret-123" not in caplog.text and "@" not in caplog.text  # rien de sensible journalisé


def test_unknown_day_or_other_user_gives_nothing() -> None:
    assert run(USER, _window("1999-01-01")).items == []
    assert run("u_inconnu", _window(available_days()[0])).items == []


def test_demo_health_and_scopes() -> None:
    assert DemoFixturesConnector().health().ok
    assert not DemoFixturesConnector(Path("dossier-absent")).health().ok
    assert DemoFixturesConnector().required_scopes() == []


def test_window_requires_a_timezone() -> None:
    with pytest.raises(ValueError):
        Window(datetime(2026, 10, 5), datetime(2026, 10, 6))  # noqa: DTZ001 (voulu)
    with pytest.raises(ValueError):
        Window(datetime(2026, 10, 6, tzinfo=UTC), datetime(2026, 10, 5, tzinfo=UTC))


def test_collection_window_is_8h30_in_paris_the_24h_before() -> None:
    window = collection_window(date(2026, 10, 26))  # lendemain du changement d'heure
    assert window.end.isoformat() == "2026-10-26T08:30:00+01:00"
    assert (window.end - window.start).total_seconds() == 24 * 3600


def test_cli_can_run_a_single_source(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--user", USER, "--day", available_days()[0], "--source", "demo"]) == 0
    assert capsys.readouterr().out.strip()


def test_cli_prints_one_json_line_per_item(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--user", USER, "--day", available_days()[0]]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines and all(Item.model_validate_json(line) for line in lines)
