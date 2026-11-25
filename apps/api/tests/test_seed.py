"""Amorçage des données de démonstration (L3.8, P3)."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest
from alliance_api.seed import main, seed
from alliance_api.stores import DemoItemStore

from apps.api.tests.factories import USER_A, make_item, write_day
from contracts.demo_data import DEMO_DIR, available_days
from packages.collect.core.audit import AuditLog

VEILLE, JOUR = "2026-09-29", "2026-09-30"


@pytest.fixture
def source(tmp_path: Path) -> Path:
    root = tmp_path / "source"
    write_day(root, USER_A, VEILLE, [make_item(USER_A, 1)])
    write_day(root, USER_A, JOUR, [make_item(USER_A, 2), make_item(USER_A, 3)])
    (root / "pas-un-utilisateur").mkdir()
    return root


def test_collecte_les_journees_avec_leur_journal(source: Path, tmp_path: Path) -> None:
    data = tmp_path / "data"
    assert seed(data, source) == 2
    assert [d.isoformat() for d in DemoItemStore(data).days(USER_A)] == [VEILLE, JOUR]
    assert not (data / USER_A / JOUR / "truth.json").exists()  # la vérité terrain reste à l'éval
    events = AuditLog(data).read(USER_A)
    assert [(e.actor, e.count) for e in events] == [("connector:demo", 1), ("connector:demo", 2)]
    paris = events[1].ts.astimezone(ZoneInfo("Europe/Paris"))
    assert paris.replace(tzinfo=None) == datetime.fromisoformat(f"{JOUR}T08:30:00")


def test_sans_effet_sur_les_journees_presentes(source: Path, tmp_path: Path) -> None:
    data = tmp_path / "data"
    seed(data, source)
    journal = (data / USER_A / "audit.jsonl").read_text(encoding="utf-8")
    write_day(source, USER_A, "2026-10-01", [make_item(USER_A, 4)])
    assert seed(data, source) == 1  # seule la nouvelle journée
    assert (data / USER_A / "audit.jsonl").read_text(encoding="utf-8").startswith(journal)
    assert seed(data, source) == 0


def test_journees_de_demonstration_completes(tmp_path: Path) -> None:
    assert seed(tmp_path, DEMO_DIR) == len(available_days("u_demo_001"))


def test_ligne_de_commande(
    source: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.delenv("DATA_DIR", raising=False)
    assert main([]) == 0 and "amorçage ignoré" in capsys.readouterr().err
    monkeypatch.setattr("alliance_api.seed.seed", lambda data: seed(data, source))
    assert main(["--data-dir", str(tmp_path / "data")]) == 0
    assert "2 journée(s) ajoutée(s)" in capsys.readouterr().err
