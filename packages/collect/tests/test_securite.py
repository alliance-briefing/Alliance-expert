"""Isolation, rétention et suppression (guide L1.7, critères d'acceptation)."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest
from alliance_api.audit import LogAuditSink
from alliance_api.auth import DEMO_HEADER
from alliance_api.main import create_app
from alliance_api.stores import DemoItemStore
from fastapi.testclient import TestClient

from contracts.demo_data import available_days, load_items
from contracts.models import Item
from packages.collect.admin import main as admin
from packages.collect.cli import main as collect
from packages.collect.core.retention import (
    RetentionPolicy,
    delete_user,
    load_policy,
    purge,
    without_content,
)
from packages.collect.core.store import write_day
from packages.collect.tests.test_normalize import VALID

A, B = "u_alice", "u_bob"
DAY = date(2026, 10, 21)


def _item(user: str, n: int, **changes) -> Item:
    raw = VALID | {
        "item_id": f"mail:{user}-{n}",
        "owner_user_id": user,
        "acl": {"visible_to": [user]},
    }
    return Item.model_validate(raw | changes)


def _audit(root: Path, user: str, *timestamps: str) -> Path:
    path = root / user / "audit.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps({"ts": ts, "user_id": user}) + "\n" for ts in timestamps))
    return path


@pytest.fixture
def root(tmp_path: Path) -> Path:
    write_day([_item(A, 1)], tmp_path, A, DAY)
    write_day([_item(B, 1)], tmp_path, B, DAY)
    return tmp_path


def _api(root: Path) -> TestClient:
    app = create_app(item_store=DemoItemStore(root), audit=LogAuditSink(), auth_mode="demo")
    return TestClient(app)


def test_intrusion_a_cannot_read_b_through_the_api(root: Path) -> None:
    client = _api(root)
    assert client.get(f"/items/mail:{A}-1", headers={DEMO_HEADER: A}).status_code == 200
    stolen = client.get(f"/items/mail:{B}-1", headers={DEMO_HEADER: A})
    assert stolen.status_code == 403 and B not in stolen.text


def test_intrusion_a_cannot_read_b_through_the_files(root: Path) -> None:
    assert {i.owner_user_id for i in load_items(DAY.isoformat(), A, root)} == {A}
    with pytest.raises(ValueError):
        write_day([_item(B, 2)], root, A, DAY)  # un Item de B n'entre pas dans le dossier de A
    for bad in ("../u_bob", "u_bob/..", "", ".."):
        with pytest.raises(ValueError):
            delete_user(root, bad)


def test_delete_user_leaves_nothing_and_spares_the_others(root: Path) -> None:
    _audit(root, A, "2026-10-21T06:30:00+00:00")
    removed = delete_user(root, A)
    assert {p.name for p in removed} == {"items.jsonl", "audit.jsonl"}
    assert not (root / A).exists() and available_days(A, root) == []
    assert not any(A in p.read_text() for p in root.rglob("*") if p.is_file())
    assert load_items(DAY.isoformat(), B, root)  # B intact
    assert delete_user(root, A) == []  # une deuxième fois : rien, sans erreur
    client = _api(root)
    assert client.get(f"/items/mail:{A}-1", headers={DEMO_HEADER: A}).status_code == 404


def test_delete_user_command_asks_for_yes(root: Path, capsys) -> None:
    assert admin(["--data-dir", str(root), "delete-user", "--user", A]) == 0
    assert "aperçu : 1 fichier(s)" in capsys.readouterr().out and (root / A).exists()
    assert admin(["--data-dir", str(root), "delete-user", "--user", A, "--yes"]) == 0
    assert not (root / A).exists()


def test_purge_removes_days_after_30_days_and_audit_after_90(root: Path) -> None:
    write_day([_item(A, 2)], root, A, date(2026, 9, 21))  # 30 jours avant le 21/10 : gardée
    write_day([_item(A, 3)], root, A, date(2026, 9, 20))  # 31 jours : supprimée
    path = _audit(root, A, "2026-07-22T21:59:00+00:00", "2026-07-22T22:00:00+00:00")
    report = purge(root, DAY, RetentionPolicy())
    assert [p.name for p in report.days_removed] == ["2026-09-20"]
    assert available_days(A, root) == ["2026-09-21", "2026-10-21"]
    # 90 jours avant le 21/10 : le 23/07 à minuit, heure de Paris (22:00 UTC la veille)
    assert report.audit_lines_removed == 1
    assert path.read_text().count("\n") == 1 and "22:00:00" in path.read_text()
    assert purge(root, DAY, RetentionPolicy()).days_removed == []  # rien de plus


def test_purge_command(root: Path, capsys) -> None:
    write_day([_item(A, 3)], root, A, date(2026, 9, 1))
    assert admin(["--data-dir", str(root), "purge", "--today", "2026-10-21"]) == 0
    assert "1 journée(s) supprimée(s)" in capsys.readouterr().out


def test_body_is_not_stored_unless_the_policy_allows_it(tmp_path, monkeypatch) -> None:
    item = _item(A, 1, content_text="Corps complet du message")
    assert without_content([item], RetentionPolicy())[0].content_text is None
    assert without_content([item], RetentionPolicy(keep_content_text=True))[0].content_text
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    assert (
        collect(["--user", "u_demo_001", "--day", "2026-10-21", "--source", "demo", "--write"]) == 0
    )
    stored = load_items("2026-10-21", "u_demo_001", tmp_path)
    assert stored and all(i.content_text is None for i in stored)


def test_policy_comes_from_collect_toml(tmp_path) -> None:
    assert load_policy() == RetentionPolicy()  # collect.toml du dépôt = valeurs du guide
    config = tmp_path / "collect.toml"
    config.write_text("[retention]\nsnippet_days = 7\n")
    assert load_policy(config) == RetentionPolicy(snippet_days=7)
    assert load_policy(tmp_path / "absent.toml") == RetentionPolicy()


def test_admin_requires_a_data_dir(monkeypatch) -> None:
    monkeypatch.delenv("DATA_DIR", raising=False)
    with pytest.raises(SystemExit):
        admin(["purge"])
