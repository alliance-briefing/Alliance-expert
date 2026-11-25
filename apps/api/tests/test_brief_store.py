"""Briefs sur disque : FileBriefStore et BRIEFS_DIR (L3.8, P2)."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date
from pathlib import Path

import pytest
from alliance_api.auth import DEMO_HEADER
from alliance_api.briefs import FileBriefStore, MemoryBriefStore
from alliance_api.main import create_app
from alliance_api.stores import DemoItemStore
from fastapi.testclient import TestClient

from apps.api.tests.factories import DAY, USER_A, USER_B, make_item
from contracts.models import Brief, Item

JOUR = date.fromisoformat(DAY)


def _generator(items: Sequence[Item], user: str, day: date | str, **kw: object) -> Brief:
    """Un brief minimal qui cite le premier item du jour."""
    entry = {"rank": 1, "priority_score": 80, "priority_reasons": ["raison"], "text": "Texte",
             "claims": [], "recommended_action": None, "source_item_ids": [items[0].item_id]}  # fmt: skip
    return Brief.model_validate(
        {
            "brief_id": f"b_{day}_{user}",
            "owner_user_id": user,
            "date": str(day),
            "generated_at": f"{day}T08:31:10+02:00",
            "generator": {"llm": "modele-local"},
            "sections": [{"kind": "actions", "entries": [entry]}],
            "stats": {
                "items_in": len(items),
                "items_out": 1,
                "tokens_in": 0,
                "tokens_out": 0,
                "latency_ms": 0,
                "cost_eur": 0.0,
            },
        }
    )


def test_ecrit_relu_puis_remplace(tmp_path: Path) -> None:
    store = FileBriefStore(tmp_path)
    first = _generator([make_item(USER_A, 1)], USER_A, DAY)
    store.put(first)
    assert FileBriefStore(tmp_path).get(USER_A, JOUR) == first  # « après redémarrage »
    second = _generator([make_item(USER_A, 2)], USER_A, DAY)
    store.put(second)
    assert store.get(USER_A, JOUR) == second
    assert [p.name for p in (tmp_path / USER_A).iterdir()] == [f"{DAY}.json"]  # aucun .tmp


def test_absent_illisible_ou_identifiant_invalide(tmp_path: Path) -> None:
    store = FileBriefStore(tmp_path)
    assert store.get(USER_A, JOUR) is None
    (tmp_path / USER_A).mkdir()
    (tmp_path / USER_A / f"{DAY}.json").write_text('{"brief_id": "coupé', encoding="utf-8")
    assert store.get(USER_A, JOUR) is None
    with pytest.raises(ValueError, match="invalide"):
        store.get("../u_test_a", JOUR)


def _client(data_root: Path) -> TestClient:
    app = create_app(item_store=DemoItemStore(data_root), generator=_generator, auth_mode="demo")
    return TestClient(app)


def test_brief_conserve_au_redemarrage(
    data_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("BRIEFS_DIR", str(tmp_path / "briefs"))
    headers = {DEMO_HEADER: USER_A}
    assert _client(data_root).post("/briefs", json={"date": DAY}, headers=headers).is_success
    relance = _client(data_root)  # nouveau processus, même dossier
    assert isinstance(relance.app.state.briefs, FileBriefStore)
    assert relance.get(f"/briefs/{DAY}", headers=headers).json()["brief_id"] == f"b_{DAY}_{USER_A}"
    assert relance.get(f"/briefs/{DAY}", headers={DEMO_HEADER: USER_B}).status_code == 404


def test_en_memoire_sans_variable(data_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("BRIEFS_DIR", raising=False)
    assert isinstance(_client(data_root).app.state.briefs, MemoryBriefStore)
