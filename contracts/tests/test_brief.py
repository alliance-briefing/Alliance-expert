"""Contrat B — Brief : fixtures valides acceptées, ≥ 15 cas invalides rejetés (guide §2.3, L3.1).

Validation croisée A ↔ B : tout item cité par un brief de démonstration existe dans la journée
et appartient au propriétaire du brief.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from contracts.demo_data import load_items
from contracts.models import Brief

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "brief"
VALID = sorted((FIXTURES / "valid").glob("*.json"))
INVALID = sorted((FIXTURES / "invalid").glob("*.json"))


@pytest.mark.parametrize("path", VALID, ids=[p.stem for p in VALID])
def test_valid_briefs_pass(path: Path) -> None:
    Brief.model_validate_json(path.read_text("utf-8"))


def test_at_least_15_invalid_cases() -> None:
    assert len(INVALID) >= 15  # exigence L3.1


@pytest.mark.parametrize("path", INVALID, ids=[p.stem for p in INVALID])
def test_invalid_briefs_are_rejected(path: Path) -> None:
    case = json.loads(path.read_text("utf-8"))
    with pytest.raises(ValidationError):
        Brief.model_validate(case["brief"])


@pytest.mark.parametrize("path", VALID, ids=[p.stem for p in VALID])
def test_cited_items_exist_and_belong_to_owner(path: Path) -> None:
    brief = Brief.model_validate_json(path.read_text("utf-8"))
    items = {item.item_id: item for item in load_items(brief.date.isoformat(), brief.owner_user_id)}
    for item_id in brief.cited_item_ids():
        assert item_id in items, f"{item_id} absent de la journée {brief.date}"
        assert brief.owner_user_id in items[item_id].acl.visible_to


def test_briefs_are_immutable() -> None:
    brief = Brief.model_validate_json(VALID[0].read_text("utf-8"))
    with pytest.raises(ValidationError):
        brief.owner_user_id = "u_autre"  # type: ignore[misc]
