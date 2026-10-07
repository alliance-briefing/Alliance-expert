"""Tests des fixtures du dossier contracts/ (guide §2.5 et L3.1).

- toutes les fixtures de démonstration valident le JSON Schema et le modèle ;
- les JSON Schema publiés sont à jour avec les modèles pydantic ;
- fixtures/valid passe, fixtures/invalid (≥ 15 cas) est entièrement rejeté ;
- le validateur détecte une fiche cassée, un fichier modifié, manquant ou en trop.

Aucune dépendance au code des lots : ce dossier se teste seul.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from pydantic import ValidationError

from contracts.demo_data import available_days, load_items, load_truth
from contracts.export_schemas import SCHEMAS_DIR, export
from contracts.models import Item
from contracts.validate_fixtures import DEFAULT_DIR, main, validate_dir

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
INVALID = sorted((FIXTURES / "invalid").glob("*.json"))
VALID = sorted((FIXTURES / "valid").glob("*.json"))


def test_demo_fixtures_are_valid() -> None:
    report = validate_dir(DEFAULT_DIR)
    assert report.ok, report.errors[:10]
    assert report.items > 4000


def test_published_schemas_are_up_to_date(tmp_path: Path) -> None:
    for fresh in export(tmp_path):
        published = SCHEMAS_DIR / fresh.name
        assert published.read_text("utf-8") == fresh.read_text("utf-8"), (
            f"{published.name} est périmé : lancer `uv run python -m contracts.export_schemas`"
        )


@pytest.mark.parametrize("path", VALID, ids=[p.stem for p in VALID])
def test_valid_examples_pass(path: Path) -> None:
    Item.model_validate_json(path.read_text("utf-8"))


def test_at_least_15_invalid_cases() -> None:
    assert len(INVALID) >= 15  # exigence L3.1


@pytest.mark.parametrize("path", INVALID, ids=[p.stem for p in INVALID])
def test_invalid_examples_are_rejected(path: Path) -> None:
    case = json.loads(path.read_text("utf-8"))
    with pytest.raises(ValidationError):
        Item.model_validate(case["item"])


def test_demo_data_helpers() -> None:
    days = available_days()
    assert len(days) == 30 and days == sorted(days)
    items = load_items(days[0])
    assert items and all(isinstance(item, Item) for item in items)
    assert 3 <= len(load_truth(days[0])["expected_brief"]) <= 7
    assert available_days("u_inconnu") == []


def _copy(tmp_path: Path) -> Path:
    copy = tmp_path / "demo"
    shutil.copytree(DEFAULT_DIR, copy)
    return copy


def test_validator_rejects_invalid_item(tmp_path: Path) -> None:
    copy = _copy(tmp_path)
    target = copy / "u_demo_001" / available_days()[0] / "items.jsonl"
    lines = target.read_text("utf-8").splitlines()
    item = json.loads(lines[0])
    item["source"] = "fax"  # source inconnue du contrat
    del item["owner_user_id"]  # champ obligatoire
    lines[0] = json.dumps(item, ensure_ascii=False)
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")

    text = "\n".join(validate_dir(copy).errors)
    assert "schéma, champ source" in text
    assert "owner_user_id" in text
    assert "sha256" in text  # la modification manuelle est aussi détectée
    assert main([str(copy)]) == 1


def test_validator_detects_missing_and_extra_files(tmp_path: Path) -> None:
    copy = _copy(tmp_path)
    day = copy / "u_demo_001" / available_days()[0]
    (day / "truth.json").unlink()
    (day / "notes.txt").write_text("ajout manuel", encoding="utf-8")
    text = "\n".join(validate_dir(copy).errors)
    assert "truth.json: listé dans le manifeste mais absent" in text
    assert "notes.txt: présent mais absent du manifeste" in text
