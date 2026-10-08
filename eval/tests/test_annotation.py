import json
from pathlib import Path

import pytest

from contracts.demo_data import DEFAULT_USER, load_items
from eval import annotate
from eval.annotation import Annotation, check_against_day, cohen_kappa

DAY = "2026-10-07"


@pytest.fixture(scope="module")
def items() -> list:
    return load_items(DAY)


def _annotation(items: list, n: int = 3, annotator: str = "alice", **changes: object) -> dict:
    ids = [item.item_id for item in items]
    data = {
        "schema": "annotation/1.0",
        "user_id": DEFAULT_USER,
        "date": DAY,
        "annotator": annotator,
        "annotated_at": "2026-10-08",
        "must_include": [
            {"rank": r, "section": "actions", "item_ids": [ids[r]], "why": "relance client"}
            for r in range(1, n + 1)
        ],
        "must_exclude": [ids[0]],
    }
    return data | changes


def test_annotation_valide(items: list) -> None:
    annotation = Annotation.model_validate(_annotation(items))
    assert check_against_day(annotation, items) == []


@pytest.mark.parametrize("n", [2, 8])
def test_entre_3_et_7_elements(items: list, n: int) -> None:
    with pytest.raises(ValueError):
        Annotation.model_validate(_annotation(items, n=n))


def test_rangs_sans_trou(items: list) -> None:
    data = _annotation(items)
    data["must_include"][2]["rank"] = 5
    with pytest.raises(ValueError, match="rangs"):
        Annotation.model_validate(data)


def test_item_inclus_et_exclu_refuse(items: list) -> None:
    data = _annotation(items, must_exclude=[items[1].item_id])
    with pytest.raises(ValueError, match="inclus et exclus"):
        Annotation.model_validate(data)


def test_pas_d_adresse_e_mail_comme_annotateur(items: list) -> None:
    with pytest.raises(ValueError):
        Annotation.model_validate(_annotation(items, annotator="alice@ecole.fr"))


def test_item_inconnu_signale(items: list) -> None:
    data = _annotation(items)
    data["must_include"][0]["item_ids"] = ["mail:n-existe-pas"]
    problems = check_against_day(Annotation.model_validate(data), items)
    assert problems and "n'existe pas" in problems[0]


def test_kappa_accord_parfait_et_desaccord(items: list) -> None:
    a = Annotation.model_validate(_annotation(items))
    assert cohen_kappa(a, a, items) == pytest.approx(1.0)
    shifted = _annotation(items, annotator="bob")
    for entry, item in zip(shifted["must_include"], items[10:], strict=False):
        entry["item_ids"] = [item.item_id]
    assert cohen_kappa(a, Annotation.model_validate(shifted), items) < 0.1


def test_fiche_sans_verite_terrain(monkeypatch: pytest.MonkeyPatch) -> None:
    opened: list[str] = []
    real_open, real_read = Path.open, Path.read_text
    monkeypatch.setattr(
        Path, "open", lambda p, *a, **k: opened.append(p.name) or real_open(p, *a, **k)
    )
    monkeypatch.setattr(
        Path, "read_text", lambda p, *a, **k: opened.append(p.name) or real_read(p, *a, **k)
    )
    sheet = annotate.sheet(DAY)
    assert opened == ["items.jsonl"]
    assert "truth.json" not in opened
    assert "## Mails" in sheet and "mail:dm1-msg-000003" in sheet


def test_modele_puis_verifie(tmp_path: Path, items: list) -> None:
    folder = tmp_path / "dev"
    folder.mkdir()
    (folder / f"{DAY}.alice.json").write_text(json.dumps(_annotation(items)), encoding="utf-8")
    assert annotate.verify(tmp_path) == []
    template = annotate.template(DAY, "bob")
    (folder / f"{DAY}.bob.json").write_text(json.dumps(template), encoding="utf-8")
    assert any("bob" in problem for problem in annotate.verify(tmp_path))
