import json
from pathlib import Path

import pytest

from contracts.demo_data import DEFAULT_USER, load_items
from eval import annotate
from eval.annotation import Annotation
from eval.cases_metric import PairScore, agreement, b_cubed, pairs, predicted_cases, score_day

DAY = "2026-10-07"


@pytest.fixture(scope="module")
def items() -> list:
    return load_items(DAY)


@pytest.fixture(scope="module")
def predicted(items: list) -> list[list[str]]:
    return [case for case in predicted_cases(items) if len(case) > 1]


def _annotation(items: list, cases: list[list[str]] | None, annotator: str = "alice") -> dict:
    ids = [item.item_id for item in items]
    return {
        "schema": "annotation/1.1",
        "user_id": DEFAULT_USER,
        "date": DAY,
        "annotator": annotator,
        "annotated_at": "2026-10-17",
        "must_include": [
            {"rank": r, "section": "actions", "item_ids": [ids[r]], "why": "relance client"}
            for r in range(1, 4)
        ],
        "cases": cases,
    }


def test_paires_sans_ordre_ni_doublon() -> None:
    assert pairs([["b", "a", "c"], ["d"]]) == {("a", "b"), ("a", "c"), ("b", "c")}
    assert pairs([["a", "b"], ["b", "a"]]) == {("a", "b")}


def test_score_vide_non_defini() -> None:
    score = PairScore(0, 0, 0)
    assert (score.precision, score.recall, score.f1) == (None, None, None)
    assert not score.passes()


def test_seuils_l2_1() -> None:
    assert PairScore(predicted=10, expected=10, common=9).passes()
    assert not PairScore(predicted=10, expected=12, common=9).passes()  # rappel 0,75
    assert not PairScore(predicted=11, expected=9, common=9).passes()  # précision 0,82
    assert PairScore(4, 5, 3) + PairScore(6, 5, 6) == PairScore(10, 10, 9)


def test_regroupement_identique_a_l_annotation(items: list, predicted: list) -> None:
    assert predicted, "la journée de démo doit contenir des affaires de plusieurs items"
    score = score_day(Annotation.model_validate(_annotation(items, predicted)), items)
    assert (score.precision, score.recall) == (1.0, 1.0)


def test_affaire_oubliee_ou_fusion_abusive(items: list, predicted: list) -> None:
    missed = score_day(Annotation.model_validate(_annotation(items, predicted[1:])), items)
    assert missed.precision < 1 and missed.recall == 1
    merged = [predicted[0] + predicted[1], *predicted[2:]]
    over = score_day(Annotation.model_validate(_annotation(items, merged)), items)
    assert over.precision == 1 and over.recall < 1


def test_affaires_non_annotees_refusees(items: list) -> None:
    with pytest.raises(ValueError, match="non annotées"):
        score_day(Annotation.model_validate(_annotation(items, None)), items)


def test_b_cubed() -> None:
    ids = ["a", "b", "c", "d"]
    assert b_cubed([["a", "b"]], [["a", "b"]], ids) == (1.0, 1.0)
    precision, recall = b_cubed([["a", "b", "c", "d"]], [], ids)
    assert (precision, recall) == (0.25, 1.0)


def test_accord_symetrique(items: list, predicted: list) -> None:
    a = Annotation.model_validate(_annotation(items, predicted))
    b = Annotation.model_validate(_annotation(items, predicted[1:], annotator="bob"))
    assert agreement(a, a) == 1.0
    assert agreement(a, b) == agreement(b, a) < 1


def test_commande_regroupement_sans_verite_terrain(
    tmp_path: Path, items: list, predicted: list, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / f"{DAY}.alice.json").write_text(
        json.dumps(_annotation(items, predicted)), encoding="utf-8"
    )
    (tmp_path / f"{DAY}.bob.json").write_text(
        json.dumps(_annotation(items, None, annotator="bob")), encoding="utf-8"
    )
    opened: list[str] = []
    real_open, real_read = Path.open, Path.read_text
    monkeypatch.setattr(
        Path, "open", lambda p, *a, **k: opened.append(p.name) or real_open(p, *a, **k)
    )
    monkeypatch.setattr(
        Path, "read_text", lambda p, *a, **k: opened.append(p.name) or real_read(p, *a, **k)
    )
    lines, total = annotate.grouping(tmp_path)
    assert "truth.json" not in opened
    assert len(lines) == 1 and lines[0].startswith(f"{DAY}.alice.json")
    assert total.passes()
    assert annotate.main(["regroupement", "--annotations", str(tmp_path)]) == 0
