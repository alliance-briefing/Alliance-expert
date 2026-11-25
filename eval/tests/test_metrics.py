import math
from datetime import datetime

import pytest

from contracts.models import Brief
from eval.annotation import Annotation
from eval.metrics import bootstrap_ci, injections_cited, match, ndcg, reading_order, score_day

DAY = "2026-10-13"
USER = "u_demo_001"


def _entry(rank: int, score: int, *item_ids: str) -> dict:
    return {
        "rank": rank,
        "priority_score": score,
        "priority_reasons": ["raison"],
        "text": "texte",
        "source_item_ids": list(item_ids),
    }


def _brief(sections: dict[str, list[list[str]]]) -> Brief:
    """Brief minimal : {section: [items de l'entrée 1, items de l'entrée 2, …]}."""
    return Brief.model_validate(
        {
            "brief_id": f"b_{DAY}_{USER}",
            "owner_user_id": USER,
            "date": DAY,
            "generated_at": datetime.fromisoformat(f"{DAY}T08:30:00+02:00"),
            "generator": {"llm": "rules"},
            "sections": [
                {
                    "kind": kind,
                    "entries": [_entry(r, 90 - r, *ids) for r, ids in enumerate(entries, start=1)],
                }
                for kind, entries in sections.items()
            ],
            "stats": {
                "items_in": 50,
                "items_out": 10,
                "tokens_in": 0,
                "tokens_out": 0,
                "latency_ms": 5,
                "cost_eur": 0,
            },
        }
    )


def _annotation(expected: list[list[str]], exclude: list[str] | None = None) -> Annotation:
    return Annotation.model_validate(
        {
            "schema": "annotation/1.1",
            "user_id": USER,
            "date": DAY,
            "annotator": "alice",
            "annotated_at": "2026-11-01",
            "must_include": [
                {"rank": r, "section": "actions", "item_ids": ids, "why": "important"}
                for r, ids in enumerate(expected, start=1)
            ],
            "must_exclude": exclude or [],
        }
    )


def test_ordre_de_lecture_suit_les_sections_du_contrat_puis_le_rang() -> None:
    brief = _brief({"actions": [["mail:a"], ["mail:b"]], "overdue": [["task:c"]]})
    assert [e.source_item_ids[0] for e in reading_order(brief)] == ["mail:a", "mail:b", "task:c"]


def test_un_element_n_est_credite_qu_une_fois() -> None:
    annotation = _annotation([["mail:a", "file:a"], ["mail:b"], ["mail:c"]])
    brief = _brief({"actions": [["mail:a"], ["file:a"], ["mail:b"]]})
    found = match(reading_order(brief), annotation.must_include)
    assert [hit.rank if hit else None for hit in found] == [1, None, 2]


def test_une_entree_qui_touche_deux_elements_prend_le_mieux_classe() -> None:
    annotation = _annotation([["mail:a"], ["mail:b"], ["mail:c"]])
    brief = _brief({"actions": [["mail:b", "mail:a"], ["mail:b"]]})
    found = match(reading_order(brief), annotation.must_include)
    assert [hit.rank if hit else None for hit in found] == [1, 2]


def test_brief_parfait() -> None:
    annotation = _annotation([["mail:a"], ["mail:b"], ["mail:c"]], exclude=["mail:z"])
    score = score_day(_brief({"actions": [["mail:a"], ["mail:b"], ["mail:c"]]}), annotation)
    assert (score.recall, score.precision, score.ndcg) == (1.0, 1.0, 1.0)
    assert score.faults == () and score.missed == ()


def test_ordre_inverse_baisse_le_ndcg_sans_toucher_rappel_ni_precision() -> None:
    annotation = _annotation([["mail:a"], ["mail:b"], ["mail:c"]])
    score = score_day(_brief({"actions": [["mail:c"], ["mail:b"], ["mail:a"]]}), annotation)
    assert (score.recall, score.precision) == (1.0, 1.0)
    expected = (1 + 2 / math.log2(3) + 3 / 2) / (3 + 2 / math.log2(3) + 1 / 2)
    assert score.ndcg == pytest.approx(expected)


def test_element_au_dela_du_top_7_est_manque_mais_place() -> None:
    annotation = _annotation([["mail:a"], ["mail:b"], ["mail:c"]])
    filler = [[f"mail:x{i}"] for i in range(7)]
    score = score_day(
        _brief({"actions": [["mail:a"], ["mail:b"], *filler, ["mail:c"]]}), annotation
    )
    assert score.recall == pytest.approx(2 / 3)
    assert score.precision == pytest.approx(2 / 7)
    assert [e.rank for e in score.missed] == [3]
    assert (3, 10) in score.placed


def test_fautes_comptees_et_reperees_dans_le_top_3() -> None:
    annotation = _annotation([["mail:a"], ["mail:b"], ["mail:c"]], exclude=["mail:z", "mail:y"])
    brief = _brief({"actions": [["mail:z"], ["mail:a"], ["mail:b"], ["mail:y"]]})
    score = score_day(brief, annotation)
    assert score.faults == ("mail:z", "mail:y")
    assert score.faults_top == 1


def test_brief_vide() -> None:
    score = score_day(_brief({}), _annotation([["mail:a"], ["mail:b"], ["mail:c"]]))
    assert (score.recall, score.precision, score.ndcg, score.entries) == (0.0, 0.0, 0.0, 0)


def test_ndcg_avec_moins_de_5_elements_de_reference() -> None:
    annotation = _annotation([["mail:a"], ["mail:b"], ["mail:c"]])
    found = match(
        reading_order(_brief({"actions": [["mail:a"], ["mail:b"], ["mail:c"]]})),
        annotation.must_include,
    )
    assert ndcg(found, 3) == pytest.approx(1.0)


def test_injection_citee() -> None:
    brief = _brief({"actions": [["mail:inj", "mail:a"]]})
    assert injections_cited(brief, ["mail:inj", "mail:autre"]) == ["mail:inj"]


def test_bootstrap_reproductible_et_encadre_la_moyenne() -> None:
    values = [0.1, 0.4, 0.5, 0.9, 0.3, 0.7]
    low, high = bootstrap_ci(values, seed=0)
    assert bootstrap_ci(values, seed=0) == (low, high)
    assert low <= sum(values) / len(values) <= high
    assert bootstrap_ci([0.5] * 5, seed=1) == (0.5, 0.5)
