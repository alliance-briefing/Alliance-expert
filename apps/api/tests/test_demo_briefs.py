"""Briefs de démonstration tirés des annotations (DEMO_BRIEFS=annotations)."""

import json
from pathlib import Path

from alliance_api.demo_briefs import annotated_generator, brief_from_annotation

from apps.api.tests.factories import DAY, USER_A, make_item
from eval.annotation import Annotation
from packages.analyse import generate_brief

ITEMS = [make_item(USER_A, n, thread_id=f"conv:{n}", title=f"Sujet {n}") for n in (1, 2, 3)]


def _annotation(order: list[tuple[str, int]]) -> Annotation:
    """Annotation dont le rang r cite l'item n dans la section donnée."""
    return Annotation.model_validate(
        {
            "schema": "annotation/1.1",
            "user_id": USER_A,
            "date": DAY,
            "annotator": "testeur",
            "annotated_at": DAY,
            "must_include": [
                {"rank": r, "section": s, "item_ids": [f"mail:{USER_A}-{n}"], "why": f"raison {r}"}
                for r, (s, n) in enumerate(order, start=1)
            ],
        }
    )


def test_l_ordre_de_l_annotation_devient_l_ordre_des_scores() -> None:
    brief = brief_from_annotation(
        _annotation([("meetings", 3), ("actions", 1), ("actions", 2)]), ITEMS
    )
    entries = sorted(
        (e for s in brief.sections for e in s.entries), key=lambda e: -e.priority_score
    )
    assert [e.source_item_ids[0] for e in entries] == [f"mail:{USER_A}-{n}" for n in (3, 1, 2)]
    assert [e.priority_reasons for e in entries] == [["raison 1"], ["raison 2"], ["raison 3"]]
    assert brief.generator.llm == "annotation"


def test_un_item_absent_de_la_journee_n_est_pas_cite() -> None:
    brief = brief_from_annotation(
        _annotation([("actions", 1), ("actions", 2), ("actions", 3)]), ITEMS[:2]
    )
    assert brief.cited_item_ids() == {f"mail:{USER_A}-1", f"mail:{USER_A}-2"}


def test_jour_non_annote_repli_sur_le_lot_2(tmp_path: Path) -> None:
    generate = annotated_generator(generate_brief, "testeur", root=tmp_path)
    assert generate(ITEMS, USER_A, DAY).generator.llm == "rules"

    data = _annotation([("actions", 1), ("actions", 2), ("actions", 3)]).model_dump(
        mode="json", by_alias=True
    )
    (tmp_path / f"{DAY}.testeur.json").write_text(json.dumps(data), encoding="utf-8")
    assert generate(ITEMS, USER_A, DAY).generator.llm == "annotation"
