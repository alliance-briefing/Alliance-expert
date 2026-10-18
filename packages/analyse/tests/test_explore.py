"""Outil d'exploration des embeddings (L2.6), avec le faux modèle : sans téléchargement."""

from __future__ import annotations

import numpy as np
import pytest

from packages.analyse.aggregate.cases import group_into_cases
from packages.analyse.rag.embeddings import FastEmbedder, HashEmbedder
from packages.analyse.rag.explore import (
    Model2VecEmbedder,
    closest_unlinked,
    main,
    make_embedder,
    split_scores,
    summary,
    title_text,
)
from packages.analyse.tests.factories import make_mail

MORNING = "2026-10-07T08:00:00+02:00"
MANAGER = {"name": "Manager", "email": "manager@agence.test", "role": "to"}


def person(name: str) -> dict:
    return {"name": name, "email": f"{name}@garage.test", "role": "from"}


def vectors_of(items: list) -> np.ndarray:
    return HashEmbedder().embed([item.title for item in items])


# --- paires ---------------------------------------------------------------------------------------


def test_pairs_of_the_same_case_are_never_proposed() -> None:
    items = [
        make_mail("a", MORNING, title="Devis carrosserie", thread_id="c1"),
        make_mail("b", MORNING, title="Devis carrosserie", thread_id="c1"),
        make_mail("c", MORNING, title="Devis carrosserie urgent"),
    ]

    pairs = closest_unlinked(items, vectors_of(items), top=10)

    assert [(p.first.item_id, p.second.item_id) for p in pairs] == [
        ("mail:a", "mail:c"),
        ("mail:b", "mail:c"),
    ]


def test_pairs_are_sorted_by_similarity_then_by_id() -> None:
    items = [
        make_mail("x", MORNING, title="congés"),
        make_mail("a", MORNING, title="devis carrosserie kia"),
        make_mail("b", MORNING, title="devis carrosserie kia niro"),
    ]

    pairs = closest_unlinked(items, vectors_of(items), top=1)

    assert (pairs[0].first.item_id, pairs[0].second.item_id) == ("mail:a", "mail:b")


def test_shared_participant_ignores_the_manager() -> None:
    items = [
        make_mail("a", MORNING, title="Devis", participants=[MANAGER, person("garage")]),
        make_mail("b", MORNING, title="Devis", participants=[MANAGER, person("garage")]),
        make_mail("c", MORNING, title="Devis", participants=[MANAGER, person("assureur")]),
    ]

    pairs = {
        (p.first.item_id, p.second.item_id): p
        for p in closest_unlinked(items, vectors_of(items), top=10)
    }

    assert pairs[("mail:a", "mail:b")].shared_participant is True
    assert pairs[("mail:a", "mail:c")].shared_participant is False


def test_days_apart_and_different_references_are_reported() -> None:
    items = [
        make_mail("a", "2026-10-02T08:00:00+02:00", title="Dossier 2026-5394 : RAS"),
        make_mail("b", MORNING, title="Dossier 2026-4752 : RAS"),
    ]

    (pair,) = closest_unlinked(items, vectors_of(items), top=5)

    assert pair.days_apart == 5
    assert pair.different_references is True


# --- distribution ---------------------------------------------------------------------------------


def test_scores_are_split_between_linked_and_other_pairs() -> None:
    items = sorted(
        [
            make_mail("a", MORNING, title="Devis", thread_id="c1"),
            make_mail("b", MORNING, title="Devis", thread_id="c1"),
            make_mail("c", MORNING, title="Congés"),
        ],
        key=lambda item: item.item_id,
    )

    linked, others, _, _ = split_scores(vectors_of(items), group_into_cases(items), items)

    assert linked.size == 1 and others.size == 2
    assert linked[0] == pytest.approx(1.0)


def test_summary() -> None:
    assert summary(np.array([])) == "aucune paire"
    assert summary(np.array([0.2, 0.4])).startswith("n=2, min 0.20")


# --- modèles et texte -------------------------------------------------------------------------------


def test_make_embedder() -> None:
    assert isinstance(make_embedder("hash"), HashEmbedder)
    assert isinstance(make_embedder("model2vec"), Model2VecEmbedder)
    assert isinstance(make_embedder("intfloat/multilingual-e5-large"), FastEmbedder)


def test_title_text_drops_the_snippet() -> None:
    mail = make_mail("a", MORNING, title="RE: Devis", snippet="Bonjour")

    assert title_text(mail) == "Devis"


def test_model2vec_when_installed() -> None:
    pytest.importorskip("model2vec")
    vectors = Model2VecEmbedder().embed(["Devis carrosserie", "Congés"])

    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0)


# --- commande ----------------------------------------------------------------------------------------


def test_command_runs_on_a_demo_day(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["2026-10-07", "--modele", "hash", "--top", "3"]) == 0

    out = capsys.readouterr().out
    assert "# Embeddings — 2026-10-07 (u_demo_001)" in out
    assert "paires déjà reliées (même affaire)" in out
    assert "3 paires les plus proches non reliées" in out
