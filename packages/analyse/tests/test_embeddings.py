"""Embeddings (L2.6) : faux modèle, texte vectorisé, cache, similarités, vrai modèle si installé."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest

from packages.analyse.rag.embeddings import (
    FastEmbedder,
    HashEmbedder,
    embed_items,
    item_text,
    similarities,
)
from packages.analyse.tests.factories import OTHER_USER, make_mail

MORNING = "2026-10-07T08:00:00+02:00"


class SpyEmbedder(HashEmbedder):
    """HashEmbedder qui note les textes qu'on lui demande de calculer."""

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[list[str]] = []

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        self.calls.append(list(texts))
        return super().embed(texts)


# --- faux modèle ------------------------------------------------------------------------------


def test_hash_embedder_is_deterministic_and_normalized() -> None:
    texts = ["Devis carrosserie Kia Niro", "Congés de fin d'année"]

    first, second = HashEmbedder().embed(texts), HashEmbedder().embed(texts)

    assert np.array_equal(first, second)
    assert np.allclose(np.linalg.norm(first, axis=1), 1.0)


def test_texts_sharing_words_are_closer() -> None:
    vectors = HashEmbedder().embed(
        [
            "devis carrosserie du Kia Niro",
            "relance devis carrosserie Kia Niro",
            "newsletter barème kilométrique",
        ]
    )

    scores = similarities(vectors)

    assert scores[0, 1] > scores[0, 2]
    assert np.allclose(np.diag(scores), 1.0)
    assert np.allclose(scores, scores.T)


def test_an_empty_text_gives_a_null_vector() -> None:
    assert not HashEmbedder().embed([""]).any()


# --- texte vectorisé ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("title", "snippet", "expected"),
    [
        ("RE: TR : Devis 2026-0412", "Bonjour", "Devis 2026-0412\nBonjour"),
        ("Fwd: re:Point flash", "", "Point flash"),
        ("", "Merci de valider", "Merci de valider"),
        ("", "", ""),
    ],
)
def test_item_text_is_the_title_without_reply_prefixes_then_the_snippet(
    title: str, snippet: str, expected: str
) -> None:
    assert item_text(make_mail("m1", MORNING, title=title, snippet=snippet)) == expected


# --- cache --------------------------------------------------------------------------------------


def test_cached_items_are_not_computed_again() -> None:
    spy, cache = SpyEmbedder(), {}
    items = [make_mail("a", MORNING, title="Devis"), make_mail("b", MORNING, title="Congés")]

    first = embed_items(items, spy, cache)
    second = embed_items([*items, make_mail("c", MORNING, title="Facture")], spy, cache)

    assert spy.calls == [["Devis", "Congés"], ["Facture"]]
    assert np.array_equal(first["mail:a"], second["mail:a"])
    assert set(second) == {"mail:a", "mail:b", "mail:c"}


def test_an_updated_item_is_computed_again() -> None:
    spy, cache = SpyEmbedder(), {}
    embed_items([make_mail("a", MORNING, title="Devis")], spy, cache)

    # make_mail aligne updated_at sur la date donnée : même item_id, mis à jour plus tard.
    updated = make_mail("a", "2026-10-07T09:00:00+02:00", title="Devis validé")
    embed_items([updated], spy, cache)

    assert spy.calls == [["Devis"], ["Devis validé"]]


def test_the_cache_is_separated_by_user() -> None:
    spy, cache = SpyEmbedder(), {}
    embed_items([make_mail("a", MORNING, title="Devis")], spy, cache)

    embed_items([make_mail("a", MORNING, title="Devis", owner_user_id=OTHER_USER)], spy, cache)

    assert len(spy.calls) == 2
    assert {key[1] for key in cache} == {"u_demo_001", OTHER_USER}


def test_nothing_to_compute_does_not_call_the_model() -> None:
    spy = SpyEmbedder()

    assert embed_items([], spy) == {}
    assert spy.calls == []


# --- vrai modèle (seulement si le groupe « embeddings » est installé) -------------------------


def test_real_model_puts_a_french_paraphrase_closer_than_an_unrelated_text() -> None:
    pytest.importorskip("fastembed")
    vectors = FastEmbedder().embed(
        [
            "Devis de réparation de la carrosserie du Kia Niro",
            "Estimation du coût des travaux sur le Kia Niro",
            "Newsletter : nouveau barème des frais kilométriques",
        ]
    )

    scores = similarities(vectors)

    assert vectors.shape == (3, 384)
    assert scores[0, 1] > scores[0, 2]


def test_real_model_accepts_an_empty_list() -> None:
    assert FastEmbedder().embed([]).shape == (0, 0)
