"""Index RAG par utilisateur (L2.6, R1) : isolation, pas de fuite du futur, suppression."""

from __future__ import annotations

from datetime import date

import numpy as np
import pytest

from contracts.demo_data import available_days, load_items
from packages.analyse.rag.embeddings import HashEmbedder
from packages.analyse.rag.index import IndexStore, UserIndex
from packages.analyse.tests.factories import OTHER_USER, USER, make_mail

DAY = date(2026, 10, 7)


def past_mail(native_id: str, title: str, when: str = "2026-10-05T09:00:00+02:00", **fields):
    return make_mail(native_id, when, title=title, **fields)


def ids(hits: list) -> list[str]:
    return [hit.item.item_id for hit in hits]


@pytest.fixture
def store() -> IndexStore:
    return IndexStore(HashEmbedder())


# --- recherche ------------------------------------------------------------------------------------


def test_the_closest_past_item_comes_first(store: IndexStore) -> None:
    index = store.for_user(USER)
    index.add(
        [
            past_mail("devis", "devis carrosserie kia niro"),
            past_mail("conges", "planning des congés de fin d'année"),
        ]
    )
    query = make_mail("q", "2026-10-07T08:00:00+02:00", title="relance devis carrosserie kia")

    (hits,) = index.search_items([query], DAY)

    assert ids(hits) == ["mail:devis", "mail:conges"]
    assert hits[0].score > hits[1].score


def test_k_and_ties_are_deterministic(store: IndexStore) -> None:
    index = store.for_user(USER)
    index.add([past_mail(n, "même titre") for n in ("c", "a", "b", "d")])
    query = make_mail("q", "2026-10-07T08:00:00+02:00", title="même titre")

    (hits,) = index.search_items([query], DAY, k=3)

    assert ids(hits) == ["mail:a", "mail:b", "mail:c"]


def test_excluded_items_are_never_returned(store: IndexStore) -> None:
    index = store.for_user(USER)
    index.add([past_mail("a", "devis"), past_mail("b", "devis")])
    query = make_mail("q", "2026-10-07T08:00:00+02:00", title="devis")

    (hits,) = index.search_items([query], DAY, exclude={"mail:a"})

    assert ids(hits) == ["mail:b"]


def test_an_empty_index_or_query_returns_nothing(store: IndexStore) -> None:
    index = store.for_user(USER)

    assert index.search(np.ones((2, 256)), DAY) == [[], []]
    assert index.search_items([], DAY) == []


# --- pas de fuite du futur ----------------------------------------------------------------------


def test_only_items_created_before_the_brief_day_are_returned(store: IndexStore) -> None:
    index = store.for_user(USER)
    index.add(
        [
            past_mail("veille", "devis", "2026-10-06T23:59:00+02:00"),
            past_mail("jour-meme", "devis", "2026-10-07T07:00:00+02:00"),
            past_mail("lendemain", "devis", "2026-10-08T09:00:00+02:00"),
            # 22:30 UTC le 06/10 = 00:30 le 07/10 à Paris : déjà le jour du brief.
            past_mail("minuit-utc", "devis", "2026-10-06T22:30:00+00:00"),
        ]
    )
    query = make_mail("q", "2026-10-07T08:00:00+02:00", title="devis")

    (hits,) = index.search_items([query], DAY, k=10)

    assert ids(hits) == ["mail:veille"]


def test_history_is_limited_to_the_last_days(store: IndexStore) -> None:
    index = store.for_user(USER)
    index.add(
        [
            past_mail("j-30", "devis", "2026-09-07T09:00:00+02:00"),
            past_mail("j-31", "devis", "2026-09-06T09:00:00+02:00"),
        ]
    )
    query = make_mail("q", "2026-10-07T08:00:00+02:00", title="devis")

    (hits,) = index.search_items([query], DAY, k=10)

    assert ids(hits) == ["mail:j-30"]


# --- isolation entre utilisateurs ---------------------------------------------------------------


def test_an_index_refuses_items_of_another_user(store: IndexStore) -> None:
    index = store.for_user(USER)

    with pytest.raises(ValueError, match="autres utilisateurs"):
        index.add([past_mail("a", "devis"), past_mail("b", "devis", owner_user_id=OTHER_USER)])

    assert len(index) == 0  # rien n'est ajouté : tout ou rien


def test_an_index_refuses_a_query_of_another_user(store: IndexStore) -> None:
    index = store.for_user(USER)
    index.add([past_mail("a", "devis")])
    foreign = make_mail("q", "2026-10-07T08:00:00+02:00", title="devis", owner_user_id=OTHER_USER)

    with pytest.raises(ValueError, match="autre utilisateur"):
        index.search_items([foreign], DAY)


def test_a_query_of_user_a_never_returns_an_item_of_user_b(store: IndexStore) -> None:
    store.for_user(USER).add([past_mail("a", "réunion budget")])
    store.for_user(OTHER_USER).add(
        [past_mail("b", "devis carrosserie kia niro", owner_user_id=OTHER_USER)]
    )
    query = make_mail("q", "2026-10-07T08:00:00+02:00", title="devis carrosserie kia niro")

    (hits,) = store.for_user(USER).search_items([query], DAY, k=10)

    assert ids(hits) == ["mail:a"]
    assert all(hit.item.owner_user_id == USER for hit in hits)


# --- mises à jour et suppression ---------------------------------------------------------------


def test_a_newer_version_replaces_the_item_and_an_older_one_is_ignored(store: IndexStore) -> None:
    index = store.for_user(USER)
    index.add([past_mail("a", "devis", "2026-10-05T09:00:00+02:00")])
    index.add([past_mail("a", "devis validé", "2026-10-05T10:00:00+02:00")])
    index.add([past_mail("a", "ancienne version", "2026-10-05T08:00:00+02:00")])
    query = make_mail("q", "2026-10-07T08:00:00+02:00", title="devis validé")

    (hits,) = index.search_items([query], DAY, k=10)

    assert len(index) == 1
    assert [hit.item.title for hit in hits] == ["devis validé"]


def test_delete_user_removes_the_index_and_the_cached_vectors(store: IndexStore) -> None:
    store.for_user(USER).add([past_mail("a", "devis")])
    store.for_user(OTHER_USER).add([past_mail("b", "devis", owner_user_id=OTHER_USER)])

    store.delete_user(USER)

    assert store.users() == [OTHER_USER]
    assert len(store.for_user(USER)) == 0
    assert {key[1] for key in store._cache} == {OTHER_USER}


def test_users_share_the_model_but_not_the_data() -> None:
    store = IndexStore(HashEmbedder())

    assert isinstance(store.for_user(USER), UserIndex)
    assert store.for_user(USER) is store.for_user(USER)
    assert store.for_user(USER) is not store.for_user(OTHER_USER)


# --- données de démo ------------------------------------------------------------------------------


def test_demo_history_finds_past_items_for_today() -> None:
    days = available_days()
    index = IndexStore(HashEmbedder()).for_user(USER)
    for day in days[:5]:
        index.add(load_items(day))
    today = date.fromisoformat(days[5])
    queries = load_items(days[5])[:10]

    results = index.search_items(queries, today, exclude={item.item_id for item in queries})

    assert len(results) == 10
    for hits in results:
        assert len(hits) == 3
        assert all(hit.item.created_at.date() < today for hit in hits)
