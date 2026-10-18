"""L1.6 : tout objet sort valide ou explicitement rejeté, jamais modifié en silence."""

from __future__ import annotations

import copy
import random
from datetime import timedelta

from contracts.models import Item
from packages.collect.core.normalize import Rejection, deduplicate, normalize, normalize_all

VALID = {
    "schema_version": "1.1",
    "item_id": "mail:AAMk-1",
    "source": "mail",
    "provider": "microsoft",
    "owner_user_id": "u_demo_001",
    "title": "Devis sinistre 2026-0412",
    "snippet": "Merci de valider le devis avant demain 12h.",
    "created_at": "2026-10-21T08:14:03+00:00",
    "updated_at": "2026-10-21T08:24:33+00:00",
    "participants": [
        {"name": "Julie Marchand", "email": "julie@horizon-sud.example", "role": "from"}
    ],
    "thread_id": "conv-1",
    "native_importance": "high",
    "is_read": False,
    "url": "https://outlook.office365.com/owa/?ItemID=1",
    "acl": {"visible_to": ["u_demo_001"]},
    "collected_at": "2026-10-22T06:30:00+00:00",
    "connector": {"name": "mail-graph", "version": "0.1.0"},
}


def identity(raw: dict) -> dict:
    return raw


def test_valid_object_becomes_an_item() -> None:
    assert isinstance(normalize(VALID, identity, connector="mail-graph"), Item)


def test_too_long_snippet_is_rejected_not_truncated() -> None:
    outcome = normalize(VALID | {"snippet": "x" * 501}, identity, connector="mail-graph")
    assert isinstance(outcome, Rejection) and "snippet" in outcome.reason


def test_translation_crash_is_a_rejection() -> None:
    outcome = normalize({}, lambda raw: raw["subject"], connector="mail-graph")
    assert outcome == Rejection("mail-graph", "traduction impossible (KeyError)")


def test_rejection_never_contains_the_data() -> None:
    bad = VALID | {"participants": [{"name": "X", "email": "secret@perso", "role": "boss"}]}
    outcome = normalize(bad, identity, connector="mail-graph")
    assert isinstance(outcome, Rejection)
    assert "secret" not in outcome.reason and "@" not in outcome.reason


def test_item_built_without_validation_is_checked_again() -> None:
    sneaky = Item.model_construct(**(VALID | {"snippet": "x" * 600}))
    assert isinstance(normalize({}, lambda _: sneaky, connector="x"), Rejection)


def test_normalize_all_splits_items_and_rejections() -> None:
    result = normalize_all([VALID, VALID | {"url": "http://x.example"}], identity, connector="m")
    assert len(result.items) == 1 and len(result.rejections) == 1


def test_deduplicate_keeps_the_latest_version() -> None:
    old = Item.model_validate(VALID)
    new = old.model_copy(update={"updated_at": old.updated_at + timedelta(hours=1), "title": "v2"})
    other = Item.model_validate(VALID | {"item_id": "mail:AAMk-2"})
    assert [i.title for i in deduplicate([old, other, new])] == ["v2", VALID["title"]]


# --- Test de propriété (guide L1.6) : 1 000 objets fournisseur aléatoires --------------------


def _mutate(rng: random.Random) -> dict:
    data = copy.deepcopy(VALID)
    for _ in range(rng.randint(0, 3)):
        key = rng.choice(list(data))
        match rng.randint(0, 7):
            case 0:
                data.pop(key, None)  # champ manquant
            case 1:
                data[key] = None  # champ obligatoire à null
            case 2:
                data[key] = rng.choice([42, 3.5, True, [], {}])  # mauvais type
            case 3:
                data["snippet"] = "é" * rng.randint(0, 800)  # extrait trop long ou non
            case 4:
                data["url"] = rng.choice(
                    ["http://x.example", "pas-une-url", "https://ok.example/1"]
                )
            case 5:
                data["created_at"] = rng.choice(
                    ["2026-10-21T08:00:00", "2026-10-21T08:00:00+02:00"]
                )
            case 6:
                data[f"inconnu_{rng.randint(0, 9)}"] = "x"  # champ hors contrat
            case 7:
                data["title"] = "".join(
                    chr(rng.randint(0, 0x2FF)) for _ in range(rng.randint(0, 30))
                )
    return data


def test_property_1000_random_objects() -> None:
    rng = random.Random(2026)
    kinds = {"item": 0, "rejet": 0}
    for _ in range(1000):
        raw = _mutate(rng)
        outcome = normalize(raw, identity, connector="fuzz")
        if isinstance(outcome, Item):
            kinds["item"] += 1
            # jamais modifié en silence : les textes ressortent à l'identique
            for key in ("item_id", "title", "snippet", "owner_user_id"):
                assert getattr(outcome, key) == raw[key]
            assert Item.model_validate(outcome.model_dump()) == outcome
        else:
            assert isinstance(outcome, Rejection) and outcome.reason
            kinds["rejet"] += 1
    assert kinds["item"] > 50 and kinds["rejet"] > 50  # les deux cas sont réellement testés
