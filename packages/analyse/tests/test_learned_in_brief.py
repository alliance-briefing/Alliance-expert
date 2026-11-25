"""Score appris des affaires dans le brief (scoring/learned.py, M8) : meilleur élément, quatre
sources, raisons, fenêtres du brief, repli, isolation. Fausse compréhension N : pas de
téléchargement, probabilités lues sur des mots du texte ; vrais poids de priorité."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date

import numpy as np
import pytest

from packages.analyse import generate_brief
from packages.analyse.aggregate.cases import group_into_cases
from packages.analyse.brief import _mails_wording
from packages.analyse.scoring.learned import (
    GENERATOR,
    REASONS,
    SCORE_ENV,
    LearnedScorer,
    learned_case_scores,
    on_brief,
    persona_for,
    viewpoint_for,
)
from packages.analyse.scoring.ml import comprendre
from packages.analyse.scoring.ml.persona import DEMO_CAMILLE
from packages.analyse.scoring.score import score_case
from packages.analyse.tests.factories import (
    COLLECTED_AT,
    OTHER_USER,
    USER,
    at,
    make_item,
    make_mail,
    make_task,
)

DAY = date(2026, 10, 7)
NOW = at(COLLECTED_AT)
N1 = "bernard.mercier@reseau-expertis.test"  # N+1 de Camille (DEMO_CAMILLE)
COLLEAGUE = "sophie.nguyen@reseau-expertis.test"
CAMILLE = DEMO_CAMILLE.address
WORDS = ("merci de", "?", "panne")  # colonnes du faux vecteur


class SpyEmbedder:
    """Un vecteur de 0 et de 1 : le texte contient-il chacun des WORDS ?"""

    name = "espion"

    def __init__(self, fail: bool = False) -> None:
        self.texts: list[str] = []
        self.fail = fail

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        if self.fail:
            raise RuntimeError("modèle introuvable")
        self.texts += texts
        return np.array([[float(w in t.lower()) for w in WORDS] for t in texts])


class FakeModel:
    """Probabilités d'une cible lues sur le faux vecteur (les 3 premières colonnes)."""

    def __init__(self, target: str) -> None:
        self.classes = comprendre.TARGETS[target]
        self.target = target

    def proba(self, X: np.ndarray) -> np.ndarray:
        out = np.full((len(X), len(self.classes)), 0.0)
        for n, (ask, question, panne) in enumerate(X[:, :3]):
            if self.target == "action":
                pick = "decider" if ask else "repondre" if question else "informer"
                out[n, self.classes.index(pick)] = 0.9
                out[n, self.classes.index("rien")] = 0.1
            elif self.target == "urgence":
                out[n, self.classes.index("normal")] = 1.0
            else:  # alerte : 0,4 pour une panne, assez pour compter, pas pour une raison
                p = 0.4 if panne else 0.01
                out[n] = (1 - p, p)
        return out


class FakeUnderstanding:
    fiche, texte, embedder_name = "hybride", "normalise", "faux"

    def __init__(self) -> None:
        self.models = {t: FakeModel(t) for t in comprendre.TARGETS}


def scorer(embedder: SpyEmbedder | None = None) -> LearnedScorer:
    return LearnedScorer(embedder or SpyEmbedder(), FakeUnderstanding())  # type: ignore[arg-type]


def mail(native_id: str, title: str, sender: str = COLLEAGUE, **fields):
    people = [
        {"name": "Expéditeur", "email": sender, "role": "from"},
        {"name": "Camille Laurent", "email": CAMILLE, "role": "to"},
    ]
    fields.setdefault("participants", people)
    return make_mail(
        native_id, fields.pop("created_at", "2026-10-06T15:00:00+02:00"), title=title, **fields
    )


def scores(items, **kwargs):
    return scorer(**kwargs).score_cases(group_into_cases(items), items, USER, NOW)


# --- score d'une affaire -----------------------------------------------------------------------


def test_case_takes_the_score_of_its_best_element() -> None:
    ask = mail("m1", "Dossier 2026-1111 : merci de valider le rapport", sender=N1)
    info = mail(
        "m2", "Dossier 2026-1111 : pour information", created_at="2026-10-06T16:00:00+02:00"
    )
    alone = {i.item_id: s for i in (ask, info) for s in scores([i]).values()}

    (case,) = scores([ask, info]).values()

    assert case.best_item_id == ask.item_id
    assert case.score == max(s.score for s in alone.values()) > alone[info.item_id].score
    assert "décision ou validation attendue" in case.reasons


def test_reasons_are_readable_and_skip_improbable_predictions() -> None:
    (case,) = scores([mail("m1", "Panne du serveur, pouvez-vous regarder ?", sender=N1)]).values()

    assert case.reasons and set(case.reasons) <= set(REASONS.values())
    assert "réponse attendue" in case.reasons
    assert "alerte (sécurité, incident, injection)" not in case.reasons  # probabilité 0,4


def test_tasks_and_meetings_are_scored_too() -> None:
    late = make_task("t1", "2026-10-05T12:00:00+02:00", title="Valider le devis 2026-2222")
    meeting = make_item(
        "calendar",
        "c1",
        title="Point budget",
        start_at="2026-10-07T14:00:00+02:00",
        end_at="2026-10-07T15:00:00+02:00",
    )

    (task_case,) = scores([late]).values()
    (meeting_case,) = scores([meeting]).values()

    assert "tâche en retard" in task_case.reasons and task_case.best_item_id == late.item_id
    assert meeting_case.best_item_id == meeting.item_id


@pytest.mark.parametrize(
    "item",
    [
        make_task("t1", "2026-10-09T12:00:00+02:00"),
        make_task("t2", "2026-10-05T12:00:00+02:00", completed=True),
        mail("m1", "Ancien échange", is_read=True, created_at="2026-09-20T10:00:00+02:00"),
        mail("m2", "Après le brief", created_at="2026-10-07T10:00:00+02:00"),
    ],
    ids=["tâche dans 2 jours", "tâche terminée", "mail lu ancien", "mail futur"],
)
def test_case_without_an_element_on_the_brief_keeps_the_rule_score(item) -> None:
    assert not on_brief(item, NOW)
    assert scores([item]) == {}


def test_scores_are_deterministic() -> None:
    items = [mail("m1", "Merci de valider ?"), mail("m2", "Point d'étape", sender=N1)]

    assert scores(items) == scores(items)


# --- isolation, persona ------------------------------------------------------------------------


def test_items_of_another_user_are_never_read() -> None:
    spy = SpyEmbedder()
    other = mail("x1", "Secret de l'autre utilisateur", owner_user_id=OTHER_USER)
    mine = mail("m1", "Merci de valider")

    scorer(spy).score_cases(group_into_cases([mine, other]), [mine, other], USER, NOW)

    assert spy.texts and not any("Secret" in text for text in spy.texts)


def test_persona_is_configured_for_demo_and_deduced_otherwise() -> None:
    items = [
        mail("m1", "A", participants=[{"name": "X", "email": "x@a.test", "role": "from"},
                                      {"name": "Moi", "email": "moi@ent.test", "role": "to"}]),
        mail("m2", "B", participants=[{"name": "Y", "email": "y@a.test", "role": "from"},
                                      {"name": "Moi", "email": "moi@ent.test", "role": "to"}]),
    ]  # fmt: skip

    assert persona_for(USER, items) == DEMO_CAMILLE
    deduced = persona_for("u_inconnu", items)
    assert (deduced.address, deduced.internal_domain, deduced.hierarchy) == (
        "moi@ent.test",
        "ent.test",
        frozenset(),
    )
    assert persona_for("u_inconnu", []).address == ""


def test_the_cards_name_the_relation_of_each_correspondent() -> None:
    vp = viewpoint_for(DEMO_CAMILLE, [mail("m1", "A", sender=N1), mail("m2", "B")], NOW)

    assert vp.people[N1] == "votre N+1" and vp.people[COLLEAGUE] == "collègue"
    assert CAMILLE not in vp.people and vp.moment == NOW
    assert comprendre.relation(f"Bernard Mercier, {vp.people[N1]}") == "hierarchie"


# --- repli -------------------------------------------------------------------------------------


def test_failing_model_falls_back_to_rules_with_a_warning() -> None:
    items = [mail("m1", "Merci de valider")]
    cases = group_into_cases(items)

    learned, warning = learned_case_scores(cases, items, USER, NOW, scorer(SpyEmbedder(fail=True)))

    assert learned is None
    assert warning and "RuntimeError" in warning


def test_rules_only_when_asked(monkeypatch: pytest.MonkeyPatch) -> None:
    items = [mail("m1", "Merci de valider")]
    cases = group_into_cases(items)
    monkeypatch.setenv(SCORE_ENV, "regles")

    assert learned_case_scores(cases, items, USER, NOW) == (None, None)
    assert learned_case_scores(cases, items, USER, NOW, None) == (None, None)


# --- dans le brief -----------------------------------------------------------------------------


def test_brief_uses_the_learned_score() -> None:
    ask = mail("m1", "Merci de valider le planning ?", sender=N1)
    (expected,) = scores([ask]).values()

    brief = generate_brief([ask], USER, DAY, now=NOW, scorer=scorer())

    (entry,) = brief.sections[0].entries
    assert entry.priority_score == expected.score
    assert entry.priority_reasons == expected.reasons
    assert brief.warnings == []
    assert (brief.generator.llm, brief.generator.llm_version) == (GENERATOR, "espion")


def test_brief_keeps_rule_scores_and_warns_when_the_model_fails() -> None:
    ask = mail("m1", "Merci de valider le planning ?")
    (case,) = group_into_cases([ask])

    brief = generate_brief([ask], USER, DAY, now=NOW, scorer=scorer(SpyEmbedder(fail=True)))

    assert brief.sections[0].entries[0].priority_score == score_case(case, NOW)[0]
    assert any("score appris indisponible" in w for w in brief.warnings)
    assert brief.generator.llm == "rules"  # le pied de page ne ment pas en cas de repli


def test_read_mail_is_never_announced_as_unread() -> None:
    read = mail("m1", "Merci de valider le devis ?", sender=N1, is_read=True)

    brief = generate_brief([read], USER, DAY, now=NOW, scorer=scorer())

    (entry,) = brief.sections[0].entries
    assert entry.text == "Mail : Merci de valider le devis ?"
    assert "non lu" not in " ".join(entry.priority_reasons)


def test_wording_of_a_case_shown_by_another_element() -> None:
    file = make_item("file", "f1", title="Rapport_2026-3333.docx")

    assert _mails_wording([file]) == (
        "Fichier : Rapport_2026-3333.docx",
        "dernière mise à jour le 01/10",
    )
