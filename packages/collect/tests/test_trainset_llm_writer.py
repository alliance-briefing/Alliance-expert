"""Écrivain LLM (M3) : fiche, contrôles, nouvelles rédactions, sonde. Sans Ollama (client factice)."""

from __future__ import annotations

import json
from collections import Counter
from datetime import date

import pytest

from packages.collect.trainset import llm_writer as L
from packages.collect.trainset import sonde
from packages.collect.trainset.facts import Fact
from packages.collect.trainset.world import all_personas, at

DAY = date(2026, 11, 4)


@pytest.fixture(scope="module")
def persona():
    return all_personas()[0]


@pytest.fixture(autouse=True)
def any_length(monkeypatch):
    """Les textes des tests sont courts : la longueur n'est testée que par son propre test."""
    monkeypatch.setattr(L, "LENGTHS", (((1, 1000), 1),))
    monkeypatch.setattr(L, "SHORT", (1, 1000))


def _mail(persona, act: str = "faire", **kw) -> Fact:
    kw.setdefault("author", persona.team[0])
    kw.setdefault("obj", persona.metier.objects[0])
    kw.setdefault("recipients", [persona.manager])
    kw.setdefault("meta", {"tu": True, "qui": persona.clients[0].name})
    return Fact(
        item_id="mail:t-1",
        source="mail",
        at=at(DAY, 9.0),
        case_id="c",
        case_type="test",
        act=act,
        **kw,
    )


def _good(persona, fact: Fact) -> str:
    return (
        f"Bonjour {persona.manager.first},\n\nPeux-tu m'envoyer {fact.obj.phrase} d'ici "
        f"vendredi 6 novembre ? Merci !\n\n{fact.author.first}"
    )


class FakeOllama:
    """Rend les réponses prévues, dans l'ordre, et garde les graines reçues."""

    def __init__(self, answers: list[dict]) -> None:
        self.answers = list(answers)
        self.seeds: list[int] = []

    def identity(self) -> dict:
        return {"model": L.MODEL, "digest": "sha256-test", "ollama": "0.0.0"}

    def chat(self, conversation: list[dict], seed: int) -> dict:
        self.seeds.append(seed)
        return self.answers.pop(0)


# --- la fiche ----------------------------------------------------------------------------------


def test_les_fonctions_inclusives_sont_accordees():
    title = "Directeur·rice administratif·ve et financier·ère"
    assert L.gendered(title, True) == "Directrice administrative et financière"
    assert L.gendered(title, False) == "Directeur administratif et financier"
    assert L.gendered("Chef·fe de chantier", True) == "Cheffe de chantier"
    assert L.gendered("Responsable commercial·e", True) == "Responsable commerciale"


def test_la_fiche_decrit_l_acte_sans_nommer_l_etiquette(persona):
    fact = _mail(persona, form="question", deadline=at(date(2026, 11, 6), 12.0))
    sh = L.LLMWriter(persona, client=FakeOllama([])).sheet(fact)
    first = persona.manager.first
    assert f"Demande : que {first} " in sh.text
    assert f"Qui agit : c'est {first} qui doit agir" in sh.text
    assert "À écrire : « vendredi 6 novembre »" in sh.text
    assert (sh.target, sh.signer) == (first, fact.author.first)
    assert "Contexte : " in sh.text and "Style : " in sh.text and " mots" in sh.text
    banned = ("Faire", "S'informer", "étiquette", "une action concrète", "se préparer")
    assert not any(word in sh.text for word in banned)
    assert sh.text == L.LLMWriter(persona, client=FakeOllama([])).sheet(fact).text  # graine


def test_deux_exemples_d_autres_actes_precedent_la_fiche(persona):
    fact = _mail(persona, "decider")
    sh = L.LLMWriter(persona, client=FakeOllama([])).sheet(fact)
    acts = {e["id"]: e["act"] for e in L.EXAMPLES}
    assert len(sh.examples) == L.N_EXAMPLES and "decider" not in {acts[e] for e in sh.examples}
    conversation = L.messages(sh)
    assert [m["role"] for m in conversation] == [
        "system",
        "user",
        "assistant",
        "user",
        "assistant",
        "user",
    ]
    assert conversation[-1]["content"] == sh.text
    assert set(json.loads(conversation[2]["content"])) == {"objet", "corps"}


def test_les_exemples_n_emploient_que_des_noms_du_monde():
    """Condition de la détection : un nom d'exemple recopié hors de sa fiche est « inventé »."""
    for e in L.EXAMPLES:
        for line in e["fiche"].splitlines():
            if line.startswith(("De : ", "À : ")):
                name = line.split(" : ", 1)[1].split(",")[0]
                assert set(name.split()) <= L._NAMES, (e["id"], name)


def test_la_copie_dit_a_qui_s_adresse_la_demande(persona):
    member = persona.team[1]
    fact = _mail(persona, "decider", to_manager=False, addressee=member, recipients=[member])
    text = L.LLMWriter(persona, client=FakeOllama([])).sheet(fact).text
    assert f"la demande s'adresse à {member.first}, pas à {persona.manager.first}" in text


def test_seuls_les_mails_rediges_par_gabarits_sont_reecrits(persona):
    assert L.rewritable(_mail(persona))
    assert not L.rewritable(_mail(persona, sent=True))
    assert not L.rewritable(_mail(persona, trap="injection"))
    assert not L.rewritable(_mail(persona, meta={"kind": "fixed"}))


# --- les contrôles -----------------------------------------------------------------------------


@pytest.fixture
def case(persona):
    fact = _mail(persona, form="question", deadline=at(date(2026, 11, 6), 12.0))
    return fact, L.LLMWriter(persona, client=FakeOllama([])).sheet(fact)


def test_un_texte_fidele_passe(persona, case):
    fact, sh = case
    assert L.check(fact, sh, "Devis à m'envoyer", _good(persona, fact)) == []


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        (lambda t: t.replace(" d'ici vendredi 6 novembre", ""), "manque « vendredi 6 novembre »"),
        (lambda t: t.replace("Merci !", "C'est urgent, merci !"), "ton changé"),
        (lambda t: t.replace("Merci !", "Vois avec Mathilde Lemoine."), "nom inventé"),
        (lambda t: t.replace("Merci !", "Il coûte 1200 euros."), "chiffres inventés"),
        (lambda t: t.replace("Merci !", "Écris à x.y@exemple.test."), "adresse ou lien"),
        (lambda t: t.replace("Merci !", "On en parle lundi."), "date inventée : lundi"),
        (lambda t: t.replace("Peux-tu m'envoyer", "Tu en penses quoi,"), "acte changé"),
        (lambda t: t.replace("Peux-tu", "Pouvez-vous"), "tutoiement ou vouvoiement"),
        (lambda t: t + "\n请回复", "caractères d'une autre langue"),
        # Défauts vus dans la sonde v2 (consigne mail-v2, 14/20).
        (lambda t: t.replace("Merci !", "Fais-moi signe dès que tu as un moment."), "ton changé"),
        (lambda t: t.replace("Merci !", "Quand tu es libre."), "ton changé"),
        (lambda t: t.replace("Merci !", "Je vais t'envoyer le reste."), "rôle inversé"),
        (lambda t: t.replace("Peux-tu", "Demande : peux-tu"), "intitulé de la fiche"),
        (lambda t: t.split("\n\n", 1)[1], "pas de salutation"),
        (lambda t: t.rsplit("\n\n", 1)[0], "pas de signature"),
        (lambda t: t.replace("Merci !", "Bisous !"), "registre familier"),
    ],
)
def test_un_texte_infidele_est_rejete(persona, case, change, reason):
    fact, sh = case
    reasons = L.check(fact, sh, "Devis", change(_good(persona, fact)))
    assert any(r.startswith(reason) for r in reasons), reasons


def test_un_mail_trop_court_est_rejete(persona, case):
    """v3 : des mails d'une phrase, pas plus utiles que les gabarits."""
    fact, sh = case
    sh.words = (70, 120)
    reasons = L.check(fact, sh, "Devis", _good(persona, fact))
    assert any(r.startswith("longueur hors bornes") for r in reasons)


def test_une_phrase_d_exemple_recopiee_est_rejetee(persona, case):
    fact, sh = case
    copied = "Il ne manque plus que ta signature. Tant qu'elle n'y est pas, je ne peux pas"
    text = _good(persona, fact).replace("Merci !", copied)
    reasons = L.check(fact, sh, "Devis", text)
    assert "exemple recopié : decider-indirecte" in reasons


def test_l_objet_du_fait_doit_figurer(persona, case):
    fact, sh = case
    text = _good(persona, fact).replace(fact.obj.phrase, "le document")
    assert any(r.startswith("objet absent") for r in L.check(fact, sh, "Document", text))


def test_un_mail_d_information_ne_demande_rien(persona):
    fact = _mail(persona, "informer")
    sh = L.LLMWriter(persona, client=FakeOllama([])).sheet(fact)
    text = f"Salut {persona.manager.first},\n\nPour info, {fact.obj.phrase} est bouclé."
    sign = f"\n\n{fact.author.first}"
    assert L.check(fact, sh, "Pour info", text + sign) == []
    reasons = L.check(fact, sh, "Pour info", text + " Tu me confirmes ?" + sign)
    assert "demande ajoutée à un mail qui n'en porte pas" in reasons


def test_le_calme_doit_etre_ecrit(persona):
    fact = _mail(persona, "relire", tone="sans_urgence")
    sh = L.LLMWriter(persona, client=FakeOllama([])).sheet(fact)
    calm = sh.must[0][0]
    text = f"Salut {persona.manager.first},\n\nTu peux relire {fact.obj.phrase} ?"
    sign = f"\n\n{fact.author.first}"
    assert any(r.startswith("manque") for r in L.check(fact, sh, "Relecture", text + sign))
    assert L.check(fact, sh, "Relecture", f"{text} {calm.capitalize()}.{sign}") == []


def test_une_demande_de_retour_n_est_rejetee_que_si_elle_change_l_etiquette(persona):
    """Relire + « fais-moi signe » → Répondre (plus engageant) ; Faire le reste."""
    ask = "Fais-moi signe quand c'est fait."
    relire = _mail(persona, "relire")
    sh = L.LLMWriter(persona, client=FakeOllama([])).sheet(relire)
    text = f"Salut {persona.manager.first},\n\nTu peux relire {relire.obj.phrase} ? {ask}"
    text += f"\n\n{relire.author.first}"
    assert "demande de retour ajoutée" in L.check(relire, sh, "Relecture", text)
    faire = _mail(persona, form="question", deadline=at(date(2026, 11, 6), 12.0))
    sh = L.LLMWriter(persona, client=FakeOllama([])).sheet(faire)
    text = _good(persona, faire).replace("Merci !", ask)
    assert L.check(faire, sh, "Devis", text) == []


def test_le_role_inverse_est_rejete_pour_une_decision(persona):
    """Sonde v2 : le N+1 écrit « je n'ai pas encore donné mon accord » au lieu de le demander."""
    fact = _mail(persona, "decider", form="indirecte")
    sh = L.LLMWriter(persona, client=FakeOllama([])).sheet(fact)
    head = f"Salut {persona.manager.first},\n\n"
    sign = f"\n\n{fact.author.first}"
    wrong = f"Je n'ai pas encore donné mon accord pour {fact.obj.phrase}."
    right = f"Il ne manque plus que ton accord pour {fact.obj.phrase}."
    assert any(r.startswith("rôle inversé") for r in L.check(fact, sh, "x", head + wrong + sign))
    assert L.check(fact, sh, "x", head + right + sign) == []


# --- nouvelles rédactions et traçabilité -------------------------------------------------------


def test_un_texte_rejete_est_reecrit_avec_une_autre_graine(persona):
    fact = _mail(persona, form="question", deadline=at(date(2026, 11, 6), 12.0))
    bad = {"objet": "Devis", "corps": "C'est urgent."}
    client = FakeOllama([bad, {"objet": "Devis", "corps": _good(persona, fact)}])
    draft = L.LLMWriter(persona, client=client).draft(fact)
    assert draft.ok and draft.tries == 2
    assert len(set(client.seeds)) == 2 and draft.seed == client.seeds[-1]
    assert draft.rejects[0] and draft.rejects[1] == []
    assert (draft.model, draft.digest, draft.prompt_version) == (
        L.MODEL,
        "sha256-test",
        L.PROMPT_VERSION,
    )


def test_apres_trois_rejets_l_item_garde_le_texte_des_gabarits(persona):
    fact = _mail(persona, form="question")
    client = FakeOllama([{"objet": "x", "corps": "C'est urgent."}] * L.MAX_TRIES)
    item, used, draft = L.LLMWriter(persona, client=client).write(fact)
    assert not draft.ok and draft.tries == L.MAX_TRIES
    assert used[-1].endswith(":rejete") and "urgent" not in item.content_text.lower()


def test_l_item_porte_le_texte_du_llm_et_reste_valide(persona):
    fact = _mail(persona, form="question", deadline=at(date(2026, 11, 6), 12.0))
    client = FakeOllama([{"objet": "Devis à m'envoyer", "corps": _good(persona, fact)}])
    item, used, draft = L.LLMWriter(persona, client=client).write(fact)
    assert draft.ok and used == [f"llm:{L.PROMPT_VERSION}"]
    assert item.title == "Devis à m'envoyer"
    assert "vendredi 6 novembre" in item.content_text and item.snippet


# --- la sonde ----------------------------------------------------------------------------------


def test_la_sonde_tire_vingt_faits_varies():
    chosen = sonde.pick()
    facts = [f for _, _, f in chosen]
    assert len({f.item_id for f in facts}) == sonde.N_CASES
    assert all(p.split == "train" and L.rewritable(f) for _, p, f in chosen)
    assert {f.act for f in facts} >= {"decider", "faire", "repondre", "relire", "preparer"}
    assert {f.form for f in facts} == {"directe", "question", "indirecte"}
    tones = Counter(f.tone for f in facts)
    assert tones["urgent"] >= 1 and tones["sans_urgence"] >= 2
    assert sum(not f.meta.get("tu", True) for f in facts) >= 3
    assert sum(not f.to_manager for f in facts) >= 2
    assert max(Counter(p.metier.key for _, p, _ in chosen).values()) <= sonde.MAX_PER_METIER
    second = sonde.pick(draw=2)  # 20 autres faits : l'annotateur ne les a jamais vus
    assert len(second) == sonde.N_CASES
    assert not {f.item_id for f in facts} & {f.item_id for _, _, f in second}


def test_la_carte_ne_montre_aucune_etiquette(persona):
    fact = _mail(persona)
    draft = L.Draft(fact.item_id, "Objet", "Corps", True, "m", "d", "v", 1, 1, 1.0, [[]])
    row = sonde.card("sonde-01", persona, fact, draft)
    assert set(row) == {"item_id", "source", "title", "when", "fields", "text", "context"}
    assert row["item_id"] == "sonde-01" and "mail:" not in json.dumps(row)


def test_la_comparaison_compte_les_textes_conformes(tmp_path):
    expected = [
        {"case_id": f"sonde-{n:02d}", "emplacement": "x", "action": "faire", "urgence": "normal"}
        | {"alerte": False}
        for n in range(1, 3)
    ]
    (tmp_path / "attendu.jsonl").write_text(
        "".join(json.dumps(e) + "\n" for e in expected), encoding="utf-8"
    )
    answers = [
        {"item_id": "sonde-01", "action": "faire", "urgence": "normal", "alerte": False},
        {"item_id": "sonde-02", "action": "faire", "urgence": "urgent", "alerte": False},
    ]
    (tmp_path / "cases_labels_theo.jsonl").write_text(
        "".join(json.dumps(a) + "\n" for a in answers), encoding="utf-8"
    )
    report = sonde.compare(tmp_path)
    assert "theo : 1 conformes sur 2 annotés" in report
    assert "≠ sonde-02 [x] urgence urgent au lieu de normal" in report
