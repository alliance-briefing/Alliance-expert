"""Écrivain LLM (M3) : fiche, contrôles, nouvelles rédactions, sonde. Sans Ollama (client factice)."""

from __future__ import annotations

import json
import random
import re
from collections import Counter
from datetime import date
from pathlib import Path

import pytest

from contracts.models import Item
from packages.collect.trainset import build, llm_corpus, load, sonde
from packages.collect.trainset import llm_writer as L
from packages.collect.trainset.facts import Fact
from packages.collect.trainset.world import all_personas, at

DAY = date(2026, 11, 4)


@pytest.fixture(scope="module")
def persona():
    return all_personas()[0]


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


def _pieces(fact: Fact, **change: str) -> dict:
    """Des morceaux fidèles au fait « envoyer l'objet, question, d'ici vendredi 6 novembre »."""
    return {
        "objet": "Pièce manquante au dossier",
        "contexte": "Le dossier fournisseur avance bien et il ne reste plus qu'une pièce pour le compléter.",
        "demande": f"Peux-tu m'envoyer {fact.obj.phrase} d'ici vendredi 6 novembre ?",
        "ton": "",
        "conclusion": "Le reste du dossier est déjà rangé dans l'espace partagé.",
    } | change


class FakeOllama:
    """Rend les réponses prévues, dans l'ordre, et garde les graines reçues."""

    model = L.MODEL

    def __init__(self, answers: list[dict]) -> None:
        self.answers = list(answers)
        self.seeds: list[int] = []
        self.conversations: list[list[dict]] = []

    def identity(self) -> dict:
        return {"model": L.MODEL, "digest": "sha256-test", "ollama": "0.0.0"}

    def chat(self, conversation: list[dict], seed: int) -> dict:
        self.seeds.append(seed)
        self.conversations.append(conversation)
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
    assert "À écrire dans la demande : « vendredi 6 novembre »" in sh.text
    assert (sh.target, sh.signer) == (first, fact.author.first)
    assert "Contexte : " in sh.text and "Style : " in sh.text
    assert first in sh.frame["greeting"] or "Bonjour" in sh.frame["greeting"]
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
    assert set(json.loads(conversation[2]["content"])) == set(L.PIECES)
    assert "À écrire : " not in conversation[1]["content"]  # fiche d'exemple au format v7


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
        # Sonde v7 : formule de calme passée dans un mail neutre.
        (lambda t: t.replace(" ? Merci", ", à l'heure qui te convient ? Merci"), "ton changé"),
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
    assert any(r.startswith("demande ajoutée à un mail qui n'en porte pas") for r in reasons)


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
    reasons = L.check(relire, sh, "Relecture", text)
    assert any(r.startswith("demande de retour ajoutée") for r in reasons)
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


def _deadline_fact(persona) -> Fact:
    return _mail(persona, form="question", deadline=at(date(2026, 11, 6), 12.0))


def test_un_texte_rejete_est_reecrit_avec_une_autre_graine(persona):
    fact = _deadline_fact(persona)
    bad = _pieces(fact, demande="C'est urgent.")
    client = FakeOllama([bad, _pieces(fact)])
    draft = L.LLMWriter(persona, client=client).draft(fact)
    assert draft.ok and draft.tries == 2
    assert len(set(client.seeds)) == 2 and draft.seed == client.seeds[-1]
    assert draft.rejects[0] and draft.rejects[1] == []
    assert (draft.model, draft.digest, draft.prompt_version) == (
        L.MODEL,
        "sha256-test",
        L.PROMPT_VERSION,
    )


def test_le_mail_est_assemble_par_le_code_autour_des_morceaux(persona):
    """v7 : salutation, ordre, signature viennent du code ; le LLM n'écrit que les morceaux."""
    fact = _deadline_fact(persona)
    draft = L.LLMWriter(persona, client=FakeOllama([_pieces(fact)])).draft(fact)
    sh = L.LLMWriter(persona, client=FakeOllama([])).sheet(fact)
    assert draft.ok and draft.objet == "Pièce manquante au dossier"
    assert draft.corps.startswith(sh.frame["greeting"])
    assert draft.corps.endswith(sh.signature)
    pieces = _pieces(fact)
    assert pieces["demande"] in draft.corps and pieces["contexte"] in draft.corps


def test_un_morceau_facultatif_qui_echoue_est_retire_sans_refuser_le_mail(persona):
    fact = _deadline_fact(persona)
    asking = _pieces(fact, conclusion="Tu peux aussi regarder le planning ?")
    draft = L.LLMWriter(persona, client=FakeOllama([asking])).draft(fact)
    assert draft.ok and draft.tries == 1
    assert "planning" not in draft.corps
    assert draft.dropped and draft.dropped[0].startswith("conclusion : demande ajoutée")


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        ({"demande": "Peux-tu m'envoyer le document ?"}, "manque « vendredi 6 novembre »"),
        ({"contexte": "Le dossier avance, mais c'est urgent et il faut le traiter vite."}, "ton"),
        (
            {"demande": "Merci de m'envoyer le devis Froid Service d'ici vendredi 6 novembre."},
            "forme",
        ),
        ({"ton": "C'est urgent."}, None),  # ton neutre : un ton rendu est ignoré
    ],
)
def test_le_controle_par_morceau(persona, change, reason):
    fact = _deadline_fact(persona)
    sh = L.LLMWriter(persona, client=FakeOllama([])).sheet(fact)
    kept, reasons, dropped = L.judge(fact, sh, _pieces(fact, **change))
    if reason is None:
        assert reasons == [] and "ton" not in kept
    else:
        assert any(reason in r for r in reasons + dropped), (reasons, dropped)


def test_un_contexte_qui_repete_la_demande_est_retire(persona):
    """Essai v7 : « la procédure est maintenant à jour », écrit dans le contexte et la demande."""
    fact = _deadline_fact(persona)
    echo = f"Pour mémoire, j'attends toujours {fact.obj.phrase} d'ici vendredi 6 novembre."
    repeated = _pieces(fact, contexte=echo)
    kept, _, dropped = L.judge(
        fact, L.LLMWriter(persona, client=FakeOllama([])).sheet(fact), repeated
    )
    assert "contexte" not in kept and "contexte : répète la demande" in dropped


# --- v8 : leçons du corpus v7 (919 textes) -----------------------------------------------------


def test_un_mail_sans_demande_voit_d_abord_un_exemple_sans_demande(persona):
    """v7 : la demande restait vide pour 60 % des « S'informer », qui ne voyaient que des
    exemples de mails qui demandent quelque chose."""
    by_id = {e["id"]: e for e in L.EXAMPLES}
    for act, other in (("informer", "rien"), ("rien", "informer")):
        sh = L.LLMWriter(persona, client=FakeOllama([])).sheet(_mail(persona, act))
        assert by_id[sh.examples[0]]["act"] == other
        assert L.NO_REQUEST_PIECE[act] in sh.text
    sh = L.LLMWriter(persona, client=FakeOllama([])).sheet(_mail(persona, "faire"))
    assert L.NO_REQUEST_PIECE["informer"] not in sh.text


def test_la_reecriture_rappelle_toujours_le_tu_ou_le_vous():
    """v7 : 29 rejets sur 113 n'avaient plus que du vous dans un mail tutoyé."""
    assert "jamais vous" in L.repair(["demande : vide"], tu=True)
    assert "jamais tu" in L.repair(["demande : vide"], tu=False)


@pytest.mark.parametrize(
    ("act", "tone", "kind", "text", "ok"),
    [
        # Un remerciement n'est pas une demande ; « merci de m'envoyer » en est une.
        ("rien", "normal", "demande", "Merci de m'avoir envoyé la confirmation.", True),
        ("rien", "normal", "demande", "Merci de m'envoyer la confirmation.", False),
        # Les mots de l'acte Faire et Répondre vus dans le corpus v7.
        ("faire", "normal", "demande", "Merci de contacter le client à ce sujet.", True),
        ("faire", "normal", "demande", "Mets à jour le tableau de suivi.", True),
        ("repondre", "normal", "demande", "Peux-tu me faire savoir si le planning tient.", True),
        # Une formule du même ton dans la demande est permise ; d'un autre ton, non.
        ("decider", "urgent", "demande", "Merci de le valider dès que possible.", True),
        ("decider", "normal", "demande", "Merci de le valider dès que possible.", False),
        ("decider", "sans_urgence", "demande", "Merci de le valider quand tu as un moment.", True),
        ("decider", "urgent", "demande", "Merci de le valider quand tu as un moment.", False),
        # Un objet sans mots-outils reste du français.
        ("decider", "normal", "objet", "Budget formation : choix attendu entre versions", True),
    ],
)
def test_les_controles_corriges_en_v8(persona, act, tone, kind, text, ok):
    fact = _mail(persona, act, tone=tone)
    sh = L.LLMWriter(persona, client=FakeOllama([])).sheet(fact)
    reasons = L.check_piece(kind, text, fact, sh)
    assert (reasons == []) == ok, reasons


@pytest.mark.parametrize(
    "text",
    [
        "Merci de le valider dans les meilleurs délais.",
        "Merci de le valider dans les plus brefs délais.",
        "Merci de le valider dès que cela sera possible.",
        "Merci de le valider dès que vous le pourrez.",
    ],
)
def test_les_formules_d_urgence_lues_dans_l_echantillon(persona, text):
    """Échantillon M3 relu à l'aveugle : lues urgentes dans des mails neutres."""
    fact = _mail(persona, "decider")
    sh = L.LLMWriter(persona, client=FakeOllama([])).sheet(fact)
    assert any("ton changé" in r for r in L.check_piece("demande", text, fact, sh))


def test_l_intitule_a_ecrire_recopie_sans_deux_points_est_refuse(persona):
    fact = _mail(persona, "relire", tone="sans_urgence")
    sh = L.LLMWriter(persona, client=FakeOllama([])).sheet(fact)
    text = f"À écrire {sh.tone_mark[0]}."
    assert "intitulé de la fiche recopié" in L.check_piece("ton", text, fact, sh)


def test_un_remerciement_perd_son_contexte_et_sa_conclusion(persona):
    """Échantillon M3 : le contexte d'un mail « Rien » y ajoutait un fait (lu S'informer)."""
    fact = _mail(persona, "rien")
    sh = L.LLMWriter(persona, client=FakeOllama([])).sheet(fact)
    pieces = {
        "objet": "Bien reçu",
        "contexte": "J'ai passé la commande auprès du fournisseur ce matin.",
        "demande": f"Merci pour {fact.obj.phrase}, c'est bien noté.",
        "ton": "",
        "conclusion": "Je te tiens au courant des prochaines étapes.",
    }
    kept, reasons, dropped = L.judge(fact, sh, pieces)
    assert reasons == [] and set(kept) == {"objet", "demande"}
    assert [d.split(" : ")[0] for d in dropped] == ["contexte", "conclusion"]


def test_une_urgence_niee_se_lit_calme():
    """v7 : « ce n'est pas une demande urgente » était lu comme urgent."""
    text = L._norm("Ce n'est pas une demande urgente.")
    assert L._CALM.search(text) and not L._URGENT.search(L._CALM.sub(" ", text))


def test_la_reecriture_guidee_renvoie_les_morceaux_refuses_et_les_mots_fautifs(persona):
    """v6 : au lieu d'un nouveau tirage, le modèle corrige ses propres morceaux."""
    fact = _deadline_fact(persona)
    urgent = _pieces(fact, demande=f"Au plus vite, peux-tu m'envoyer {fact.obj.phrase} d'ici "
                                   "vendredi 6 novembre ?")  # fmt: skip
    client = FakeOllama([urgent, _pieces(fact)])
    draft = L.LLMWriter(persona, client=client).draft(fact)
    assert draft.ok and draft.tries == 2
    first, second = client.conversations
    assert second[: len(first)] == first
    assert second[-2] == {"role": "assistant", "content": json.dumps(urgent, ensure_ascii=False)}
    assert "« au plus vite »" in second[-1]["content"]


def test_apres_trois_rejets_l_item_garde_le_texte_des_gabarits(persona):
    fact = _mail(persona, form="question")
    client = FakeOllama([_pieces(fact, demande="C'est urgent.")] * L.MAX_TRIES)
    item, used, draft = L.LLMWriter(persona, client=client).write(fact)
    assert not draft.ok and draft.tries == L.MAX_TRIES
    assert used[-1].endswith(":rejete") and "urgent" not in item.snippet.lower()


def test_l_item_a_la_forme_du_connecteur_mail(persona):
    """Comme mail_graph : objet, extrait nettoyé de 500 caractères au plus, pas de corps."""
    fact = _deadline_fact(persona)
    client = FakeOllama([_pieces(fact)])
    item, used, draft = L.LLMWriter(persona, client=client).write(fact)
    assert draft.ok and used == [f"llm:{L.PROMPT_VERSION}"]
    assert item.title == "Pièce manquante au dossier" and item.content_text is None
    assert "vendredi 6 novembre" in item.snippet and len(item.snippet) <= 500
    assert persona.manager.first not in item.snippet.split(".")[0]  # salutation retirée


def test_ce_qui_porte_l_etiquette_doit_tenir_dans_l_extrait(persona, case):
    """Le modèle ne lit que 500 caractères : une échéance écrite plus loin est perdue."""
    fact, sh = case
    filler = "Le dossier avance bien et toute l'équipe a fait le point sur l'organisation. " * 8
    text = _good(persona, fact).replace(" d'ici vendredi 6 novembre ?", " ?")
    text = text.replace("Merci !", f"{filler}\n\nÀ faire d'ici vendredi 6 novembre.")
    reasons = L.check(fact, sh, "Devis", text)
    assert "« vendredi 6 novembre » hors de l'extrait lu" in reasons


@pytest.mark.parametrize(
    ("act", "added", "reason"),
    [
        # Défauts vus avec Gemma (essai v4).
        ("relire", "Merci de me confirmer ton accord.", "décision ajoutée"),
        ("relire", "Le document est ici : [Lien vers le document].", "texte à trous"),
        ("preparer", "N'hésite pas à me faire signe.", "demande de retour ajoutée"),
    ],
)
def test_les_garde_fous_de_l_essai_gemma(persona, act, added, reason):
    fact = _mail(persona, act)
    sh = L.LLMWriter(persona, client=FakeOllama([])).sheet(fact)
    text = f"Salut {persona.manager.first},\n\nTu peux relire {fact.obj.phrase} pour la réunion ?"
    text += f" {added}\n\n{fact.author.first}"
    assert any(r.startswith(reason) for r in L.check(fact, sh, "Relecture", text))


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


class SameAnswer(FakeOllama):
    """Rend toujours le même mail : assez pour faire tourner une sonde entière."""

    def chat(self, conversation: list[dict], seed: int) -> dict:
        self.seeds.append(seed)
        return dict.fromkeys(L.PIECES, "Un texte.")


def test_la_sonde_n_ecrit_a_annoter_que_ce_que_voit_le_modele(tmp_path):
    summary = sonde.run(out=tmp_path, client=SameAnswer([]))
    rows = [json.loads(x) for x in (tmp_path / "a_annoter.jsonl").read_text("utf-8").splitlines()]
    assert len(rows) == sonde.N_CASES and summary["cases"] == sonde.N_CASES
    assert all(set(r) == {"case_id", "persona", "jour", "item"} for r in rows)
    assert all(r["item"]["content_text"] is None for r in rows)
    expected = [json.loads(x) for x in (tmp_path / "attendu.jsonl").read_text("utf-8").splitlines()]
    assert all({"action", "urgence", "alerte", "extrait"} <= set(e) for e in expected)
    assert [r["case_id"] for r in rows] == [e["case_id"] for e in expected]


# --- le corpus de nuit -------------------------------------------------------------------------


def test_le_corpus_couvre_tous_les_mails_dans_un_ordre_fixe():
    todo = llm_corpus.plan()
    assert todo and all(L.rewritable(f) for _, f in todo)
    assert {p.split for p, _ in todo} == {"train", "tgen"}
    assert len({f.item_id for _, f in todo}) == len(todo)
    first = [f.item_id for _, f in todo[:50]]
    assert first == [f.item_id for _, f in llm_corpus.plan()[:50]]  # graine fixe
    assert len({p.metier.key for p, _ in todo[:50]}) >= 8  # le début est déjà varié


def test_le_corpus_reprend_la_ou_il_s_est_arrete(tmp_path):
    """Arrêt puis relance : rien n'est réécrit, rien n'est perdu."""
    quiet = []
    first = llm_corpus.run(out=tmp_path, limit=3, client=SameAnswer([]), log=quiet.append)
    assert first["textes"] == 3 and first["ecrits"] == 3
    with (tmp_path / "textes.jsonl").open("a", encoding="utf-8") as file:
        file.write('{"item_id": "mail:coupé')  # arrêt brutal au milieu d'une ligne
    second = llm_corpus.run(out=tmp_path, limit=2, client=SameAnswer([]), log=quiet.append)
    assert second["ecrits"] == 5
    ids = llm_corpus.done_ids(tmp_path / "textes.jsonl")
    assert len(ids) == 5
    assert ids == {f.item_id for _, f in llm_corpus.plan()[:5]}
    rows = (tmp_path / "seances.jsonl").read_text("utf-8").splitlines()
    assert len(rows) == 2 and json.loads(rows[0])["consigne"] == L.PROMPT_VERSION


# --- l'assemblage du corpus (sans GPU) ---------------------------------------------------------

_ACT_OF = (  # l'acte d'une fiche, retrouvé dans sa ligne « Demande »
    ("decider", r"valide|signe|approuve|donne son accord|choisisse"),
    ("faire", r"envoie|transmette|appelle|mette à jour|dépose"),
    ("repondre", r"donne son avis|dise si|confirme|donne ses disponibilités|propose une date"),
    ("relire", r"relise|vérifie"),
    ("preparer", r"vienne à la réunion|présente|prépare ses remarques"),
)
_VERB = {
    "decider": "valider {o}",
    "faire": "mettre à jour {o}",
    "repondre": "donner un avis sur {o}",
    "relire": "relire {o}",
    "preparer": "préparer {o} pour la réunion",
}


class FromSheet(FakeOllama):
    """Écrit des morceaux sobres à partir de la fiche seule, comme un LLM docile : assez pour
    fabriquer un `textes.jsonl` dont la plupart des textes passent les contrôles."""

    def __init__(self) -> None:
        super().__init__([])

    def chat(self, conversation: list[dict], seed: int) -> dict:
        self.seeds.append(seed)
        sheet = [m["content"] for m in conversation if m["role"] == "user"][L.N_EXAMPLES]

        def line(label: str) -> str:
            found = re.search(rf"^{label} : (.*)$", sheet, flags=re.MULTILINE)
            return found.group(1) if found else ""

        tu = line("Adresse").startswith("tutoiement")
        o = line("Sujet")
        mark = re.search(r"« (.*) »", line("À écrire dans la demande"))
        when = f" d'ici {mark.group(1)}" if mark else ""
        tone = re.search(r"« (.*) »", line("À écrire dans le ton"))
        act = next((a for a, verbs in _ACT_OF if re.search(verbs, line("Demande"))), None)
        if "Information donnée" in sheet:
            demande = f"{'Je te' if tu else 'Je vous'} confirme que le point sur {o} est fait."
        elif act is None:
            demande = f"Merci pour {o}, bien reçu de mon côté."
        else:
            verb = _VERB[act].format(o=o)
            form = line("Forme")
            if form.startswith("question"):
                demande = f"{'Peux-tu' if tu else 'Pouvez-vous'} {verb}{when} ?"
            elif form.startswith("directe"):
                demande = f"Merci de {verb}{when}."
            else:
                demande = f"Il reste à {verb}{when}."
        return {
            "objet": "Point sur " + re.sub(r"^(le |la |les |l')", "", o),
            "contexte": "",
            "demande": demande,
            "ton": f"Pour être clair : {tone.group(1)}." if tone else "",
            "conclusion": "",
        }


N_TEXTS = 40


def _copy(rows: list[dict]) -> list[dict]:
    return json.loads(json.dumps(rows))


@pytest.fixture(scope="module")
def plan_facts():
    return {f.item_id: (p, f) for p, f in llm_corpus.plan()}


def _as_generated(row: dict, plan_facts: dict, **pieces: str) -> dict:
    """La ligne qu'aurait écrite le PC GPU avec ces morceaux, tous gardés : le mail et
    l'extrait enregistrés sont rassemblés à partir d'eux."""
    p, f = plan_facts[row["item_id"]]
    sh = L.LLMWriter(p, client=FakeOllama([])).sheet(f)
    row["llm"]["pieces"] |= pieces
    kept = {
        k: v
        for k, v in row["llm"]["pieces"].items()
        if v and not (k == "ton" and f.tone == "normal")
    }
    objet, corps = L.assemble(sh, kept)
    row["llm"] |= {"objet": objet, "corps": corps}
    row["extrait"] = L.extract(corps)
    return row


def _write_rows(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    path.write_text(text, encoding="utf-8", newline="\n")
    return path


def _assembled(out: Path, row: dict) -> tuple[Item, list[str]]:
    """L'Item assemblé d'un fait et ses marques d'écrivain (scénario)."""
    item = next(i for i in load.load_items(row["persona"], out) if i.item_id == row["item_id"])
    scenario = load.load_scenario(row["persona"], out)
    marks = next(r["templates"] for r in scenario if r["item_id"] == row["item_id"])
    return item, marks


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    """Un `textes.jsonl` factice (les 40 premiers faits du plan) et son assemblage."""
    out = tmp_path_factory.mktemp("mail-v7")
    llm_corpus.run(out=out, limit=N_TEXTS, client=FromSheet(), log=lambda _: None)
    rows = [json.loads(x) for x in (out / "textes.jsonl").read_text("utf-8").splitlines()]
    return out, rows, llm_corpus.assemble_items(out=out)


@pytest.fixture(scope="module")
def templates():
    """Le corpus des gabarits, par persona : la référence de l'assemblage."""
    return {p.key: build.build_persona(p) for p in all_personas()}


def test_l_assemblage_ecrit_tout_le_corpus_en_item_valides(corpus, templates):
    """100 % des lignes sont des Item du contrat A : mêmes éléments, même ordre, mêmes
    étiquettes que le corpus des gabarits ; seuls le texte et la marque d'écrivain changent."""
    out, _, manifest = corpus
    assert set(load.personas(data=out)) == set(templates)
    for key, (items, scenario) in templates.items():
        assembled = load.load_items(key, out)  # valide chaque ligne (contrat A)
        assert [i.item_id for i in assembled] == [i.item_id for i in items]
        same = [{k: v for k, v in r.items() if k != "templates"} for r in scenario]
        got = [
            {k: v for k, v in r.items() if k != "templates"} for r in load.load_scenario(key, out)
        ]
        assert got == same
    c = manifest["llm"]
    assert c["ecrits"] == N_TEXTS and c["a_reecrire"] == len(llm_corpus.plan())
    assert c["retenus"] + c["rejetes"] + c["recontroles"] == N_TEXTS
    assert not c["complet"] and c["absents"] == c["a_reecrire"] - N_TEXTS
    by_split = manifest["llm_par_split"]
    assert by_split["train"]["ecrits"] + by_split["tgen"]["ecrits"] == N_TEXTS


def test_un_texte_retenu_donne_l_objet_et_l_extrait_du_llm(corpus):
    out, rows, manifest = corpus
    kept = [r for r in rows if r["llm"]["ok"]]
    assert len(kept) >= N_TEXTS // 2  # le faux client est docile
    assert manifest["llm"]["retenus"] == len(kept) and manifest["llm"]["recontroles"] == 0
    for row in kept:
        item, marks = _assembled(out, row)
        assert item.snippet == row["extrait"] and item.content_text is None
        assert item.title.endswith(row["llm"]["objet"])  # « Re: » ajouté pour une relance
        assert marks == [f"llm:{L.PROMPT_VERSION}"]


def test_un_texte_absent_garde_exactement_les_gabarits(corpus, templates):
    out, rows, _ = corpus
    written = {r["item_id"] for r in rows}
    key = next(iter(templates))
    items, _ = templates[key]
    by_id = {i.item_id: i for i in load.load_items(key, out)}
    marks = {r["item_id"]: r["templates"] for r in load.load_scenario(key, out)}
    absent = [i for i in items if f"llm:{L.PROMPT_VERSION}:absent" in marks[i.item_id]]
    assert absent and not {i.item_id for i in absent} & written
    assert all(by_id[i.item_id] == i for i in absent)


def test_un_texte_rejete_ou_refuse_au_recontrole_garde_les_gabarits(
    corpus, templates, plan_facts, tmp_path
):
    """Rejeté au moment de l'écrire, ou accepté alors mais refusé par les contrôles
    d'aujourd'hui (morceau fautif, mail rassemblé différent) : texte des gabarits."""
    _, rows, _ = corpus
    rows = _copy(rows)
    rejected, faulty, altered = [r for r in rows if r["llm"]["ok"]][:3]
    rejected["llm"]["ok"] = False
    _as_generated(faulty, plan_facts, demande="C'est urgent.")
    altered["llm"]["corps"] += "\nPS : un mot ajouté après coup."
    out = tmp_path / "mail-v7"
    manifest = llm_corpus.assemble_items(_write_rows(out / "textes.jsonl", rows), out)
    assert manifest["llm"]["recontroles"] == 2
    detail = {d["item_id"]: d["raisons"] for d in manifest["recontroles_detail"]}
    assert detail[altered["item_id"]] == ["mail rassemblé différent de celui enregistré"]
    assert all(r.startswith("demande : ") for r in detail[faulty["item_id"]])
    for row, status in ((rejected, "rejete"), (faulty, "recontrole"), (altered, "recontrole")):
        items, _ = templates[row["persona"]]
        item, marks = _assembled(out, row)
        assert item == next(i for i in items if i.item_id == row["item_id"])
        assert marks[-1] == f"llm:{L.PROMPT_VERSION}:{status}"


def test_une_regle_nouvelle_raccourcit_le_mail_sans_le_rejeter(corpus, plan_facts, tmp_path):
    """Un mail « Rien » écrit avec un contexte (accepté sur le PC GPU) : l'assemblage
    d'aujourd'hui retire le contexte, garde le texte du LLM et le compte comme raccourci."""
    _, rows, _ = corpus
    rows = _copy(rows)
    thanks = next(r for r in rows if r["llm"]["ok"] and plan_facts[r["item_id"]][1].act == "rien")
    added = "Le dossier avance bien de mon côté et toute l'équipe a fait le point cette semaine."
    _as_generated(thanks, plan_facts, contexte=added)
    assert "avance bien" in thanks["llm"]["corps"]
    out = tmp_path / "mail-v8"
    manifest = llm_corpus.assemble_items(_write_rows(out / "textes.jsonl", rows), out)
    assert manifest["llm"]["raccourcis"] == 1 and manifest["llm"]["recontroles"] == 0
    item, marks = _assembled(out, thanks)
    assert marks == [f"llm:{L.PROMPT_VERSION}"] and "avance bien" not in item.snippet
    assert thanks["llm"]["pieces"]["demande"] in item.snippet


def test_le_mail_enregistre_doit_s_expliquer_par_ses_morceaux(corpus, plan_facts):
    """La preuve que les deux PC concordent : une combinaison des morceaux rendus redonne
    exactement le mail enregistré."""
    _, rows, _ = corpus
    row = _copy([next(r for r in rows if r["llm"]["ok"])])[0]
    p, f = plan_facts[row["item_id"]]
    sh = L.LLMWriter(p, client=FakeOllama([])).sheet(f)
    d = llm_corpus._draft_of(row)
    assert llm_corpus.generation_kept(sh, d) is not None
    other = L.Draft(**(row["llm"] | {"corps": d.corps.replace(d.pieces["demande"], "Autre.")}))
    assert llm_corpus.generation_kept(sh, other) is None


def test_l_assemblage_est_deterministe(corpus, tmp_path):
    """Mêmes textes, mêmes octets, quel que soit l'ordre des lignes ; une ligne coupée par
    un arrêt brutal (au milieu du fichier après une relance) est ignorée."""
    _, rows, first = corpus
    shuffled = list(rows)
    random.Random(1).shuffle(shuffled)
    out = tmp_path / "mail-v7"
    path = _write_rows(out / "textes.jsonl", shuffled)
    text = path.read_text("utf-8")
    cut = text.index("\n", len(text) // 2) + 1
    path.write_text(text[:cut] + '{"item_id": "mail:coupé\n' + text[cut:], "utf-8")
    second = llm_corpus.assemble_items(path, out)
    assert second["personas"] == first["personas"]  # empreintes des items et du scénario
    assert second["llm"] == first["llm"] and second["textes"]["illisibles"] == 1


def test_le_corpus_compresse_redonne_le_meme_corpus(corpus, tmp_path):
    """La version versionnée (.gz) : stable octet pour octet, et assemblée à l'identique
    (mêmes empreintes, même empreinte des textes)."""
    _, rows, first = corpus
    work = tmp_path / "mail-v8"
    path = _write_rows(work / "textes.jsonl", rows)
    packed = llm_corpus.compress(work)
    assert packed.read_bytes() == llm_corpus.compress(work).read_bytes()  # sans date
    assert llm_corpus.texts_bytes(packed) == path.read_bytes()
    path.unlink()  # la génération est finie : seul le .gz reste
    assert llm_corpus.default_texts(work) == packed
    again = llm_corpus.assemble_items(out=work)
    assert again["personas"] == first["personas"] and again["llm"] == first["llm"]
    assert again["textes"]["sha256"] == first["textes"]["sha256"]


def _first_llm(rows: list[dict], **change) -> list[dict]:
    return [rows[0] | {"llm": rows[0]["llm"] | change}, *rows[1:]]


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda rows: [*rows, rows[0]], "deux fois"),
        (lambda rows: _first_llm(rows, prompt_version="mail-v6"), "consignes"),
        (lambda rows: _first_llm(rows, digest="sha256-autre"), "plusieurs modèles"),
        (lambda rows: [rows[0] | {"llm": {"ok": True}}, *rows[1:]], "illisible"),
        (lambda rows: [], "aucun texte"),
    ],
)
def test_l_assemblage_refuse_un_corpus_douteux(corpus, tmp_path, change, message):
    _, rows, _ = corpus
    path = _write_rows(tmp_path / "textes.jsonl", change(_copy(rows)))
    with pytest.raises(ValueError, match=message):
        llm_corpus.assemble_items(path, tmp_path)
    assert not (tmp_path / "manifest.json").exists()  # rien d'écrit à moitié


def test_l_assemblage_refuse_un_fait_inconnu_du_simulateur(corpus, tmp_path):
    _, rows, _ = corpus
    stranger = _copy(rows[:1])[0] | {"item_id": "mail:inconnu-1"}
    path = _write_rows(tmp_path / "textes.jsonl", [*rows, stranger])
    with pytest.raises(ValueError, match="inconnu du simulateur"):
        llm_corpus.assemble_items(path, tmp_path)
    assert not (tmp_path / "items").exists()


def test_la_comparaison_de_deux_consignes_est_appariee(corpus, tmp_path):
    """Seuls les faits écrits par les deux consignes sont comparés ; l'ancienne est aussi
    rejugée avec les contrôles d'aujourd'hui."""
    out, rows, _ = corpus
    _write_rows(tmp_path / "ancienne" / "textes.jsonl", rows[:25])
    result = llm_corpus.compare(tmp_path / "ancienne", out)
    assert result.startswith("25 faits écrits par les deux consignes")
    assert "ancienne rejugée" in result and "Par acte" in result
    empty = tmp_path / "vide"
    _write_rows(empty / "textes.jsonl", [])
    assert llm_corpus.compare(empty, out).startswith("Aucun fait")


def test_le_client_hors_ligne_n_appelle_jamais_le_llm(corpus):
    client = L.Offline(L.MODEL, "sha256-test")
    with pytest.raises(RuntimeError, match="hors ligne"):
        client.chat([], seed=0)
    _, _, manifest = corpus  # l'assemblage a tourné avec ce client : aucun appel
    assert manifest["modele"] == L.MODEL and manifest["empreinte"] == "sha256-test"
