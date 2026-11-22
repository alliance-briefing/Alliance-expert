"""L'écrivain LLM (M3) : il réécrit un fait du simulateur, il ne simule rien.

Le simulateur décide des faits et des étiquettes ; ce module ne touche que le texte libre
des mails rédigés par gabarits (objet et corps). Les automates, les pièges, les messages
fixes et les réponses du manager restent tirés des banques : leur forme fait partie du fait.

Pour chaque fait : une fiche en mots simples (jamais le nom de l'étiquette), une consigne
versionnée, deux exemples écrits à la main (`llm_exemples.py`), un appel à l'API locale
d'Ollama (bibliothèque standard, température et graine fixées), puis des contrôles
déterministes. Un texte rejeté est réécrit avec une autre graine,
`MAX_TRIES` fois au plus. Le modèle exact (empreinte), la version de la consigne et la graine
sont gardés à côté de chaque texte.

Un LLM ne vérifie pas un LLM : les contrôles sont des règles, et l'acte réellement compris
se mesure par l'annotation humaine à l'aveugle (sonde).
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import time
import unicodedata
import urllib.request
from dataclasses import dataclass, field

from contracts.models import Item
from packages.collect.connectors.mail_graph.mapping import snippet as graph_snippet
from packages.collect.trainset import phrases as P
from packages.collect.trainset.facts import AVIS, COPIE_SUIVI, Fact
from packages.collect.trainset.llm_exemples import EXAMPLES, MORCEAUX
from packages.collect.trainset.metiers import METIERS
from packages.collect.trainset.world import FEMALE, LAST, MALE, MONTHS, WEEKDAYS, Persona, fr_day
from packages.collect.trainset.writer import Writer, _short

OLLAMA_URL = "http://localhost:11434"
# Gemma 2 9B plutôt que Qwen 2.5 7B (décision de Théo, 02/11) : sur les 20 faits de la sonde
# v2, en v4, 13/20 contre 11/20 aux contrôles et un français nettement meilleur, pour un
# appel 1,7 fois plus lent. Qwen reste utilisable avec --modele.
MODEL = "gemma2:9b"
# v2 (sonde : 14/20) : fautes de ton, rôles inversés, fiche recopiée.
# v3 (essai) : justes mais courts, la ligne « Demande » mise à l'impératif : des gabarits.
# v4 : des mails réalistes. Deux exemples écrits à la main avant chaque fiche, un contexte,
# un style et une longueur tirés par fait (et contrôlée), les garde-fous de la v3.
# v5 : le connecteur mail ne garde qu'un extrait de 500 caractères (#114) : la demande,
# l'échéance et le ton doivent tenir au début du mail, et sont contrôlés dans l'extrait.
# v6 : réécriture guidée. Un mail refusé revient au modèle avec les raisons du refus et les
# mots fautifs (essai v5 : 13/20, surtout des formules d'urgence dans des mails neutres).
# v7 : écriture par morceaux. Le modèle n'écrit plus le mail mais ses morceaux (objet,
# contexte, demande, ton, conclusion), contrôlés un par un ; le code assemble la salutation,
# l'ordre, la signature (v6 : 11/20, Gemma raccourcit ou dérive dès qu'on corrige le style,
# et le style embrouillait l'annotation).
# v8 : mails sans demande. Corpus v7 (919 premiers textes) : 87,7 % passent ; la demande est
# vide au 1er essai pour 60 % des « S'informer » et 40 % des « Rien » (aucun exemple sans
# demande n'était montré), et la réécriture glissait alors au vous. Un exemple sans demande
# est désormais montré, la fiche dit ce que porte le morceau, la réécriture rappelle tu ou vous.
PROMPT_VERSION = "mail-v8"
TEMPERATURE = 0.8
TOP_P = 0.9
MAX_TRIES = 3
MAX_TOKENS = 500
NUM_CTX = 4096  # consigne + 2 exemples + fiche ≈ 1 500 jetons : fixé, pas laissé au défaut
N_EXAMPLES = 2
PIECES = ("objet", "contexte", "demande", "ton", "conclusion")

SYSTEM = """Tu écris les morceaux d'un e-mail professionnel réaliste, en français, pour entraîner un assistant de tri. On te donne une fiche ; un programme assemblera tes morceaux et ajoutera lui-même la salutation et la signature.

Remplis chaque morceau :
- objet : l'objet du mail, court (3 à 10 mots).
- contexte : 1 ou 2 phrases (15 à 35 mots) qui situent le dossier : ce qui s'est passé, où on en est. Aucune demande, aucune question, aucune échéance, aucune formule d'urgence ou de calme.
- demande : 1 ou 2 phrases (10 à 35 mots) qui portent la demande de la fiche, reformulée avec tes propres mots, dans la forme indiquée, avec l'échéance s'il y en a une. Pour une information, la phrase qui la donne ; pour un remerciement, la phrase de remerciement. Jamais vide. Rien d'autre : ni urgence, ni calme, ni autre demande, ni « n'hésite pas », ni offre d'aide.
- ton : si le ton de la fiche est urgent ou calme, une courte phrase qui le dit, avec la formule « À écrire » qui lui est destinée (la formule seule, sans les mots « À écrire ») ; si le ton est neutre, une chaîne vide.
- conclusion : une courte phrase neutre (5 à 20 mots) sur un aspect pratique du dossier, sans demande, sans question, sans formule de délai ; ou une chaîne vide.

Règles :
1. N'invente aucun fait vérifiable : personne, entreprise, adresse, téléphone, lien, montant, chiffre, date ou jour de la semaine absents de la fiche. Les détails généraux, sans chiffre ni nom, sont permis.
2. La personne qui agit est celle de la ligne « Qui agit » : l'expéditeur ne fait pas l'acte à sa place, ne dit pas qu'il va le faire et ne donne pas lui-même la décision.
3. Respecte le tutoiement ou le vouvoiement de la fiche dans chaque morceau. Registre professionnel : pas de « bisous », d'émoticône ni d'abréviation.
4. Écris avec tes propres mots : ne recopie ni les intitulés de la fiche (« Demande : »…) ni les phrases des exemples.
Réponds uniquement en JSON avec les cinq clés : objet, contexte, demande, ton, conclusion."""

RELATION = {
    "hierarchie": "son N+1 (supérieur hiérarchique)",
    "pair": "un pair (même niveau, autre service)",
    "equipe": "un membre de son équipe",
    "externe_client": "un client externe",
    "externe_fournisseur": "un fournisseur externe",
}
# Ce que l'expéditeur demande, au subjonctif, avec la personne qui agit : jamais le nom d'une
# catégorie (la v2 disait « une décision : … », que le LLM recopiait). {x} : la personne qui
# agit, {a} : l'expéditeur, {o} : l'objet avec son article (il suit « concernant » ou
# « pour », jamais « de » ni « à »). Pas de « lire {o} avant la réunion » : trop proche de Relire.
# Règles du 23/11 : les demandes de la fiche qui attendent un avis ou un accord, et les
# phrases qui disent que la copie sert à suivre (lues dans l'extrait, comme le verrait le modèle).
_AVIS_SHEET = re.compile(r"donne son avis concernant|lui convient")
_COPY_FOLLOW = re.compile(
    r"copie[^.!?\n]{0,80}(suiv|au courant|inform|connaissance|surveill)"
    r"|suivi[^.!?\n]{0,40}copie",
    re.IGNORECASE,
)

ACTS = {
    "decider": (
        "que {x} valide {o}",
        "que {x} signe {o}",
        "que {x} approuve {o}",
        "que {x} donne son accord pour engager {o}",
        "que {x} choisisse entre les deux versions proposées pour {o}",
    ),
    "faire": (
        "que {x} envoie {o} à {a}",
        "que {x} transmette {o} à {qui}",
        "que {x} appelle {qui} concernant {o}",
        "que {x} mette à jour {o}",
        "que {x} dépose {o} sur le portail",
    ),
    "repondre": (
        "que {x} donne son avis concernant {o}",
        "que {x} dise si {o} lui convient",
        "que {x} confirme que {o} est toujours d'actualité",
        "que {x} donne ses disponibilités {jour} pour un point concernant {o}",
        "que {x} propose une date pour {o}",
    ),
    "relire": (
        "que {x} relise {o}",
        "que {x} vérifie les chiffres concernant {o}",
        "que {x} relise {o} et corrige directement dans le document",
    ),
    "preparer": (
        "que {x} vienne à la réunion du {jour} avec une proposition concernant {o}",
        "que {x} présente {o} à la réunion du {jour} (dix minutes)",
        "que {x} prépare ses remarques concernant {o} pour la réunion du {jour}",
    ),
}
WHO = {  # qui agit : la v2 laissait le LLM inverser les rôles (le N+1 qui « n'a pas validé »)
    "decider": "c'est {x} qui décide ; {a} attend sa décision et ne décide pas {soi}",
    "repondre": "c'est {x} qui répond ; {a} attend sa réponse",
}
WHO_DEFAULT = "c'est {x} qui doit agir ; {a} ne le fait pas {soi} et ne propose pas de le faire"
SECOND = {
    "faire": "que {x} envoie aussi le tableau à jour",
    "relire": "que {x} jette aussi un œil à la pièce jointe",
    "repondre": "que {x} dise si {jour} lui va pour en parler",
}
ADDRESS = {  # Qwen glissait du tu au vous en cours de mail : les formes sont rappelées
    True: "tutoiement (tu, te, ton, ta, tes) du début à la fin, jamais vous",
    False: "vouvoiement (vous, votre, vos) du début à la fin, jamais tu",
}
FORM = {
    "directe": "directe (impératif ou « merci de… »), sans question",
    "question": "question",
    "indirecte": "indirecte (ne formule pas la demande : décris ce qui manque ou ce qui bloque)",
}
# Le contexte de l'affaire, vu par l'expéditeur : des faits du simulateur, jamais inventés,
# qui donnent au LLM de quoi écrire une vraie introduction.
CASE_CONTEXT = {
    "validation": (
        "{a} a préparé {o} ; sans décision, la suite est en attente.",
        "{o} est prêt du côté de {a}, qui attend une décision pour lancer la suite.",
    ),
    "demande_client": (
        "le client a besoin de cet élément pour avancer de son côté.",
        "le client suit ce dossier de près.",
    ),
    "reunion_preparee": ("la réunion du {jour} réunit l'équipe et la direction.",),
    "relecture": (
        "{a} a terminé une version presque finale de {o}.",
        "{o} doit ensuite être diffusé ; {a} l'a rédigé.",
    ),
    "demande_info": (
        "{a} doit répondre à son tour à quelqu'un et a besoin de cet élément.",
        "{a} fait le point sur plusieurs dossiers en cours.",
    ),
    "delegation_cc": ("{a} met {m} en copie pour qu'il ou elle suive le sujet.",),
    "fond": ("{a} fait un point d'étape.", "c'est un échange courant sur un dossier suivi."),
}
STYLES = (
    "direct et efficace",
    "chaleureux et un peu bavard",
    "formel et précis",
    "pressé, phrases courtes",
    "cordial et posé",
)
# Bornes en mots de chaque morceau : au-delà, la demande sortirait de l'extrait de 500
# caractères ; en deçà, le morceau n'apporte rien. Un morceau facultatif hors bornes est retiré.
PIECE_WORDS = {
    "objet": (2, 14),
    "contexte": (8, 45),
    "demande": (5, 45),
    "ton": (2, 25),
    "conclusion": (3, 25),
}
REQUEST_FIRST_SHARE = 0.3  # la demande avant le contexte, pour varier l'ordre du mail
REPEAT_RUN = 6  # mots consécutifs communs avec la demande : le morceau la répète
TONE = {
    "normal": "neutre (aucune formule de délai, ni urgente ni calme)",
    "urgent": "urgent : dis-le clairement (urgent, au plus vite, ça bloque…)",
    "sans_urgence": "calme, sans urgence",
}
MUST_CALM = {
    True: ("rien d'urgent", "pas pressé", "quand tu as un moment", "aucune urgence"),
    False: ("rien d'urgent", "rien ne presse", "quand vous aurez un moment", "aucune urgence"),
}
MUST_URGENT = ("urgent", "au plus vite", "ça bloque", "c'est pressé", "en priorité")
# v8 : ce que porte le morceau « demande » d'un mail qui ne demande rien (S'informer, Rien).
NO_REQUEST_PIECE = {
    "informer": "Morceau demande : la phrase qui donne cette information (jamais vide), sans "
    "rien demander ni proposer d'aide.",
    "rien": "Morceau demande : la phrase de remerciement ou d'accusé de réception (jamais "
    "vide), sans rien demander ni proposer d'aide.",
}


def rewritable(fact: Fact) -> bool:
    """Les seuls faits que le LLM réécrit : les mails rédigés par gabarits."""
    return fact.source == "mail" and not (fact.meta.get("kind") or fact.sent or fact.trap)


def gendered(title: str, female: bool) -> str:
    """« Directeur·rice » → Directrice ou Directeur ; « financier·ère », « Chef·fe »…"""

    def one(m: re.Match) -> str:
        base, suffix = m.group(1), m.group(2)
        if not female:
            return base
        for old, new in (("eur", "rice"), ("f", "ve"), ("er", "ère")):
            if new.endswith(suffix) and base.endswith(old):
                return base[: -len(old)] + new
        return base + suffix

    return re.sub(r"(\w+)·(\w+)", one, title)


# --- la fiche ----------------------------------------------------------------------------------


@dataclass
class Sheet:
    text: str
    must: list[tuple[str, ...]]  # chaque élément : la forme montrée, puis les variantes admises
    style: dict = field(default_factory=dict)
    target: str = ""  # prénom de la personne saluée
    signer: str = ""  # prénom de l'expéditeur, attendu dans la signature
    words: tuple[int, int] = (0, 10_000)  # longueur demandée du corps, en mots (v4 à v6)
    examples: list[str] = field(default_factory=list)  # identifiants des exemples montrés
    deadline: tuple[str, ...] | None = None  # à écrire dans la demande (v7)
    tone_mark: tuple[str, ...] | None = None  # à écrire dans le ton (v7)
    signature: str = ""  # signature complète, ajoutée par le code (v7)
    frame: dict = field(default_factory=dict)  # salutation, fin, second acte, ordre (v7)


def sheet(persona: Persona, fact: Fact, rng: random.Random) -> Sheet:
    """La fiche d'un fait : tout ce que le texte doit dire, et rien d'autre."""
    m, a = persona.manager, fact.author
    tu = fact.meta.get("tu", True)
    o = fact.obj.phrase if fact.obj else "le dossier"
    target = fact.addressee or m
    slots = {
        "o": o,
        "qui": fact.meta.get("qui", ""),
        "jour": fact.meta.get("jour", "demain"),
        "x": target.first,
        "a": a.first,
        "m": m.first,
        "soi": "elle-même" if a.female else "lui-même",
    }
    must: list[tuple[str, ...]] = []
    lines = [
        f"Entreprise : {persona.metier.company}",
        f"De : {a.name}, {gendered(a.role, a.female)} — pour {m.first} : {RELATION[a.kind]}",
    ]
    if fact.to_manager:
        lines.append(f"À : {m.name}, {gendered(m.role, m.female)} (destinataire direct)")
    else:
        lines.append(
            f"À : {target.name}, {target.role} (équipe de {m.first}) ; {m.name} est seulement "
            f"en copie : la demande s'adresse à {target.first}, pas à {m.first}"
        )
    lines.append(f"Adresse : {ADDRESS[tu]}")
    lines.append(f"Sujet : {o}")
    if fact.meta.get("relance"):
        context = "c'est une relance : {a} a déjà écrit il y a deux jours et n'a pas eu de réponse."
    else:
        context = rng.choice(CASE_CONTEXT.get(fact.case_type, CASE_CONTEXT["fond"]))
    if fact.act != "rien":
        context = context.format(**slots)
        lines.append(f"Contexte : {context[:1].upper()}{context[1:]}")
    if fact.act in ACTS:
        lines.append("Demande : " + rng.choice(ACTS[fact.act]).format(**slots) + ".")
        lines.append("Qui agit : " + WHO.get(fact.act, WHO_DEFAULT).format(**slots) + ".")
        lines.append(f"Forme : {FORM[fact.form]}")
    elif fact.act == "informer":
        lines.append(
            f"Information donnée : {o} est {rng.choice(P.STATUTS)}. "
            f"{target.first} n'a rien à faire ni à répondre."
        )
        lines.append(NO_REQUEST_PIECE["informer"])
    else:
        lines.append(
            f"Rien n'est demandé : {a.first} remercie ou accuse réception concernant {o}, "
            "sans nouvelle information."
        )
        lines.append(NO_REQUEST_PIECE["rien"])
    # v7 : le second acte éventuel est ajouté par le code (banque des gabarits), pas par le LLM.
    if fact.deadline:
        day = fact.deadline.date()
        shown = fr_day(day)
        lines.append(f"Échéance : {shown}")
        weekday, number, month = shown.split()
        must.append((shown, f"{number} {month}", f"{weekday} {number}", f"{number}/{day.month}"))
    elif fact.act not in ("informer", "rien"):
        lines.append("Échéance : aucune, n'en mets pas")
    lines.append(f"Ton : {TONE[fact.tone]}")
    deadline_mark = must[0] if must else None
    tone_mark = None
    if fact.tone == "sans_urgence":
        tone_mark = (rng.choice(MUST_CALM[tu]),)
    elif fact.tone == "urgent":
        tone_mark = (rng.choice(MUST_URGENT),)
    if tone_mark:
        must.append(tone_mark)
    if deadline_mark:
        lines.append(f"À écrire dans la demande : « {deadline_mark[0]} »")
    if tone_mark:
        lines.append(f"À écrire dans le ton : « {tone_mark[0]} »")
    # Le style : une personnalité tirée par fait. Longueur, ouverture et fautes ne sont plus
    # demandées (v7) : le code assemble le mail, et les fautes embrouillaient l'annotation.
    style = {"trait": rng.choice(STYLES)}
    lines.append(f"Style : {style['trait']}")
    if tu:
        signature = a.first
    elif a.kind.startswith("externe"):
        signature = f"{a.name}\n{a.role.split(', ', 1)[-1]}"
    else:
        signature = f"{a.name}\n{gendered(a.role, a.female)}"
    lines.append(f"Signature : {signature.replace(chr(10), ', ')}")
    # Ce que le code ajoute autour des morceaux (v7), tiré par fait : salutation, second acte
    # (banque des gabarits), formule de fin, et l'ordre contexte / demande.
    greetings = [g for g in (P.GREETING_TU if tu else P.GREETING_VOUS) if g]
    civ = "Madame" if target.female else "Monsieur"
    second = ""
    if fact.secondary:
        second = P.resolve(P.SECOND[fact.secondary], tu).format(**slots)
        lines.append(f"Phrase ajoutée par le programme après la demande : « {second} »")
    frame = {
        "greeting": rng.choice(greetings).format(prenom=target.first, nom=target.last, civ=civ),
        "closing": P.resolve(rng.choice([c for c in P.CLOSING if c]), tu),
        "second": second,
        "request_first": fact.act in ACTS and rng.random() < REQUEST_FIRST_SHARE,
    }
    # Deux exemples d'autres actes (la forme d'un vrai mail, sans souffler la demande), au
    # même tu ou vous d'abord : un exemple au vous faisait glisser Qwen du tu au vous.
    pool = [e for e in EXAMPLES if e["act"] != fact.act]
    rng.shuffle(pool)
    pool.sort(key=lambda e: ("tutoiement" in e["fiche"]) != tu)
    # v8 : un mail sans demande voit d'abord l'exemple de l'autre acte sans demande (sinon,
    # tous ses exemples demandent quelque chose et le morceau « demande » restait vide). Tri
    # stable, sans nouveau tirage : les autres choix de la fiche ne changent pas.
    if fact.act in NO_REQUEST_PIECE:
        pool.sort(key=lambda e: e["act"] not in NO_REQUEST_PIECE)
    examples = [e["id"] for e in pool[:N_EXAMPLES]]
    return Sheet(
        "\n".join(lines),
        must,
        style,
        target.first,
        a.first,
        examples=examples,
        deadline=deadline_mark,
        tone_mark=tone_mark,
        signature=signature,
        frame=frame,
    )


def example_sheet(fiche: str) -> str:
    """Une fiche d'exemple (écrite pour la v4) au format v7 : style réduit à la personnalité,
    « À écrire » rangé dans la demande (une date) ou dans le ton (le reste)."""
    fiche = re.sub(r"^(Style : [^;\n]+).*$", r"\1", fiche, flags=re.MULTILINE)

    def where(m: re.Match) -> str:
        dated = any(month in m.group(1) for month in MONTHS)
        return f"À écrire dans {'la demande' if dated else 'le ton'} : « {m.group(1)} »"

    return re.sub(r"^À écrire : « (.+) »$", where, fiche, flags=re.MULTILINE)


def messages(sh: Sheet) -> list[dict]:
    """La conversation envoyée au modèle : consigne, exemples (fiche → morceaux), la fiche."""
    by_id = {e["id"]: e for e in EXAMPLES}
    out = [{"role": "system", "content": SYSTEM}]
    for eid in sh.examples:
        answer = json.dumps(MORCEAUX[eid], ensure_ascii=False)
        out += [
            {"role": "user", "content": example_sheet(by_id[eid]["fiche"])},
            {"role": "assistant", "content": answer},
        ]
    return out + [{"role": "user", "content": sh.text}]


# --- les contrôles -----------------------------------------------------------------------------


def _norm(text: str) -> str:
    """Minuscules, sans accents, apostrophes droites : pour comparer sans chipoter."""
    text = text.replace("’", "'").replace("œ", "oe").lower()
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()


_NAMES = set(FEMALE) | set(MALE) | set(LAST)
_ORGS = {o for m in METIERS for o in (m.company, *m.clients, *m.suppliers)}
_AMBIGUOUS = {"Petit", "Blanc"}  # aussi des mots courants en début de phrase
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+(\.[\w-]+)+")
_URL = re.compile(r"https?://|www\.", re.IGNORECASE)
# Alphabets d'autres langues (cyrillique, hébreu et arabe, kana, idéogrammes, hangûl).
_FOREIGN = re.compile(
    "["
    + "".join(
        f"{chr(a)}-{chr(b)}"
        for a, b in (
            (0x400, 0x4FF),
            (0x590, 0x6FF),
            (0x3040, 0x30FF),
            (0x3400, 0x9FFF),
            (0xAC00, 0xD7AF),
        )
    )
    + "]"
)
_FRENCH = set(
    re.findall(
        r"\w+",
        "le la les de des du un une et est pour que qui je tu vous a en sur merci ce on pas ne me "
        "te nous au avec il bonjour ton votre mon ma si d l j c n qu",
    )
)
_URGENT = re.compile(
    r"\burgen|au plus vite|des que possible|des que (tu peux|vous pouvez)|rapidement"
    r"|priorit|sans tarder|\bpresse\b|immediatement|\basap\b|ca bloque|bloquant"
    # Échantillon M3 (v8, relu à l'aveugle) : lues comme urgentes dans des mails neutres.
    r"|brefs? delais|meilleurs delais|des que (cela|ce) sera possible"
    r"|des que (tu|vous) (le |la |les )?(pourras|pourrez)"
)
# Le calme écrit, sous toutes ses formes : la sonde v2 a laissé passer « dès que tu as un
# moment » et « quand tu es libre », que le guide range dans Sans urgence.
_CALM = re.compile(
    # Corpus v7 : « ce n'est pas une demande urgente » était lu comme urgent.
    r"rien d'urgent|pas (d'|une |un )?((demande|sujet|dossier|chose) )?urgen\w*|aucune urgence|sans urgence|pas presse\w*|rien ne presse"
    r"|(quand|des que|lorsque) (tu as|vous avez|vous aurez|tu auras) (un moment|le temps|un instant)"
    r"|quand (tu es|tu seras|vous etes|vous serez) (libre|dispo)|quand (tu peux|vous pouvez)"
    r"|a l'occasion|a ta convenance|a votre convenance|ca peut attendre|prends ton temps"
    r"|prenez votre temps|pas de stress"
    # Sonde v7 : « à l'heure qui te convient » est passé dans un mail neutre.
    r"|(a l'heure|au moment|le jour|quand) (qui|que) (te|vous) (convient|conviendra|arrange)"
    r"|quand (tu veux|vous voulez|tu voudras|vous voudrez)|des que tu peux te liberer"
    r"|sans te presser|sans vous presser|tranquillement"
)
# Familles de mots qui portent chaque acte (texte normalisé). L'acte d'un texte qui n'en
# contient aucune a sans doute changé : le LLM a reformulé la demande en une autre.
# Corpus v7 : « contacter X », « mets à jour », « faire parvenir » (Faire) et « faire savoir »
# (Répondre) étaient rejetés alors que l'acte est le même : ajoutés.
_ACT_WORDS = {
    "decider": r"valid|sign|approuv|accord|feu vert|\bvisa\b|tranch|arbitr|decision|decid|engag"
    r"|choisi|choix",
    "faire": r"envo|transm|appel|\bmet\w* a jour|mise a jour|depos|complet|faire suivre|relanc"
    r"|saisi|imprim|reserv|organis|inscri|contact|parvenir",
    "repondre": r"avis|\bdire\b|dis-moi|dites-moi|confirm|dispo|\bdate\b|repon|retour|pens"
    r"|prefer|convien|propos|faire savoir|moi savoir|nous savoir|\?",
    "relire": r"reli|relecture|verifi|\boeil\b|corrig",
    "preparer": r"reunion|seance|comite|present|prepar",
}
# « merci de m'avoir envoyé » remercie, il ne demande rien (corpus v7, mail « Rien » rejeté).
_REQUEST = re.compile(
    r"\?|merci de(?! (m'|t'|l'|nous |vous )?avoir\b)|peux-tu|pouvez-vous|pourrais|pourriez|tu peux|vous pouvez|j'ai besoin"
    r"|il faut que|n'hesite|dis-moi|dites-moi|tiens-moi|tenez-moi|fais-moi|faites-moi"
)
# Une demande de retour fait passer Préparer, Relire, S'informer et Rien à Répondre (plus
# engageant) : interdite pour ces actes. Pour Décider et Faire, elle ne change pas l'étiquette.
_FEEDBACK = re.compile(
    r"fai(s|tes)[- ]moi signe|laisse(z)?[- ]moi savoir|me laisser savoir|tien(s|ez)[- ]moi"
    r"|di(s|tes)[- ]moi quand|preven(s|ez)[- ]moi|confirme(z)?[- ]moi|me (dire|confirmer) quand"
    r"|revien(s|ez) vers moi|me faire (signe|savoir|part)|me tenir (au courant|informee?)"
    r"|me (le |la )?confirmer"
)
REPLY_FLIPS = ("preparer", "relire", "informer", "rien")
# Une décision demandée en plus fait passer tout autre acte à Décider (le plus engageant) :
# « merci de la relire et de me confirmer ton accord » (Gemma, essai v4) se lit Décider.
_DECIDE_ASK = re.compile(
    r"\b(ton|votre) (accord|feu vert|aval|validation|visa|approbation|signature)\b"
    r"|(peux-tu|pouvez-vous|pourrais-tu|pourriez-vous|merci de|il faut) (le |la |les )?"
    r"(valider|signer|approuver|arbitrer|trancher)"
)
# Texte à trous laissé par le modèle : « [Lien vers le devis] » (Gemma, essai v4).
_PLACEHOLDER = re.compile(r"\[[^\]\n]{1,60}\]|\{[^}\n]{1,60}\}|<[^>\n]{1,60}>")
# Rôle inversé : l'expéditeur décide ou agit à la place de la personne de la fiche.
_SELF = {
    "decider": re.compile(
        r"\b(mon accord|ma validation|ma signature|ma decision|mon feu vert|mon visa|je valide"
        r"|je signe|j'approuve|je donne mon|je n'ai pas (encore )?(donne|valide|signe))"
    ),
    "faire": re.compile(
        r"\bje (vais|peux) (te |t'|vous |lui |l'|le |la |les )*(envoy|transm|appel|mettre a jour"
        r"|depos)|je m'en (occupe|charge)"
    ),
}
# Intitulés de la fiche recopiés dans le corps (« tu te prépares : … » en v2).
_LEAK = re.compile(
    r"\b(demande|qui agit|forme|ton|echeance|sujet|objet|adresse|longueur|signature|a ecrire"
    r"|information donnee|seconde demande|expediteur|destinataire|entreprise)\s?:"
)
_GREETING = re.compile(r"(bonjour|bonsoir|salut|hello|cher|chere|madame|monsieur)\b")
_CASUAL = re.compile(r"\bbis(ous?|es|ou)\b|\bfav'|\bbiz\b|\bmdr\b|\blol\b|\bxoxo\b")
_EMOJI = re.compile(
    "[" + "".join(f"{chr(a)}-{chr(b)}" for a, b in ((0x2600, 0x27BF), (0x1F300, 0x1FAFF))) + "]"
)
_TU = re.compile(r"\b(tu|toi|ton|ta|tes)\b|\bt'")
COPY_RUN = 8  # mots consécutifs repris d'un exemple : copie


def _runs(text: str, n: int = COPY_RUN) -> set[tuple[str, ...]]:
    words = re.findall(r"\w+", _norm(text))
    return {tuple(words[i : i + n]) for i in range(len(words) - n + 1)}


_EXAMPLE_RUNS = {
    e["id"]: _runs(e["corps"]) | _runs(" ".join(MORCEAUX[e["id"]].values())) for e in EXAMPLES
}
_VOUS = re.compile(r"\b(vous|votre|vos)\b")


def _quote(pattern: re.Pattern, text: str, limit: int = 3) -> str:
    """Les passages qui déclenchent une règle, cités : « au plus vite », « ca bloque »."""
    found = list(dict.fromkeys(m.group(0).strip() for m in pattern.finditer(text)))
    return ", ".join(f"« {x} »" for x in found[:limit])


def _present(alternatives: tuple[str, ...], text: str) -> bool:
    return any(_norm(x) in text for x in alternatives)


def _object_words(fact: Fact) -> tuple[list[str], list[str]]:
    """Noms propres de l'objet (tous exigés) et autres mots pleins (la moitié exigée)."""
    words = re.findall(r"\w+", _short(fact))
    proper = [w for w in words if w[:1].isupper() or w.isdigit()]
    common = [w for w in words if w not in proper and len(w) >= 4]
    return proper, common


def _invented(text: str, sh: Sheet) -> list[str]:
    """Rien d'inventé : personnes, organisations, coordonnées, chiffres, jours, mois absents
    de la fiche."""
    reasons = []
    low, sheet_low = _norm(text), _norm(sh.text)
    for name in sorted(_NAMES - set(re.findall(r"\w+", sh.text))):
        hits = re.finditer(rf"(?<![\w-]){name}(?![\w-])", text)
        if name in _AMBIGUOUS:  # seulement précédé d'un prénom : « Julie Petit »
            hits = (h for h in hits if re.search(r"[A-ZÀ-Ý]\w+\s*$", text[: h.start()]))
        if next(hits, None):
            reasons.append(f"nom inventé : {name}")
    reasons += [
        f"organisation inventée : {org}"
        for org in sorted(_ORGS)
        if org in text and org not in sh.text
    ]
    if _EMAIL.search(text) or _URL.search(text):
        reasons.append("adresse ou lien inventé")
    invented = sorted(set(re.findall(r"\d+", text)) - set(re.findall(r"\d+", sh.text)))
    if invented:
        reasons.append(f"chiffres inventés : {', '.join(invented)}")
    for word in (*WEEKDAYS, *MONTHS):
        w = _norm(word)
        if re.search(rf"\b{w}\b", low) and not re.search(rf"\b{w}\b", sheet_low):
            reasons.append(f"date inventée : {word}")
    return reasons


def extract(corps: str) -> str:
    """Ce que le modèle verra du corps : l'extrait du vrai connecteur mail (`mail_graph`),
    sans salutation, citation ni signature, sur une ligne, 500 caractères au plus."""
    return graph_snippet({"body": {"contentType": "text", "content": corps}})


def check(fact: Fact, sh: Sheet, objet: str, corps: str) -> list[str]:
    """Les raisons de rejeter un texte (liste vide : il passe).

    Ce qui porte l'étiquette (demande, échéance, ton, objet) est cherché dans l'objet et
    l'extrait que lira le modèle ; ce qui ne doit figurer nulle part (invention, autre
    demande, registre), dans tout le mail.
    """
    reasons = []
    full = f"{objet}\n{corps}"
    low = _norm(full)
    body = _norm(corps)
    seen = _norm(f"{objet}\n{extract(corps)}")
    if not objet.strip() or len(objet) > 90 or "\n" in objet.strip():
        reasons.append("objet vide, trop long ou sur plusieurs lignes")
    # La longueur demandée, avec de la marge : un mail de 3 lignes n'apporte rien de plus
    # qu'un gabarit, un mail fleuve finit par inventer.
    n_words = len(re.findall(r"\w+", corps))
    low_words, high_words = round(sh.words[0] * 0.75), round(sh.words[1] * 1.4)
    if not low_words <= n_words <= high_words:
        reasons.append(
            f"longueur hors bornes ({n_words} mots, attendu {sh.words[0]}-{sh.words[1]})"
        )
    runs = _runs(corps)
    copied = [eid for eid, example in _EXAMPLE_RUNS.items() if runs & example]
    if copied:
        reasons.append(f"exemple recopié : {', '.join(copied)}")
    if _FOREIGN.search(full):
        reasons.append("caractères d'une autre langue")
    tokens = re.findall(r"[a-z]+", body)
    if tokens and sum(t in _FRENCH for t in tokens) / len(tokens) < 0.15:
        reasons.append("pas en français")
    for x in sh.must:
        if not _present(x, body):
            reasons.append(f"manque « {x[0]} »")
        elif not _present(x, seen):
            reasons.append(f"« {x[0]} » hors de l'extrait lu")
    proper, common = _object_words(fact)
    missing = [w for w in proper if _norm(w) not in seen]
    if missing or (common and sum(_norm(w)[:5] in seen for w in common) * 2 < len(common)):
        reasons.append(f"objet absent de l'extrait lu ({_short(fact)})")
    reasons += _invented(full, sh)
    # Le ton écrit : c'est lui seul qui porte l'urgence. Neutre : rien nulle part ; urgent ou
    # calme : écrit dans l'extrait lu.
    urgent_anywhere = bool(_URGENT.search(_CALM.sub(" ", low)))
    calm_seen = bool(_CALM.search(seen))
    urgent_seen = bool(_URGENT.search(_CALM.sub(" ", seen)))
    # Les mots fautifs sont cités : la réécriture guidée les renvoie au modèle.
    if fact.tone == "normal" and (urgent_anywhere or _CALM.search(low)):
        words = _quote(_URGENT, _CALM.sub(" ", low)) or _quote(_CALM, low)
        reasons.append(f"ton changé ({words} écrit alors que le ton est neutre)")
    if fact.tone == "sans_urgence" and urgent_anywhere:
        reasons.append(f"ton changé ({_quote(_URGENT, _CALM.sub(' ', low))} dans un mail calme)")
    elif fact.tone == "sans_urgence" and not calm_seen:
        reasons.append("ton changé (le calme n'est pas écrit dans l'extrait lu)")
    if fact.tone == "urgent" and not urgent_seen:
        reasons.append("ton changé (l'urgence n'est pas écrite dans l'extrait lu)")
    # L'acte demandé, lisible dans l'extrait ; aucun acte plus engageant ajouté.
    if fact.act in _ACT_WORDS and not re.search(_ACT_WORDS[fact.act], seen):
        reasons.append("acte changé (aucun mot de l'acte demandé dans l'extrait lu)")
    if "decider" not in (fact.act, fact.secondary) and _DECIDE_ASK.search(body):
        reasons.append(f"décision ajoutée ({_quote(_DECIDE_ASK, body)} : acte plus engageant)")
    if _PLACEHOLDER.search(full):
        reasons.append(f"texte à trous ({_quote(_PLACEHOLDER, full)})")
    if fact.act in ("informer", "rien") and _REQUEST.search(body):
        reasons.append(f"demande ajoutée à un mail qui n'en porte pas ({_quote(_REQUEST, body)})")
    elif fact.act in REPLY_FLIPS and _FEEDBACK.search(body):
        reasons.append(f"demande de retour ajoutée ({_quote(_FEEDBACK, body)})")
    if fact.act in ACTS and fact.form == "question" and "?" not in seen:
        reasons.append("forme changée (la demande devait être une question)")
    if fact.act in _SELF and _SELF[fact.act].search(body):
        reasons.append("rôle inversé (l'expéditeur décide ou agit lui-même)")
    tu = fact.meta.get("tu", True)
    if (tu and _VOUS.search(body)) or (not tu and _TU.search(body)):
        reasons.append("tutoiement ou vouvoiement non respecté")
    # La forme d'un mail : intitulés de la fiche, salutation, signature, registre.
    if _LEAK.search(body):
        reasons.append("intitulé de la fiche recopié")
    lines = [line for line in body.splitlines() if line.strip()]
    if lines and not (_GREETING.match(lines[0]) or _norm(sh.target) in lines[0]):
        reasons.append("pas de salutation")
    if sh.signer and not any(_norm(sh.signer) in line for line in lines[-3:]):
        reasons.append("pas de signature")
    if _CASUAL.search(body) or _EMOJI.search(corps):
        reasons.append("registre familier")
    return reasons


# --- les morceaux (v7) -------------------------------------------------------------------------


# Échantillon M3 (v8, relu à l'aveugle) : environ 20 mails « Rien » sur 68 lus « S'informer »,
# le contexte ou la conclusion y ajoutant un fait (« j'ai passé la commande », « je te tiens au
# courant »). Un remerciement n'a ni l'un ni l'autre : ils sont toujours retirés, comme un
# morceau facultatif qui échoue. Aucune règle ne sait dire « pas d'information nouvelle ».
NO_INFO_DROP = {
    "contexte": "un remerciement n'apporte pas de contexte (retiré d'office)",
    "conclusion": "un remerciement n'apporte pas de conclusion (retirée d'office)",
}


def required(kind: str, fact: Fact) -> bool:
    """Un morceau sans lequel le mail ne dit pas le fait : refusé, il faut réécrire. Les autres
    (contexte, conclusion) sont simplement retirés s'ils échouent."""
    return kind in ("objet", "demande") or (kind == "ton" and fact.tone != "normal")


def check_piece(kind: str, text: str, fact: Fact, sh: Sheet) -> list[str]:
    """Les raisons de refuser un morceau (liste vide : il passe)."""
    reasons = []
    low = _norm(text)
    n = len(re.findall(r"\w+", text))
    lo, hi = PIECE_WORDS[kind]
    if not lo <= n <= hi:
        reasons.append(f"longueur hors bornes ({n} mots, attendu {lo} à {hi})")
    if kind == "objet" and "\n" in text.strip():
        reasons.append("objet sur plusieurs lignes")
    if _FOREIGN.search(text):
        reasons.append("caractères d'une autre langue")
    tokens = re.findall(r"[a-z]+", low)
    # Un objet est un groupe nominal sans mots-outils (« Bon de commande Voirie Plus : choix
    # attendu ») : la part de mots courants n'y dit rien de la langue (corpus v7).
    french = sum(t in _FRENCH for t in tokens) / len(tokens) if tokens else 1
    if kind != "objet" and len(tokens) >= 6 and french < 0.15:
        reasons.append("pas en français")
    reasons += _invented(text, sh)
    if _PLACEHOLDER.search(text):
        reasons.append(f"texte à trous ({_quote(_PLACEHOLDER, text)})")
    tu = fact.meta.get("tu", True)
    if (tu and _VOUS.search(low)) or (not tu and _TU.search(low)):
        reasons.append("tutoiement ou vouvoiement non respecté")
    if _LEAK.search(low):
        reasons.append("intitulé de la fiche recopié")
    if _CASUAL.search(low) or _EMOJI.search(text):
        reasons.append("registre familier")
    # Le ton n'est porté que par son morceau ; les autres sont neutres.
    calm, urgent = _CALM.search(low), _URGENT.search(_CALM.sub(" ", low))
    if kind == "ton":
        if fact.tone == "urgent" and (calm or not urgent):
            reasons.append("ton changé (l'urgence n'est pas écrite)")
        if fact.tone == "sans_urgence" and (urgent or not calm):
            reasons.append("ton changé (le calme n'est pas écrit)")
        if sh.tone_mark and not _present(sh.tone_mark, low):
            reasons.append(f"manque « {sh.tone_mark[0]} »")
        if "?" in text:
            reasons.append("question dans la phrase de ton")
        # Échantillon M3 : « À écrire quand tu as un moment », recopié sans les deux-points.
        if "a ecrire" in low and "intitulé de la fiche recopié" not in reasons:
            reasons.append("intitulé de la fiche recopié")
    # Une formule du même ton dans la demande (« signe-le dès que possible » dans un mail
    # urgent) ne change pas l'étiquette : permise depuis la v8 (corpus v7 : 18 rejets, dont
    # une majorité de ce cas). Le contrôle du mail entier garde la cohérence du ton.
    elif urgent and not (kind in ("objet", "demande") and fact.tone == "urgent"):
        reasons.append(f"ton changé ({_quote(_URGENT, _CALM.sub(' ', low))} hors du ton)")
    elif calm and not (kind == "demande" and fact.tone == "sans_urgence"):
        reasons.append(f"ton changé ({_quote(_CALM, low)} hors du ton)")
    if kind in ("contexte", "conclusion"):
        for pattern in (_REQUEST, _FEEDBACK, _DECIDE_ASK):
            if pattern.search(low):
                reasons.append(f"demande ajoutée ({_quote(pattern, low)})")
                break
    if kind == "demande":
        reasons += _request_reasons(text, low, fact, sh)
    return reasons


def _request_reasons(text: str, low: str, fact: Fact, sh: Sheet) -> list[str]:
    """Ce que doit porter la phrase de demande : l'acte, l'échéance, la forme, rien de plus."""
    reasons = []
    if fact.act in _ACT_WORDS and not re.search(_ACT_WORDS[fact.act], low):
        reasons.append("acte changé (aucun mot de l'acte demandé)")
    if sh.deadline and not _present(sh.deadline, low):
        reasons.append(f"manque « {sh.deadline[0]} »")
    if fact.act in ACTS and fact.form == "question" and "?" not in text:
        reasons.append("forme changée (la demande devait être une question)")
    if fact.act in ACTS and fact.form != "question" and "?" in text:
        reasons.append("forme changée (la demande ne devait pas être une question)")
    if "decider" not in (fact.act, fact.secondary) and _DECIDE_ASK.search(low):
        reasons.append(f"décision ajoutée ({_quote(_DECIDE_ASK, low)} : acte plus engageant)")
    if fact.act in ("informer", "rien") and _REQUEST.search(low):
        reasons.append(f"demande ajoutée à un mail qui n'en porte pas ({_quote(_REQUEST, low)})")
    elif fact.act in REPLY_FLIPS and _FEEDBACK.search(low):
        reasons.append(f"demande de retour ajoutée ({_quote(_FEEDBACK, low)})")
    if fact.act in _SELF and _SELF[fact.act].search(low):
        reasons.append("rôle inversé (l'expéditeur décide ou agit lui-même)")
    return reasons


def assemble(sh: Sheet, pieces: dict[str, str]) -> tuple[str, str]:
    """Le mail assemblé par le code : salutation, contexte et demande (dans un ordre tiré par
    fait), second acte, ton, conclusion, formule de fin et signature."""
    f = sh.frame
    core = [pieces.get("contexte", ""), pieces.get("demande", "")]
    if f.get("request_first"):
        core.reverse()
    first = " ".join(x for x in (*core, f.get("second", ""), pieces.get("ton", "")) if x)
    paragraphs = [f.get("greeting", ""), first, pieces.get("conclusion", "")]
    paragraphs.append("\n".join(x for x in (f.get("closing", ""), sh.signature) if x))
    return pieces.get("objet", ""), "\n\n".join(p for p in paragraphs if p)


# --- Ollama ------------------------------------------------------------------------------------


class Ollama:
    """Client minimal de l'API HTTP locale d'Ollama (urllib, aucune dépendance)."""

    def __init__(self, model: str = MODEL, url: str = OLLAMA_URL, timeout: float = 300) -> None:
        self.model, self.url, self.timeout = model, url.rstrip("/"), timeout

    def _call(self, path: str, body: dict | None = None) -> dict:
        data = json.dumps(body).encode() if body is not None else None
        request = urllib.request.Request(
            self.url + path, data=data, headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            return json.loads(response.read())

    def identity(self) -> dict:
        """Le modèle exact : nom, empreinte, version d'Ollama."""
        tags = self._call("/api/tags")["models"]
        found = next((m for m in tags if m["name"] == self.model), None)
        if found is None:
            raise RuntimeError(f"modèle absent d'Ollama : {self.model} (ollama pull {self.model})")
        version = self._call("/api/version")["version"]
        return {"model": self.model, "digest": found["digest"], "ollama": version}

    def chat(self, conversation: list[dict], seed: int) -> dict:
        body = {
            "model": self.model,
            "stream": False,
            "format": {
                "type": "object",
                "properties": {kind: {"type": "string"} for kind in PIECES},
                "required": list(PIECES),
            },
            "options": {
                "temperature": TEMPERATURE,
                "top_p": TOP_P,
                "seed": seed,
                "num_predict": MAX_TOKENS,
                "num_ctx": NUM_CTX,
            },
            "messages": conversation,
        }
        return json.loads(self._call("/api/chat", body)["message"]["content"])


class Offline:
    """Client hors ligne, pour assembler le corpus sans GPU (`llm_corpus --assembler`) : il
    rend l'identité du modèle enregistrée avec les textes et refuse tout appel. Aucun texte
    n'est rédigé ailleurs que sur le PC qui a le LLM."""

    def __init__(self, model: str, digest: str) -> None:
        self.model, self.digest = model, digest

    def identity(self) -> dict:
        return {"model": self.model, "digest": self.digest, "ollama": None}

    def chat(self, conversation: list[dict], seed: int) -> dict:
        raise RuntimeError("client hors ligne : aucun appel au LLM pendant l'assemblage")


# --- l'écrivain --------------------------------------------------------------------------------


@dataclass
class Draft:
    """Un texte rédigé et sa traçabilité (ce qui est écrit à côté de chaque texte)."""

    item_id: str
    objet: str
    corps: str
    ok: bool
    model: str
    digest: str
    prompt_version: str
    seed: int  # graine de l'essai retenu
    tries: int
    seconds: float  # temps total de rédaction, essais rejetés compris
    rejects: list[list[str]]  # raisons de rejet de chaque essai
    examples: list[str] = field(default_factory=list)  # exemples montrés au modèle
    pieces: dict = field(default_factory=dict)  # morceaux rendus par le modèle (dernier essai)
    dropped: list[str] = field(default_factory=list)  # morceaux facultatifs retirés, et pourquoi


def repair(reasons: list[str], tu: bool = True) -> str:
    """Le message de réécriture guidée : les raisons du refus, telles que les contrôles les
    donnent (avec les mots fautifs), et la consigne de ne corriger que ces points.

    v8 : il rappelle toujours le tu ou le vous (corpus v7 : 29 rejets sur 113 n'avaient plus
    que du « vous » dans un mail tutoyé, glissé pendant la réécriture)."""
    listed = "\n".join(f"- {r}" for r in reasons)
    tone = any("ton changé" in r for r in reasons)
    return (
        f"Ces morceaux sont refusés :\n{listed}\n"
        "Réécris-les en corrigeant seulement ces points, sans raccourcir les autres morceaux."
        + (" Hors du morceau « ton », aucune formule de délai ni de calme." if tone else "")
        + f" Dans tous les morceaux : {ADDRESS[tu]}."
        + " Réponds uniquement en JSON avec les cinq clés."
    )


def judge(fact: Fact, sh: Sheet, pieces: dict[str, str]) -> tuple[dict, list[str], list[str]]:
    """Les morceaux gardés, les raisons de refuser l'essai, les morceaux facultatifs retirés.

    Un morceau requis qui échoue fait refuser l'essai ; un facultatif est retiré. Le mail
    assemblé repasse ensuite les contrôles du mail entier (extrait lu, inventions, copie).
    """
    kept, reasons, dropped = {}, [], []
    for kind in PIECES:
        text = pieces.get(kind, "").strip()
        if kind == "ton" and fact.tone == "normal":
            continue  # ton neutre : rien à écrire, un ton rendu quand même est ignoré
        if kind in NO_INFO_DROP and fact.act == "rien":
            if text:
                dropped.append(f"{kind} : {NO_INFO_DROP[kind]}")
            continue
        if not text:
            if required(kind, fact):
                reasons.append(f"{kind} : vide")
            continue
        problems = check_piece(kind, text, fact, sh)
        if not problems:
            kept[kind] = text
        elif required(kind, fact):
            reasons += [f"{kind} : {p}" for p in problems]
        else:
            dropped.append(f"{kind} : {'; '.join(problems)}")
    # Un contexte ou une conclusion qui répète la demande (« la procédure est à jour », deux
    # fois, essai v7) n'apporte rien et brouille la lecture : retiré.
    request_runs = _runs(kept.get("demande", ""), REPEAT_RUN)
    for kind in ("contexte", "conclusion"):
        if kind in kept and _runs(kept[kind], REPEAT_RUN) & request_runs:
            dropped.append(f"{kind} : répète la demande")
            del kept[kind]
    if not reasons:
        reasons = check(fact, sh, *assemble(sh, kept))
    return kept, reasons, dropped


def llm_seed(seed: int, item_id: str, attempt: int) -> int:
    digest = hashlib.blake2b(f"trainset:llm:{seed}:{item_id}:{attempt}".encode(), digest_size=4)
    return int.from_bytes(digest.digest(), "big") >> 1  # entier positif sur 31 bits


class LLMWriter:
    """Remplace le texte des gabarits par celui du LLM, fait par fait, étiquettes inchangées."""

    def __init__(
        self, persona: Persona, seed: int = 0, client: Ollama | Offline | None = None
    ) -> None:
        self.p = persona
        self.seed = seed
        self.client = client or Ollama()
        self.identity = self.client.identity()
        self.templates = Writer(persona, seed)

    def sheet(self, fact: Fact) -> Sheet:
        return sheet(self.p, fact, random.Random(f"trainset:llm:{self.seed}:{fact.item_id}"))

    def draft(self, fact: Fact) -> Draft:
        sh = self.sheet(fact)
        first = messages(sh)
        conversation = first
        rejects: list[list[str]] = []
        start = time.perf_counter()
        pieces: dict[str, str] = {}
        kept: dict[str, str] = {}
        dropped: list[str] = []
        for attempt in range(MAX_TRIES):
            seed = llm_seed(self.seed, fact.item_id, attempt)
            try:
                out = self.client.chat(conversation, seed)
                pieces = {kind: str(out[kind]).strip() for kind in PIECES}
                kept, reasons, dropped = judge(fact, sh, pieces)
                readable = True
            except (KeyError, TypeError, json.JSONDecodeError) as exc:
                reasons, readable = [f"réponse illisible : {exc}"], False
            rejects.append(reasons)
            if not reasons:
                break
            # Réécriture guidée : les morceaux refusés et leurs raisons, plutôt qu'un nouveau
            # tirage (Gemma refaisait souvent la même faute de ton).
            conversation = first
            if readable:
                conversation = first + [
                    {"role": "assistant", "content": json.dumps(pieces, ensure_ascii=False)},
                    {"role": "user", "content": repair(reasons, fact.meta.get("tu", True))},
                ]
        # Refusé : le mail montre quand même tous les morceaux rendus (la sonde mesure le LLM).
        objet, corps = assemble(sh, kept if not rejects[-1] else pieces)
        return Draft(
            item_id=fact.item_id,
            objet=objet,
            corps=corps,
            ok=not rejects[-1],
            model=self.identity["model"],
            digest=self.identity["digest"],
            prompt_version=PROMPT_VERSION,
            seed=seed,
            tries=len(rejects),
            seconds=round(time.perf_counter() - start, 2),
            rejects=rejects,
            examples=sh.examples,
            pieces=pieces,
            dropped=dropped,
        )

    def item(self, fact: Fact, d: Draft) -> Item:
        """L'Item de l'écrivain par gabarits, avec l'objet et l'extrait du texte du LLM, sous
        la forme du connecteur mail (`content_text` vide, extrait de 500 caractères)."""
        item, _ = self.templates.write(fact)
        title = d.objet
        if fact.meta.get("relance") and not re.match(r"(re ?:|relance)", _norm(title)):
            title = "Re: " + title
        fields = item.model_dump() | {"title": title, "snippet": extract(d.corps)}
        return Item.model_validate(fields)

    def marks(self, fact: Fact, item: Item) -> list[str]:
        """Les marques de lecture du texte retenu (règles du 23/11, guide) : la fiche
        demandait un avis ou un accord ; le texte dit que la copie est « pour suivre »."""
        out = []
        if fact.act == "repondre" and _AVIS_SHEET.search(self.sheet(fact).text):
            out.append(AVIS)
        if not fact.to_manager and _COPY_FOLLOW.search(item.snippet):
            out.append(COPIE_SUIVI)
        return out

    def choose(self, fact: Fact, d: Draft | None) -> tuple[Item | None, list[str]]:
        """L'Item retenu : le texte du LLM s'il a passé les contrôles, sinon celui des gabarits.

        Une seule règle pour la rédaction sur le PC GPU (`write`) et pour l'assemblage hors
        ligne à partir de `textes.jsonl` (`llm_corpus.assemble_items`).
        """
        item, used = self.templates.write(fact)
        if item is None or d is None or not rewritable(fact):
            return item, used
        if not d.ok:
            return item, used + [f"llm:{d.prompt_version}:rejete"]
        item = self.item(fact, d)
        return item, [f"llm:{d.prompt_version}", *self.marks(fact, item)]

    def write(self, fact: Fact) -> tuple[Item | None, list[str], Draft | None]:
        """L'Item (texte du LLM s'il passe les contrôles, sinon celui des gabarits)."""
        d = self.draft(fact) if rewritable(fact) else None  # un mail est toujours collecté
        item, used = self.choose(fact, d)
        return item, used, d
