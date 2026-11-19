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
from packages.collect.core.text import normalize_whitespace, strip_quotes_and_signature, truncate
from packages.collect.trainset import phrases as P
from packages.collect.trainset.facts import Fact
from packages.collect.trainset.llm_exemples import EXAMPLES
from packages.collect.trainset.metiers import METIERS
from packages.collect.trainset.world import FEMALE, LAST, MALE, MONTHS, WEEKDAYS, Persona, fr_day
from packages.collect.trainset.writer import Writer, _short

OLLAMA_URL = "http://localhost:11434"
MODEL = "qwen2.5:7b-instruct-q4_K_M"
# v2 (sonde : 14/20) : fautes de ton, rôles inversés, fiche recopiée.
# v3 (essai) : justes mais courts, la ligne « Demande » mise à l'impératif : des gabarits.
# v4 : des mails réalistes. Deux exemples écrits à la main avant chaque fiche, un contexte,
# un style et une longueur tirés par fait (et contrôlée), les garde-fous de la v3.
PROMPT_VERSION = "mail-v4"
TEMPERATURE = 0.8
TOP_P = 0.9
MAX_TRIES = 3
MAX_TOKENS = 700
NUM_CTX = 4096  # consigne + 2 exemples + fiche ≈ 1 650 jetons : fixé, pas laissé au défaut
N_EXAMPLES = 2

SYSTEM = """Tu écris des e-mails professionnels réalistes, en français, pour entraîner un assistant de tri. On te donne une fiche ; tu écris l'e-mail que l'expéditeur enverrait vraiment.

Ce qui rend l'e-mail réaliste :
- une introduction qui situe le dossier (pourquoi on écrit maintenant), puis la demande, puis une phrase de conclusion ;
- des phrases variées, avec la personnalité indiquée dans « Style » ; la longueur indiquée est obligatoire ;
- la demande reformulée avec tes propres mots : ne recopie jamais la ligne « Demande » ni les intitulés de la fiche (« Demande : », « Qui agit : »…).

Ce qui est interdit, car l'e-mail doit rester fidèle à la fiche :
1. Inventer un fait vérifiable : personne, entreprise, adresse, téléphone, lien, montant, chiffre, date ou jour de la semaine absents de la fiche. Tu peux ajouter des détails généraux, sans chiffre ni nom.
2. Ajouter une autre demande que celle de la fiche, même polie : pas de « fais-moi signe », « tiens-moi au courant », « dis-moi si tu as des questions », pas de question en plus.
3. Inverser les rôles : la personne qui agit est celle de la ligne « Qui agit ». L'expéditeur ne fait pas l'acte, ne dit pas qu'il va le faire et ne donne pas lui-même la décision.
4. Changer le ton. Ton neutre : aucune formule de délai ni d'humeur, ni d'urgence (urgent, au plus vite, dès que possible, rapidement, en priorité) ni de calme (quand tu as un moment, quand tu es libre, rien ne presse). Une échéance s'écrit simplement : ce n'est pas une urgence. Ton urgent ou calme : la formule « À écrire » le dit.
5. Changer d'adresse : le tutoiement ou le vouvoiement de la fiche, du début à la fin.
6. Un registre familier : pas de « bisous », d'émoticône ni d'abréviation.

Les éléments « À écrire » apparaissent dans le corps, avec ces mots-là. Commence par une salutation et termine par la signature de la fiche. Les exemples montrent le ton juste : ne recopie jamais leurs phrases.
Réponds uniquement en JSON : {"objet": "...", "corps": "..."}, avec des sauts de ligne entre les paragraphes du corps."""

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
OPENINGS = (
    "commence par situer le dossier",
    "commence par une nouvelle du dossier",
    "commence par rappeler votre dernier échange, sans date",
    "entre directement dans le sujet après la salutation",
)
# Longueur du corps en mots (salutation et signature comprises), contrôlée : les textes courts
# de la v3 n'apportaient rien de plus que les gabarits.
# Qwen 7B ne tient pas 120 à 180 mots (essai : 70 à 80) : bandes à sa portée.
LENGTHS = (((50, 80), 3), ((70, 110), 5), ((100, 150), 2))
SHORT = (35, 60)  # un remerciement ou un accusé de réception reste bref
DETAIL_SHARE = 0.4  # une phrase sur un aspect pratique du dossier, sans chiffre ni nom
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
TYPO_SHARE = 0.2  # une ou deux fautes de frappe, comme un mail écrit vite


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
    words: tuple[int, int] = (0, 10_000)  # longueur demandée du corps, en mots
    examples: list[str] = field(default_factory=list)  # identifiants des exemples montrés


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
    else:
        lines.append(
            f"Rien n'est demandé : {a.first} remercie ou accuse réception concernant {o}, "
            "sans nouvelle information."
        )
    if fact.secondary in SECOND:
        lines.append(
            "Seconde demande, plus courte, à la fin : "
            + SECOND[fact.secondary].format(**slots)
            + "."
        )
    elif fact.secondary:
        lines.append(f"Information à ajouter à la fin : {slots['qui']} est en congé jusqu'à lundi")
    if fact.deadline:
        day = fact.deadline.date()
        shown = fr_day(day)
        lines.append(f"Échéance : {shown}")
        weekday, number, month = shown.split()
        must.append((shown, f"{number} {month}", f"{weekday} {number}", f"{number}/{day.month}"))
    elif fact.act not in ("informer", "rien"):
        lines.append("Échéance : aucune, n'en mets pas")
    lines.append(f"Ton : {TONE[fact.tone]}")
    if fact.tone == "sans_urgence":
        must.append((rng.choice(MUST_CALM[tu]),))
    elif fact.tone == "urgent":
        must.append((rng.choice(MUST_URGENT),))
    lines += [f"À écrire : « {x[0]} »" for x in must]
    bands, weights = [b for b, _ in LENGTHS], [w for _, w in LENGTHS]
    words = SHORT if fact.act == "rien" else rng.choices(bands, weights)[0]
    if fact.meta.get("relance"):
        opening = "rappelle d'abord le message resté sans réponse"
    else:
        opening = rng.choice(OPENINGS)
    trait = rng.choice(STYLES)
    if trait.startswith("pressé"):  # des phrases courtes ne font pas un mail long
        words = min(words, bands[0])
    style = {
        "trait": trait,
        "ouverture": None if fact.act == "rien" else opening,
        "mots": words,
        "detail": fact.act != "rien" and rng.random() < DETAIL_SHARE,
        "fautes": rng.random() < TYPO_SHARE,
    }
    parts = [style["trait"], style["ouverture"], f"{words[0]} à {words[1]} mots"]
    lines.append("Style : " + " ; ".join(p for p in parts if p))
    if style["detail"]:
        lines.append(
            "Détail : ajoute une phrase sur un aspect pratique du dossier (organisation, équipe, "
            "suite prévue), sans chiffre ni nom, sans nouvelle demande"
        )
    if style["fautes"]:
        lines.append(
            "Fautes : glisse une ou deux petites fautes de frappe, comme un mail écrit vite"
        )
    if tu:
        signature = a.first
    elif a.kind.startswith("externe"):
        signature = f"{a.name}, {a.role.split(', ', 1)[-1]}"
    else:
        signature = f"{a.name}, {gendered(a.role, a.female)}"
    lines.append(f"Signature : {signature}")
    # Deux exemples d'autres actes (la forme d'un vrai mail, sans souffler la demande), au
    # même tu ou vous d'abord : un exemple au vous faisait glisser Qwen du tu au vous.
    pool = [e for e in EXAMPLES if e["act"] != fact.act]
    rng.shuffle(pool)
    pool.sort(key=lambda e: ("tutoiement" in e["fiche"]) != tu)
    examples = [e["id"] for e in pool[:N_EXAMPLES]]
    return Sheet("\n".join(lines), must, style, target.first, a.first, words, examples)


def messages(sh: Sheet) -> list[dict]:
    """La conversation envoyée au modèle : consigne, exemples (fiche → mail), puis la fiche."""
    by_id = {e["id"]: e for e in EXAMPLES}
    out = [{"role": "system", "content": SYSTEM}]
    for eid in sh.examples:
        e = by_id[eid]
        answer = json.dumps({"objet": e["objet"], "corps": e["corps"]}, ensure_ascii=False)
        out += [{"role": "user", "content": e["fiche"]}, {"role": "assistant", "content": answer}]
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
)
# Le calme écrit, sous toutes ses formes : la sonde v2 a laissé passer « dès que tu as un
# moment » et « quand tu es libre », que le guide range dans Sans urgence.
_CALM = re.compile(
    r"rien d'urgent|pas (d')?urgen\w*|aucune urgence|sans urgence|pas presse\w*|rien ne presse"
    r"|(quand|des que|lorsque) (tu as|vous avez|vous aurez|tu auras) (un moment|le temps|un instant)"
    r"|quand (tu es|tu seras|vous etes|vous serez) (libre|dispo)|quand (tu peux|vous pouvez)"
    r"|a l'occasion|a ta convenance|a votre convenance|ca peut attendre|prends ton temps"
    r"|prenez votre temps|pas de stress"
)
# Familles de mots qui portent chaque acte (texte normalisé). L'acte d'un texte qui n'en
# contient aucune a sans doute changé : le LLM a reformulé la demande en une autre.
_ACT_WORDS = {
    "decider": r"valid|sign|approuv|accord|feu vert|\bvisa\b|tranch|arbitr|decision|decid|engag"
    r"|choisi|choix",
    "faire": r"envo|transm|appel|mettre a jour|mise a jour|depos|complet|faire suivre|relanc"
    r"|saisi|imprim|reserv|organis|inscri",
    "repondre": r"avis|\bdire\b|dis-moi|dites-moi|confirm|dispo|\bdate\b|repon|retour|pens"
    r"|prefer|convien|propos|\?",
    "relire": r"reli|relecture|verifi|\boeil\b|corrig",
    "preparer": r"reunion|seance|comite|present|prepar",
}
_REQUEST = re.compile(
    r"\?|merci de|peux-tu|pouvez-vous|pourrais|pourriez|tu peux|vous pouvez|j'ai besoin"
    r"|il faut que|n'hesite|dis-moi|dites-moi|tiens-moi|tenez-moi|fais-moi|faites-moi"
)
# Une demande de retour fait passer Préparer, Relire, S'informer et Rien à Répondre (plus
# engageant) : interdite pour ces actes. Pour Décider et Faire, elle ne change pas l'étiquette.
_FEEDBACK = re.compile(
    r"fai(s|tes)[- ]moi signe|laisse(z)?[- ]moi savoir|me laisser savoir|tien(s|ez)[- ]moi"
    r"|di(s|tes)[- ]moi quand|preven(s|ez)[- ]moi|confirme(z)?[- ]moi|me (dire|confirmer) quand"
    r"|revien(s|ez) vers moi"
)
REPLY_FLIPS = ("preparer", "relire", "informer", "rien")
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


_EXAMPLE_RUNS = {e["id"]: _runs(e["corps"]) for e in EXAMPLES}
_VOUS = re.compile(r"\b(vous|votre|vos)\b")


def _present(alternatives: tuple[str, ...], text: str) -> bool:
    return any(_norm(x) in text for x in alternatives)


def _object_words(fact: Fact) -> tuple[list[str], list[str]]:
    """Noms propres de l'objet (tous exigés) et autres mots pleins (la moitié exigée)."""
    words = re.findall(r"\w+", _short(fact))
    proper = [w for w in words if w[:1].isupper() or w.isdigit()]
    common = [w for w in words if w not in proper and len(w) >= 4]
    return proper, common


def check(fact: Fact, sh: Sheet, objet: str, corps: str) -> list[str]:
    """Les raisons de rejeter un texte (liste vide : il passe)."""
    reasons = []
    full = f"{objet}\n{corps}"
    low = _norm(full)
    body = _norm(corps)
    sheet_low = _norm(sh.text)
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
    proper, common = _object_words(fact)
    missing = [w for w in proper if _norm(w) not in low]
    if missing or (common and sum(_norm(w)[:5] in low for w in common) * 2 < len(common)):
        reasons.append(f"objet absent ({_short(fact)})")
    # Rien d'inventé : personnes, organisations, coordonnées, chiffres, jours, mois.
    for name in sorted(_NAMES - set(re.findall(r"\w+", sh.text))):
        hits = re.finditer(rf"(?<![\w-]){name}(?![\w-])", full)
        if name in _AMBIGUOUS:  # seulement précédé d'un prénom : « Julie Petit »
            hits = (h for h in hits if re.search(r"[A-ZÀ-Ý]\w+\s*$", full[: h.start()]))
        if next(hits, None):
            reasons.append(f"nom inventé : {name}")
    reasons += [
        f"organisation inventée : {org}"
        for org in sorted(_ORGS)
        if org in full and org not in sh.text
    ]
    if _EMAIL.search(full) or _URL.search(full):
        reasons.append("adresse ou lien inventé")
    numbers = set(re.findall(r"\d+", sh.text))
    invented = sorted(set(re.findall(r"\d+", full)) - numbers)
    if invented:
        reasons.append(f"chiffres inventés : {', '.join(invented)}")
    for word in (*WEEKDAYS, *MONTHS):
        w = _norm(word)
        if re.search(rf"\b{w}\b", low) and not re.search(rf"\b{w}\b", sheet_low):
            reasons.append(f"date inventée : {word}")
    # Le ton écrit : c'est lui seul qui porte l'urgence.
    calm = bool(_CALM.search(low))
    urgent = bool(_URGENT.search(_CALM.sub(" ", low)))
    if fact.tone == "normal" and (urgent or calm):
        reasons.append("ton changé (urgence ou calme écrit alors que le ton est normal)")
    if fact.tone == "sans_urgence" and (urgent or not calm):
        reasons.append("ton changé (le calme n'est pas écrit)")
    if fact.tone == "urgent" and not urgent:
        reasons.append("ton changé (l'urgence n'est pas écrite)")
    # L'acte demandé.
    if fact.act in _ACT_WORDS and not re.search(_ACT_WORDS[fact.act], body):
        reasons.append("acte changé (aucun mot de l'acte demandé)")
    if fact.act in ("informer", "rien") and _REQUEST.search(body):
        reasons.append("demande ajoutée à un mail qui n'en porte pas")
    elif fact.act in REPLY_FLIPS and _FEEDBACK.search(body):
        reasons.append("demande de retour ajoutée")
    if fact.act in ACTS and fact.form == "question" and "?" not in corps:
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
                "properties": {"objet": {"type": "string"}, "corps": {"type": "string"}},
                "required": ["objet", "corps"],
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


def llm_seed(seed: int, item_id: str, attempt: int) -> int:
    digest = hashlib.blake2b(f"trainset:llm:{seed}:{item_id}:{attempt}".encode(), digest_size=4)
    return int.from_bytes(digest.digest(), "big") >> 1  # entier positif sur 31 bits


class LLMWriter:
    """Remplace le texte des gabarits par celui du LLM, fait par fait, étiquettes inchangées."""

    def __init__(self, persona: Persona, seed: int = 0, client: Ollama | None = None) -> None:
        self.p = persona
        self.seed = seed
        self.client = client or Ollama()
        self.identity = self.client.identity()
        self.templates = Writer(persona, seed)

    def sheet(self, fact: Fact) -> Sheet:
        return sheet(self.p, fact, random.Random(f"trainset:llm:{self.seed}:{fact.item_id}"))

    def draft(self, fact: Fact) -> Draft:
        sh = self.sheet(fact)
        conversation = messages(sh)
        rejects: list[list[str]] = []
        start = time.perf_counter()
        objet = corps = ""
        for attempt in range(MAX_TRIES):
            seed = llm_seed(self.seed, fact.item_id, attempt)
            try:
                out = self.client.chat(conversation, seed)
                objet, corps = str(out["objet"]).strip(), str(out["corps"]).strip()
                reasons = check(fact, sh, objet, corps)
            except (KeyError, TypeError, json.JSONDecodeError) as exc:
                reasons = [f"réponse illisible : {exc}"]
            rejects.append(reasons)
            if not reasons:
                break
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
        )

    def write(self, fact: Fact) -> tuple[Item, list[str], Draft | None]:
        """L'Item (texte du LLM s'il passe les contrôles, sinon celui des gabarits)."""
        item, used = self.templates.write(fact)
        if not rewritable(fact):
            return item, used, None
        d = self.draft(fact)
        if not d.ok:
            return item, used + [f"llm:{PROMPT_VERSION}:rejete"], d
        title = d.objet
        if fact.meta.get("relance") and not re.match(r"(re ?:|relance)", _norm(title)):
            title = "Re: " + title
        clean = strip_quotes_and_signature(normalize_whitespace(d.corps)) or d.corps
        fields = item.model_dump() | {
            "title": title,
            "snippet": truncate(re.sub(r"\s+", " ", clean).strip()),
            "content_text": clean,
        }
        return Item.model_validate(fields), [f"llm:{PROMPT_VERSION}"], d
