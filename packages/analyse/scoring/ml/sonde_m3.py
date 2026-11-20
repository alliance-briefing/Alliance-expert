"""Acceptation de l'écrivain LLM (M3) : sonde, diversité, échantillon relu à l'aveugle.

Le trainset n'a pas le droit d'importer le lot 2 : il écrit ses textes et le corpus assemblé
(`llm_corpus --assembler`), ce module les mesure et en fait des séries pour `annoter.py`, avec
des cartes (`card`) qui montrent exactement ce que verra le modèle (objet, extrait de 500
caractères), jamais une étiquette (`check_blind`).

**Sonde** (20 textes, avant le corpus) :

    uv run python -m packages.analyse.scoring.ml.sonde_m3 --serie mail-v5-tirage2
    uv run python -m packages.analyse.scoring.ml.annoter --serie sonde-m3-mail-v5-tirage2 --annotateur theo
    uv run python -m packages.analyse.scoring.ml.sonde_m3 --comparer mail-v5-tirage2

Critère fixé d'avance : un texte est conforme si l'action, l'urgence et l'alerte annotées sont
celles du simulateur ; la sonde passe avec au moins 16 conformes sur 20, pour chaque
annotateur. Seules les premières présentations comptent (pas les répétitions cachées).

**Corpus** (critères d'acceptation de M3, plan du modèle), sur le corpus LLM assemblé :

    uv run python -m packages.analyse.scoring.ml.sonde_m3 --diversite [consigne]
    uv run python -m packages.analyse.scoring.ml.sonde_m3 --echantillon [consigne]
    uv run python -m packages.analyse.scoring.ml.annoter --serie m3-mail-v8-echantillon --annotateur theo
    uv run python -m packages.analyse.scoring.ml.sonde_m3 --repartir [consigne]   # à deux
    uv run python -m packages.analyse.scoring.ml.sonde_m3 --conformite [consigne]
    uv run python -m packages.analyse.scoring.ml.sonde_m3 --relecture [consigne]  # selon le guide

- diversité : « bien plus de modèles de phrases distincts qu'avec les gabarits » (mesure de
  l'ADR-0003), comparée sur les mêmes mails écrits des deux façons ;
- échantillon : 5 % du corpus LLM attendu (400 au plus), tiré au hasard parmi les textes
  retenus, relu à l'aveugle ; critère : au moins 90 % conformes.
"""

from __future__ import annotations

import json
import math
import random
import re
import sys
from collections import Counter
from datetime import date, datetime
from pathlib import Path

from contracts.models import Item
from packages.analyse.scoring.ml.annoter import (
    ANNOTATORS,
    SERIES,
    card,
    final_labels,
    first_answers,
    load_series,
    viewpoint,
    write_series,
)
from packages.analyse.scoring.ml.sample import _CALENDAR, template
from packages.collect.trainset import load
from packages.collect.trainset.build import DATA as TEMPLATES
from packages.collect.trainset.llm_corpus import OUT as LLM
from packages.collect.trainset.llm_writer import (
    PROMPT_VERSION,
    _norm,  # minuscules, sans accents
)
from packages.collect.trainset.metiers import METIERS
from packages.collect.trainset.sonde import OUT as SONDES
from packages.collect.trainset.sonde import brief_day
from packages.collect.trainset.world import FEMALE, LAST, MALE

THRESHOLD = 16  # conformes sur 20, fixé avant toute mesure
KEYS = ("action", "urgence", "alerte")
CONFORMITY = 0.90  # part de textes conformes exigée sur l'échantillon (plan, M3)
SHARE, CAP = 0.05, 400  # taille de l'échantillon : 5 % du corpus LLM attendu, 400 au plus
TOP = 20  # modèles les plus fréquents dont on mesure la part
COMMON = 40  # cartes jugées par deux annotateurs, pour l'accord (kappa)


def series_name(folder: str) -> str:
    return f"sonde-m3-{folder}"


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def build_series(folder: str, sondes: Path = SONDES, series: Path = SERIES) -> Path:
    """La série d'annotation d'une sonde : une tâche « élément » par mail, à l'aveugle."""
    tasks = []
    for row in _jsonl(sondes / folder / "a_annoter.jsonl"):
        vp = viewpoint(row["persona"], date.fromisoformat(row["jour"]))
        item = Item.model_validate(row["item"])
        tasks.append(
            {"task_id": row["case_id"], "kind": "element", "context": vp.context}
            | {"card": card(item, vp)}
        )
    path = series / f"{series_name(folder)}.jsonl"
    write_series(tasks, path)  # refuse toute étiquette (check_blind)
    return path


def compare(folder: str, sondes: Path = SONDES, series: Path = SERIES) -> str:
    """Textes conformes au simulateur, par annotateur, et le détail des écarts."""
    expected = {row["case_id"]: row for row in _jsonl(sondes / folder / "attendu.jsonl")}
    by_who = first_answers(series_name(folder), series)
    if not by_who:
        return f"Aucune réponse pour la série {series_name(folder)}."
    lines = []
    for who, answers in sorted(by_who.items()):
        done = [c for c in expected if c in answers]
        ok = [c for c in done if all(answers[c][k] == expected[c][k] for k in KEYS)]
        reached = len(ok) >= THRESHOLD and len(done) == len(expected)
        lines.append(
            f"{who} : {len(ok)} conformes sur {len(done)} annotés "
            f"(critère {THRESHOLD}/{len(expected)} : {'atteint' if reached else 'non atteint'})"
        )
        for key in KEYS:
            agree = sum(answers[c][key] == expected[c][key] for c in done)
            lines.append(f"  {key:<8} {agree}/{len(done)}")
        for c in done:
            if c not in ok:
                e = expected[c]
                diff = ", ".join(
                    f"{k} {answers[c][k]} au lieu de {e[k]}" for k in KEYS if answers[c][k] != e[k]
                )
                lines.append(f"  ≠ {c} [{e['emplacement']}] {diff}")
    return "\n".join(lines)


# --- le corpus LLM assemblé --------------------------------------------------------------------


def _manifest(folder: Path) -> dict:
    path = folder / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} absent : lancer d'abord "
            "`uv run python -m packages.collect.trainset.llm_corpus --assembler`"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def retained(folder: Path) -> list[tuple[str, dict]]:
    """Les mails dont le texte du LLM est retenu : (persona, ligne du scénario)."""
    mark = [f"llm:{_manifest(folder)['consigne']}"]
    return [
        (key, row)
        for key in load.personas(data=folder)
        for row in load.load_scenario(key, folder)
        if row["templates"] == mark
    ]


# --- diversité (ADR-0003) ----------------------------------------------------------------------


def _alternation(words: set[str], flags: int = re.IGNORECASE) -> re.Pattern:
    """Une expression qui trouve ces mots ou groupes de mots, les plus longs d'abord."""
    ordered = sorted(words, key=len, reverse=True)
    return re.compile(r"(?<!\w)(" + "|".join(map(re.escape, ordered)) + r")(?!\w)", flags)


# Ce qui varie sans changer la phrase : l'objet du dossier (avec ou sans article), les
# organisations et les personnes du monde simulé. Sans ce masque, « valider le devis A » et
# « valider le devis B » compteraient pour deux modèles et gonfleraient les gabarits.
_OBJECTS = _alternation(
    {
        phrase
        for m in METIERS
        for o in m.objects
        for phrase in (o.phrase, re.sub(r"^(le |la |les |l')", "", o.phrase))
    }
)
_ORGS = _alternation({o for m in METIERS for o in (m.company, *m.clients, *m.suppliers)})
_PEOPLE = _alternation(set(FEMALE) | set(MALE) | set(LAST), flags=0)  # « Petit », pas « petit »
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
# Formules de politesse et signatures (« Merci d'avance. », « Bonne journée, Julie ») : le code
# les ajoute à l'identique aux deux corpus, elles ne disent rien du texte rédigé.
MIN_WORDS = 4


def mask(text: str) -> str:
    """Objets, organisations, personnes, nombres, jours et mois masqués (ADR-0003)."""
    text = _OBJECTS.sub("<objet>", text)
    text = _ORGS.sub("<org>", text)
    text = _PEOPLE.sub("<nom>", text)
    return re.sub(r"\d+", "#", _CALENDAR.sub("#", text))


def sentence_models(text: str) -> list[str]:
    """Les modèles de phrases d'un extrait, masqués et en minuscules. Sont écartées la phrase
    coupée par l'extrait (« … ») et les formules de moins de `MIN_WORDS` mots."""
    out = []
    for sentence in _SENTENCE_END.split(mask(text)):
        sentence = re.sub(r"\s+", " ", sentence).strip().lower()
        words = re.findall(r"[\w<>#'-]+", sentence)
        if len(words) >= MIN_WORDS and not sentence.endswith("…"):
            out.append(sentence)
    return out


def _profile(items: list[Item]) -> dict:
    sentences = Counter(s for i in items for s in sentence_models(i.snippet or ""))
    titles = Counter(template(mask(i.title or "")) for i in items)
    total = sum(sentences.values())
    return {
        "phrases": total,
        "modeles_phrases": len(sentences),
        # Ne dépend pas de la longueur des mails : le LLM écrit plus de phrases.
        "part_distincte": round(len(sentences) / total, 4) if total else None,
        f"part_top{TOP}": round(sum(n for _, n in sentences.most_common(TOP)) / total, 4)
        if total
        else None,
        "modeles_objets": len(titles),
    }


def diversity(folder: Path, templates: Path = TEMPLATES) -> dict:
    """Modèles de phrases distincts, gabarits contre LLM, sur les mêmes mails (ceux dont le
    texte du LLM est retenu) : la comparaison est appariée."""
    by_persona: dict[str, set[str]] = {}
    for key, row in retained(folder):
        by_persona.setdefault(key, set()).add(row["item_id"])
    pairs: tuple[list[Item], list[Item]] = ([], [])
    for key, ids in by_persona.items():
        for corpus, data in zip(pairs, (templates, folder), strict=True):
            corpus += [i for i in load.load_items(key, data) if i.item_id in ids]
    gab, llm = (_profile(items) for items in pairs)
    ratio = llm["modeles_phrases"] / gab["modeles_phrases"] if gab["modeles_phrases"] else None
    return {
        "mails": len(pairs[0]),
        "gabarits": gab,
        "llm": llm,
        "rapport_modeles": round(ratio, 2) if ratio else None,
    }


def diversity_report(folder: Path, templates: Path = TEMPLATES) -> str:
    d = diversity(folder, templates)
    g, m = d["gabarits"], d["llm"]
    top = f"part_top{TOP}"
    return "\n".join(
        [
            f"{d['mails']} mails écrits des deux façons (textes du LLM retenus).",
            (
                f"{'':<10} {'phrases':>8} {'modèles':>8} {'distincts':>10} {f'top {TOP}':>8} "
                f"{'objets':>8}"
            ),
            *(
                f"{name:<10} {p['phrases']:>8} {p['modeles_phrases']:>8} "
                f"{p['part_distincte']:>10.1%} {p[top]:>8.1%} {p['modeles_objets']:>8}"
                for name, p in (("gabarits", g), ("LLM", m))
            ),
            (
                f"Le LLM a {d['rapport_modeles']} fois plus de modèles de phrases distincts "
                "(phrases d'au moins 4 mots, objets, noms, nombres et dates masqués)."
            ),
        ]
    )


# --- échantillon relu à l'aveugle --------------------------------------------------------------


def sample_name(consigne: str) -> str:
    return f"m3-{consigne}-echantillon"


def draw(folder: Path, seed: int = 0) -> list[tuple[str, dict]]:
    """L'échantillon : 5 % du corpus LLM attendu à la fin (mails à réécrire × taux de
    passage), 400 au plus, tiré au hasard parmi les textes retenus.

    Le corpus est écrit dans un ordre mélangé : une partie déjà écrite est un échantillon au
    hasard du tout, et un tirage fait dessus vaut un tirage en fin de génération.
    """
    counts = _manifest(folder)["llm"]
    size = min(CAP, round(SHARE * counts["a_reecrire"] * (counts["taux_controles"] or 0)))
    pool = sorted(retained(folder), key=lambda kr: kr[1]["item_id"])
    if len(pool) < size:
        raise ValueError(f"{len(pool)} textes retenus, il en faut {size} : attendre la suite")
    return random.Random(f"m3:echantillon:{seed}").sample(pool, size)


def build_sample(
    consigne: str = PROMPT_VERSION, series: Path = SERIES, llm: Path = LLM, seed: int = 0
) -> Path:
    """La série de l'échantillon, à l'aveugle, et son tirage (`<série>.tirage.json` :
    consigne, empreinte des textes, graine, et l'élément derrière chaque carte).

    Un échantillon de mesure ne se retire pas : la série existante est refusée.
    """
    folder = llm / consigne
    name = sample_name(consigne)
    path = series / f"{name}.jsonl"
    if path.exists():
        raise FileExistsError(f"{path.name} existe : un échantillon de mesure ne se retire pas")
    chosen = draw(folder, seed)
    items: dict[str, dict[str, Item]] = {}
    tasks, behind = [], {}
    for n, (key, row) in enumerate(chosen, 1):
        if key not in items:
            items[key] = {i.item_id: i for i in load.load_items(key, folder)}
        day = brief_day(datetime.fromisoformat(row["visible_from"])).date()
        vp = viewpoint(key, day)
        task_id = f"m3-{n:03d}"  # numéro sans lien avec le profil du texte
        tasks.append(
            {"task_id": task_id, "kind": "element", "context": vp.context}
            | {"card": card(items[key][row["item_id"]], vp)}
        )
        behind[task_id] = row["item_id"]
    write_series(tasks, path)  # refuse toute étiquette (check_blind)
    manifest = _manifest(folder)
    draw_info = {
        "consigne": consigne,
        "textes_sha256": manifest["textes"]["sha256"],
        "ecrits": manifest["llm"]["ecrits"],
        "retenus": manifest["llm"]["retenus"],
        "graine": seed,
        "taille": len(tasks),
        "elements": behind,
    }
    (series / f"{name}.tirage.json").write_text(
        json.dumps(draw_info, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n"
    )
    return path


def share(
    consigne: str = PROMPT_VERSION,
    first: str = "theo",
    second: str = "paul",
    common: int = COMMON,
    series: Path = SERIES,
    seed: int = 0,
) -> dict:
    """Répartit la série entre deux annotateurs sans changer l'échantillon (champ « pour »).

    Les cartes que `first` a déjà jugées lui restent ; `common` d'entre elles, tirées au
    hasard, vont aussi à `second`, sans ses réponses, pour mesurer l'accord (kappa). Les
    cartes restantes sont coupées en deux au hasard. Chacun garde son ordre propre et ses
    répétitions cachées (`annoter.order`). Une série répartie ne se répartit pas deux fois.
    """
    name = sample_name(consigne)
    path = series / f"{name}.jsonl"
    tasks = load_series(path)
    if any("pour" in t for t in tasks):
        raise ValueError(f"{path.name} est déjà répartie")
    done = sorted(first_answers(name, series).get(first, {}))
    rest = sorted(t["task_id"] for t in tasks if t["task_id"] not in set(done))
    rng = random.Random(f"m3:repartition:{seed}")
    both = set(rng.sample(done, min(common, len(done))))
    rng.shuffle(rest)
    to_second = set(rest[: len(rest) // 2])
    for task in tasks:
        tid = task["task_id"]
        task["pour"] = [first, second] if tid in both else [second] if tid in to_second else [first]
    write_series(tasks, path)
    counts = Counter(who for t in tasks for who in t["pour"])
    info = {
        "graine": seed,
        "deja_juges": {first: len(done)},
        "cartes": {first: counts[first], second: counts[second]},
        "communes": len(both),
    }
    draw_path = series / f"{name}.tirage.json"
    draw_info = json.loads(draw_path.read_text(encoding="utf-8")) | {"repartition": info}
    draw_path.write_text(
        json.dumps(draw_info, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n"
    )
    return info


# --- relecture selon le guide ------------------------------------------------------------------

# Les règles du guide (guide-annotation.md) sur lesquelles la première passe s'est écartée,
# rappelées en tête de chaque carte, les mêmes pour toutes : un rappel propre à une carte
# dirait pourquoi elle est revenue.
GUIDE_REMINDER = (
    "Rappels du guide : c'est l'acte demandé qui compte, pas la forme de la phrase. "
    "S'informer : rien à faire, mais un contenu à lire lui est utile (dont la confirmation "
    "qu'une chose est faite) ; Rien : rien du tout. Mail reçu en copie : quand la demande "
    "s'adresse à quelqu'un d'autre, S'informer, et l'urgence reste celle du ton écrit."
)
_VALIDATE_QUESTION = re.compile(
    r"\b(valides?|validez|signes?|signez|approuves?|approuvez)-(tu|vous)\b"
    r"|\b(peux|pouvez|pourrais|pourriez)-(tu|vous) (le |la |les |l')?"
    r"(valider|signer|approuver)\b"
    r"|\b(peut|peuvent)-(il|elle|ils|elles) (etre )?(valide|signe|approuve)",
)


def review_name(consigne: str) -> str:
    return f"{sample_name(consigne)}-relecture"


def build_review(
    consigne: str = PROMPT_VERSION,
    annotator: str = "theo",
    series: Path = SERIES,
    llm: Path = LLM,
    seed: int = 0,
) -> dict:
    """La série de relecture : les cartes de l'échantillon choisies sur des critères visibles
    sans le simulateur, rejugées à l'aveugle (ni étiquette, ni ancienne réponse).

    - texte changé : le corpus corrigé ne montre plus le même texte (nouvelle carte) ;
    - copie : la carte montre « Copie : vous » ;
    - validation en question : « Valides-tu… ? », « Pouvez-vous signer… ? » ;
    - noté Rien : la vérité humaine de la première passe est Rien (la réponse humaine, jamais
      celle du simulateur).

    Les cartes dont le texte est retombé sur les gabarits sortent de la mesure (ce ne sont
    plus des textes du LLM). Toutes les cartes revues, conformes ou non à la première passe,
    sont mélangées : on ne peut pas savoir lesquelles étaient des écarts.
    """
    name = sample_name(consigne)
    path = series / f"{review_name(consigne)}.jsonl"
    if path.exists():
        raise FileExistsError(f"{path.name} existe : une relecture ne se retire pas")
    folder = llm / consigne
    draw_info = json.loads((series / f"{name}.tirage.json").read_text(encoding="utf-8"))
    behind = draw_info["elements"]
    tasks = {t["task_id"]: t for t in load_series(series / f"{name}.jsonl")}
    truth = final_labels(name, series)
    keys = {item_id.split(":")[1].rsplit("-", 1)[0] for item_id in behind.values()}
    items = {i.item_id: i for key in keys for i in load.load_items(key, folder)}
    mark = [f"llm:{_manifest(folder)['consigne']}"]
    rows = {r["item_id"]: r for key in keys for r in load.load_scenario(key, folder)}
    lots: dict[str, list[str]] = {k: [] for k in ("texte_change", "copie", "validation", "rien")}
    out_of_llm, cards = [], {}
    for task_id, item_id in behind.items():
        old = tasks[task_id]["card"]
        if rows[item_id]["templates"] != mark:
            out_of_llm.append(task_id)
            continue
        new = items[item_id]
        if (new.snippet or "") != old["text"]:
            lots["texte_change"].append(task_id)
            day = brief_day(datetime.fromisoformat(rows[item_id]["visible_from"])).date()
            cards[task_id] = card(new, viewpoint(rows[item_id]["persona"], day))
            continue
        if ["Copie", "vous"] in old["fields"]:
            lots["copie"].append(task_id)
        if "?" in old["text"] and _VALIDATE_QUESTION.search(_norm(old["text"])):
            lots["validation"].append(task_id)
        if truth.get(task_id, {}).get("action") == "rien":
            lots["rien"].append(task_id)
        if any(task_id in lot for lot in lots.values()):
            cards[task_id] = old
    order_ = sorted(cards)
    random.Random(f"m3:relecture:{seed}").shuffle(order_)
    review, sources = [], {}
    for n, task_id in enumerate(order_, 1):
        review_id = f"r-{n:03d}"
        context = f"{tasks[task_id]['context']} {GUIDE_REMINDER}"
        review.append(
            {"task_id": review_id, "kind": "element", "context": context}
            | {"card": cards[task_id], "pour": [annotator]}
        )
        sources[review_id] = task_id
    write_series(review, path)
    info = {
        "graine": seed,
        "annotateur": annotator,
        "cartes": len(review),
        "lots": {k: sorted(v) for k, v in lots.items()},
        "hors_llm": sorted(out_of_llm),
        "sources": sources,
    }
    (series / f"{review_name(consigne)}.tirage.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n"
    )
    return info


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Intervalle de confiance à 95 % d'une proportion (Wilson : juste près de 0 et de 1)."""
    if n == 0:
        return 0.0, 1.0
    p = k / n
    center = p + z * z / (2 * n)
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (center - margin) / (1 + z * z / n), (center + margin) / (1 + z * z / n)


def _rate(judged: dict[str, dict], expected: dict[str, dict]) -> tuple[list[str], list[str], str]:
    """Cartes jugées, cartes conformes, et « k conformes sur n (x %, IC 95 % a-b) »."""
    done = [t for t in expected if t in judged]
    ok = [t for t in done if all(judged[t][k] == expected[t][k] for k in KEYS)]
    low, high = wilson(len(ok), len(done))
    rate = len(ok) / len(done) if done else 0.0
    text = f"{len(ok)} conformes sur {len(done)} ({rate:.1%}, IC 95 % {low:.1%}-{high:.1%})"
    return done, ok, text


def conformity(consigne: str = PROMPT_VERSION, series: Path = SERIES, llm: Path = LLM) -> str:
    """Textes conformes au simulateur : d'abord sur la vérité humaine de la série (réponse de
    l'unique annotateur, réponse commune, ou arbitrage : `annoter.final_labels`), qui porte
    le critère ; puis par annotateur, sur ses propres cartes.

    Les étiquettes du simulateur ne sont lues qu'ici, dans le scénario, jamais écrites à
    côté de la série."""
    name = sample_name(consigne)
    behind = json.loads((series / f"{name}.tirage.json").read_text(encoding="utf-8"))["elements"]
    by_who = first_answers(name, series)
    if not by_who:
        return f"Aucune réponse pour la série {name}."
    folder = llm / consigne
    keys = {item_id.split(":")[1].rsplit("-", 1)[0] for item_id in behind.values()}
    rows = {r["item_id"]: r for key in keys for r in load.load_scenario(key, folder)}
    expected = {task: rows[item_id] for task, item_id in behind.items()}
    tasks = load_series(series / f"{name}.jsonl")
    truth = final_labels(name, series)
    judged = {t for answers in by_who.values() for t in answers}
    open_conflicts = [t for t in judged if t not in truth]  # désaccords pas encore arbitrés
    done, ok, text = _rate(truth, expected)
    reached = len(done) == len(expected) and len(ok) >= CONFORMITY * len(done)
    lines = [
        (
            f"Ensemble : {text} ; {len(open_conflicts)} désaccord(s) à arbitrer ; "
            f"{len(expected) - len(judged)} carte(s) pas encore jugée(s) ; "
            f"critère {CONFORMITY:.0%} : {'atteint' if reached else 'non atteint'}"
        )
    ]
    for key in KEYS:
        agree = sum(truth[t][key] == expected[t][key] for t in done)
        lines.append(f"  {key:<8} {agree}/{len(done)}")
    acts = Counter(expected[t]["action"] for t in done)
    for act, total in acts.most_common():
        good = sum(t in ok for t in done if expected[t]["action"] == act)
        lines.append(f"  acte {act:<9} {good}/{total} conformes")
    for t in done:
        if t not in ok:
            e = expected[t]
            diff = ", ".join(
                f"{k} {truth[t][k]} au lieu de {e[k]}" for k in KEYS if truth[t][k] != e[k]
            )
            lines.append(f"  ≠ {t} ({behind[t]}) {diff}")
    assigned = {t["task_id"]: t.get("pour", ANNOTATORS) for t in tasks}
    for annotator, answers in sorted(by_who.items()):
        mine = {t: e for t, e in expected.items() if annotator in assigned[t]}
        done_mine, _, text = _rate(answers, mine)
        lines.append(f"{annotator} : {text} ; {len(mine) - len(done_mine)} restent")
    return "\n".join(lines + _after_review(consigne, series, expected, truth, behind))


def _after_review(
    consigne: str,
    series: Path,
    expected: dict[str, dict],
    truth: dict[str, dict],
    behind: dict[str, str],
) -> list[str]:
    """La mesure après relecture : la réponse de la relecture remplace celle de la première
    passe sur les cartes revues ; les cartes sorties du corpus LLM ne comptent plus. On dit
    aussi dans quel sens la relecture a bougé les réponses (vers le simulateur ou non)."""
    path = series / f"{review_name(consigne)}.tirage.json"
    if not path.exists():
        return []
    info = json.loads(path.read_text(encoding="utf-8"))
    answers = first_answers(review_name(consigne), series).get(info["annotateur"], {})
    reviewed = {info["sources"][r]: a for r, a in answers.items() if r in info["sources"]}
    changed = set(info["lots"]["texte_change"])
    kept = {t: e for t, e in expected.items() if t not in set(info["hors_llm"])}
    corrected = {}
    for t in kept:
        if t in reviewed:
            corrected[t] = reviewed[t]
        elif t not in changed and t in truth:  # un texte changé doit être rejugé
            corrected[t] = truth[t]
    done, ok, text = _rate(corrected, kept)
    reached = len(done) == len(kept) and len(ok) >= CONFORMITY * len(done)
    lines = [
        (
            f"Après relecture selon le guide ({len(info['sources'])} cartes revues, "
            f"{len(info['hors_llm'])} sorties du corpus LLM) : {text} ; "
            f"{len(info['sources']) - len(reviewed)} carte(s) à relire ; "
            f"critère {CONFORMITY:.0%} : {'atteint' if reached else 'non atteint'}"
        )
    ]

    def conform(answer: dict | None, t: str) -> bool:
        return answer is not None and all(answer[k] == expected[t][k] for k in KEYS)

    moves = Counter()
    for t, answer in reviewed.items():
        if t in changed:
            moves["texte changé"] += 1
            continue
        before, after = conform(truth.get(t), t), conform(answer, t)
        moves[
            "vers le simulateur"
            if after and not before
            else "loin du simulateur"
            if before and not after
            else "sans effet"
        ] += 1
    lines.append("  relecture : " + ", ".join(f"{k} {n}" for k, n in sorted(moves.items())))
    for t in done:
        if t not in ok:
            e = expected[t]
            diff = ", ".join(
                f"{k} {corrected[t][k]} au lieu de {e[k]}" for k in KEYS if corrected[t][k] != e[k]
            )
            lines.append(f"  ≠ {t} ({behind[t]}) {diff}")
    return lines


def main(argv: list[str]) -> int:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    corpus = {"--diversite", "--echantillon", "--repartir", "--relecture", "--conformite"}
    if argv and argv[0] in corpus and len(argv) <= 2:
        consigne = argv[1] if len(argv) == 2 else PROMPT_VERSION
        if argv[0] == "--diversite":
            print(diversity_report(LLM / consigne))
        elif argv[0] == "--relecture":
            info = build_review(consigne)
            lots = {k: len(v) for k, v in info["lots"].items()}
            print(
                f"Série de relecture : {info['cartes']} cartes (lots, avec recouvrements : "
                f"{lots}) ; {len(info['hors_llm'])} sorties du corpus LLM."
            )
            print(
                "Annoter : uv run python -m packages.analyse.scoring.ml.annoter "
                f"--serie {review_name(consigne)} --annotateur {info['annotateur']}"
            )
        elif argv[0] == "--repartir":
            info = share(consigne)
            print(
                f"Série répartie : {info['cartes']} cartes, dont {info['communes']} communes "
                f"(déjà jugées : {info['deja_juges']})."
            )
        elif argv[0] == "--echantillon":
            path = build_sample(consigne)
            print(f"Série écrite : {path} ({len(path.read_text('utf-8').splitlines())} cartes)")
            print(
                "Annoter : uv run python -m packages.analyse.scoring.ml.annoter "
                f"--serie {sample_name(consigne)} --annotateur theo"
            )
        else:
            print(conformity(consigne))
        return 0
    if len(argv) == 2 and argv[0] == "--serie":
        path = build_series(argv[1])
        print(f"Série écrite : {path}")
        print(
            "Annoter : uv run python -m packages.analyse.scoring.ml.annoter "
            f"--serie {series_name(argv[1])} --annotateur theo"
        )
        return 0
    if len(argv) == 2 and argv[0] == "--comparer":
        print(compare(argv[1]))
        return 0
    print(
        "usage : sonde_m3 --serie <dossier> | --comparer <dossier> | --diversite [consigne] "
        "| --echantillon [consigne] | --repartir [consigne] | --relecture [consigne] "
        "| --conformite [consigne]"
    )
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
