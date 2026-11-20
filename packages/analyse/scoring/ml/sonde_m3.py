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
    uv run python -m packages.analyse.scoring.ml.sonde_m3 --conformite [consigne]

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
    SERIES,
    card,
    first_answers,
    viewpoint,
    write_series,
)
from packages.analyse.scoring.ml.sample import _CALENDAR, template
from packages.collect.trainset import load
from packages.collect.trainset.build import DATA as TEMPLATES
from packages.collect.trainset.llm_corpus import OUT as LLM
from packages.collect.trainset.llm_writer import PROMPT_VERSION
from packages.collect.trainset.metiers import METIERS
from packages.collect.trainset.sonde import OUT as SONDES
from packages.collect.trainset.sonde import brief_day
from packages.collect.trainset.world import FEMALE, LAST, MALE

THRESHOLD = 16  # conformes sur 20, fixé avant toute mesure
KEYS = ("action", "urgence", "alerte")
CONFORMITY = 0.90  # part de textes conformes exigée sur l'échantillon (plan, M3)
SHARE, CAP = 0.05, 400  # taille de l'échantillon : 5 % du corpus LLM attendu, 400 au plus
TOP = 20  # modèles les plus fréquents dont on mesure la part


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


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Intervalle de confiance à 95 % d'une proportion (Wilson : juste près de 0 et de 1)."""
    if n == 0:
        return 0.0, 1.0
    p = k / n
    center = p + z * z / (2 * n)
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (center - margin) / (1 + z * z / n), (center + margin) / (1 + z * z / n)


def conformity(consigne: str = PROMPT_VERSION, series: Path = SERIES, llm: Path = LLM) -> str:
    """Textes conformes au simulateur, par annotateur : taux, intervalle, détail par acte
    et écarts. Les étiquettes ne sont lues qu'ici, dans le scénario, jamais écrites à côté
    de la série."""
    name = sample_name(consigne)
    behind = json.loads((series / f"{name}.tirage.json").read_text(encoding="utf-8"))["elements"]
    by_who = first_answers(name, series)
    if not by_who:
        return f"Aucune réponse pour la série {name}."
    folder = llm / consigne
    keys = {item_id.split(":")[1].rsplit("-", 1)[0] for item_id in behind.values()}
    rows = {r["item_id"]: r for key in keys for r in load.load_scenario(key, folder)}
    expected = {task: rows[item_id] for task, item_id in behind.items()}
    lines = []
    for annotator, answers in sorted(by_who.items()):
        done = [t for t in expected if t in answers]
        ok = [t for t in done if all(answers[t][k] == expected[t][k] for k in KEYS)]
        low, high = wilson(len(ok), len(done))
        rate = len(ok) / len(done) if done else 0.0
        reached = rate >= CONFORMITY and len(done) == len(expected)
        lines.append(
            f"{annotator} : {len(ok)} conformes sur {len(done)} annotés ({rate:.1%}, "
            f"IC 95 % {low:.1%}-{high:.1%}) ; {len(expected) - len(done)} restent ; "
            f"critère {CONFORMITY:.0%} : {'atteint' if reached else 'non atteint'}"
        )
        for key in KEYS:
            agree = sum(answers[t][key] == expected[t][key] for t in done)
            lines.append(f"  {key:<8} {agree}/{len(done)}")
        acts = Counter(expected[t]["action"] for t in done)
        for act, total in acts.most_common():
            good = sum(t in ok for t in done if expected[t]["action"] == act)
            lines.append(f"  acte {act:<9} {good}/{total} conformes")
        for t in done:
            if t not in ok:
                e = expected[t]
                diff = ", ".join(
                    f"{k} {answers[t][k]} au lieu de {e[k]}" for k in KEYS if answers[t][k] != e[k]
                )
                lines.append(f"  ≠ {t} ({behind[t]}) {diff}")
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    corpus = {"--diversite", "--echantillon", "--conformite"}
    if argv and argv[0] in corpus and len(argv) <= 2:
        consigne = argv[1] if len(argv) == 2 else PROMPT_VERSION
        if argv[0] == "--diversite":
            print(diversity_report(LLM / consigne))
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
        "| --echantillon [consigne] | --conformite [consigne]"
    )
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
