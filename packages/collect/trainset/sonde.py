"""La sonde de M3 : 20 mails du simulateur écrits par le LLM, annotés à l'aveugle.

`uv run python -m packages.collect.trainset.sonde --tirage 2` tire 20 faits de mails variés
(actes, formes, tons, tu ou vous, relances, copie), jamais tirés avant, les fait écrire par
le LLM et écrit dans `data/sonde/<consigne>-tirage<n>/` :
- `a_annoter.jsonl` : par cas, la persona, le jour du brief et l'`Item` tel que le verrait
  le modèle (objet et extrait de 500 caractères), sans aucune étiquette ;
- `attendu.jsonl` : étiquettes du simulateur, fiche, texte complet, extrait, texte des
  gabarits et traçabilité du LLM (à ne pas ouvrir avant d'avoir annoté) ;
- `sonde.json` : modèle exact, consigne, graine, temps.

L'annotation passe par l'outil unifié du lot 2 (M4) : `packages.analyse.scoring.ml.sonde_m3`
en fait une série pour `annoter.py`, puis compare les réponses au simulateur. Critère fixé
d'avance : un texte est conforme si l'action, l'urgence et l'alerte annotées sont celles du
simulateur ; la sonde passe avec au moins 16 conformes sur 20, pour chaque annotateur.

`… sonde --essai <item_id>…` réécrit des faits connus et montre fiche, texte et contrôles,
sans rien écrire : pour vérifier une consigne sur les cas déjà ratés, jamais pour mesurer.

Historique : consigne mail-v2, tirage 1 → 14/20 (Théo seul, cartes au texte complet, avec
l'ancien `annotate.py`), voir `mail-v2-tirage1/`.
"""

from __future__ import annotations

import json
import random
import statistics
import sys
from collections.abc import Callable
from dataclasses import asdict
from datetime import datetime, timedelta
from pathlib import Path

from packages.collect.trainset import llm_writer as L
from packages.collect.trainset.cases import Sim
from packages.collect.trainset.facts import Fact, label
from packages.collect.trainset.world import Persona, all_personas
from packages.collect.trainset.writer import Writer

OUT = Path(__file__).with_name("data") / "sonde"
N_CASES = 20
MAX_PER_METIER = 2


def _tu(f: Fact) -> bool:
    return f.meta.get("tu", True)


def _relance(f: Fact) -> bool:
    return bool(f.meta.get("relance"))


def _direct(act: str, form: str | None = None, tone: str | None = None) -> Callable:
    return lambda f: (
        f.to_manager
        and f.act == act
        and (form is None or f.form == form)
        and (tone is None or f.tone == tone)
        and not _relance(f)
    )


# Un emplacement par profil voulu : la sonde couvre les 7 actions, les 3 formes, les 3 tons,
# tu et vous, les relances, le second acte et la copie (étiquette S'informer).
SLOTS: tuple[tuple[str, Callable[[Fact], bool]], ...] = (
    ("décider, question", _direct("decider", "question", "normal")),
    ("décider, directe, urgent", _direct("decider", "directe", "urgent")),
    ("décider, indirecte", _direct("decider", "indirecte")),
    ("décider, relance", lambda f: f.act == "decider" and _relance(f)),
    ("faire, question, vous", lambda f: _direct("faire", "question")(f) and not _tu(f)),
    ("faire, directe, sans urgence", _direct("faire", "directe", "sans_urgence")),
    ("faire, indirecte", _direct("faire", "indirecte")),
    ("répondre, question", _direct("repondre", "question")),
    ("répondre, directe, vous", lambda f: _direct("repondre", "directe")(f) and not _tu(f)),
    ("répondre, relance d'un client", lambda f: _relance(f) and f.author.kind == "externe_client"),
    ("second acte", lambda f: f.to_manager and f.secondary is not None),
    ("relire, sans urgence", _direct("relire", tone="sans_urgence")),
    ("relire, échéance", lambda f: _direct("relire", tone="normal")(f) and f.deadline is not None),
    ("préparer", _direct("preparer")),
    ("préparer", _direct("preparer")),
    ("s'informer, d'un pair", lambda f: _direct("informer")(f) and f.author.kind == "pair"),
    ("s'informer, de l'équipe", lambda f: _direct("informer")(f) and f.author.kind == "equipe"),
    ("rien", _direct("rien")),
    ("en copie", lambda f: not f.to_manager and f.tone == "normal"),
    ("en copie, ton marqué", lambda f: not f.to_manager and f.tone != "normal"),
)


def pick(seed: int = 0, draw: int = 1) -> list[tuple[str, Persona, Fact]]:
    """Les 20 faits : un par emplacement, personas d'entraînement, 2 au plus par métier.

    Un nouveau tirage (`draw`) prend 20 autres faits du même monde, jamais tirés avant :
    l'annotateur ne reconnaît pas les textes d'une sonde précédente.
    """
    pool = [
        (persona, fact)
        for persona in all_personas(seed)
        if persona.split == "train"
        for fact in Sim(persona, seed).run()
        if L.rewritable(fact)
    ]
    seen: set[str] = set()
    for d in range(1, draw + 1):
        chosen = _draw(pool, seed, d, seen)
        seen |= {f.item_id for _, _, f in chosen}
    return chosen


def _draw(pool: list, seed: int, draw: int, seen: set[str]) -> list[tuple[str, Persona, Fact]]:
    order = list(pool)
    key = f"trainset:sonde:{seed}" if draw == 1 else f"trainset:sonde:{seed}:{draw}"
    random.Random(key).shuffle(order)
    pool = [(p, f) for p, f in order if f.item_id not in seen]
    chosen: list[tuple[str, Persona, Fact]] = []
    for name, test in SLOTS:
        used = {f.item_id for _, _, f in chosen}
        per_metier = [p.metier.key for _, p, _ in chosen]
        hit = next(
            (
                (persona, fact)
                for persona, fact in pool
                if fact.item_id not in used
                and per_metier.count(persona.metier.key) < MAX_PER_METIER
                and test(fact)
            ),
            None,
        )
        if hit is None:
            raise RuntimeError(f"aucun fait pour l'emplacement « {name} »")
        chosen.append((name, *hit))
    return chosen


# --- génération --------------------------------------------------------------------------------


def brief_day(moment: datetime) -> datetime:
    """Le matin du brief qui montre l'élément : le jour même avant 8 h 30, sinon le suivant."""
    day = moment.replace(hour=8, minute=30, second=0, microsecond=0)
    if moment >= day:
        day += timedelta(days=1)
    while day.weekday() >= 5:
        day += timedelta(days=1)
    return day


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    text = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    path.write_text(text, encoding="utf-8", newline="\n")


def folder(draw: int = 1) -> Path:
    """Un dossier par consigne et par tirage : `data/sonde/mail-v3-tirage2/`."""
    return OUT / f"{L.PROMPT_VERSION}-tirage{draw}"


def run(
    out: Path | None = None, seed: int = 0, draw: int = 1, client: L.Ollama | None = None
) -> dict:
    out = out or folder(draw)
    client = client or L.Ollama()
    chosen = pick(seed, draw)
    order = list(range(len(chosen)))
    # Numérotation sans lien aux profils. Tirage 1 : « v2 », car la 1re numérotation a été
    # affichée avec les profils lors d'un essai, elle n'était plus aveugle.
    key = f"trainset:sonde:ordre:v2:{seed}" if draw == 1 else f"trainset:sonde:ordre:{seed}:{draw}"
    random.Random(key).shuffle(order)
    cases, expected = [], []
    started = datetime.now().astimezone()
    for n, k in enumerate(order, 1):
        slot, persona, fact = chosen[k]
        writer = L.LLMWriter(persona, seed, client)
        draft = writer.draft(fact)
        case_id = f"sonde-{n:02d}"
        template, _ = Writer(persona, seed).write(fact)
        # L'Item tel que le verrait le modèle, texte du LLM même rejeté : la sonde mesure le
        # LLM. Ni étiquette ni profil : la carte est construite côté lot 2 (annoter.py).
        item = writer.item(fact, draft)
        cases.append(
            {
                "case_id": case_id,
                "persona": persona.key,
                "jour": brief_day(fact.at).date().isoformat(),
            }
            | {"item": item.model_dump(mode="json")}
        )
        expected.append(
            {
                "case_id": case_id,
                "item_id": fact.item_id,
                "persona": persona.key,
                "emplacement": slot,
                **label(fact),
                "fait": {
                    "acte": fact.act,
                    "forme": fact.form,
                    "ton": fact.tone,
                    "tu": _tu(fact),
                    "relance": _relance(fact),
                    "copie": not fact.to_manager,
                    "second_acte": fact.secondary,
                    "echeance": fact.deadline.isoformat() if fact.deadline else None,
                    "auteur": fact.author.kind,
                    "affaire": fact.case_type,
                },
                "fiche": writer.sheet(fact).text,
                "llm": asdict(draft),
                "extrait": item.snippet,
                "gabarit": {"title": template.title, "text": template.snippet},
            }
        )
        # Rien qui relie un cas à son profil : l'annotation reste à l'aveugle.
        status = "ok" if draft.ok else "rejeté"
        print(f"{case_id}  {draft.seconds:5.1f} s  {draft.tries} essai(s)  {status}")
    out.mkdir(parents=True, exist_ok=True)
    _write_jsonl(out / "a_annoter.jsonl", cases)
    _write_jsonl(out / "attendu.jsonl", expected)
    drafts = [e["llm"] for e in expected]
    summary = {
        **client.identity(),
        "prompt_version": L.PROMPT_VERSION,
        "temperature": L.TEMPERATURE,
        "top_p": L.TOP_P,
        "max_tries": L.MAX_TRIES,
        "seed": seed,
        "draw": draw,
        "started": started.isoformat(timespec="seconds"),
        "cases": len(cases),
        "passed": sum(d["ok"] for d in drafts),
        "passed_first_try": sum(d["ok"] and d["tries"] == 1 for d in drafts),
        "seconds_per_call": round(
            sum(d["seconds"] for d in drafts) / sum(d["tries"] for d in drafts), 2
        ),
        "seconds_per_text": round(statistics.mean(d["seconds"] for d in drafts), 2),
    }
    (out / "sonde.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n"
    )
    return summary


def essai(item_ids: list[str], seed: int = 0, client: L.Ollama | None = None) -> str:
    """Réécrit des faits connus et montre fiche, texte et contrôles, sans rien écrire.

    Pour vérifier une nouvelle consigne sur les cas déjà ratés ; jamais pour une mesure :
    l'annotateur connaît ces faits.
    """
    client = client or L.Ollama()
    personas = {p.key: p for p in all_personas(seed)}
    out, drafts = [], []
    for item_id in item_ids:
        persona = personas[item_id.split(":")[1].rsplit("-", 1)[0]]
        fact = next(f for f in Sim(persona, seed).run() if f.item_id == item_id)
        writer = L.LLMWriter(persona, seed, client)
        d = writer.draft(fact)
        drafts.append(d)
        status = "ok" if d.ok else "REJETÉ"
        out += [
            "=" * 80,
            f"{item_id}  {label(fact)}  {d.tries} essai(s), {d.seconds} s, {status}",
            f"  exemples : {', '.join(d.examples)} ; {len(d.corps.split())} mots",
            *(f"  rejet : {'; '.join(r)}" for r in d.rejects if r),
            "--- fiche ---",
            writer.sheet(fact).text,
            "--- texte ---",
            f"Objet : {d.objet}",
            d.corps,
            "--- extrait lu par le modèle ---",
            L.extract(d.corps),
        ]
    if drafts:
        calls = sum(d.tries for d in drafts)
        out += [
            "=" * 80,
            (
                f"{client.model} : {sum(d.ok for d in drafts)}/{len(drafts)} passent les "
                f"contrôles ({sum(d.ok and d.tries == 1 for d in drafts)} du premier coup), "
                f"{sum(d.seconds for d in drafts) / calls:.1f} s par appel, "
                f"{statistics.mean(d.seconds for d in drafts):.1f} s par texte"
            ),
        ]
    return "\n".join(out)


def main(argv: list[str]) -> int:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    args = list(argv)
    model = L.MODEL
    if "--modele" in args:  # un autre modèle d'Ollama, pour comparer (la référence ne change pas)
        i = args.index("--modele")
        model = args[i + 1]
        del args[i : i + 2]
    client = L.Ollama(model)
    if "--essai" in args:
        print(essai(args[args.index("--essai") + 1 :], client=client))
        return 0
    draw = 1
    if "--tirage" in args:
        i = args.index("--tirage")
        draw = int(args[i + 1])
        del args[i : i + 2]
    summary = run(draw=draw, client=client)
    print(
        f"{summary['passed']}/{summary['cases']} textes passent les contrôles "
        f"({summary['passed_first_try']} du premier coup), "
        f"{summary['seconds_per_call']} s par appel, {summary['seconds_per_text']} s par texte"
    )
    print(f"→ {folder(draw)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
