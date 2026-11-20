"""Le corpus de l'écrivain LLM (M3), écrit sur plusieurs nuits.

    .venv\\Scripts\\python.exe -m packages.collect.trainset.llm_corpus

réécrit par le LLM tous les mails rédigés par gabarits des 40 personas (entraînement et
T-gen), dans un ordre mélangé à graine fixe : à tout moment, la partie déjà écrite est un
échantillon équilibré des personas, des métiers et des actes.

- Chaque texte est ajouté dès qu'il est écrit à `data/llm/<consigne>/textes.jsonl` : une
  ligne par fait, avec le brouillon complet et sa traçabilité (modèle, empreinte, consigne,
  graine, essais, raisons de rejet, morceaux retirés).
- Au lancement, les faits déjà écrits sont sautés : on arrête quand on veut (Ctrl+C,
  extinction), on relance la nuit suivante. Au pire, le texte en cours est perdu.
- Un incident d'Ollama (serveur arrêté, délai dépassé) : nouvelles tentatives espacées, puis
  arrêt propre ; relancer suffit.
- `--limite N` : s'arrête après N textes (mesure de vitesse). `--modele` : un autre modèle.

L'assemblage, sans GPU ni appel au LLM (client hors ligne) :

    uv run python -m packages.collect.trainset.llm_corpus --assembler [--textes <fichier>]

rejoue le simulateur et écrit, à côté de `textes.jsonl`, le corpus complet des 40 personas
dans la même forme que `data/` (`items/`, `scenario/`, `manifest.json`), lisible par
`load.py` avec `data=<dossier>`. Pour chaque mail réécrit, le texte du LLM n'est retenu que
s'il passe les contrôles d'aujourd'hui, rejoués sur les morceaux enregistrés, et si le mail
rassemblé est octet pour octet celui du PC GPU ; sinon, celui des gabarits. Un fichier
partiel suffit : les faits pas encore écrits gardent les gabarits (`absent`).

Comparer deux consignes sur les mêmes faits (les N premiers du plan, ordre fixe) :

    uv run python -m packages.collect.trainset.llm_corpus --comparer mail-v7 mail-v8
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import sys
import time
import urllib.error
from collections import Counter
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
from itertools import combinations
from pathlib import Path

from contracts.models import Item
from packages.collect.trainset import llm_writer as L
from packages.collect.trainset.build import _jsonl, _sha, build_persona, distribution
from packages.collect.trainset.cases import Sim
from packages.collect.trainset.facts import Fact
from packages.collect.trainset.world import Persona, all_personas

OUT = Path(__file__).with_name("data") / "llm"
TEXTS = "textes.jsonl"
PROGRESS_EVERY = 10
NETWORK_RETRIES = 3
NETWORK_PAUSE = 30.0  # secondes entre deux tentatives quand Ollama ne répond pas
# Ce qui arrive à un mail réécrit à l'assemblage : texte du LLM retenu, rejeté par les
# contrôles au moment de l'écrire, refusé par les contrôles d'aujourd'hui, pas encore écrit.
STATUSES = ("retenu", "rejete", "recontrole", "absent")


def folder() -> Path:
    return OUT / L.PROMPT_VERSION


def plan(seed: int = 0) -> list[tuple[Persona, Fact]]:
    """Tous les faits à réécrire, dans un ordre mélangé fixe (graine du corpus)."""
    todo = [
        (persona, fact)
        for persona in all_personas(seed)
        for fact in Sim(persona, seed).run()
        if L.rewritable(fact)
    ]
    random.Random(f"trainset:llm-corpus:{seed}").shuffle(todo)
    return todo


def done_ids(path: Path) -> set[str]:
    """Les faits déjà écrits. Une dernière ligne coupée (arrêt brutal) est ignorée."""
    if not path.exists():
        return set()
    ids = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            ids.add(json.loads(line)["item_id"])
        except (json.JSONDecodeError, KeyError):
            continue
    return ids


def _duration(seconds: float) -> str:
    hours, rest = divmod(int(seconds), 3600)
    return f"{hours} h {rest // 60:02d}" if hours else f"{rest // 60} min"


def _draft(writer: L.LLMWriter, fact: Fact, log: Callable[[str], None]) -> L.Draft:
    """Le brouillon d'un fait ; si Ollama ne répond pas, on attend et on réessaie."""
    for attempt in range(NETWORK_RETRIES + 1):
        try:
            return writer.draft(fact)
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            if attempt == NETWORK_RETRIES:
                raise
            log(f"Ollama ne répond pas ({exc}) : nouvel essai dans {NETWORK_PAUSE:.0f} s")
            time.sleep(NETWORK_PAUSE)
    raise AssertionError("inatteignable")


def run(
    out: Path | None = None,
    seed: int = 0,
    limit: int | None = None,
    client: L.Ollama | None = None,
    log: Callable[[str], None] = print,
) -> dict:
    """Écrit les faits restants (au plus `limit`) et rend le bilan de la séance."""
    out = out or folder()
    out.mkdir(parents=True, exist_ok=True)
    path = out / TEXTS
    client = client or L.Ollama()
    todo = plan(seed)
    done = done_ids(path)
    pending = [(p, f) for p, f in todo if f.item_id not in done]
    if limit is not None:
        pending = pending[:limit]
    log(
        f"{len(done)}/{len(todo)} déjà écrits ; cette séance : {len(pending)} "
        f"({client.model}, consigne {L.PROMPT_VERSION}) → {path}"
    )
    writers: dict[str, L.LLMWriter] = {}
    started, n, passed, seconds = time.perf_counter(), 0, 0, 0.0
    session = {"debut": datetime.now().astimezone().isoformat(timespec="seconds")}
    try:
        with path.open("a", encoding="utf-8", newline="\n") as file:
            if path.stat().st_size and not path.read_bytes().endswith(b"\n"):
                file.write("\n")  # ligne coupée par un arrêt brutal : on ne colle rien derrière
            for persona, fact in pending:
                if persona.key not in writers:
                    writers[persona.key] = L.LLMWriter(persona, seed, client)
                d = _draft(writers[persona.key], fact, log)
                row = {"item_id": fact.item_id, "persona": persona.key, "split": persona.split}
                row |= {"extrait": L.extract(d.corps), "llm": asdict(d)}
                file.write(json.dumps(row, ensure_ascii=False) + "\n")
                file.flush()  # chaque texte est sur le disque avant le suivant
                n, passed, seconds = n + 1, passed + d.ok, seconds + d.seconds
                if n % PROGRESS_EVERY == 0 or n == len(pending):
                    rate = (time.perf_counter() - started) / n
                    left = len(todo) - len(done) - n
                    log(
                        f"{len(done) + n}/{len(todo)} écrits · {passed / n:.0%} passent les "
                        f"contrôles · {rate:.1f} s par texte · reste environ {_duration(left * rate)}"
                    )
    except KeyboardInterrupt:
        log("Arrêt demandé : tout ce qui est écrit est gardé, relancer reprendra ici.")
    elapsed = time.perf_counter() - started
    session |= {
        "fin": datetime.now().astimezone().isoformat(timespec="seconds"),
        **client.identity(),
        "consigne": L.PROMPT_VERSION,
        "textes": n,
        "passent": passed,
        "secondes_par_texte": round(elapsed / n, 2) if n else None,
        "secondes_llm": round(seconds, 1),
    }
    with (out / "seances.jsonl").open("a", encoding="utf-8", newline="\n") as file:
        file.write(json.dumps(session, ensure_ascii=False) + "\n")
    total_done = len(done) + n
    log(f"Séance finie : {n} textes en {_duration(elapsed)} ; corpus {total_done}/{len(todo)}.")
    return session | {"ecrits": total_done, "total": len(todo)}


# --- l'assemblage (sans GPU) -------------------------------------------------------------------


def read_texts(path: Path) -> tuple[dict[str, dict], int]:
    """Les lignes de `textes.jsonl` par fait, et le nombre de lignes illisibles.

    Une ligne coupée par un arrêt brutal est ignorée (comme `done_ids`) : le fait sera
    réécrit à la relance. Un fait écrit deux fois est une erreur : on ne choisit pas.
    """
    rows: dict[str, dict] = {}
    unreadable = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            item_id = row["item_id"]
        except (json.JSONDecodeError, KeyError, TypeError):
            unreadable += 1
            continue
        if item_id in rows:
            raise ValueError(f"fait écrit deux fois dans {path.name} : {item_id}")
        rows[item_id] = row
    return rows, unreadable


def _draft_of(row: dict) -> L.Draft:
    try:
        return L.Draft(**row["llm"])
    except (KeyError, TypeError) as exc:  # champs ajoutés ou retirés depuis la rédaction
        raise ValueError(f"brouillon illisible pour {row['item_id']} : {exc}") from exc


def _identity(rows: dict[str, dict]) -> tuple[str, str]:
    """Le modèle et son empreinte : une seule consigne et un seul modèle par corpus."""
    drafts = [_draft_of(r) for r in rows.values()]
    versions = {d.prompt_version for d in drafts}
    if versions != {L.PROMPT_VERSION}:
        raise ValueError(
            f"consignes {sorted(versions)} : ce code assemble la consigne {L.PROMPT_VERSION} "
            "(la fiche et les contrôles en font partie)"
        )
    models = {(d.model, d.digest) for d in drafts}
    if len(models) != 1:
        raise ValueError(f"plusieurs modèles dans le même corpus : {sorted(models)}")
    return models.pop()


def generation_kept(sh: L.Sheet, d: L.Draft) -> set[str] | None:
    """Les morceaux que le PC GPU a gardés : la combinaison des morceaux rendus qui redonne
    exactement le mail enregistré (au plus 8 essais : contexte, ton, conclusion facultatifs).
    None : aucune, la fiche ou l'assemblage diffèrent entre les deux PC."""
    present = [k for k in L.PIECES if d.pieces.get(k)]
    optional = [k for k in present if k in ("contexte", "ton", "conclusion")]
    base = [k for k in present if k not in optional]
    for n in range(len(optional) + 1):
        for extra in combinations(optional, n):
            kept = {k: d.pieces[k] for k in (*base, *extra)}
            if L.assemble(sh, kept) == (d.objet, d.corps):
                return set(kept)
    return None


def recheck(
    writer: L.LLMWriter, fact: Fact, d: L.Draft, extrait: str | None = None
) -> tuple[list[str], L.Draft]:
    """Les raisons de ne plus retenir un texte accepté au moment de l'écrire, et le
    brouillon à utiliser.

    Le mail enregistré doit d'abord s'expliquer par la fiche et l'assemblage d'ici : la preuve
    que les deux PC concordent. Puis les contrôles d'aujourd'hui sont rejoués sur les morceaux
    enregistrés : un morceau requis refusé fait retomber sur les gabarits ; un morceau
    facultatif retiré par une règle nouvelle raccourcit le mail. Une règle d'aujourd'hui peut
    retirer un morceau, jamais en rajouter un que le PC GPU avait retiré.
    """
    sh = writer.sheet(fact)
    stored = generation_kept(sh, d)
    if stored is None:
        return ["mail rassemblé différent de celui enregistré"], d
    if extrait is not None and L.extract(d.corps) != extrait:
        return ["extrait différent de celui enregistré"], d
    kept, reasons, _ = L.judge(fact, sh, d.pieces)
    if reasons:
        return reasons, d
    objet, corps = L.assemble(sh, {k: v for k, v in kept.items() if k in stored})
    return [], replace(d, objet=objet, corps=corps)


@dataclass
class Tally:
    """Le compte de l'assemblage : statut de chaque mail réécrit, faits rencontrés."""

    statuses: Counter = field(default_factory=Counter)  # (persona, statut) → nombre
    seen: set[str] = field(default_factory=set)
    rechecked: list[dict] = field(default_factory=list)
    shortened: list[str] = field(default_factory=list)  # retenus, raccourcis par une règle nouvelle


def _choose(
    writer: L.LLMWriter, rows: dict[str, dict], tally: Tally, fact: Fact
) -> tuple[Item | None, list[str]]:
    """L'Item d'un fait : texte du LLM retenu, sinon gabarits, avec le statut en marque."""
    if not L.rewritable(fact):
        return writer.choose(fact, None)
    row = rows.get(fact.item_id)
    d = None
    if row is None:
        status = "absent"
    else:
        tally.seen.add(fact.item_id)
        if row.get("persona") != writer.p.key:
            raise ValueError(f"{fact.item_id} : persona {row.get('persona')} ≠ {writer.p.key}")
        d = _draft_of(row)
        reasons: list[str] = []
        if d.ok:
            reasons, used = recheck(writer, fact, d, row.get("extrait"))
            if not reasons and used.corps != d.corps:
                tally.shortened.append(fact.item_id)
            d = used
        if reasons:
            tally.rechecked.append({"item_id": fact.item_id, "raisons": reasons})
        status = "rejete" if not d.ok else "recontrole" if reasons else "retenu"
    tally.statuses[writer.p.key, status] += 1
    if status in ("retenu", "rejete"):
        return writer.choose(fact, d)
    item, used = writer.choose(fact, None)
    return item, used + [f"llm:{L.PROMPT_VERSION}:{status}"]


def _counts(statuses: Counter, keys: set[str] | None = None) -> dict:
    c = Counter()
    for (key, status), n in statuses.items():
        if keys is None or key in keys:
            c[status] += n
    written = c["retenu"] + c["rejete"] + c["recontrole"]
    return {
        "a_reecrire": written + c["absent"],
        "ecrits": written,
        **{f"{s}s": c[s] for s in STATUSES},
        "taux_controles": round(c["retenu"] / written, 4) if written else None,
    }


def assemble_items(texts: Path | None = None, out: Path | None = None, seed: int = 0) -> dict:
    """Écrit le corpus LLM en `Item` (`items/`, `scenario/`, `manifest.json`) et rend le
    manifeste. Tout est calculé avant d'écrire : un refus ne laisse pas un corpus à moitié."""
    out = out or folder()
    texts = texts or out / TEXTS
    rows, unreadable = read_texts(texts)
    if not rows:
        raise ValueError(f"aucun texte lisible dans {texts}")
    model, digest = _identity(rows)
    client = L.Offline(model, digest)
    tally = Tally()
    rendered: dict[str, tuple[Persona, str, str, list[dict]]] = {}
    for persona in all_personas(seed):
        writer = L.LLMWriter(persona, seed, client)

        def write(fact: Fact, writer: L.LLMWriter = writer) -> tuple[Item | None, list[str]]:
            return _choose(writer, rows, tally, fact)

        items, scenario = build_persona(persona, seed, write)
        items_text = _jsonl([i.model_dump(mode="json") for i in items])
        rendered[persona.key] = (persona, items_text, _jsonl(scenario), scenario)
    unknown = sorted(set(rows) - tally.seen)
    if unknown:
        raise ValueError(
            f"{len(unknown)} texte(s) d'un fait inconnu du simulateur (le monde a changé ?) : "
            + ", ".join(unknown[:5])
        )
    manifest: dict = {
        "seed": seed,
        "consigne": L.PROMPT_VERSION,
        "modele": model,
        "empreinte": digest,
        "textes": {
            "sha256": hashlib.sha256(texts.read_bytes()).hexdigest(),
            "lignes": len(rows),
            "illisibles": unreadable,
        },
    }
    splits: dict[str, set[str]] = {}
    for key, (persona, *_) in rendered.items():
        splits.setdefault(persona.split, set()).add(key)
    llm = _counts(tally.statuses)
    manifest["llm"] = llm | {"complet": llm["absents"] == 0, "raccourcis": len(tally.shortened)}
    manifest["llm_par_split"] = {s: _counts(tally.statuses, k) for s, k in sorted(splits.items())}
    manifest["recontroles_detail"] = tally.rechecked
    manifest["personas"] = {
        key: {
            "split": persona.split,
            "elements": len(scenario),
            "llm_retenus": tally.statuses[key, "retenu"],
            "items_sha256": _sha(items_text),
            "scenario_sha256": _sha(scenario_text),
        }
        for key, (persona, items_text, scenario_text, scenario) in rendered.items()
    }
    manifest["distribution"] = {
        split: distribution([r for k in keys for r in rendered[k][3]])
        for split, keys in sorted(splits.items())
    }
    for sub in ("items", "scenario"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    for key, (_, items_text, scenario_text, _) in rendered.items():
        for sub, text in (("items", items_text), ("scenario", scenario_text)):
            (out / sub / f"{key}.jsonl").write_text(text, encoding="utf-8", newline="\n")
    (out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return manifest


# --- comparer deux consignes -------------------------------------------------------------------


def _reason_kind(reason: str) -> str:
    """« demande : ton changé (« au plus vite » hors du ton) » → « demande : ton changé »."""
    return re.sub(r"\s*\(.*$", "", re.sub(r"« [^»]*»", "«…»", reason)).strip()


def compare(old: Path, new: Path, seed: int = 0) -> str:
    """Deux consignes sur les mêmes faits : contrôles passés, essais, temps, par acte.

    L'ordre du plan est fixe : les N premiers textes de chaque consigne portent sur les mêmes
    faits, la comparaison est appariée. L'ancienne consigne est aussi rejugée avec les
    contrôles d'aujourd'hui (fiche d'aujourd'hui, morceaux du dernier essai), pour séparer ce
    que gagne la consigne de ce que gagnent les contrôles.
    """
    rows_old, _ = read_texts(old / TEXTS)
    rows_new, _ = read_texts(new / TEXTS)
    common = sorted(set(rows_old) & set(rows_new))
    if not common:
        return f"Aucun fait écrit à la fois dans {old.name} et {new.name}."
    facts = {f.item_id: (p, f) for p, f in plan(seed)}
    client = L.Offline(L.MODEL, "")
    writers: dict[str, L.LLMWriter] = {}

    def rejudged(row: dict) -> bool:
        p, f = facts[row["item_id"]]
        w = writers.setdefault(p.key, L.LLMWriter(p, seed, client))
        return not L.judge(f, w.sheet(f), row["llm"]["pieces"])[1]

    old_drafts = [_draft_of(rows_old[i]) for i in common]
    new_drafts = [_draft_of(rows_new[i]) for i in common]
    variants = [
        (f"{old.name}", [d.ok for d in old_drafts], old_drafts),
        (f"{old.name} rejugée", [rejudged(rows_old[i]) for i in common], None),
        (f"{new.name}", [d.ok for d in new_drafts], new_drafts),
    ]
    n = len(common)
    lines = [f"{n} faits écrits par les deux consignes (comparaison appariée)."]
    for label, oks, drafts in variants:
        line = f"{label:<22} {sum(oks):5d}/{n} passent ({sum(oks) / n:.1%})"
        if drafts:
            first = sum(d.ok and d.tries == 1 for d in drafts) / n
            calls = sum(d.tries for d in drafts) / n
            seconds = sum(d.seconds for d in drafts) / n
            line += f" · {first:.0%} du 1er coup · {calls:.2f} appels · {seconds:.1f} s par texte"
        lines.append(line)
    acts = Counter(facts[i][1].act for i in common)
    lines.append("Par acte : " + " | ".join(label for label, _, _ in variants))
    for act, total in acts.most_common():
        rates = []
        for _, oks, _ in variants:
            ok = sum(o for o, i in zip(oks, common, strict=True) if facts[i][1].act == act)
            rates.append(f"{ok / total:.0%}")
        lines.append(f"  {act:<9} ({total:4d})  " + " | ".join(rates))
    reasons = Counter(
        kind for d in new_drafts if not d.ok for kind in {_reason_kind(r) for r in d.rejects[-1]}
    )
    lines.append(f"Raisons des rejets de {new.name} (dernier essai) :")
    lines += [f"  {count:4d}  {kind}" for kind, count in reasons.most_common(8)]
    return "\n".join(lines)


def _assemble_main(args: list[str]) -> int:
    texts = Path(args[args.index("--textes") + 1]) if "--textes" in args else None
    try:
        m = assemble_items(texts)
    except (ValueError, FileNotFoundError) as exc:
        print(f"Assemblage refusé : {exc}")
        return 1
    c = m["llm"]
    print(
        f"{c['ecrits']}/{c['a_reecrire']} mails écrits par {m['modele']} ({m['consigne']}) ; "
        f"retenus {c['retenus']}, rejetés {c['rejetes']}, refusés au recontrôle "
        f"{c['recontroles']}, pas encore écrits {c['absents']} (gabarits) ; "
        f"{m['textes']['illisibles']} ligne(s) illisible(s)."
    )
    if c["taux_controles"] is not None:
        print(
            f"Passent les contrôles : {c['taux_controles']:.1%} des textes écrits (objectif 95 %)."
        )
    print(f"→ {folder()}")
    return 0


def main(argv: list[str]) -> int:
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)  # type: ignore[attr-defined]
    args = list(argv)
    if "--assembler" in args:
        return _assemble_main(args)
    if "--comparer" in args:
        i = args.index("--comparer")
        print(compare(OUT / args[i + 1], OUT / args[i + 2]))
        return 0
    limit = None
    if "--limite" in args:
        i = args.index("--limite")
        limit = int(args[i + 1])
        del args[i : i + 2]
    model = L.MODEL
    if "--modele" in args:
        i = args.index("--modele")
        model = args[i + 1]
        del args[i : i + 2]
    try:
        run(limit=limit, client=L.Ollama(model))
    except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
        print(f"Ollama ne répond plus ({exc}). Ouvrir l'application Ollama, puis relancer.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
