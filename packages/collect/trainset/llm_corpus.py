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

L'assemblage en `Item` du contrat A (texte du LLM s'il passe les contrôles, sinon celui des
gabarits) se fait ensuite, sans GPU, à partir de ce fichier.
"""

from __future__ import annotations

import json
import random
import sys
import time
import urllib.error
from collections.abc import Callable
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from packages.collect.trainset import llm_writer as L
from packages.collect.trainset.cases import Sim
from packages.collect.trainset.facts import Fact
from packages.collect.trainset.world import Persona, all_personas

OUT = Path(__file__).with_name("data") / "llm"
PROGRESS_EVERY = 10
NETWORK_RETRIES = 3
NETWORK_PAUSE = 30.0  # secondes entre deux tentatives quand Ollama ne répond pas


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
    path = out / "textes.jsonl"
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


def main(argv: list[str]) -> int:
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)  # type: ignore[attr-defined]
    args = list(argv)
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
