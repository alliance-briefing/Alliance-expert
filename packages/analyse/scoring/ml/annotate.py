"""Page d'annotation locale : on répond oui / non à une question par mail, au clavier.

`uv run python -m packages.analyse.scoring.ml.annotate` puis http://127.0.0.1:8765

Chaque réponse est ajoutée en fin de `scoring/annotations/intent_labels.jsonl` (rien n'est
jamais réécrit : fermer la page ne perd rien, la dernière réponse à une question compte).
Environ 10 % des mails reviennent une seconde fois, sans l'annoncer, pour mesurer la
cohérence de l'annotateur. Le serveur n'écoute que la machine locale.
"""

from __future__ import annotations

import json
import random
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from packages.analyse.scoring.ml.sample import ANNOTATIONS, SAMPLE_PATH, load_sample

LABELS_PATH = ANNOTATIONS / "intent_labels.jsonl"
PAGE = Path(__file__).with_name("annotate.html")
VALUES = ("oui", "non", "?")
REPEAT_SHARE = 0.10


@dataclass(frozen=True)
class Feature:
    key: str
    question: str
    guide: str  # le critère précis, affiché sous la question, pour rester cohérent


FEATURES = (
    Feature(
        "action",
        "Le mail demande-t-il à Hélène de faire quelque chose ?",
        "Oui : envoyer, valider, relire, payer, confirmer, traiter… Non : informer, remercier, "
        "confirmer ce qui est déjà fait, demander à quelqu'un d'autre, inviter à une réunion.",
    ),
    Feature(
        "reponse",
        "L'expéditeur attend-il une réponse d'Hélène ?",
        "Oui : une question posée, une demande d'avis ou de disponibilité. "
        "Non : un message qui se suffit à lui-même. Invitation à une réunion : oui.",
    ),
    Feature(
        "probleme",
        "Le mail signale-t-il un problème ?",
        "Oui : incident, retard, erreur, refus, insatisfaction, alerte de sécurité. "
        "Non : un fonctionnement normal, même s'il demande une action, une invitation.",
    ),
    Feature(
        "echeance",
        "Le mail fixe-t-il une échéance à Hélène ?",
        "Oui : une date ou un délai à respecter (« avant jeudi », « d'ici le 12 »). "
        "Non : la date d'une réunion (invitation comprise), d'un voyage ou d'un événement passé.",
    ),
)
_BY_KEY = {feature.key: feature for feature in FEATURES}


def queue(sample: list[dict], feature: str, seed: int = 0) -> list[tuple[str, bool]]:
    """Ordre fixe des mails pour une question : (item_id, est_une_répétition).

    Les répétitions sont insérées dans la seconde moitié, loin de leur première apparition.
    """
    rng = random.Random(f"{seed}:{feature}")
    ids = [row["item_id"] for row in sample]
    rng.shuffle(ids)
    half = len(ids) // 2
    repeats = rng.sample(ids[:half], round(len(ids) * REPEAT_SHARE))
    tasks: list[tuple[str, bool]] = [(item_id, False) for item_id in ids]
    for n, item_id in enumerate(repeats):
        tasks.insert(half + n * (len(tasks) - half) // len(repeats) + n, (item_id, True))
    return tasks


class LabelStore:
    """Réponses en JSONL, ajoutées en fin de fichier ; la dernière pour une clé compte."""

    def __init__(self, path: Path = LABELS_PATH) -> None:
        self.path = path
        self.answers: dict[tuple[str, str, bool], str] = {}
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line:
                    row = json.loads(line)
                    self.answers[(row["item_id"], row["feature"], row["repeat"])] = row["value"]

    def record(self, item_id: str, feature: str, value: str, repeat: bool) -> None:
        row = {
            "item_id": item_id,
            "feature": feature,
            "value": value,
            "repeat": repeat,
            "at": datetime.now(UTC).isoformat(timespec="seconds"),
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8", newline="\n") as file:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
        self.answers[(item_id, feature, repeat)] = value


class Annotator:
    """Ce que la page demande : l'état des questions, la tâche suivante, une réponse."""

    def __init__(self, sample: list[dict], store: LabelStore) -> None:
        self.sample = {row["item_id"]: row for row in sample}
        self.store = store
        self.queues = {f.key: queue(sample, f.key) for f in FEATURES}

    def _progress(self, feature: str) -> tuple[int, int]:
        tasks = self.queues[feature]
        done = sum((item_id, feature, rep) in self.store.answers for item_id, rep in tasks)
        return done, len(tasks)

    def state(self) -> list[dict]:
        out = []
        for f in FEATURES:
            done, total = self._progress(f.key)
            out.append(
                {"key": f.key, "question": f.question, "guide": f.guide}
                | {"done": done, "total": total}
            )
        return out

    def next_task(self, feature: str) -> dict | None:
        for item_id, repeat in self.queues[feature]:
            if (item_id, feature, repeat) not in self.store.answers:
                row = self.sample[item_id]
                return {"feature": feature, "repeat": repeat} | {
                    k: row[k] for k in ("item_id", "sender", "sender_kind", "title", "text")
                }
        return None

    def answer(self, payload: dict) -> None:
        feature, item_id, value = (
            payload.get("feature"),
            payload.get("item_id"),
            payload.get("value"),
        )
        if feature not in _BY_KEY or item_id not in self.sample or value not in VALUES:
            raise ValueError("réponse invalide")
        self.store.record(item_id, feature, value, bool(payload.get("repeat")))


def make_handler(annotator: Annotator) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def _send(self, status: int, body: bytes, kind: str = "application/json") -> None:
            self.send_response(status)
            self.send_header("Content-Type", f"{kind}; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _json(self, data: object, status: int = HTTPStatus.OK) -> None:
            self._send(status, json.dumps(data, ensure_ascii=False).encode())

        def do_GET(self) -> None:
            url = urlparse(self.path)
            if url.path == "/":
                self._send(HTTPStatus.OK, PAGE.read_bytes(), "text/html")
            elif url.path == "/api/state":
                self._json(annotator.state())
            elif url.path == "/api/next":
                feature = parse_qs(url.query).get("feature", [""])[0]
                if feature not in _BY_KEY:
                    self._json({"error": "question inconnue"}, HTTPStatus.BAD_REQUEST)
                else:
                    self._json({"task": annotator.next_task(feature)})
            else:
                self._json({"error": "introuvable"}, HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:
            if urlparse(self.path).path != "/api/answer":
                self._json({"error": "introuvable"}, HTTPStatus.NOT_FOUND)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                annotator.answer(json.loads(self.rfile.read(length) or b"{}"))
            except (ValueError, json.JSONDecodeError) as exc:
                self._json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
                return
            self._json({"ok": True})

        def log_message(self, format: str, *args: object) -> None:
            pass  # pas de journal à chaque touche

    return Handler


def serve(port: int = 8765, sample_path: Path = SAMPLE_PATH, labels: Path = LABELS_PATH):
    annotator = Annotator(load_sample(sample_path), LabelStore(labels))
    return ThreadingHTTPServer(("127.0.0.1", port), make_handler(annotator))


def main(argv: list[str]) -> int:
    port = int(argv[0]) if argv else 8765
    server = serve(port)
    print(f"Annotation : http://127.0.0.1:{port}  (Ctrl+C pour arrêter)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
