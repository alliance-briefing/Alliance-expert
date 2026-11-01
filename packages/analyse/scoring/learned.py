"""Score appris des affaires dans le brief (L2.2, pipeline A ; ADR-0003).

Chaque mail reçu de l'affaire reçoit le score appris (`scoring/ml/` : intentions MiniLM,
fiche lisible, poids Bradley-Terry). L'affaire prend le score de son **meilleur mail** :
un fil plein de relances sans intérêt ne monte pas, un mail urgent suffit. Les signaux
que le modèle ne voit pas (tâche en retard ou à échéance, réunion dans les 48 h) sont
ajoutés par-dessus, avec les points du score par règles. Une affaire sans mail à noter
garde le score par règles.

Mails notés : reçus (pas envoyés par le manager), hors envoi en masse, non lus ou reçus
dans les 3 jours avant `now` (la fenêtre des mails de l'entraînement).

Choix du score (A/B, réunion du 14/10) : variable d'environnement `ALLIANCE_SCORE`,
`appris` (défaut) ou `regles`. Si le modèle manque ou échoue, le brief garde le score par
règles et l'annonce dans un warning : il ne casse jamais.
"""

from __future__ import annotations

import os
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import cache
from typing import TYPE_CHECKING, Literal

from contracts.models import Item
from packages.analyse.aggregate.cases import Case
from packages.analyse.scoring.score import task_and_meeting_signals

if TYPE_CHECKING:
    from packages.analyse.rag.embeddings import Embedder
    from packages.analyse.scoring.ml.features import MailFeatures
    from packages.analyse.scoring.ml.graph_convert import GraphDataset
    from packages.analyse.scoring.ml.intents import IntentModel
    from packages.analyse.scoring.ml.learned_score import LearnedScore
    from packages.analyse.scoring.ml.persona import Persona

SCORE_ENV = "ALLIANCE_SCORE"
CACHE_ENV = "FASTEMBED_CACHE_PATH"  # dossier du modèle (rempli à la construction de l'image)
RECENT = timedelta(days=3)
MAX_REASONS = 3
UNAVAILABLE = "score appris indisponible ({}) : score par règles"
# generator.llm d'un brief noté par le score appris ; l'interface affiche « par le modèle … ».
GENERATOR = "MiniLM (score de priorité appris)"

# Raisons du modèle (noms des caractéristiques, `features.VECTOR_NAMES`) en clair.
REASONS = {
    "expéditeur : hiérarchie": "message de la hiérarchie",
    "expéditeur : direction": "message d'un membre de la direction",
    "expéditeur : client": "message d'un client",
    "expéditeur : fournisseur": "message d'un fournisseur",
    "expéditeur : automate": "message automatique",
    "demande une action": "demande une action",
    "attend une réponse": "attend une réponse",
    "incident": "signale un incident ou un blocage",
    "alerte": "alerte de sécurité ou bancaire",
    "échéance : dépassée": "échéance dépassée",
    "échéance : aujourd'hui ou demain": "échéance aujourd'hui ou demain",
    "échéance : ≤ 3 jours": "échéance dans 3 jours ou moins",
    "échéance : plus tard": "échéance à venir",
    "balle dans son camp": "pas encore répondu",
    "fil long (≥ 3 messages)": "échange de 3 messages ou plus",
    "hiérarchie dans le fil": "la hiérarchie participe à l'échange",
    "reçu depuis moins de 24 h": "reçu depuis moins de 24 h",
    "réunion dans les 48 h": "réunion liée dans les 48 h",
    "demande d'action du N+1": "demande d'action de la hiérarchie",
}


@dataclass(frozen=True)
class CaseScore:
    """Score d'une affaire, ses raisons, et le mail qui le porte (affiché dans l'entrée)."""

    score: int
    reasons: list[str]
    best_mail_id: str
    model: str  # modèle d'embeddings des intentions (generator.llm_version du brief)


class LearnedScorer:
    """Note les affaires avec le score appris. Modèles chargés une fois, au premier appel."""

    def __init__(
        self,
        embedder: Embedder | None = None,
        intents: IntentModel | None = None,
        weights: LearnedScore | None = None,
    ) -> None:
        from packages.analyse.scoring.ml.intents import IntentModel
        from packages.analyse.scoring.ml.learned_score import LearnedScore

        self.intents = intents or IntentModel.load()
        self.weights = weights or LearnedScore.load()
        if embedder is None:
            from packages.analyse.rag.embeddings import FastEmbedder

            embedder = FastEmbedder(self.intents.embedder_name, os.environ.get(CACHE_ENV))
        self.embedder = embedder

    def score_cases(
        self, cases: Sequence[Case], items: Sequence[Item], user: str, now: datetime
    ) -> dict[str, CaseScore]:
        """{case_id: CaseScore} des affaires qui ont au moins un mail à noter.

        `items` : tous les items de l'utilisateur (contexte des fils) ; aucun autre.
        """
        from packages.analyse.scoring.ml.features import mail_features
        from packages.analyse.scoring.ml.graph_convert import dataset_from_items
        from packages.analyse.scoring.ml.intents import LEARNED, intent_text

        mine = [item for item in items if item.owner_user_id == user]
        persona = persona_for(user, mine)
        dataset = dataset_from_items(mine, persona.address)
        to_score = {
            case.case_id: [m for m in case.items if _scorable(m, dataset, now)] for case in cases
        }
        mails = sorted(
            {m.item_id: m for ms in to_score.values() for m in ms}.values(),
            key=lambda m: m.item_id,
        )
        if not mails:
            return {}
        texts = [intent_text(m.title, m.content_text or m.snippet) for m in mails]
        predicted = self.intents.predict(self.embedder.embed(texts), texts)
        probs = {
            f: {m.item_id: float(p) for m, p in zip(mails, predicted[f], strict=True)}
            for f in LEARNED
        }
        features = {f.item_id: f for f in mail_features(mails, dataset, now, persona, probs)}
        scored = {item_id: self.weights.score(f) for item_id, f in features.items()}

        out = {}
        for case in cases:
            if not to_score[case.case_id]:
                continue
            # Meilleur mail ; à score égal, le plus petit item_id (déterministe).
            best = min(to_score[case.case_id], key=lambda m: (-scored[m.item_id][0], m.item_id))
            score, names = scored[best.item_id]
            signals = sorted(task_and_meeting_signals(case, now), key=lambda s: -s.weight)
            reasons = [s.reason for s in signals] + [
                REASONS[n] for n in names if n in REASONS and _shown(n, features[best.item_id])
            ]
            total = max(0, min(100, score + sum(s.weight for s in signals)))
            out[case.case_id] = CaseScore(
                total, reasons[:MAX_REASONS], best.item_id, self.embedder.name
            )
        return out


def _shown(name: str, f: MailFeatures) -> bool:
    """Une intention apprise n'est une raison que si le modèle la juge probable : une
    probabilité de 0,4 compte dans le score, mais « signale un incident » serait faux."""
    probability = {
        "demande une action": f.action,
        "attend une réponse": f.reponse,
        "incident": f.incident,
        "demande d'action du N+1": f.action,
    }.get(name, 1.0)
    return probability >= 0.5


def _scorable(item: Item, dataset: GraphDataset, now: datetime) -> bool:
    from packages.analyse.scoring.ml.noise import noise_reason

    return (
        item.source == "mail"
        and not dataset.extras[item.item_id].is_sent
        and item.created_at <= now
        and (item.is_read is False or now - item.created_at <= RECENT)
        and noise_reason(item) is None
    )


def persona_for(user: str, items: Sequence[Item]) -> Persona:
    """Qui est qui pour `user`. Le contrat A ne donne ni l'adresse du manager ni son
    organigramme : configuration connue pour les jeux de démo, sinon l'adresse déduite
    (destinataire principal le plus fréquent), sans hiérarchie ni clients."""
    from packages.analyse.scoring.ml.persona import DEMO_CAMILLE, GRAPH_HELENE, Persona

    known = {"u_demo_001": DEMO_CAMILLE, "u_graph_helene": GRAPH_HELENE}
    if user in known:
        return known[user]
    counts = Counter(
        p.email.lower()
        for item in items
        if item.source == "mail"
        for p in item.participants
        if p.role == "to"
    )
    address = min(counts, key=lambda e: (-counts[e], e)) if counts else ""
    empty: frozenset[str] = frozenset()
    return Persona(address, address.partition("@")[2], empty, empty, empty)


@cache
def _default_scorer() -> LearnedScorer:
    return LearnedScorer()


def learned_case_scores(
    cases: Sequence[Case],
    items: Sequence[Item],
    user: str,
    now: datetime,
    scorer: LearnedScorer | Literal["auto"] | None = "auto",
) -> tuple[dict[str, CaseScore], str | None]:
    """(scores appris, warning éventuel). `scorer` : "auto" (selon `ALLIANCE_SCORE`),
    None (score par règles seul) ou un scorer donné (tests)."""
    if scorer == "auto":
        if os.environ.get(SCORE_ENV, "appris") == "regles":
            return {}, None
        try:
            scorer = _default_scorer()
        except Exception as error:  # noqa: BLE001 — le brief ne casse jamais : repli annoncé
            return {}, UNAVAILABLE.format(type(error).__name__)
    if scorer is None:
        return {}, None
    try:
        return scorer.score_cases(cases, items, user, now), None
    except Exception as error:  # noqa: BLE001 — modèle absent, téléchargement impossible…
        return {}, UNAVAILABLE.format(type(error).__name__)


def download_model() -> None:
    """Télécharge le modèle d'embeddings (construction de l'image Docker)."""
    LearnedScorer().embedder.embed(["préchargement"])
