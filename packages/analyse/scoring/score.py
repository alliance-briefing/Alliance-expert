"""Score de priorité expliqué pour une affaire (L2.2).

Classe les affaires de 0 à 100 avec des raisons lisibles,
indépendamment du LLM. Mêmes entrées → même sortie (déterministe).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from packages.analyse.aggregate.cases import Case


@dataclass(frozen=True)
class ScoringConfig:
    """Configuration du barème qui modifiable sans toucher au code."""

    frequent_contacts: frozenset[str] = frozenset()
    hierarchy: frozenset[str] = frozenset()
    blacklist: frozenset[str] = frozenset()
    urgency_keywords: frozenset[str] = frozenset(
        {
            "urgent",
            "urgente",
            "urgence",
            "asap",
            "dès que possible",
            "deadline",
            "échéance",
            "impératif",
        }
    )
    noise_keywords: frozenset[str] = frozenset(
        {
            "newsletter",
            "noreply",
            "no-reply",
            "notification",
            "unsubscribe",
            "désabonner",
        }
    )


def score_case(
    case: Case,
    now: datetime,
    config: ScoringConfig | None = None,
) -> tuple[int, list[str]]:
    """Retourne (score 0-100, raisons 1-4) pour une affaire.

    Args:
        case: l'affaire à scorer
        now: l'instant de référence (jamais datetime.now() dans le code)
        config: barème configurable, valeurs par défaut si None

    Returns:
        (score, reasons) — score entier 0-100, reasons liste de 1 à 4 strings
    """
    if config is None:
        config = ScoringConfig()

    score = 50  # score de départ neutre
    reasons: list[str] = []

    # Signal 1 : bruit (newsletters, notifications automatiques)
    all_text = f"{' '.join(item.title for item in case.items)} {' '.join(item.snippet for item in case.items)}".lower()
    sender_emails = {
        p.email.lower() for item in case.items for p in item.participants if p.role == "from"
    }

    is_noise = any(keyword in all_text for keyword in config.noise_keywords) or any(
        keyword in email for email in sender_emails for keyword in config.noise_keywords
    )

    if is_noise:
        score -= 30
        reasons.append("message automatique ou newsletter")

    # --- Signal 2 : importance native ---
    for item in case.items:
        if item.native_importance == "high":
            score += 20
            reasons.append("marqué comme important")
            break
        elif item.native_importance == "low":
            score -= 10
            break

    # --- Signal 3 : mots d'urgence dans le titre ---
    all_titles = " ".join(item.title for item in case.items).lower()
    if any(keyword in all_titles for keyword in config.urgency_keywords):
        score += 15
        reasons.append("contient des mots signalant l'urgence")

    # --- Signal 4 : échéance proche ou tâche en retard ---
    for item in case.items:
        if item.source == "task" and item.is_overdue(now):
            score += 25
            reasons.append("tâche en retard")
            break
        elif item.due_at is not None:
            delta = (item.due_at - now).total_seconds() / 3600  # en heures
            if 0 < delta <= 24:
                score += 20
                reasons.append("échéance dans moins de 24 h")
                break
            elif 0 < delta <= 48:
                score += 10
                reasons.append("échéance dans moins de 48 h")
                break

    # Signal 5 : expéditeur important
    if sender_emails & config.blacklist:
        score -= 20
        reasons.append("expéditeur bloqué")
    elif sender_emails & config.hierarchy:
        score += 25
        reasons.append("message de la hiérarchie")
    elif sender_emails & config.frequent_contacts:
        score += 10
        reasons.append("contact fréquent")

    # --- Signal 6 : réunion à venir dans les 48 h ---
    for item in case.items:
        if item.source == "calendar" and item.start_at is not None:
            delta = (item.start_at - now).total_seconds() / 3600
            if 0 < delta <= 48:
                score += 15
                reasons.append("réunion à venir dans les 48 h")
                break

    # --- Signal 7 : mail non lu adressé directement ---
    for item in case.items:
        if item.source == "mail" and item.is_read is False:
            to_emails = {p.email.lower() for p in item.participants if p.role == "to"}
            if config.frequent_contacts & to_emails or config.hierarchy & to_emails:
                score += 10
                reasons.append("mail non lu adressé directement")
                break

    # garantir les bornes 0-100
    score = max(0, min(100, score))

    # garantir au moins 1 raison
    if not reasons:
        reasons = ["élément professionnel standard"]

    return score, reasons[:4]  # max 4 raisons
