"""POST /feedback et GET /feedback : retour de l'utilisateur sur une entrée du brief (#93).

Une entrée est désignée par (brief_id, section, rank), comme dans le contrat B. Le serveur
recopie les `item_ids` de l'entrée dans le retour : l'évaluation (L3.6) peut ainsi relier un
verdict aux items même si le brief est régénéré plus tard et que les rangs changent.

L'API parle JSON. L'interface (sans JavaScript, ADR 0001) appelle `record` et `given` depuis
ses propres routes de formulaire, comme elle appelle `briefs.generate` et `briefs.view`.

Journal : identifiants seulement, sur le logger. Le contrat C (AuditEvent) ne connaît pas
encore d'action « feedback » ; l'ajouter demande une PR `contract-change` séparée.
"""

from __future__ import annotations

import logging
import re
import threading
from datetime import UTC, datetime
from datetime import date as Date
from pathlib import Path
from typing import Annotated, Literal, Protocol

from fastapi import APIRouter, Query, Request
from pydantic import AwareDatetime, BaseModel, ConfigDict, Field
from starlette.datastructures import State

from alliance_api.auth import CurrentUser, requires, valid_user_id
from alliance_api.errors import ApiError
from contracts.models.brief import SECTION_ORDER, SectionKind

logger = logging.getLogger("alliance_api")
router = APIRouter()

Verdict = Literal["useful", "not_useful", "wrong"]
_BRIEF_ID = re.compile(r"^b_(\d{4}-\d{2}-\d{2})_(.+)$")


class FeedbackRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    brief_id: str = Field(max_length=100)
    section: SectionKind
    rank: int = Field(ge=1, le=1000)
    verdict: Verdict


class FeedbackView(BaseModel):
    """Ce que voit le client : son propre retour, sans identifiant d'utilisateur."""

    model_config = ConfigDict(frozen=True)

    brief_id: str
    section: SectionKind
    rank: int
    verdict: Verdict
    item_ids: list[str]
    ts: AwareDatetime


class Feedback(FeedbackView):
    """Retour stocké (une ligne JSONL relue par l'évaluation)."""

    user_id: str

    def key(self) -> tuple[str, str, str, int]:
        return (self.user_id, self.brief_id, self.section, self.rank)

    def view(self) -> FeedbackView:
        return FeedbackView.model_validate(self.model_dump(exclude={"user_id"}))


class FeedbackStore(Protocol):
    def put(self, feedback: Feedback) -> None: ...
    def for_brief(self, user: str, brief_id: str) -> list[Feedback]:
        """Le dernier verdict de `user` sur chaque entrée du brief, dans l'ordre du brief."""
        ...


def _latest(records: list[Feedback], brief_id: str) -> list[Feedback]:
    """Dernier verdict par entrée (un nouveau clic remplace le précédent), trié section puis rang."""
    latest = {r.key(): r for r in records if r.brief_id == brief_id}
    return sorted(latest.values(), key=lambda r: (SECTION_ORDER.index(r.section), r.rank))


class MemoryFeedbackStore:
    """Retours dans le processus : perdus au redémarrage. Tests et démonstration."""

    def __init__(self) -> None:
        self._records: list[Feedback] = []

    def put(self, feedback: Feedback) -> None:
        self._records.append(feedback)

    def for_brief(self, user: str, brief_id: str) -> list[Feedback]:
        return _latest([r for r in self._records if r.user_id == user], brief_id)


class JsonlFeedbackStore:
    """Un fichier `<dossier>/<user>.jsonl` par utilisateur, en ajout seul : l'historique des
    clics est conservé (changements d'avis compris) ; la lecture garde le dernier verdict.
    Le nom de fichier est sûr : l'identifiant d'utilisateur a été validé par `auth`."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self._lock = threading.Lock()

    def _file(self, user: str) -> Path:
        if not valid_user_id(user):
            raise ValueError("identifiant d'utilisateur invalide")
        return self.root / f"{user}.jsonl"

    def put(self, feedback: Feedback) -> None:
        path = self._file(feedback.user_id)
        with self._lock:
            self.root.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as out:
                out.write(feedback.model_dump_json() + "\n")

    def for_brief(self, user: str, brief_id: str) -> list[Feedback]:
        path = self._file(user)
        if not path.exists():
            return []
        lines = path.read_text(encoding="utf-8").splitlines()
        return _latest([Feedback.model_validate_json(line) for line in lines if line], brief_id)


def _own_day(user: str, brief_id: str) -> Date:
    """Le jour du brief s'il appartient à `user` ; sinon 404, sans dire si le brief existe
    chez quelqu'un d'autre."""
    match = _BRIEF_ID.match(brief_id)
    if match is None or match.group(2) != user:
        raise ApiError(404, "brief_not_found", "brief inconnu")
    try:
        return Date.fromisoformat(match.group(1))
    except ValueError:
        raise ApiError(404, "brief_not_found", "brief inconnu") from None


def record(state: State, user: str, body: FeedbackRequest) -> Feedback:
    """Enregistre le verdict de `user` sur une entrée de son brief (API et interface)."""
    brief = state.briefs.get(user, _own_day(user, body.brief_id))
    if brief is None:
        raise ApiError(404, "brief_not_found", "brief inconnu")
    entry = next(
        (
            entry
            for section in brief.sections
            if section.kind == body.section
            for entry in section.entries
            if entry.rank == body.rank
        ),
        None,
    )
    if entry is None:
        raise ApiError(422, "unknown_entry", "aucune entrée à cette section et à ce rang")
    feedback = Feedback(
        **body.model_dump(),
        item_ids=list(entry.source_item_ids),
        ts=datetime.now(UTC),
        user_id=user,
    )
    state.feedback.put(feedback)
    logger.info(
        "feedback user=%s brief=%s section=%s rank=%d verdict=%s",
        user,
        body.brief_id,
        body.section,
        body.rank,
        body.verdict,
    )
    return feedback


def given(state: State, user: str, brief_id: str) -> list[Feedback]:
    """Verdicts déjà donnés par `user` sur son brief (pour l'état `aria-pressed`)."""
    _own_day(user, brief_id)
    return state.feedback.for_brief(user, brief_id)


@router.post(
    "/feedback",
    status_code=201,
    response_model=FeedbackView,
    dependencies=[requires("feedback:own")],
)
def create_feedback(body: FeedbackRequest, user: CurrentUser, request: Request) -> FeedbackView:
    return record(request.app.state, user, body).view()


@router.get("/feedback", response_model=list[FeedbackView], dependencies=[requires("feedback:own")])
def list_feedback(
    user: CurrentUser,
    request: Request,
    brief_id: Annotated[str, Query(max_length=100)],
) -> list[FeedbackView]:
    return [f.view() for f in given(request.app.state, user, brief_id)]
