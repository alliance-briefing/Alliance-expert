"""GET /items/{item_id} : métadonnées et lien source d'un item (pour « Voir la source »)."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Path, Request
from pydantic import BaseModel

from alliance_api.audit import api_event
from alliance_api.auth import CurrentUser
from alliance_api.errors import ApiError
from contracts.models import Item
from contracts.models.item import Source

router = APIRouter()

ItemId = Annotated[str, Path(pattern=r"^(mail|calendar|task|file):\S{1,200}$")]


class ItemView(BaseModel):
    """Ce que l'interface affiche d'un item : ni corps du message, ni participants, ni ACL.

    `title` et `snippet` sont du texte brut, potentiellement piégé : à échapper à l'affichage.
    """

    item_id: str
    source: Source
    title: str
    snippet: str
    url: str
    created_at: datetime
    updated_at: datetime
    start_at: datetime | None
    end_at: datetime | None
    due_at: datetime | None
    collected_at: datetime
    connector: str

    @classmethod
    def of(cls, item: Item) -> ItemView:
        return cls(
            **item.model_dump(
                include={
                    "item_id",
                    "source",
                    "title",
                    "snippet",
                    "created_at",
                    "updated_at",
                    "start_at",
                    "end_at",
                    "due_at",
                    "collected_at",
                }
            ),
            url=str(item.url),
            connector=f"{item.connector.name}/{item.connector.version}",
        )


@router.get("/items/{item_id}", response_model=ItemView)
def get_item(item_id: ItemId, user: CurrentUser, request: Request) -> ItemView:
    state = request.app.state
    item = state.items.find(item_id)
    if item is None:
        state.audit.write(api_event(user, "view", [item_id], 0, "item_not_found"))
        raise ApiError(404, "item_not_found", "item inconnu")
    if item.owner_user_id != user:
        # L'identifiant tenté est journalisé : utile pour repérer un essai d'intrusion.
        state.audit.write(api_event(user, "view", [item_id], 0, "forbidden"))
        raise ApiError(403, "forbidden", "cet item appartient à un autre utilisateur")
    state.audit.write(api_event(user, "view", [item_id], 1))
    return ItemView.of(item)
