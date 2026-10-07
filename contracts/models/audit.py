"""Contrat C — AuditEvent : journal écrit par les lots 1, 2 et 3 (guide §2.4).

On journalise des identifiants et des compteurs, jamais du contenu :
pas de corps de message, pas de jeton, pas d'adresse e-mail complète.
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, field_validator, model_validator

from contracts.models.item import Source

_ACTOR = r"^(connector:[a-z0-9_-]+|analyse|api)$"
_ITEM_ID = re.compile(r"^(mail|calendar|task|file):\S+$")


class AuditEvent(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    ts: AwareDatetime
    actor: str = Field(pattern=_ACTOR)
    user_id: str = Field(min_length=1)
    action: Literal["collect", "generate", "view"]
    source: Source | None = None
    item_ids: list[str] = Field(default_factory=list)
    count: int = Field(ge=0)
    status: Literal["ok", "error"]
    error_code: str | None = None

    @field_validator("item_ids")
    @classmethod
    def _ids_only(cls, values: list[str]) -> list[str]:
        for value in values:
            if not _ITEM_ID.match(value):
                raise ValueError(f"identifiant d'item invalide : {value!r}")
        return values

    @model_validator(mode="after")
    def _no_personal_data(self) -> AuditEvent:
        for value in (self.user_id, self.error_code or "", *self.item_ids):
            if "@" in value:
                raise ValueError("adresse e-mail interdite dans le journal d'audit")
        if self.status == "error" and not self.error_code:
            raise ValueError("un événement en erreur exige un error_code")
        return self
