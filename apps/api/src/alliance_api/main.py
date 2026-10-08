"""Application FastAPI (L3.2). `create_app` reçoit ses dépendances pour être testable."""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI

from alliance_api import briefs, errors, items, ui
from alliance_api.audit import AuditSink, LogAuditSink
from alliance_api.auth import CurrentUser
from alliance_api.briefs import BriefGenerator, BriefStore, MemoryBriefStore
from alliance_api.stores import DemoItemStore, ItemStore
from packages.analyse import generate_brief


def create_app(
    *,
    item_store: ItemStore | None = None,
    brief_store: BriefStore | None = None,
    generator: BriefGenerator | None = None,
    audit: AuditSink | None = None,
    auth_mode: str | None = None,
) -> FastAPI:
    app = FastAPI(title="Alliance Briefing API")
    app.state.items = item_store or _default_item_store()
    app.state.briefs = brief_store or MemoryBriefStore()
    app.state.generator = generator or generate_brief
    app.state.audit = audit or LogAuditSink()
    app.state.auth_mode = auth_mode if auth_mode is not None else os.getenv("AUTH_MODE", "")
    errors.install(app)
    app.include_router(briefs.router)
    app.include_router(items.router)
    app.include_router(ui.router)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/me")
    def me(user: CurrentUser) -> dict[str, str]:
        """Utilisateur connecté (l'interface s'en sert pour savoir si la session est valide)."""
        return {"user_id": user}

    return app


def _default_item_store() -> ItemStore:
    """Items lus dans DATA_DIR (<user>/<AAAA-MM-JJ>/items.jsonl, écrit par le pipeline du
    lot 1) ; sans DATA_DIR, les journées de démonstration de contracts/fixtures/demo."""
    data_dir = os.getenv("DATA_DIR")
    return DemoItemStore(Path(data_dir)) if data_dir else DemoItemStore()


app = create_app()
