"""Application FastAPI (L3.2). `create_app` reçoit ses dépendances pour être testable."""

from __future__ import annotations

import os

from fastapi import FastAPI

from alliance_api import errors
from alliance_api.auth import CurrentUser


def create_app(*, auth_mode: str | None = None) -> FastAPI:
    app = FastAPI(title="Alliance Briefing API")
    app.state.auth_mode = auth_mode if auth_mode is not None else os.getenv("AUTH_MODE", "")
    errors.install(app)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/me")
    def me(user: CurrentUser) -> dict[str, str]:
        """Utilisateur connecté (l'interface s'en sert pour savoir si la session est valide)."""
        return {"user_id": user}

    return app


app = create_app()
