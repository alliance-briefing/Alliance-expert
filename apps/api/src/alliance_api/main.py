"""Application FastAPI (L3.2). `create_app` reçoit ses dépendances pour être testable."""

from __future__ import annotations

import os
from datetime import timedelta
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from alliance_api import briefs, errors, feedback, items, ui
from alliance_api.audit import AuditSink, LogAuditSink
from alliance_api.auth import CurrentUser, install_csrf
from alliance_api.briefs import BriefGenerator, BriefStore, MemoryBriefStore
from alliance_api.feedback import FeedbackStore, JsonlFeedbackStore, MemoryFeedbackStore
from alliance_api.permissions import DEFAULT_ROLES_FILE, Policy, load_policy
from alliance_api.sessions import MemorySessionStore, SessionStore
from alliance_api.stores import DemoItemStore, ItemStore
from packages.analyse import generate_brief


def create_app(
    *,
    item_store: ItemStore | None = None,
    brief_store: BriefStore | None = None,
    generator: BriefGenerator | None = None,
    audit: AuditSink | None = None,
    feedback_store: FeedbackStore | None = None,
    session_store: SessionStore | None = None,
    policy: Policy | None = None,
    auth_mode: str | None = None,
) -> FastAPI:
    app = FastAPI(title="Alliance Briefing API")
    app.state.items = item_store or _default_item_store()
    app.state.briefs = brief_store or MemoryBriefStore()
    app.state.generator = generator or _default_generator()
    app.state.audit = audit or LogAuditSink()
    app.state.feedback = feedback_store or _default_feedback_store()
    app.state.sessions = session_store or _default_session_store()
    app.state.policy = policy or load_policy(Path(os.getenv("ROLES_FILE") or DEFAULT_ROLES_FILE))
    # Origines acceptées en plus de celle de l'application (proxy, nom de domaine) ; virgules.
    app.state.allowed_origins = {o for o in os.getenv("ALLOWED_ORIGINS", "").split(",") if o}
    app.state.auth_mode = auth_mode if auth_mode is not None else os.getenv("AUTH_MODE", "")
    install_csrf(app)  # avant errors.install : la corrélation l'enveloppe
    errors.install(app)
    app.include_router(briefs.router)
    app.include_router(items.router)
    app.include_router(feedback.router)
    app.include_router(ui.router)
    app.mount("/static", StaticFiles(directory=ui.STATIC_DIR), name="static")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/me")
    def me(user: CurrentUser) -> dict[str, str]:
        """Utilisateur connecté (l'interface s'en sert pour savoir si la session est valide)."""
        return {"user_id": user}

    return app


def _default_generator() -> BriefGenerator:
    """Le générateur du lot 2 ; avec DEMO_BRIEFS=annotations, les briefs de référence annotés
    (démonstration de l'interface uniquement, voir demo_briefs.py)."""
    if os.getenv("DEMO_BRIEFS") == "annotations":
        from alliance_api.demo_briefs import annotated_generator

        return annotated_generator(generate_brief, os.getenv("DEMO_ANNOTATOR", "guillaume-carrier"))
    return generate_brief


def _default_item_store() -> ItemStore:
    """Items lus dans DATA_DIR (<user>/<AAAA-MM-JJ>/items.jsonl, écrit par le pipeline du
    lot 1) ; sans DATA_DIR, les journées de démonstration de contracts/fixtures/demo."""
    data_dir = os.getenv("DATA_DIR")
    return DemoItemStore(Path(data_dir)) if data_dir else DemoItemStore()


def _default_feedback_store() -> FeedbackStore:
    """Retours écrits dans FEEDBACK_DIR (<user>.jsonl, relus par l'évaluation) ; sans
    FEEDBACK_DIR, en mémoire (perdus au redémarrage)."""
    feedback_dir = os.getenv("FEEDBACK_DIR")
    return JsonlFeedbackStore(Path(feedback_dir)) if feedback_dir else MemoryFeedbackStore()


def _default_session_store() -> SessionStore:
    """Sessions de 10 h au plus et 1 h d'inactivité au plus, réglables."""
    return MemorySessionStore(
        max_age=timedelta(hours=float(os.getenv("SESSION_MAX_HOURS", "10"))),
        idle=timedelta(minutes=float(os.getenv("SESSION_IDLE_MINUTES", "60"))),
    )


app = create_app()
