"""Erreurs au format standard du guide (L3.2) : {code, message, correlation_id}.

Chaque requête reçoit un identifiant de corrélation (repris de l'en-tête X-Correlation-ID
s'il est sûr, sinon généré), renvoyé dans l'en-tête de réponse et dans le corps des erreurs.
Les messages ne recopient jamais l'entrée du client : rien à réinjecter, rien de personnel.
"""

from __future__ import annotations

import logging
import re
import uuid
from collections.abc import Awaitable, Callable

from fastapi import FastAPI, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

CORRELATION_HEADER = "X-Correlation-ID"
_SAFE_CORRELATION_ID = re.compile(r"^[A-Za-z0-9-]{1,64}$")
_HTTP_CODES = {404: "not_found", 405: "method_not_allowed"}

logger = logging.getLogger("alliance_api")


class ApiError(Exception):
    """Erreur métier renvoyée au client : statut HTTP, code stable, message lisible."""

    def __init__(self, status: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message


def correlation_id(request: Request) -> str:
    return getattr(request.state, "correlation_id", "")


def _body(request: Request, status: int, code: str, message: str) -> JSONResponse:
    cid = correlation_id(request)
    return JSONResponse(
        status_code=status,
        content={"code": code, "message": message, "correlation_id": cid},
        headers={CORRELATION_HEADER: cid},
    )


def install(app: FastAPI) -> None:
    """Branche le middleware de corrélation et les gestionnaires d'erreurs sur `app`."""

    @app.middleware("http")
    async def _correlation(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        incoming = request.headers.get(CORRELATION_HEADER, "")
        cid = incoming if _SAFE_CORRELATION_ID.match(incoming) else uuid.uuid4().hex
        request.state.correlation_id = cid
        response = await call_next(request)
        response.headers[CORRELATION_HEADER] = cid
        return response

    @app.exception_handler(ApiError)
    async def _api_error(request: Request, exc: ApiError) -> JSONResponse:
        return _body(request, exc.status, exc.code, exc.message)

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        code = _HTTP_CODES.get(exc.status_code, "http_error")
        return _body(request, exc.status_code, code, "ressource ou méthode inconnue")

    @app.exception_handler(RequestValidationError)
    async def _invalid(request: Request, exc: RequestValidationError) -> JSONResponse:
        fields = sorted({".".join(str(p) for p in err["loc"]) for err in exc.errors()})
        return _body(request, 422, "invalid_request", f"paramètres invalides : {', '.join(fields)}")

    @app.exception_handler(Exception)
    async def _unexpected(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("erreur inattendue (correlation_id=%s)", correlation_id(request))
        return _body(request, 500, "internal_error", "erreur interne")
