"""Connexion OpenID Connect (L3.4, #112, étape C) : Microsoft pour la démo, tout autre
fournisseur OpenID Connect par simple configuration (`OIDC_*`, voir infra/.env.example).

Parcours « code d'autorisation » avec PKCE :
1. `GET /auth/login` tire `state`, `nonce` et un secret PKCE, gardés côté serveur 10 min ;
   le navigateur part chez le fournisseur avec `state`, `nonce` et l'empreinte du secret ;
2. le fournisseur renvoie sur `GET /auth/callback?code&state` : `state` doit être celui du
   cookie de ce navigateur (connexion imposée par un tiers) et ne sert qu'une fois ;
3. le code est échangé contre un `id_token`, avec le secret PKCE (un code intercepté est
   inutilisable) ; on vérifie signature (clés publiées du fournisseur), aud, iss, exp, nonce ;
4. l'identité (`tid:oid` chez Microsoft) est cherchée dans USERS_FILE, hors dépôt : un compte
   absent n'entre pas, même s'il est valide chez Microsoft ;
5. la session s'ouvre comme en démo, et le travail en attente est repris (`resume.complete`).

Ce module vérifie l'identité ; les routes `/auth/login` et `/auth/callback` arrivent en C2.
Aucun jeton n'est gardé ni journalisé : seul l'identifiant interne (`u_…`) sert ensuite.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import threading
import tomllib
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Protocol

import jwt

from alliance_api.auth import valid_user_id
from alliance_api.sessions import Clock, utc_now

ATTEMPT_TTL = timedelta(minutes=10)
_REQUIRED = ("OIDC_ISSUER", "OIDC_CLIENT_ID", "OIDC_REDIRECT_URI", "USERS_FILE")


class LoginFailed(Exception):
    """Connexion refusée. `reason` va au journal ; la page n'affiche que `public`."""

    def __init__(self, reason: str, public: str = "login_failed") -> None:
        super().__init__(reason)
        self.reason = reason
        self.public = public


@dataclass(frozen=True)
class OidcConfig:
    issuer: str  # ex. https://login.microsoftonline.com/common/v2.0
    client_id: str
    redirect_uri: str  # ex. http://localhost:8000/auth/callback, déclarée chez le fournisseur
    client_secret: str | None = None  # facultatif : PKCE suffit pour un client public
    scopes: str = "openid profile email"
    identity_claims: tuple[str, ...] = ("tid", "oid")  # Microsoft ; « sub » ailleurs
    tenants: frozenset[str] = frozenset()  # locataires acceptés ; vide : tous

    @classmethod
    def from_env(cls) -> OidcConfig:
        missing = [name for name in _REQUIRED if not os.getenv(name)]
        if missing:
            raise ValueError(f"AUTH_MODE=oidc : variables manquantes {missing}")

        def items(name: str, default: str) -> tuple[str, ...]:
            return tuple(v.strip() for v in os.getenv(name, default).split(",") if v.strip())

        return cls(
            issuer=os.environ["OIDC_ISSUER"],
            client_id=os.environ["OIDC_CLIENT_ID"],
            redirect_uri=os.environ["OIDC_REDIRECT_URI"],
            client_secret=os.getenv("OIDC_CLIENT_SECRET") or None,
            scopes=os.getenv("OIDC_SCOPES", cls.scopes),
            identity_claims=items("OIDC_IDENTITY_CLAIMS", "tid,oid"),
            tenants=frozenset(items("OIDC_TENANTS", "")),
        )


class Provider(Protocol):
    def metadata(self) -> dict[str, Any]: ...  # document de découverte du fournisseur
    def signing_key(self, id_token: str) -> Any: ...  # clé publique qui a signé le jeton
    def exchange(self, form: dict[str, str]) -> dict[str, Any]: ...  # code → jetons


class HttpProvider:
    """Fournisseur réel : découverte, clés publiques (mises en cache) et échange du code."""

    def __init__(self, issuer: str) -> None:
        self._discovery = issuer.rstrip("/") + "/.well-known/openid-configuration"
        self._metadata: dict[str, Any] | None = None
        self._keys: jwt.PyJWKClient | None = None

    def metadata(self) -> dict[str, Any]:
        if self._metadata is None:
            self._metadata = _http_json(urllib.request.Request(self._discovery))
            self._keys = jwt.PyJWKClient(self._metadata["jwks_uri"], cache_keys=True)
        return self._metadata

    def signing_key(self, id_token: str) -> Any:
        self.metadata()
        assert self._keys is not None
        return self._keys.get_signing_key_from_jwt(id_token).key

    def exchange(self, form: dict[str, str]) -> dict[str, Any]:
        data = urllib.parse.urlencode(form).encode()
        url = self.metadata()["token_endpoint"]
        return _http_json(urllib.request.Request(url, data=data))


def _http_json(request: urllib.request.Request) -> dict[str, Any]:
    request.add_header("Accept", "application/json")
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as exc:
        # Seul le code OAuth (ex. invalid_grant) est gardé : jamais le corps de la réponse.
        try:
            code = json.loads(exc.read()).get("error", exc.code)
        except ValueError:
            code = exc.code
        raise LoginFailed(f"fournisseur : {code}") from None
    except (urllib.error.URLError, TimeoutError, ValueError) as exc:
        raise LoginFailed(f"fournisseur injoignable : {type(exc).__name__}") from None


@dataclass(frozen=True)
class Attempt:
    nonce: str
    verifier: str  # secret PKCE
    next_url: str
    created_at: datetime


@dataclass
class Oidc:
    config: OidcConfig
    provider: Provider
    accounts: dict[str, str]  # identité chez le fournisseur → identifiant interne
    clock: Clock = utc_now
    _attempts: dict[str, Attempt] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    @classmethod
    def from_env(cls) -> Oidc:
        config = OidcConfig.from_env()
        return cls(
            config, HttpProvider(config.issuer), load_accounts(Path(os.environ["USERS_FILE"]))
        )

    def start(self, state: str, attempt: Attempt) -> None:
        with self._lock:
            now = self.clock()
            self._attempts = {
                s: a for s, a in self._attempts.items() if now - a.created_at < ATTEMPT_TTL
            }
            self._attempts[state] = attempt

    def finish(self, state: str) -> Attempt:
        """La tentative de connexion de ce `state`, une seule fois et dans les 10 minutes."""
        with self._lock:
            attempt = self._attempts.pop(state, None)
        if attempt is None or self.clock() - attempt.created_at >= ATTEMPT_TTL:
            raise LoginFailed("state inconnu, déjà utilisé ou expiré")
        return attempt


def load_accounts(path: Path) -> dict[str, str]:
    """`[accounts]` de USERS_FILE : "<tid>:<oid>" = "u_…". Vérifié au démarrage."""
    accounts = dict(tomllib.loads(path.read_text(encoding="utf-8")).get("accounts", {}))
    invalid = sorted(u for u in accounts.values() if not valid_user_id(str(u)))
    if invalid:
        raise ValueError(f"{path} : identifiants internes invalides {invalid}")
    return accounts


def pkce_challenge(verifier: str) -> str:
    digest = hashlib.sha256(verifier.encode()).digest()
    return base64.urlsafe_b64encode(digest).rstrip(b"=").decode()


def validate_id_token(oidc: Oidc, id_token: str, nonce: str) -> dict[str, Any]:
    try:
        claims = jwt.decode(
            id_token,
            oidc.provider.signing_key(id_token),
            algorithms=["RS256"],
            audience=oidc.config.client_id,
            options={"require": ["exp", "iat", "iss", "aud", "nonce"], "verify_iss": False},
            leeway=60,
        )
    except jwt.PyJWTError as exc:
        raise LoginFailed(f"id_token refusé : {type(exc).__name__}") from None
    issuer = oidc.provider.metadata()["issuer"]
    if "{tenantid}" in issuer:  # Microsoft multi-locataire : l'émetteur dépend du locataire
        issuer = issuer.replace("{tenantid}", str(claims.get("tid", "")))
    if claims["iss"] != issuer:
        raise LoginFailed("émetteur inattendu")
    if oidc.config.tenants and claims.get("tid") not in oidc.config.tenants:
        raise LoginFailed("locataire non accepté", public="unknown_account")
    if not hmac.compare_digest(str(claims["nonce"]), nonce):
        raise LoginFailed("nonce différent")
    return claims


def identity(oidc: Oidc, claims: dict[str, Any]) -> str:
    return ":".join(str(claims.get(name, "")) for name in oidc.config.identity_claims)
