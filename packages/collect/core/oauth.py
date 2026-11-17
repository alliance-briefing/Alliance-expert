"""OAuth 2.0 Microsoft : code d'autorisation + PKCE, jetons chiffrés au repos (guide L1.2).

Déroulé d'une connexion (`login`) :
1. on tire un secret aléatoire (`verifier`) et on n'envoie à Microsoft que son empreinte
   SHA-256 (`challenge`) : c'est PKCE, qui remplace le secret client d'une application
   installée (un secret livré avec un programme n'en est plus un) ;
2. le navigateur ouvre la page de connexion de Microsoft : le mot de passe n'est tapé que
   chez Microsoft, jamais chez nous ;
3. Microsoft renvoie un code à usage unique sur `http://localhost:<port>` ; notre petit
   serveur local le reçoit et vérifie `state` (protection contre une réponse forgée) ;
4. on échange code + `verifier` contre un jeton d'accès (≈ 1 h) et un jeton de
   renouvellement, rangés chiffrés (Fernet) dans `TOKENS_DIR/<user>.json.enc`.

Aucun jeton n'est jamais journalisé ni affiché ; les erreurs ne citent que le code HTTP ou
le code d'erreur OAuth (`invalid_grant`…).
"""

from __future__ import annotations

import base64
import hashlib
import http.server
import json
import os
import re
import secrets
import socket
import tomllib
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from cryptography.fernet import Fernet, InvalidToken

from packages.collect.core.registry import CONFIG_PATH

SCOPES = (
    "offline_access",  # jeton de renouvellement : collecte sans reconnexion
    "User.Read",  # adresse de l'utilisateur (/me) : « adressé directement à lui »
    "Mail.Read",
    "Calendars.Read",
    "Tasks.Read",
    "Files.Read",
)
EXPIRY_MARGIN = timedelta(minutes=5)  # renouveler un peu avant l'expiration
DEFAULT_TOKENS_DIR = Path.home() / ".local/share/alliance-briefing/tokens"
DEFAULT_KEY_FILE = Path.home() / ".config/alliance-briefing/token_key"  # hors du dépôt
_SAFE_USER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*")

Post = Callable[[str, dict[str, str]], tuple[int, dict]]  # (url, formulaire) → (statut, JSON)
Clock = Callable[[], datetime]


class OAuthError(Exception):
    """Échec technique (réseau, réponse inattendue) : le message ne contient aucun jeton."""


class ReauthRequired(OAuthError):
    """Plus de jeton utilisable (absent, révoqué, expiré) : l'utilisateur doit se reconnecter."""

    def __init__(self, reason: str = "") -> None:
        super().__init__("reauth_required" + (f" ({reason})" if reason else ""))


@dataclass(frozen=True)
class OAuthConfig:
    client_id: str
    tenant: str = "common"  # comptes professionnels et personnels

    @property
    def authority(self) -> str:
        return f"https://login.microsoftonline.com/{self.tenant}/oauth2/v2.0"


def load_config(config_path: Path = CONFIG_PATH) -> OAuthConfig:
    """`[oauth.microsoft]` de collect.toml ; la variable MS_CLIENT_ID a priorité."""
    section: dict = {}
    if config_path.exists():
        section = (
            tomllib.loads(config_path.read_text("utf-8")).get("oauth", {}).get("microsoft", {})
        )
    client_id = os.environ.get("MS_CLIENT_ID") or section.get("client_id", "")
    if not client_id:
        raise OAuthError("client_id Microsoft absent (collect.toml ou MS_CLIENT_ID)")
    return OAuthConfig(client_id=client_id, tenant=section.get("tenant", "common"))


@dataclass(frozen=True)
class Tokens:
    access_token: str
    refresh_token: str
    expires_at: str  # ISO 8601, UTC

    def expired(self, now: datetime) -> bool:
        return now + EXPIRY_MARGIN >= datetime.fromisoformat(self.expires_at)

    def __repr__(self) -> str:  # jamais de jeton dans une trace ou un journal
        return f"Tokens(expires_at={self.expires_at!r})"


# --- PKCE et protocole ------------------------------------------------------------------


def pkce_pair() -> tuple[str, str]:
    """(verifier, challenge) : le secret reste chez nous, Microsoft ne voit que l'empreinte."""
    verifier = secrets.token_urlsafe(64)
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
    return verifier, challenge


def authorize_url(config: OAuthConfig, redirect_uri: str, challenge: str, state: str) -> str:
    params = {
        "client_id": config.client_id,
        "response_type": "code",
        "redirect_uri": redirect_uri,
        "response_mode": "query",
        "scope": " ".join(SCOPES),
        "state": state,
        "code_challenge": challenge,
        "code_challenge_method": "S256",
        "prompt": "select_account",
    }
    return f"{config.authority}/authorize?{urllib.parse.urlencode(params)}"


def urllib_post(url: str, form: dict[str, str]) -> tuple[int, dict]:
    data = urllib.parse.urlencode(form).encode()
    request = urllib.request.Request(url, data=data, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.status, json.loads(response.read() or b"{}")
    except urllib.error.HTTPError as error:
        try:
            return error.code, json.loads(error.read() or b"{}")
        except ValueError:
            return error.code, {}
    except urllib.error.URLError as error:
        raise OAuthError("serveur d'authentification injoignable") from error


def _token_request(config: OAuthConfig, form: dict[str, str], post: Post, now: datetime) -> Tokens:
    status, body = post(f"{config.authority}/token", {"client_id": config.client_id, **form})
    if status == 200 and body.get("access_token"):
        return Tokens(
            access_token=body["access_token"],
            refresh_token=body.get("refresh_token") or form.get("refresh_token", ""),
            expires_at=(now + timedelta(seconds=int(body.get("expires_in", 3600)))).isoformat(),
        )
    code = str(body.get("error") or status)
    if code in {"invalid_grant", "interaction_required", "consent_required"}:
        raise ReauthRequired(code)
    raise OAuthError(f"échec de l'échange de jeton ({code})")


def exchange_code(
    config: OAuthConfig, code: str, verifier: str, redirect_uri: str, post: Post, now: datetime
) -> Tokens:
    form = {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": redirect_uri,
        "code_verifier": verifier,
        "scope": " ".join(SCOPES),
    }
    return _token_request(config, form, post, now)


def refresh(config: OAuthConfig, tokens: Tokens, post: Post, now: datetime) -> Tokens:
    if not tokens.refresh_token:
        raise ReauthRequired("pas de jeton de renouvellement")
    form = {
        "grant_type": "refresh_token",
        "refresh_token": tokens.refresh_token,
        "scope": " ".join(SCOPES),
    }
    return _token_request(config, form, post, now)


# --- Rangement chiffré -----------------------------------------------------------------


def _read_key(path: Path) -> str:
    return path.read_text("ascii").strip() if path.exists() else ""


def save_key(key: str, path: Path = DEFAULT_KEY_FILE) -> None:
    """Range la clé dans un fichier lisible par le seul propriétaire ; n'écrase jamais."""
    if path.exists():
        raise OAuthError(
            f"une clé existe déjà dans {path} : la remplacer rendrait les jetons illisibles"
        )
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.touch(mode=0o600)
    path.write_text(key + "\n", "ascii")


class TokenStore:
    """Un fichier chiffré par utilisateur ; la clé Fernet vient de TOKEN_KEY (jamais du dépôt)."""

    def __init__(
        self,
        folder: Path | str | None = None,
        key: str | None = None,
        key_file: Path = DEFAULT_KEY_FILE,
    ) -> None:
        self.folder = Path(folder or os.environ.get("TOKENS_DIR") or DEFAULT_TOKENS_DIR)
        self._key = key or os.environ.get("TOKEN_KEY", "") or _read_key(key_file)

    def _fernet(self) -> Fernet:
        if not self._key:
            raise OAuthError("clé absente : `auth keygen --save` ou variable TOKEN_KEY")
        return Fernet(self._key.encode())

    def path(self, user: str) -> Path:
        if not _SAFE_USER.fullmatch(user):
            raise ValueError("identifiant utilisateur invalide pour un nom de fichier")
        return self.folder / f"{user}.json.enc"

    def save(self, user: str, tokens: Tokens) -> None:
        self.folder.mkdir(parents=True, exist_ok=True, mode=0o700)
        path = self.path(user)
        tmp = path.with_suffix(".tmp")
        tmp.write_bytes(self._fernet().encrypt(json.dumps(asdict(tokens)).encode()))
        tmp.chmod(0o600)  # lisible par le seul propriétaire
        tmp.replace(path)

    def load(self, user: str) -> Tokens:
        path = self.path(user)
        if not path.exists():
            raise ReauthRequired("aucune connexion enregistrée")
        try:
            return Tokens(**json.loads(self._fernet().decrypt(path.read_bytes())))
        except InvalidToken as error:  # mauvaise clé ou fichier altéré
            raise ReauthRequired("jetons illisibles") from error

    def delete(self, user: str) -> bool:
        path = self.path(user)
        existed = path.exists()
        path.unlink(missing_ok=True)
        return existed


def access_token(
    user: str,
    store: TokenStore,
    config: OAuthConfig,
    post: Post = urllib_post,
    clock: Clock = lambda: datetime.now(UTC),
) -> str:
    """Un jeton d'accès valide pour `user`, renouvelé (et réenregistré) s'il a expiré."""
    tokens = store.load(user)
    now = clock()
    if tokens.expired(now):
        tokens = refresh(config, tokens, post, now)
        store.save(user, tokens)
    return tokens.access_token


# --- Connexion interactive -------------------------------------------------------------


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def wait_for_code(port: int, state: str, timeout: float = 300) -> str:
    """Attend la redirection de Microsoft sur 127.0.0.1:`port` et renvoie le code reçu."""
    received: dict[str, str] = {}

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # nom imposé par http.server
            query = dict(urllib.parse.parse_qsl(urllib.parse.urlparse(self.path).query))
            received.update(query)
            ok = query.get("state") == state and "code" in query
            self.send_response(200 if ok else 400)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            message = "Connexion réussie, vous pouvez fermer cet onglet." if ok else "Échec."
            self.wfile.write(f"<p>{message}</p>".encode())

        def log_message(self, *args: object) -> None:  # l'URL contient le code : pas de journal
            pass

    with http.server.HTTPServer(("127.0.0.1", port), Handler) as server:
        server.timeout = timeout
        server.handle_request()
    if received.get("state") != state:
        raise OAuthError("réponse de connexion invalide (state)")
    if "code" not in received:
        raise ReauthRequired(received.get("error", "connexion refusée"))
    return received["code"]


def login(
    user: str,
    store: TokenStore,
    config: OAuthConfig,
    open_browser: Callable[[str], object],
    post: Post = urllib_post,
    clock: Clock = lambda: datetime.now(UTC),
    port: int | None = None,
) -> Tokens:
    """Connexion complète : navigateur, code reçu en local, échange PKCE, jetons rangés."""
    port = port or _free_port()
    redirect_uri = f"http://localhost:{port}"
    verifier, challenge = pkce_pair()
    state = secrets.token_urlsafe(16)
    open_browser(authorize_url(config, redirect_uri, challenge, state))
    code = wait_for_code(port, state)
    tokens = exchange_code(config, code, verifier, redirect_uri, post, clock())
    store.save(user, tokens)
    return tokens
