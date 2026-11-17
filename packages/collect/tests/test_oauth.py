"""OAuth Microsoft : PKCE, jetons chiffrés, renouvellement, révocation (guide L1.2).

Aucun appel réseau : Microsoft est simulé. Le jeton factice est une chaîne repérable,
cherchée ensuite dans les journaux et la sortie (« grep automatisé » du guide).
"""

from __future__ import annotations

import base64
import hashlib
import logging
import threading
import urllib.parse
import urllib.request
from datetime import UTC, datetime, timedelta

import pytest
from cryptography.fernet import Fernet

from packages.collect import auth as auth_cli
from packages.collect.connectors.calendar_graph.connector import (
    GraphCalendarConnector,
    graph_source,
)
from packages.collect.core import oauth
from packages.collect.core.auth import AuthError, get_access_token
from packages.collect.core.connector import collection_window
from packages.collect.core.oauth import (
    OAuthConfig,
    OAuthError,
    ReauthRequired,
    Tokens,
    TokenStore,
    access_token,
    authorize_url,
    exchange_code,
    load_config,
    pkce_pair,
)
from packages.collect.core.runner import run
from packages.collect.tests.test_graph_http import FakeGraph, _ok

ACCESS = "ACCESS-tok-9f8e7d"
REFRESH = "REFRESH-tok-1a2b3c"
NOW = datetime(2026, 10, 21, 6, 30, tzinfo=UTC)
CONFIG = OAuthConfig(client_id="00000000-0000-0000-0000-000000000001")
USER = "u_claire"


class FakeMicrosoft:
    """Point /token simulé : renvoie les réponses prévues et garde les formulaires reçus."""

    def __init__(self, *responses: tuple[int, dict]) -> None:
        self.responses = list(responses)
        self.forms: list[dict[str, str]] = []

    def __call__(self, url: str, form: dict[str, str]) -> tuple[int, dict]:
        assert url == f"{CONFIG.authority}/token"
        self.forms.append(form)
        return self.responses.pop(0)


def _granted(access: str = ACCESS, refresh: str = REFRESH) -> tuple[int, dict]:
    return 200, {"access_token": access, "refresh_token": refresh, "expires_in": 3600}


@pytest.fixture
def store(tmp_path) -> TokenStore:
    return TokenStore(tmp_path / "tokens", Fernet.generate_key().decode())


def _saved(store: TokenStore, expires_at: datetime) -> None:
    store.save(USER, Tokens(ACCESS, REFRESH, expires_at.isoformat()))


def test_pkce_sends_only_the_fingerprint_of_the_secret() -> None:
    verifier, challenge = pkce_pair()
    assert 43 <= len(verifier) <= 128
    expected = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=")
    assert challenge == expected.decode()
    url = authorize_url(CONFIG, "http://localhost:5000", challenge, "etat")
    query = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
    assert query["code_challenge"] == [challenge] and query["code_challenge_method"] == ["S256"]
    assert verifier not in url and query["state"] == ["etat"]


def test_only_read_scopes_are_requested() -> None:
    scopes = set(oauth.SCOPES)
    assert scopes == {
        "offline_access",
        "User.Read",
        "Mail.Read",
        "Calendars.Read",
        "Tasks.Read",
        "Files.Read",
    }
    assert not any("Write" in s or "Send" in s for s in scopes)


def test_code_is_exchanged_with_the_verifier_and_no_secret() -> None:
    microsoft = FakeMicrosoft(_granted())
    tokens = exchange_code(CONFIG, "code-1", "verif", "http://localhost:5000", microsoft, NOW)
    form = microsoft.forms[0]
    assert form["grant_type"] == "authorization_code" and form["code_verifier"] == "verif"
    assert "client_secret" not in form
    assert tokens.access_token == ACCESS and tokens.expires_at == "2026-10-21T07:30:00+00:00"


def test_tokens_are_encrypted_and_private_on_disk(store: TokenStore) -> None:
    _saved(store, NOW)
    raw = store.path(USER).read_bytes()
    assert ACCESS.encode() not in raw and REFRESH.encode() not in raw
    assert store.path(USER).stat().st_mode & 0o777 == 0o600
    assert store.load(USER).access_token == ACCESS
    assert ACCESS not in repr(store.load(USER))
    other_key = TokenStore(store.folder, Fernet.generate_key().decode())
    with pytest.raises(ReauthRequired):
        other_key.load(USER)  # mauvaise clé : illisible, pas de plantage
    with pytest.raises(ValueError):
        store.path("../u_bob")


def test_valid_token_is_used_without_calling_microsoft(store: TokenStore) -> None:
    _saved(store, NOW + timedelta(hours=1))
    microsoft = FakeMicrosoft()
    assert access_token(USER, store, CONFIG, microsoft, lambda: NOW) == ACCESS
    assert microsoft.forms == []


def test_expired_token_is_renewed_without_intervention(store: TokenStore) -> None:
    _saved(store, NOW + timedelta(minutes=2))  # dans la marge de 5 min : renouvelé
    microsoft = FakeMicrosoft(_granted("ACCESS-neuf", "REFRESH-neuf"))
    assert access_token(USER, store, CONFIG, microsoft, lambda: NOW) == "ACCESS-neuf"
    assert microsoft.forms[0]["grant_type"] == "refresh_token"
    assert microsoft.forms[0]["refresh_token"] == REFRESH
    assert store.load(USER).refresh_token == "REFRESH-neuf"  # rotation enregistrée


def test_revoked_access_gives_reauth_required(store: TokenStore) -> None:
    _saved(store, NOW - timedelta(hours=1))
    microsoft = FakeMicrosoft((400, {"error": "invalid_grant", "error_description": REFRESH}))
    with pytest.raises(ReauthRequired, match="reauth_required") as error:
        access_token(USER, store, CONFIG, microsoft, lambda: NOW)
    assert REFRESH not in str(error.value)


def test_server_error_does_not_leak_the_token(store: TokenStore) -> None:
    _saved(store, NOW - timedelta(hours=1))
    with pytest.raises(OAuthError) as error:
        access_token(USER, store, CONFIG, FakeMicrosoft((503, {})), lambda: NOW)
    assert not isinstance(error.value, ReauthRequired)
    assert ACCESS not in str(error.value) and REFRESH not in str(error.value)


def test_never_connected_is_reauth_required(store: TokenStore) -> None:
    with pytest.raises(ReauthRequired):
        store.load("u_inconnu")


def _browser_that_logs_in(code: str = "code-1", state: str | None = None):
    """Simule l'utilisateur : Microsoft renvoie le navigateur sur la redirection locale."""

    def open_browser(url: str) -> None:
        query = dict(urllib.parse.parse_qsl(urllib.parse.urlparse(url).query))
        target = f"{query['redirect_uri']}?code={code}&state={state or query['state']}"

        def follow() -> None:
            for _ in range(50):
                try:
                    urllib.request.urlopen(target, timeout=2).read()
                    return
                except OSError:
                    threading.Event().wait(0.05)

        threading.Thread(target=follow, daemon=True).start()

    return open_browser


def test_full_login_stores_the_tokens(store: TokenStore) -> None:
    microsoft = FakeMicrosoft(_granted())
    tokens = oauth.login(USER, store, CONFIG, _browser_that_logs_in(), microsoft, lambda: NOW)
    assert tokens.access_token == ACCESS and store.load(USER).refresh_token == REFRESH
    assert microsoft.forms[0]["code"] == "code-1"
    assert microsoft.forms[0]["redirect_uri"].startswith("http://localhost:")


def test_forged_redirect_is_refused(store: TokenStore) -> None:
    with pytest.raises(OAuthError, match="state"):
        oauth.login(USER, store, CONFIG, _browser_that_logs_in(state="faux"), FakeMicrosoft())
    with pytest.raises(ReauthRequired):
        store.load(USER)


def test_config_comes_from_collect_toml(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("MS_CLIENT_ID", raising=False)
    assert load_config().client_id == "2cc83c27-af71-4bfc-9338-52885df8f00d"
    monkeypatch.setenv("MS_CLIENT_ID", "autre-id")
    assert load_config().client_id == "autre-id"
    monkeypatch.delenv("MS_CLIENT_ID")
    with pytest.raises(OAuthError):
        load_config(tmp_path / "absent.toml")


@pytest.fixture
def env(monkeypatch, tmp_path) -> TokenStore:
    key = Fernet.generate_key().decode()
    monkeypatch.setenv("TOKEN_KEY", key)
    monkeypatch.setenv("TOKENS_DIR", str(tmp_path / "tokens"))
    monkeypatch.delenv("GRAPH_ACCESS_TOKEN", raising=False)
    return TokenStore()


def test_cli_login_status_logout(env: TokenStore, monkeypatch, capsys) -> None:
    real_login = oauth.login
    monkeypatch.setattr(
        auth_cli,
        "login",
        lambda user, store, config, browser: real_login(
            user, store, config, browser, FakeMicrosoft(_granted()), lambda: NOW
        ),
    )
    assert auth_cli.main(["login", "--user", USER], open_browser=_browser_that_logs_in()) == 0
    assert auth_cli.main(["status", "--user", USER]) == 0
    assert auth_cli.main(["logout", "--user", USER]) == 0
    assert auth_cli.main(["status", "--user", USER]) == 1
    out = capsys.readouterr()
    assert "connecté" in out.out and "jetons effacés" in out.out and "reauth_required" in out.err
    assert ACCESS not in out.out + out.err and REFRESH not in out.out + out.err


def test_keygen_prints_a_usable_key(capsys) -> None:
    assert auth_cli.main(["keygen"]) == 0
    Fernet(capsys.readouterr().out.strip().encode())  # clé Fernet valide


def test_keygen_save_writes_a_private_key_file_once(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.delenv("TOKEN_KEY", raising=False)
    key_file = tmp_path / "config" / "token_key"
    assert auth_cli.main(["keygen", "--save"], key_file=key_file) == 0
    assert key_file.stat().st_mode & 0o777 == 0o600
    assert auth_cli.main(["keygen", "--save"], key_file=key_file) == 1  # jamais écrasée
    store = TokenStore(tmp_path / "tokens", key_file=key_file)  # clé lue dans le fichier
    store.save(USER, Tokens(ACCESS, REFRESH, NOW.isoformat()))
    assert store.load(USER).access_token == ACCESS
    assert key_file.read_text().strip() not in capsys.readouterr().out


def test_missing_key_is_an_explicit_error(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("TOKEN_KEY", raising=False)
    store = TokenStore(tmp_path / "tokens", key_file=tmp_path / "absente")
    with pytest.raises(OAuthError, match="clé absente"):
        store.save(USER, Tokens(ACCESS, REFRESH, NOW.isoformat()))


def test_connectors_get_the_stored_token_and_never_print_it(env, capsys, caplog) -> None:
    """Collecte complète avec jetons rangés : aucun jeton dans les journaux ni la sortie."""
    caplog.set_level(logging.DEBUG)
    env.save(USER, Tokens(ACCESS, REFRESH, (datetime.now(UTC) + timedelta(hours=1)).isoformat()))
    assert get_access_token(USER) == ACCESS
    fake = FakeGraph(_ok([]))
    run(
        USER,
        collection_window(NOW.date()),
        [GraphCalendarConnector(source=graph_source(send=fake))],
    )
    assert fake.requests[0][1]["Authorization"] == f"Bearer {ACCESS}"  # bien utilisé…
    out = capsys.readouterr()
    for text in (caplog.text, out.out, out.err):  # … mais jamais écrit
        assert ACCESS not in text and REFRESH not in text


def test_without_connection_connectors_get_reauth_required(env) -> None:
    with pytest.raises(AuthError, match="reauth_required"):
        get_access_token("u_jamais_connecte")
