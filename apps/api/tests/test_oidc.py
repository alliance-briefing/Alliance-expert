"""Vérification d'identité OpenID Connect (L3.4, #112, étape C1), avec un faux fournisseur qui
signe de vrais jetons : aucun appel réseau, aucun compte réel."""

import base64
import hashlib
import time
from pathlib import Path
from typing import Any

import jwt
import pytest
from alliance_api.oidc import (
    Attempt,
    LoginFailed,
    Oidc,
    OidcConfig,
    identity,
    load_accounts,
    pkce_challenge,
    validate_id_token,
)
from cryptography.hazmat.primitives.asymmetric import rsa

from apps.api.tests.factories import USER_A
from apps.api.tests.test_resume import Clock

TID, OID, CLIENT = "tenant-demo", "oid-claire", "client-demo"
KEY = rsa.generate_private_key(public_exponent=65537, key_size=2048)
OTHER_KEY = rsa.generate_private_key(public_exponent=65537, key_size=2048)


class FakeProvider:
    """Se comporte comme Microsoft multi-locataire : l'émetteur dépend du locataire."""

    def __init__(self) -> None:
        self.nonce = ""  # celui que le navigateur a transmis à l'autorisation
        self.claims: dict[str, Any] = {}  # surcharges du prochain jeton (None = absent)
        self.key = KEY

    def metadata(self) -> dict[str, Any]:
        return {"issuer": "https://login.example/{tenantid}/v2.0"}

    def signing_key(self, id_token: str) -> Any:
        return KEY.public_key()

    def exchange(self, form: dict[str, str]) -> dict[str, Any]:
        now = int(time.time())
        claims = {
            "iss": f"https://login.example/{TID}/v2.0",
            "aud": CLIENT,
            "iat": now,
            "exp": now + 600,
            "nonce": self.nonce,
            "tid": TID,
            "oid": OID,
        } | self.claims
        claims = {k: v for k, v in claims.items() if v is not None}
        return {"id_token": jwt.encode(claims, self.key, algorithm="RS256")}


def make_oidc(provider: FakeProvider, **config: Any) -> Oidc:
    settings = {
        "issuer": "https://login.example/common/v2.0",
        "client_id": CLIENT,
        "redirect_uri": "http://testserver/auth/callback",
    }
    return Oidc(OidcConfig(**settings | config), provider, accounts={f"{TID}:{OID}": USER_A})


@pytest.fixture
def provider() -> FakeProvider:
    provider = FakeProvider()
    provider.nonce = "nonce-1"
    return provider


def _check(oidc: Oidc, provider: FakeProvider) -> dict[str, Any]:
    token = provider.exchange({})["id_token"]
    return validate_id_token(oidc, token, "nonce-1")


def test_jeton_valide_et_identite(provider: FakeProvider) -> None:
    oidc = make_oidc(provider)
    claims = _check(oidc, provider)
    assert oidc.accounts[identity(oidc, claims)] == USER_A


@pytest.mark.parametrize(
    "claims",
    [
        {"aud": "autre-application"},
        {"exp": int(time.time()) - 3600},
        {"nonce": "rejoue"},
        {"nonce": None},
        {"iss": "https://pirate.example/v2.0"},
        {"iss": f"https://login.example/{TID}/v2.0", "tid": "autre-locataire"},
    ],
)
def test_jeton_invalide_refuse(provider: FakeProvider, claims: dict[str, Any]) -> None:
    provider.claims = claims
    with pytest.raises(LoginFailed) as refused:
        _check(make_oidc(provider), provider)
    assert refused.value.public == "login_failed"


def test_jeton_signe_par_une_autre_cle(provider: FakeProvider) -> None:
    provider.key = OTHER_KEY
    with pytest.raises(LoginFailed, match="InvalidSignatureError"):
        _check(make_oidc(provider), provider)


def test_locataire_non_accepte(provider: FakeProvider) -> None:
    with pytest.raises(LoginFailed) as refused:
        _check(make_oidc(provider, tenants=frozenset({"alliance"})), provider)
    assert refused.value.public == "unknown_account"


def test_identite_configurable(provider: FakeProvider) -> None:
    provider.claims = {"sub": "sujet-1"}
    oidc = make_oidc(provider, identity_claims=("sub",))
    assert identity(oidc, _check(oidc, provider)) == "sujet-1"


def test_tentative_a_usage_unique_et_limitee_a_10_minutes(provider: FakeProvider) -> None:
    clock = Clock()
    oidc = make_oidc(provider)
    oidc.clock = clock
    oidc.start("s1", Attempt("n", "v", "/", clock()))
    oidc.start("s2", Attempt("n", "v", "/", clock()))
    assert oidc.finish("s1").next_url == "/"
    with pytest.raises(LoginFailed):
        oidc.finish("s1")
    clock.advance(minutes=11)
    with pytest.raises(LoginFailed):
        oidc.finish("s2")


def test_pkce_empreinte_s256() -> None:
    verifier = "secret-pkce"
    digest = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest())
    assert pkce_challenge(verifier) == digest.decode().rstrip("=")  # sans « = » final


def test_configuration_lue_dans_l_environnement(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OIDC_ISSUER", "https://login.example/common/v2.0")
    monkeypatch.setenv("OIDC_CLIENT_ID", CLIENT)
    monkeypatch.setenv("OIDC_REDIRECT_URI", "http://localhost:8000/auth/callback")
    monkeypatch.setenv("USERS_FILE", "users.toml")
    monkeypatch.setenv("OIDC_TENANTS", "alliance, autre")
    config = OidcConfig.from_env()
    assert config.tenants == {"alliance", "autre"} and config.identity_claims == ("tid", "oid")
    monkeypatch.delenv("OIDC_CLIENT_ID")
    with pytest.raises(ValueError, match="OIDC_CLIENT_ID"):
        OidcConfig.from_env()  # configuration incomplète : l'application ne démarre pas


def test_fichier_des_comptes_verifie(tmp_path: Path) -> None:
    path = tmp_path / "users.toml"
    path.write_text('[accounts]\n"t:o" = "u_claire"\n', encoding="utf-8")
    assert load_accounts(path) == {"t:o": "u_claire"}
    path.write_text('[accounts]\n"t:o" = "../admin"\n', encoding="utf-8")
    with pytest.raises(ValueError, match="invalides"):
        load_accounts(path)
