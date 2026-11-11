"""Collecte locale des mails, sans compte réel et sans modification du code commun."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

from contracts.models import ConnectorInfo, Item
from packages.collect.core.connector import Connector, Health, Window
from packages.collect.synth.clock import PARIS
from packages.collect.synth.generator import GENERATOR_VERSION, Params, build_world
from packages.collect.synth.model import SimMail
from packages.collect.synth.people import Person
from packages.collect.synth.window import mail_item

DEFAULT_FIXTURES = Path(__file__).resolve().parents[4] / "contracts" / "fixtures" / "demo"


def select_mails(mails: Iterable[SimMail], manager: Person, now: datetime) -> list[SimMail]:
    """Mails reçus sur (now - 24 h, now], ou sans réponse et âgés d'au moins 7 jours.

    Les anciens mails doivent être adressés directement au manager et provenir
    d'un humain, comme pour le report du générateur. Une réponse future ne compte
    pas encore. Les durées sont calculées en UTC, même au changement d'heure.
    """
    if now.utcoffset() is None:
        raise ValueError("la référence de collecte doit avoir un fuseau horaire")
    reference = now.astimezone(UTC)
    recent_start = reference - timedelta(hours=24)
    old_end = reference - timedelta(days=7)
    selected: dict[str, SimMail] = {}
    for mail in mails:
        if mail.sent_at.utcoffset() is None:
            raise ValueError("la date du mail doit avoir un fuseau horaire")
        if mail.answered_at is not None and mail.answered_at.utcoffset() is None:
            raise ValueError("la date de réponse doit avoir un fuseau horaire")
        if mail.sender == manager or manager not in [*mail.to, *mail.cc]:
            continue
        sent = mail.sent_at.astimezone(UTC)
        if sent > reference:
            continue
        recent = sent > recent_start
        unanswered = mail.answered_at is None or mail.answered_at.astimezone(UTC) > reference
        old_pending = (
            sent <= old_end and unanswered and manager in mail.to and not mail.sender.automated
        )
        if recent or old_pending:
            selected.setdefault(mail.native_id, mail)
    return sorted(
        selected.values(), key=lambda mail: (mail.sent_at.astimezone(UTC), mail.native_id)
    )


def load_test_world(fixtures_dir: Path, user_id: str):
    """Reconstitue les SimMail bruts à partir des paramètres du jeu livré.

    Les exports Item ne contiennent ni html ni answered_at : on ne peut pas
    déduire ces champs de is_read, du snippet ou du corrigé truth.json.
    """
    manifest = json.loads((fixtures_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest["generator"] != {"name": "collect.synth", "version": GENERATOR_VERSION}:
        raise ValueError("version du générateur incompatible avec le jeu de test")
    raw = manifest["params"]
    if user_id not in raw["users"]:
        raise ValueError("manager absent du jeu de test")
    params = Params(
        seed=raw["seed"],
        days=raw["days"],
        start=date.fromisoformat(raw["start"]),
        users=tuple(raw["users"]),
        with_raw=raw["with_raw"],
    )
    return build_world(user_id, params)


class LocalMailConnector(Connector):
    name = "mail-local"
    version = "0.1.0"

    def __init__(self, fixtures_dir: Path | str = DEFAULT_FIXTURES) -> None:
        self.fixtures_dir = Path(fixtures_dir)

    def required_scopes(self) -> list[str]:
        return []

    def health(self) -> Health:
        if not (self.fixtures_dir / "manifest.json").is_file():
            return Health("down", "manifeste de données de test absent")
        return Health("ok")

    def collect(self, user_id: str, window: Window) -> Iterator[Item]:
        world = load_test_world(self.fixtures_dir, user_id)
        if window.end.astimezone(PARIS).date() not in world.export_days:
            raise ValueError("date absente du jeu de test")
        for mail in select_mails(world.mails, world.me, window.end)[-500:]:
            # mail_item appelle déjà core.text.make_snippet(mail.html).
            item = mail_item(world, mail, window.end)
            yield Item(
                **item.model_dump(exclude={"connector"}),
                connector=ConnectorInfo(name=self.name, version=self.version),
            )
