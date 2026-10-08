"""Fabrique d'Item synthétiques : deux utilisateurs (A et B), pour les tests d'isolation."""

from __future__ import annotations

import json
from pathlib import Path

from contracts.models import Item

FIXTURE = Path(__file__).parents[3] / "contracts" / "fixtures" / "valid" / "item-mail.json"
DAY = "2026-09-30"
USER_A, USER_B = "u_test_a", "u_test_b"
XSS = "<script>alert('xss')</script>"


def make_item(user: str, n: int, **changes: object) -> Item:
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    data.update(
        item_id=f"mail:{user}-{n}",
        owner_user_id=user,
        acl={"visible_to": [user]},
        url=f"https://outlook.reseau-expertis.test/mail/id/{user}-{n}",
        **changes,
    )
    return Item.model_validate(data)


def write_day(root: Path, user: str, day: str, items: list[Item]) -> None:
    folder = root / user / day
    folder.mkdir(parents=True)
    lines = (item.model_dump_json() for item in items)
    (folder / "items.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (folder / "truth.json").write_text('{"secret": "vérité terrain"}', encoding="utf-8")
