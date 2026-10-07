"""Fixtures de l'API : données synthétiques dans un dossier temporaire, client de test."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
from alliance_api.audit import LogAuditSink
from alliance_api.auth import DEMO_HEADER
from alliance_api.main import create_app
from alliance_api.stores import DemoItemStore
from fastapi.testclient import TestClient

from apps.api.tests.factories import DAY, USER_A, USER_B, XSS, make_item, write_day


@pytest.fixture
def data_root(tmp_path: Path) -> Path:
    write_day(tmp_path, USER_A, DAY, [make_item(USER_A, 1, snippet=XSS), make_item(USER_A, 2)])
    write_day(tmp_path, USER_B, DAY, [make_item(USER_B, 1)])
    return tmp_path


@pytest.fixture
def audit() -> LogAuditSink:
    return LogAuditSink()


@pytest.fixture
def client(data_root: Path, audit: LogAuditSink) -> TestClient:
    app = create_app(item_store=DemoItemStore(data_root), audit=audit, auth_mode="demo")
    return TestClient(app)


@pytest.fixture
def as_user() -> Callable[[str], dict[str, str]]:
    return lambda user: {DEMO_HEADER: user}
