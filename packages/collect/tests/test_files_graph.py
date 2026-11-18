"""Connecteur fichiers OneDrive (guide L1.5) : fenêtre de 48 h, extraits, accès retiré."""

from __future__ import annotations

import io
import json
import logging
import zipfile
from datetime import date
from urllib.parse import urlparse

import pytest

from contracts.models import Item
from packages.collect.connectors.files_graph import GraphFilesConnector
from packages.collect.connectors.files_graph.connector import (
    DEMO_USER,
    DOCX_MAX_SIZE,
    MAX_FILES,
    TEXT_BYTES,
    DriveSource,
    Gone,
    file_source,
    graph_source,
)
from packages.collect.connectors.files_graph.mapping import human_size, path
from packages.collect.core.connector import collection_window
from packages.collect.core.runner import run
from packages.collect.tests.test_graph_http import TOKEN

WINDOW = collection_window(date(2026, 10, 21))  # 21/10 à 8h30, Paris (06:30 UTC)


def _collect(connector: GraphFilesConnector | None = None, user: str = DEMO_USER) -> list[Item]:
    return list((connector or GraphFilesConnector(source=file_source())).collect(user, WINDOW))


def _file(file_id: str, name: str, modified: str = "2026-10-20T10:00:00Z", size: int = 100):
    return {
        "id": file_id,
        "name": name,
        "size": size,
        "file": {},
        "createdDateTime": modified,
        "lastModifiedDateTime": modified,
        "webUrl": f"https://onedrive.live.com/?id={file_id}",
        "parentReference": {"path": "/drive/root:/Dossier"},
    }


def _source(files: list[dict], contents: dict[str, bytes]) -> tuple[DriveSource, list]:
    reads: list[tuple[str, int | None]] = []

    def read(user: str, item: dict, limit: int | None) -> bytes:
        reads.append((item["id"], limit))
        if item["id"] not in contents:
            raise Gone
        return contents[item["id"]][:limit] if limit else contents[item["id"]]

    return DriveSource(lambda user: files, read), reads


def test_recent_files_only_newest_first_without_folders_deleted_or_old() -> None:
    items = _collect()
    assert [i.title for i in items] == [
        "Budget prévisionnel 2027.csv",
        "Compte rendu cadrage 2026-10-20.txt",
        "Plaquette Transports Marchetti.pdf",
    ]  # pas : dossier, supprimé, trop ancien (15/10), accès retiré (f_contrat)
    assert all(i.source == "file" and i.provider == "microsoft" for i in items)


def test_snippet_gives_path_type_size_modifier_then_content() -> None:
    item = next(i for i in _collect() if i.item_id == "file:f_cr_noviel")
    assert item.snippet.startswith(
        "Cabinet Noviel/Compte rendu cadrage 2026-10-20.txt · TXT · 212 o · "
        "modifié par Hélène MARCHAND — Réunion de cadrage Cabinet Noviel"
    )
    assert "\n" not in item.snippet and len(item.snippet) <= 500
    assert [(p.email, p.role) for p in item.participants] == [
        ("helene.marchand@example.com", "modifier")
    ]
    assert str(item.url) == "https://onedrive.live.com/?id=f_cr_noviel"
    assert item.created_at.isoformat() == "2026-10-20T17:30:00+02:00"


def test_50_mb_file_is_never_downloaded_metadata_only() -> None:
    big = _file("big", "Plaquette.pdf", size=50 * 1024 * 1024)
    huge_docx = _file("huge", "Rapport.docx", size=DOCX_MAX_SIZE + 1)
    source, reads = _source([big, huge_docx], {"big": b"%PDF", "huge": b"PK"})
    items = _collect(GraphFilesConnector(source=source))
    assert sorted(reads) == [("big", 1), ("huge", 1)]  # un octet : l'accès est vérifié, c'est tout
    assert all(" — " not in i.snippet for i in items)  # métadonnées seules
    assert "50 Mo" in next(i for i in items if i.title == "Plaquette.pdf").snippet


def test_text_files_read_only_the_first_bytes() -> None:
    long_text = ("ligne de budget ; " * 2000).encode()
    source, reads = _source([_file("t", "Notes.md")], {"t": long_text})
    (item,) = _collect(GraphFilesConnector(source=source))
    assert reads == [("t", TEXT_BYTES)] and len(item.snippet) <= 500


def test_docx_text_is_extracted() -> None:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(
            "word/document.xml",
            "<w:document><w:body><w:p><w:r><w:t>Note de cadrage</w:t></w:r></w:p>"
            "<w:p><w:r><w:t>Livrable attendu le 3 novembre.</w:t></w:r></w:p></w:body></w:document>",
        )
    source, _ = _source([_file("d", "Note.docx")], {"d": buffer.getvalue()})
    (item,) = _collect(GraphFilesConnector(source=source))
    assert item.snippet.endswith("— Note de cadrage Livrable attendu le 3 novembre.")


def test_unreadable_docx_keeps_metadata() -> None:
    source, _ = _source([_file("d", "Abîmé.docx")], {"d": b"pas un zip"})
    (item,) = _collect(GraphFilesConnector(source=source))
    assert " — " not in item.snippet


def test_capped_at_100_most_recent(caplog: pytest.LogCaptureFixture) -> None:
    files = [
        _file(f"f{n:03d}", f"{n}.pdf", f"2026-10-20T{n // 60:02d}:{n % 60:02d}:00Z")
        for n in range(130)
    ]
    source, _ = _source(files, {f["id"]: b"x" for f in files})
    with caplog.at_level(logging.WARNING):
        items = _collect(GraphFilesConnector(source=source))
    assert len(items) == MAX_FILES and items[0].title == "129.pdf"
    assert "30 fichier(s) au-delà du plafond" in caplog.text


def test_window_is_48h_before_collection_time_both_ends() -> None:
    files = [
        _file("in", "dedans.pdf", "2026-10-21T06:30:00Z"),  # pile à l'heure de collecte
        _file("edge", "limite.pdf", "2026-10-19T06:30:00Z"),  # pile 48 h avant : exclu
        _file("after", "apres.pdf", "2026-10-21T06:31:00Z"),  # après la collecte : exclu
    ]
    source, _ = _source(files, {f["id"]: b"x" for f in files})
    assert [i.title for i in _collect(GraphFilesConnector(source=source))] == ["dedans.pdf"]


def test_read_only_scope_and_test_data_isolated() -> None:
    assert GraphFilesConnector(source=file_source()).required_scopes() == ["Files.Read"]
    assert _collect(user="u_autre") == []


def test_helpers() -> None:
    assert human_size(126) == "126 o" and human_size(2048) == "2 Ko"
    assert human_size(52428800) == "50 Mo" and human_size(1572864) == "1.5 Mo"
    assert path({"name": "a.txt", "parentReference": {"path": "/drive/root:"}}) == "a.txt"
    assert path({"name": "a.txt"}) == "a.txt"


class FakeDrive:
    """Graph simulé : delta, métadonnées d'un fichier, téléchargement pré-signé."""

    def __init__(self, files: list[dict], status: dict[str, int], content: bytes = b"") -> None:
        self.files, self.status, self.content = files, status, content
        self.requests: list[tuple[str, dict]] = []

    def __call__(self, url: str, headers: dict) -> tuple[int, dict, bytes]:
        self.requests.append((url, headers))
        parsed = urlparse(url)
        if parsed.path.endswith("/root/delta"):
            return 200, {}, json.dumps({"value": self.files}).encode()
        if parsed.netloc == "download.example":
            return 206, {}, self.content[:TEXT_BYTES]
        file_id = parsed.path.rsplit("/", 1)[-1]
        code = self.status.get(file_id, 200)
        body = {
            "id": file_id,
            "@microsoft.graph.downloadUrl": f"https://download.example/{file_id}",
        }
        return code, {}, json.dumps(body if code == 200 else {"error": {}}).encode()


def _graph(fake: FakeDrive) -> GraphFilesConnector:
    return GraphFilesConnector(source=graph_source(token_for=lambda user: TOKEN, send=fake))


def test_graph_reads_content_by_presigned_link_with_range_and_no_token() -> None:
    fake = FakeDrive([_file("abc", "Notes.txt")], {}, b"Compte rendu fictif")
    (item,) = _collect(_graph(fake))
    assert item.snippet.endswith("— Compte rendu fictif")
    url, headers = fake.requests[-1]
    assert url == "https://download.example/abc"
    assert headers == {"Range": f"bytes=0-{TEXT_BYTES - 1}"}  # surtout pas le jeton
    assert all(h.get("Authorization") == f"Bearer {TOKEN}" for _, h in fake.requests[:-1])


def test_graph_access_removed_or_deleted_gives_no_item_and_no_crash() -> None:
    files = [_file("revoked", "Contrat.pdf"), _file("gone", "Vieux.txt"), _file("ok", "Ok.pdf")]
    fake = FakeDrive(files, {"revoked": 403, "gone": 404})
    assert [i.title for i in _collect(_graph(fake))] == ["Ok.pdf"]


def test_graph_server_error_is_a_warning_for_the_runner(monkeypatch) -> None:
    monkeypatch.setattr("time.sleep", lambda s: None)
    fake = FakeDrive([_file("x", "x.pdf")], {"x": 400})
    result = run(DEMO_USER, WINDOW, [_graph(fake)])
    assert result.items == [] and len(result.warnings) == 1


def test_health() -> None:
    assert GraphFilesConnector(source=file_source()).health().ok

    def broken(user: str) -> list[dict]:
        raise OSError

    assert not GraphFilesConnector(source=DriveSource(broken, lambda *a: b"")).health().ok
