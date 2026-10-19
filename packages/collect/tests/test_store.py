"""Écriture de la journée dans DATA_DIR, au format lu par l'API (contracts.demo_data)."""

from __future__ import annotations

from datetime import date

import pytest

from contracts.demo_data import available_days, load_items
from packages.collect.cli import main
from packages.collect.connectors.calendar_graph.connector import DEMO_USER
from packages.collect.core.connector import collection_window
from packages.collect.core.runner import run
from packages.collect.core.store import day_path, write_day

DAY = date(2026, 10, 21)


@pytest.fixture(scope="module")
def items() -> list:
    return run(DEMO_USER, collection_window(DAY)).items


def test_written_day_is_read_back_identically_by_the_api_loader(tmp_path, items) -> None:
    assert items, "le jeu de David doit donner des réunions ce jour-là"
    path = write_day(items, tmp_path, DEMO_USER, DAY)
    assert path == tmp_path / DEMO_USER / "2026-10-21" / "items.jsonl"
    assert available_days(DEMO_USER, tmp_path) == ["2026-10-21"]
    assert load_items("2026-10-21", DEMO_USER, tmp_path) == sorted(items, key=lambda i: i.item_id)


def test_same_collection_gives_the_same_file_and_rewriting_replaces(tmp_path, items) -> None:
    path = write_day(items, tmp_path, DEMO_USER, DAY)
    first = path.read_bytes()
    write_day(list(reversed(items)), tmp_path, DEMO_USER, DAY)
    assert path.read_bytes() == first  # trié, pas ajouté à la suite
    write_day(items[:1], tmp_path, DEMO_USER, DAY)
    assert len(load_items("2026-10-21", DEMO_USER, tmp_path)) == 1
    assert [p.name for p in path.parent.iterdir()] == ["items.jsonl"]  # pas de temporaire oublié


def test_empty_collection_still_marks_the_day(tmp_path) -> None:
    write_day([], tmp_path, DEMO_USER, DAY)
    assert available_days(DEMO_USER, tmp_path) == ["2026-10-21"]
    assert load_items("2026-10-21", DEMO_USER, tmp_path) == []


def test_refuses_another_users_item_and_unsafe_folder_names(tmp_path, items) -> None:
    stranger = items[0].model_copy(update={"owner_user_id": "u_autre"})
    with pytest.raises(ValueError, match="autre utilisateur"):
        write_day([*items, stranger], tmp_path, DEMO_USER, DAY)
    for bad in ["../u_demo_graph", "a/b", "", ".."]:
        with pytest.raises(ValueError):
            day_path(tmp_path, bad, DAY)
    assert not any(tmp_path.iterdir())  # rien n'a été écrit


def test_cli_writes_the_day_for_the_graph_demo_user(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    assert main(["--user", DEMO_USER, "--day", "2026-10-21", "--write"]) == 0
    assert capsys.readouterr().out == ""  # écrit dans le fichier, pas à l'écran
    written = load_items("2026-10-21", DEMO_USER, tmp_path)
    assert written and {i.source for i in written} == {"calendar"}
    assert all(i.owner_user_id == DEMO_USER and i.schema_version == "1.1" for i in written)


def test_cli_write_without_data_dir_is_an_error(monkeypatch) -> None:
    monkeypatch.delenv("DATA_DIR", raising=False)
    with pytest.raises(SystemExit) as exit_info:
        main(["--user", DEMO_USER, "--day", "2026-10-21", "--write"])
    assert exit_info.value.code == 2
