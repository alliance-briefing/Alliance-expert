import json
from pathlib import Path

import pytest

from contracts.demo_data import DEMO_DIR
from contracts.models import Brief
from eval.report import evaluate, render
from eval.tests.test_metrics import _annotation, _brief

INJECTION = "mail:dm1-msg-000207"  # injection de prompt du 2026-10-13 (truth.json)
REFERENCE = [["mail:dm1-msg-000230"], ["mail:dm1-msg-000065"], ["task:dm1-tsk-000012"]]
VERSION = ("abc1234", "2026-11-25")


@pytest.fixture
def annotations(tmp_path: Path) -> Path:
    folder = tmp_path / "dev"
    folder.mkdir()
    annotation = _annotation(REFERENCE, exclude=["mail:dm1-msg-001993"])
    (folder / "2026-10-13.alice.json").write_text(annotation.model_dump_json(by_alias=True))
    return folder


def _generator(*entries: list[str]):
    def generate(items, user, day, *, now):
        return _brief({"actions": [list(e) for e in entries]})

    return generate


def test_rapport_identique_a_chaque_execution(annotations: Path) -> None:
    generate = _generator(*REFERENCE)
    first = render(evaluate(annotations, DEMO_DIR, generate, "dev", seed=0), VERSION)
    second = render(evaluate(annotations, DEMO_DIR, generate, "dev", seed=0), VERSION)
    assert first == second
    assert "| Rappel@7 | 1,00 |" in first
    assert "Injections de prompt neutralisées (non citées) | 1/1" in first


def test_injection_citee_et_faute_signalees(annotations: Path) -> None:
    generate = _generator(["mail:dm1-msg-001993"], [INJECTION], *REFERENCE)
    report = render(evaluate(annotations, DEMO_DIR, generate, "dev"), VERSION)
    assert "| 0/1 |" in report
    assert "1 journée(s) sur 1" in report
    assert "`mail:dm1-msg-001993`" in report


def test_jeu_cache_sans_identifiant_ni_texte_de_reference(annotations: Path) -> None:
    generate = _generator(["mail:dm1-msg-001993"], REFERENCE[0])
    report = render(evaluate(annotations, DEMO_DIR, generate, "hidden"), VERSION)
    assert "dm1-" not in report
    assert "important" not in report  # le « why » des annotations


def test_generateur_en_echec_note_sans_arreter(annotations: Path) -> None:
    def broken(items, user, day, *, now):
        raise RuntimeError("modèle absent")

    evaluation = evaluate(annotations, DEMO_DIR, broken, "dev")
    assert evaluation.days[0].error.startswith("RuntimeError")
    assert "le brief n'a pas été produit" in render(evaluation, VERSION)


def _with_claim(text: str, item_id: str = "mail:dm1-msg-000052") -> Brief:
    data = _brief({"actions": [[item_id]]}).model_dump()
    data["sections"][0]["entries"][0]["claims"] = [{"text": text, "source_item_ids": [item_id]}]
    return Brief.model_validate(data)


def test_claim_faux_compte_et_montre_sur_le_jeu_de_dev(annotations: Path) -> None:
    brief = _with_claim("Mail « Mon dossier 2026-2405 – où en est-on ? », reçu le 01/10")
    evaluation = evaluate(annotations, DEMO_DIR, lambda *a, **k: brief, "dev")
    report = render(evaluation, VERSION)
    assert "| Claims soutenus par leur source | 0,00 (0/1) |" in report
    assert "| Briefs avec au moins un claim non soutenu | 1,00 (1/1) |" in report
    assert "reçu le 01/10" in report and "introuvable dans les sources" in report
    hidden = render(evaluate(annotations, DEMO_DIR, lambda *a, **k: brief, "hidden"), VERSION)
    assert "Mon dossier" not in hidden and "entrée 1 (claim)" in hidden


def test_claim_vrai(annotations: Path) -> None:
    brief = _with_claim("Mail « Mon dossier 2026-2405 – où en est-on ? », reçu le 12/10 à 17:17")
    report = render(evaluate(annotations, DEMO_DIR, lambda *a, **k: brief, "dev"), VERSION)
    assert "| Claims soutenus par leur source | 1,00 (1/1) |" in report
    assert "Aucun : chaque claim" in report


def test_annotations_du_depot_lisibles() -> None:
    folder = Path(__file__).parents[1] / "annotations" / "dev"
    for path in folder.glob("*.json"):
        assert json.loads(path.read_text(encoding="utf-8"))["date"] in path.name
