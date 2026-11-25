import json
from pathlib import Path

import pytest

from contracts.demo_data import DEMO_DIR
from eval import compare
from eval.report import evaluate, summary
from eval.tests.test_report import REFERENCE, VERSION, _generator, annotations  # noqa: F401


def _summary(**metrics: float | None) -> dict:
    base = {key: 0.5 for key in compare.METRICS}
    return {
        "schema": "eval-summary/1.0",
        "code": "abc1234",
        "set": "dev",
        "generators": ["rules"],
        "days": ["2026-10-13"],
        "errors": [],
        "metrics": base | metrics,
    }


def _run(tmp_path: Path, base: dict, head: dict, *flags: str) -> int:
    paths = []
    for name, content in (("base", base), ("head", head)):
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(content), encoding="utf-8")
        paths.append(str(path))
    return compare.main([*paths, *flags])


def test_perte_de_plus_de_3_points_est_une_regression() -> None:
    rows = compare.compare(_summary(recall=0.60), _summary(recall=0.56))
    assert [r.key for r in rows if r.regression] == ["recall"]


def test_perte_de_3_points_tout_juste_n_alerte_pas() -> None:
    rows = compare.compare(_summary(ndcg=0.60), _summary(ndcg=0.57))
    assert not any(r.regression for r in rows)


def test_metrique_ou_plus_bas_vaut_mieux() -> None:
    rows = compare.compare(_summary(days_with_top_fault=0.0), _summary(days_with_top_fault=0.17))
    assert [r.key for r in rows if r.regression] == ["days_with_top_fault"]
    rows = compare.compare(_summary(days_with_top_fault=0.5), _summary(days_with_top_fault=0.0))
    assert not any(r.regression for r in rows)


def test_metrique_non_mesuree_ignoree() -> None:
    rows = compare.compare(_summary(injections_neutralized=None), _summary())
    assert not any(r.regression for r in rows)


def test_changement_de_generateur_note_sans_masquer_la_regression() -> None:
    head = _summary(recall=0.1) | {"generators": ["rules"]}
    base = _summary() | {"generators": ["MiniLM (score de priorité appris) · modele"]}
    assert compare.invalid_reasons(base, head) == []
    assert "MiniLM" in (compare.generator_note(base, head) or "")
    rows = compare.compare(base, head)
    table = compare.render(base, head, rows, [])
    assert "Régression de plus de 3 points : Rappel@7" in table
    assert "le générateur change" in table


def test_strict_sort_en_erreur_seulement_s_il_y_a_regression(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    job_summary = tmp_path / "resume.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(job_summary))
    assert _run(tmp_path, _summary(), _summary(), "--strict") == 0
    code = _run(tmp_path, _summary(recall=0.9), _summary(recall=0.5), "--strict", "--github")
    out = capsys.readouterr().out
    assert code == 1
    assert "::warning title=Régression de l'évaluation::Rappel@7" in out
    assert "| Rappel@7 | 0,90 | 0,50 | -40 pt | ❌ |" in out
    assert "Régression de plus de 3 points : Rappel@7" in job_summary.read_text(encoding="utf-8")


def test_comparaison_non_valable_jamais_bloquante(tmp_path: Path) -> None:
    head = _summary(recall=0.0) | {"days": ["2026-10-13", "2026-10-14"]}
    assert _run(tmp_path, _summary(recall=0.9), head, "--strict") == 0


def test_resume_json_du_rapport(annotations: Path) -> None:  # noqa: F811
    evaluation = evaluate(annotations, DEMO_DIR, _generator(*REFERENCE), "dev")
    data = summary(evaluation, VERSION)
    assert set(compare.METRICS) <= set(data["metrics"])
    assert data["metrics"]["recall"] == 1.0
    assert data["metrics"]["days_with_top_fault"] == 0.0
    assert data["metrics"]["claims_supported"] is None  # aucun claim dans ce brief
    assert data["days"] == ["2026-10-13"] and data["generators"] == ["rules"]
