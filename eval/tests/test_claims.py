import pytest

from contracts.demo_data import DEFAULT_USER, load_items
from contracts.models import Item
from eval.claims import PARIS, check_brief, check_text, title_of
from eval.tests.test_metrics import _brief

DAY = "2026-10-13"
MAIL = "mail:dm1-msg-000052"  # « Mon dossier 2026-2405 – où en est-on ? », reçu le 12/10 à 17:17
MEETING = "calendar:dm1-evt-000076"  # « Point téléphonique – Loïc Ngoma », le 13/10 à 11:15
LATE_TASK = "task:dm1-tsk-000029"  # « Valider le devis 2026-3025 (8 300 €) », échue le 09/10
NESTED = "mail:dm1-msg-001991"  # non lu, titre qui contient lui-même « … »


@pytest.fixture(scope="module")
def items() -> dict[str, Item]:
    return {item.item_id: item for item in load_items(DAY)}


def _check(items: dict[str, Item], text: str, *cited: str) -> str | None:
    return check_text(text, cited, items, DEFAULT_USER)


def _first(items: dict[str, Item], predicate) -> Item:
    return next(i for i in items.values() if predicate(i))


def test_les_gabarits_du_brief_sont_soutenus(items: dict[str, Item]) -> None:
    title = title_of(items[NESTED])
    cases = [
        ("Mail « Mon dossier 2026-2405 – où en est-on ? », reçu le 12/10 à 17:17", MAIL),
        ("Mail : Mon dossier 2026-2405 – où en est-on ?", MAIL),
        ("Réunion « Point téléphonique – Loïc Ngoma » le 13/10 à 11:15", MEETING),
        ("Réunion à 11:15 : Point téléphonique – Loïc Ngoma", MEETING),
        ("Tâche « Valider le devis 2026-3025 (8 300 €) », échéance dépassée le 09/10", LATE_TASK),
        ("Tâche en retard : Valider le devis 2026-3025 (8 300 €)", LATE_TASK),
        (f"Mail non lu « {title} », reçu le 12/10 à 17:54", NESTED),
    ]
    for text, item_id in cases:
        assert _check(items, text, item_id) is None, text


@pytest.mark.parametrize(
    ("text", "cited", "reason"),
    [
        ("Mail « Mon dossier 2026-2405 »", MAIL, "titre cité absent"),
        ("Mail « Mon dossier 2026-2405 – où en est-on ? », reçu le 11/10", MAIL, "date"),
        ("Mail « Mon dossier 2026-2405 – où en est-on ? », reçu le 12/10 à 09:00", MAIL, "date"),
        ("Réunion « Point téléphonique – Loïc Ngoma » le 13/10 à 14:00", MEETING, "date"),
        ("Réunion à 10:00 : Point téléphonique – Loïc Ngoma", MEETING, "heure"),
        ("Réunion « Mon dossier 2026-2405 – où en est-on ? »", MAIL, "type"),
        ("Tâche « Valider le devis 2026-3025 (8 300 €) », échéance le 10/10", LATE_TASK, "date"),
        ("Mail « Mon dossier 2026-2405 – où en est-on ? » : 12 500 € en jeu", MAIL, "nombre"),
        ("Réunion « Point téléphonique – Loïc Ngoma » annulée", MEETING, "annulée"),
        ("Mail « X »", "mail:inconnu-0001", "absent de la journée"),
    ],
)
def test_un_fait_faux_rend_le_claim_non_soutenu(
    items: dict[str, Item], text: str, cited: str, reason: str
) -> None:
    problem = _check(items, text, cited)
    assert problem is not None and reason in problem


def test_non_lu_faux_sur_un_mail_lu(items: dict[str, Item]) -> None:
    read = _first(items, lambda i: i.source == "mail" and i.is_read is True)
    assert "non lu" in (_check(items, f"Mail non lu : {title_of(read)}", read.item_id) or "")


def test_en_retard_faux_sur_une_tache_a_venir(items: dict[str, Item]) -> None:
    task = _first(items, lambda i: i.source == "task" and not i.is_overdue(i.collected_at))
    problem = _check(items, f"Tâche en retard : {title_of(task)}", task.item_id)
    assert problem is not None and "en retard" in problem


def test_item_d_un_autre_utilisateur(items: dict[str, Item]) -> None:
    other = items[MAIL].model_copy(update={"owner_user_id": "u_autre"})
    problem = check_text("Mail : x", [MAIL], {MAIL: other}, DEFAULT_USER)
    assert problem is not None and "autre utilisateur" in problem


def test_heure_de_paris(items: dict[str, Item]) -> None:
    start = items[MEETING].start_at.astimezone(PARIS)
    assert start.strftime("%d/%m %H:%M") == "13/10 11:15"


def test_bilan_d_un_brief(items: dict[str, Item]) -> None:
    brief = _brief({"actions": [[MAIL], [MEETING]]})  # textes « texte » : rien d'affirmé
    stats = check_brief(brief, list(items.values()))
    assert (stats.claims, stats.texts, stats.texts_ok) == (0, 2, 2)
