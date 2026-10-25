"""Tests de la conversion Graph → Item (données d'entraînement du score appris).

Les messages Graph sont écrits ici à la main, au format du MCP `ms365` ; le jeu complet de
David n'est lu que par le dernier test, sauté si `ALLIANCE_GRAPH_DIR` n'est pas défini.
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path

import pytest

from packages.analyse.scoring.ml.graph_convert import (
    OWNER,
    load_graph_dataset,
    mail_to_item,
    main,
)

ME = "helene.marchand@verdier.example"
INBOX, SENT = "inbox-id", "sent-id"
COLLECTED = datetime(2026, 10, 7, 22, 0, tzinfo=UTC)


def _who(address: str, name: str = "Quelqu'un") -> dict:
    return {"emailAddress": {"name": name, "address": address}}


def _mail(mail_id: str = "m1", **extra: object) -> dict:
    data: dict[str, object] = {
        "id": mail_id,
        "createdDateTime": "2026-10-07T08:00:00Z",
        "lastModifiedDateTime": "2026-10-07T08:05:00Z",
        "receivedDateTime": "2026-10-07T08:00:00Z",
        "sentDateTime": "2026-10-07T07:59:58Z",
        "subject": "Validation du devis",
        "bodyPreview": "Bonjour Hélène, peux-tu valider le devis avant jeudi ?",
        "body": {
            "contentType": "html",
            "content": "<html><body><div>Bonjour Hélène,</div><div>Peux-tu valider le devis "
            "avant jeudi ?</div><div>Cordialement,</div><div>Bruno</div></body></html>",
        },
        "importance": "normal",
        "parentFolderId": INBOX,
        "conversationId": "conv-1",
        "isRead": False,
        "webLink": "https://outlook.office365.com/owa/?ItemID=m1",
        "inferenceClassification": "focused",
        "hasAttachments": False,
        "categories": [],
        "flag": {"flagStatus": "notFlagged"},
        "from": _who("bruno.lefevre@verdier.example", "Bruno LEFÈVRE"),
        "toRecipients": [_who(ME, "Hélène MARCHAND")],
        "ccRecipients": [],
        "bccRecipients": [],
    }
    data.update(extra)
    return data


def _event(event_id: str = "e1", **extra: object) -> dict:
    data: dict[str, object] = {
        "id": event_id,
        "createdDateTime": "2026-10-01T08:00:00Z",
        "lastModifiedDateTime": "2026-10-01T08:00:00Z",
        "subject": "Comité de direction",
        "bodyPreview": "",
        "importance": "normal",
        "sensitivity": "normal",
        "isAllDay": False,
        "isCancelled": False,
        "seriesMasterId": None,
        "webLink": "https://outlook.office365.com/owa/?itemid=e1",
        "originalStartTimeZone": "Romance Standard Time",
        "responseStatus": {"response": "accepted"},
        "start": {"dateTime": "2026-10-07T12:00:00.0000000", "timeZone": "UTC"},
        "end": {"dateTime": "2026-10-07T13:00:00.0000000", "timeZone": "UTC"},
        "location": {"displayName": "Salle Rhône"},
        "onlineMeeting": None,
        "organizer": _who("bruno.lefevre@verdier.example", "Bruno LEFÈVRE"),
        "attendees": [_who(ME, "Hélène MARCHAND")],
    }
    data.update(extra)
    return data


def _convert(mail: dict):
    return mail_to_item(mail, OWNER, ME, SENT, COLLECTED)


def test_mail_recu_devient_un_item_valide_avec_ses_extras():
    item, extras = _convert(_mail())
    assert item.item_id == "mail:m1"
    assert (item.source, item.provider, item.owner_user_id) == ("mail", "microsoft", OWNER)
    assert item.acl.visible_to == [OWNER]
    assert item.thread_id == "conv-1"
    assert item.is_read is False
    assert item.created_at == datetime(2026, 10, 7, 8, 0, tzinfo=UTC)
    # Salutation et signature retirées par le code du lot 1
    assert item.snippet == "Peux-tu valider le devis avant jeudi ?"
    assert [(p.role, p.email) for p in item.participants] == [
        ("from", "bruno.lefevre@verdier.example"),
        ("to", ME),
    ]
    assert extras.recipient_role == "to"
    assert not extras.is_sent


def test_mail_envoye_date_d_envoi_et_role_from():
    mail = _mail(
        parentFolderId=SENT,
        **{"from": _who(ME, "Hélène MARCHAND")},
        toRecipients=[_who("bruno.lefevre@verdier.example")],
    )
    item, extras = _convert(mail)
    assert item.created_at == datetime(2026, 10, 7, 7, 59, 58, tzinfo=UTC)
    assert extras.is_sent
    assert extras.recipient_role == "from"


def test_place_du_manager_cc_liste_de_diffusion_et_bcc_cache():
    cc = _convert(_mail(toRecipients=[_who("a@x.example")], ccRecipients=[_who(ME)]))[1]
    assert cc.recipient_role == "cc"
    item, liste = _convert(
        _mail(toRecipients=[_who("equipe@verdier.example")], bccRecipients=[_who("b@x.example")])
    )
    assert liste.recipient_role is None
    assert "b@x.example" not in {p.email for p in item.participants}


def test_sans_objet_html_mal_forme_et_corps_vide():
    item, _ = _convert(_mail(subject=None, body={"contentType": "html", "content": "<div>Ok <b>"}))
    assert item.title == ""
    assert item.snippet == "Ok"
    vide, _ = _convert(_mail(body={"contentType": "text", "content": ""}))
    assert vide.snippet.startswith("Bonjour Hélène, peux-tu valider")  # repli sur l'aperçu


def test_modification_anterieure_a_la_reception_ramenee_a_la_reception():
    item, _ = _convert(_mail(lastModifiedDateTime="2026-10-07T07:00:00Z"))
    assert item.updated_at == item.created_at


def test_extras_gardent_les_champs_aleatoires_pour_memoire():
    _, extras = _convert(
        _mail(flag={"flagStatus": "flagged"}, categories=["À traiter"], hasAttachments=True)
    )
    assert extras.flagged and extras.categories == ("À traiter",) and extras.has_attachments


def _write_dataset(root: Path, days: dict[str, dict[str, list[dict]]]) -> Path:
    manifest = {
        "persona": {"address": ME},
        "sentFolderId": SENT,
        "jours": {day: {"window_utc": [f"{day}T00:00:00Z", f"{day}T22:00:00Z"]} for day in days},
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    for day, kinds in days.items():
        for kind, values in kinds.items():
            (root / kind).mkdir(exist_ok=True)
            (root / kind / f"{day}.json").write_text(json.dumps({"value": values}), "utf-8")
    return root


def test_jeu_complet_dedoublonne_correle_et_compte_les_refus(tmp_path, capsys):
    invitation = _mail(
        "inv",
        subject="Comité de direction",
        meetingMessageType="meetingRequest",
        startDateTime={"dateTime": "2026-10-07T12:00:00.0000000", "timeZone": "UTC"},
    )
    refuse = _mail("bad", webLink="http://non-https.example")
    root = _write_dataset(
        tmp_path,
        {
            "2026-10-07": {"mail": [_mail(), invitation, refuse], "calendar": [_event()]},
            "2026-10-08": {"calendar": [_event()]},  # même événement vu deux jours
        },
    )
    dataset = load_graph_dataset(root)
    ids = [item.item_id for item in dataset.items]
    assert ids == ["mail:m1", "mail:inv", "calendar:e1"]
    by_id = {item.item_id: item for item in dataset.items}
    assert by_id["mail:inv"].correlation_id == by_id["calendar:e1"].correlation_id
    assert set(dataset.extras) == {"mail:m1", "mail:inv"}
    assert [key for key, _ in dataset.rejected] == ["mail:bad"]

    assert main([str(root)]) == 0
    assert "refusés : 1" in capsys.readouterr().out
    assert main([]) == 2


@pytest.mark.skipif(not os.environ.get("ALLIANCE_GRAPH_DIR"), reason="jeu de David absent")
def test_jeu_de_david_converti_sans_refus():
    dataset = load_graph_dataset(os.environ["ALLIANCE_GRAPH_DIR"])
    assert not dataset.rejected
    assert sum(item.source == "mail" for item in dataset.items) == 4463
