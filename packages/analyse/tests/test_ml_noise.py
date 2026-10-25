"""Tests du filtre de bruit certain (pipeline A) : seuls les envois en masse sont écartés."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from contracts.models import Item
from packages.analyse.scoring.ml.noise import REASON, NoiseConfig, main, noise_reason, report

AT = datetime(2026, 10, 7, 8, 0, tzinfo=UTC)


def _mail(item_id: str, title: str, text: str, sender: str = "a@verdier.example") -> Item:
    return Item(
        item_id=f"mail:{item_id}",
        source="mail",
        provider="microsoft",
        owner_user_id="u1",
        title=title,
        snippet=text[:500],
        content_text=text,
        created_at=AT,
        updated_at=AT,
        participants=[{"name": "X", "email": sender, "role": "from"}],
        url="https://outlook.example/m",
        acl={"visible_to": ["u1"]},
        collected_at=AT,
        connector={"name": "test", "version": "0.1.0"},
    )


def test_lettre_d_information_ecartee_meme_si_le_lien_est_en_fin_de_corps():
    text = "Au sommaire : cybersécurité.\n" + "…\n" * 300 + "Se DÉSABONNER de cette lettre"
    assert noise_reason(_mail("1", "Veille Conseil : n° 12", text)) == REASON


@pytest.mark.parametrize(
    ("title", "text", "sender"),
    [
        ("Rappel : validation des notes de frais", "Un document est disponible.", "noreply@rh.ex"),
        ("[INC-4] Imprimante (en attente)", "Notification automatique du ticket.", "it@it.ex"),
        ("Newsletter interne", "Peux-tu relire la newsletter avant vendredi ?", "a@verdier.ex"),
        ("Nouveau catalogue 2027", "Découvrez notre catalogue 2027.", "s.petit@locespace.ex"),
    ],
)
def test_automates_actionnables_collegues_et_catalogues_gardes(title, text, sender):
    assert noise_reason(_mail("1", title, text, sender)) is None


def test_marqueurs_configurables_et_repli_sur_l_extrait():
    item = _mail("1", "Offre", "Pour ne plus recevoir nos offres, cliquez ici")
    assert noise_reason(item) is None
    assert noise_reason(item, NoiseConfig(frozenset({"ne plus recevoir"}))) == REASON
    sans_corps = item.model_copy(update={"content_text": None, "snippet": "unsubscribe here"})
    assert noise_reason(sans_corps) == REASON


def test_rapport_compte_par_expediteur_avec_un_echantillon_stable():
    items = [_mail(str(i), f"Lettre {i}", "désabonner", "news@x.example") for i in range(12)]
    items.append(_mail("h", "Question", "Peux-tu valider ?"))
    text = report(items)
    assert text.splitlines()[:2] == ["écartés : 12 sur 13", "     12  news@x.example"]
    assert text == report(items)  # graine fixée
    assert text.count("\n  - ") == 10


def test_main_ne_rapporte_que_les_mails_recus(monkeypatch, capsys):
    from packages.analyse.scoring.ml import graph_convert

    recu, envoye = _mail("r", "Lettre", "désabonner"), _mail("e", "Lettre", "désabonner")
    dataset = graph_convert.GraphDataset(
        items=[recu, envoye],
        extras={
            "mail:r": graph_convert.MailExtras("to", False, None, None, False, False, ()),
            "mail:e": graph_convert.MailExtras("from", True, None, None, False, False, ()),
        },
    )
    monkeypatch.setattr(graph_convert, "load_graph_dataset", lambda _: dataset)
    assert main(["data"]) == 0
    assert "écartés : 1 sur 1" in capsys.readouterr().out
    assert main([]) == 2
    assert "noise" in capsys.readouterr().out
