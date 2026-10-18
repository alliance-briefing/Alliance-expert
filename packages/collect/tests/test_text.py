"""Nettoyage HTML → texte : cas réels de messagerie, y compris HTML abîmé."""

from __future__ import annotations

import pytest

from packages.collect.core.text import SNIPPET_MAX_CHARS, html_to_text, make_snippet, truncate

CASES = {
    "entités et espaces insécables": (
        "<p>Montant&nbsp;: 4&nbsp;850,00&nbsp;&euro;</p>",
        "Montant : 4 850,00 €",
    ),
    "paragraphes non fermés": ("<p>un<p>deux<p>trois", "un deux trois"),
    "fermetures orphelines": ("<div>texte</span></font></td></div>", "texte"),
    "commentaire conditionnel Outlook": (
        "<!--[if mso]><p>MSO</p><![endif]--><p>visible</p>",
        "visible",
    ),
    "style et script retirés": ("<style>p{}</style><script>x=1</script><p>ok</p>", "ok"),
    "balise style non fermée": ("<p>avant</p><style>p{color:red}", "avant"),
    "salutation retirée": (
        "<p>Bonjour Camille,</p><p>Le rapport est prêt.</p>",
        "Le rapport est prêt.",
    ),
    "signature retirée": (
        "<p>Le rapport est prêt.</p><p>Cordialement,<br>Loïc Ngoma<br>Expert</p>",
        "Le rapport est prêt.",
    ),
    "citation retirée": (
        "<p>D'accord pour jeudi.</p><div>Le 12/10/2026 à 09:14, Julie a écrit :</div><blockquote>Et jeudi ?</blockquote>",
        "D'accord pour jeudi.",
    ),
    "citation style >": ("<p>Oui.</p><p>&gt; tu confirmes ?</p>", "Oui."),
    "message d'origine Outlook": (
        "<p>Vu.</p><p>-----Message d'origine-----</p><p>De : X</p>",
        "Vu.",
    ),
    "transfert conservé": (
        (
            "<p>Pour info.</p><div>---------- Message transféré ---------<br>De : Client<br>Objet : Réclamation<br>"
            "Je conteste le montant.</div>"
        ),
        "Pour info. [Message transféré] Objet : Réclamation Je conteste le montant.",
    ),
    "tableau": ("<table><tr><td>Montant</td><td>4 850 €</td></tr></table>", "Montant 4 850 €"),
    "texte qui ressemble à du code": (
        "<p>&lt;script&gt;alert(1)&lt;/script&gt;</p>",
        "<script>alert(1)</script>",
    ),
    "accents et typographie": (
        "<p>« Réunion à chaud » – œuvres, ça, Noël</p>",
        "« Réunion à chaud » – œuvres, ça, Noël",
    ),
    "emoji": ("<p>Machine réparée ☕</p>", "Machine réparée ☕"),
    "vide": ("", ""),
    "image seule": ('<img src="cid:image001.png">', ""),
    "retours Windows": ("<pre>ligne 1\r\nligne 2</pre>", "ligne 1 ligne 2"),
    "réponse mobile": ("<p>OK</p><p>Envoyé de mon iPhone</p>", "OK"),
}


@pytest.mark.parametrize(("raw", "expected"), CASES.values(), ids=CASES.keys())
def test_make_snippet(raw: str, expected: str) -> None:
    assert make_snippet(raw) == expected


def test_html_to_text_keeps_lines() -> None:
    assert html_to_text("<p>a</p><p>b</p>").split("\n") == ["a", "", "b"]


def test_truncate_on_word_boundary() -> None:
    text = "mot " * 200
    cut = truncate(text)
    assert len(cut) <= SNIPPET_MAX_CHARS
    assert cut.endswith("…") and not cut.endswith(" …")
    assert truncate("court") == "court"


def test_never_crashes_on_garbage() -> None:
    for garbage in ("<<<>>>", "<p", "</", "<!--", "<![CDATA[", "&#xZZ;", "<a href='x>lien</a>"):
        assert isinstance(make_snippet(garbage), str)
