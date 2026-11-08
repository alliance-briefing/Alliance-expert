"""Accessibilité de l'interface (L3.3, #75) : contrastes WCAG AA, noms et rôles ARIA."""

import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from apps.api.tests.factories import DAY, USER_A

CSS = (Path(__file__).parents[1] / "src/alliance_api/static/app.css").read_text(encoding="utf-8")
AA = 4.5  # texte normal ; 3:1 suffirait pour le gros texte, on vise 4,5 partout


def _luminance(hex_color: str) -> float:
    """Luminance relative WCAG 2 d'une couleur #rrggbb."""
    channels = [int(hex_color[i : i + 2], 16) / 255 for i in (1, 3, 5)]
    r, g, b = (c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in channels)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _contrast(fg: str, bg: str) -> float:
    light, dark = sorted((_luminance(fg), _luminance(bg)), reverse=True)
    return (light + 0.05) / (dark + 0.05)


def _token(name: str) -> str:
    return re.search(rf"--{name}:\s*(#[0-9a-f]{{6}})\b", CSS).group(1)


def _gradient(rule: str) -> list[str]:
    """Les deux couleurs du dégradé de fond d'une règle CSS, ex. `.btn--primary:hover`."""
    body = re.search(rf"{re.escape(rule)}\s*\{{([^}}]*)\}}", CSS).group(1)
    return re.findall(r"#[0-9a-f]{6}\b", body.split("linear-gradient", 1)[1])[:2]


def test_contraste_ratio_connu() -> None:
    # Valeurs de référence WCAG : noir sur blanc = 21:1, une couleur sur elle-même = 1:1.
    assert round(_contrast("#000000", "#ffffff"), 1) == 21.0
    assert _contrast("#8b7cff", "#8b7cff") == 1.0


@pytest.mark.parametrize(
    ("text", "background"),
    [
        ("ink", "surface"),
        ("ink", "raised"),
        ("mist", "surface"),
        ("mist", "raised"),
        ("mist", "night"),
    ],
)
def test_texte_des_variables_au_moins_aa(text: str, background: str) -> None:
    assert _contrast(_token(text), _token(background)) >= AA


@pytest.mark.parametrize("rule", [".btn--primary", ".btn--primary:hover"])
def test_bouton_principal_au_moins_aa(rule: str) -> None:
    # Texte blanc sur tout le dégradé (« Se connecter », « Générer le brief »).
    stops = _gradient(rule)
    assert len(stops) == 2
    assert all(_contrast("#ffffff", stop) >= AA for stop in stops), stops


@pytest.fixture
def page(client: TestClient) -> str:
    client.post("/login", data={"user_id": USER_A})
    response = client.post("/generate", data={"date": DAY})
    assert response.status_code == 200, response.text
    return response.text


def _text(html: str) -> str:
    return " ".join(re.sub(r"<[^>]+>", " ", html).split())


def test_lien_d_evitement_mene_au_contenu(page: str) -> None:
    assert '<a class="skip" href="#contenu">' in page
    assert 'id="contenu"' in page


def test_boutons_details_ont_un_nom_unique(page: str) -> None:
    buttons = re.findall(
        r'<button[^>]*class="entry-open"[^>]*>(.*?)</button>', page, flags=re.DOTALL
    )
    names = [_text(b) for b in buttons]
    assert names and all(n.startswith("Détails : ") for n in names)
    assert len(set(names)) == len(names)


def test_boutons_qui_ouvrent_un_panneau_l_annoncent(page: str) -> None:
    openers = re.findall(r"<(?:button|a)\b[^>]*data-open=[^>]*>", page)
    assert openers
    for tag in openers:
        assert 'aria-haspopup="dialog"' in tag or tag.startswith("<a "), tag


def test_chaque_panneau_a_un_titre_existant(page: str) -> None:
    labels = re.findall(r'<dialog class="drawer"[^>]*aria-labelledby="([^"]+)"', page)
    assert labels
    assert all(f'id="{label}"' in page for label in labels)


def test_filtres_exposent_leur_etat(page: str) -> None:
    filters = re.findall(r"<button[^>]*data-filter=[^>]*>", page)
    assert filters and all(re.search(r'aria-pressed="(true|false)"', f) for f in filters)
    assert sum('aria-pressed="true"' in f for f in filters) == 1


def test_pas_d_aria_label_sur_un_span_sans_role(page: str) -> None:
    # Les lecteurs d'écran ignorent aria-label sur un <span> sans rôle : utiliser .sr-only.
    spans = re.findall(r"<span\b[^>]*aria-label=[^>]*>", page)
    assert all("role=" in span for span in spans), spans


def test_rang_lisible_par_les_lecteurs_d_ecran(page: str) -> None:
    assert re.search(
        r'<span class="entry-rank[^"]*"><span class="sr-only">Priorité </span>1</span>', page
    )
