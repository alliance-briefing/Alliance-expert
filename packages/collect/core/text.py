"""Nettoyage des corps de messages : HTML → texte brut → extrait (snippet).

Utilisé par le générateur de données factices ET réutilisable tel quel par les
vrais connecteurs (L1.3) : même code, mêmes tests.

Principes :
- tolérant au HTML mal formé (balises non fermées, commentaires conditionnels Outlook) ;
- ne laisse passer ni script, ni style, ni commentaire ;
- réduit les citations en cascade et les signatures ;
- conserve le contenu d'un message transféré (il porte souvent l'information utile) ;
- produit un extrait ≤ 500 caractères, coupé sur une frontière de mot.
"""

from __future__ import annotations

import html
import re
from html.parser import HTMLParser

SNIPPET_MAX_CHARS = 500
ELLIPSIS = "…"

_BLOCK_TAGS = {
    "address",
    "article",
    "aside",
    "blockquote",
    "br",
    "dd",
    "div",
    "dl",
    "dt",
    "footer",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "header",
    "hr",
    "li",
    "ol",
    "p",
    "pre",
    "section",
    "table",
    "tbody",
    "td",
    "th",
    "thead",
    "tr",
    "ul",
}
_SKIP_TAGS = {"script", "style", "head", "title", "noscript", "template", "xml"}

_QUOTE_START = re.compile(
    r"^\s*("
    r"Le .{3,120} a écrit\s?:"
    r"|On .{3,120} wrote\s?:"
    r"|-{2,}\s?(Message d'origine|Original Message)\s?-{2,}"
    r"|De\s?:\s?.+"
    r"|From\s?:\s?.+"
    r"|_{8,}"
    r")\s*$",
    re.IGNORECASE,
)
_FORWARD_MARKER = re.compile(
    r"^\s*-{3,}\s?(Message transféré|Forwarded message)\s?-{3,}\s*$", re.IGNORECASE
)
_FORWARD_HEADER = re.compile(r"^\s*(De|From|À|A|To|Cc|Envoyé|Sent|Date)\s?:", re.IGNORECASE)
_SIGNATURE_START = re.compile(
    r"^\s*(--\s*|Cordialement[,.!]?|Bien cordialement[,.!]?|Bien à (vous|toi)[,.!]?"
    r"|Salutations[,.!]?|Best regards[,.!]?|Regards[,.!]?|Envoyé de mon (iPhone|Android|mobile))\s*$",
    re.IGNORECASE,
)
_GREETING = re.compile(
    r"^\s*(Bonjour|Bonsoir|Hello|Salut|Madame|Monsieur|Chère?|Dear)\b[^\n]{0,40}[,!]?\s*$",
    re.IGNORECASE,
)
_SPACES = re.compile(r"[ \t\f\v   ]+")
_MANY_NEWLINES = re.compile(r"\n{3,}")


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if tag in _SKIP_TAGS:
            self._skip_depth += 1
        elif tag in _BLOCK_TAGS:
            self.parts.append("\n")

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() in _BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in _SKIP_TAGS:
            self._skip_depth = max(0, self._skip_depth - 1)
        elif tag in _BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0:
            self.parts.append(data)

    def handle_comment(self, data: str) -> None:  # commentaires ignorés (y compris <!--[if mso]>)
        return

    def unknown_decl(self, data: str) -> None:  # <![endif]> et consorts
        return


def html_to_text(raw_html: str) -> str:
    """Convertit du HTML (même mal formé) en texte brut lisible, lignes préservées."""
    parser = _TextExtractor()
    try:
        parser.feed(raw_html)
        parser.close()
        text = "".join(parser.parts)
    except Exception:  # noqa: BLE001 — dernier recours : suppression grossière des balises
        text = html.unescape(re.sub(r"<[^>]*>", " ", raw_html))
    return normalize_whitespace(text)


def normalize_whitespace(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [_SPACES.sub(" ", line).strip() for line in text.split("\n")]
    return _MANY_NEWLINES.sub("\n\n", "\n".join(lines)).strip()


def strip_quotes_and_signature(text: str) -> str:
    """Retire les citations en cascade et la signature, garde le contenu transféré."""
    kept: list[str] = []
    in_forward = False
    for line in text.split("\n"):
        if _FORWARD_MARKER.match(line):
            in_forward = True
            kept.append("[Message transféré]")
            continue
        if in_forward:
            if _FORWARD_HEADER.match(line):
                continue  # en-têtes du message transféré : bruit
            if _SIGNATURE_START.match(line):
                break
            kept.append(line)
            continue
        if (
            line.lstrip().startswith(">")
            or _QUOTE_START.match(line)
            or _SIGNATURE_START.match(line)
        ):
            break
        kept.append(line)

    while kept and (not kept[0].strip() or _GREETING.match(kept[0])):
        kept.pop(0)
    return "\n".join(kept).strip()


def truncate(text: str, max_chars: int = SNIPPET_MAX_CHARS) -> str:
    """Coupe sur une frontière de mot, ajoute « … » si nécessaire, longueur ≤ max_chars."""
    if len(text) <= max_chars:
        return text
    cut = text[: max_chars - len(ELLIPSIS)]
    space = cut.rfind(" ")
    if space > max_chars * 0.6:
        cut = cut[:space]
    return cut.rstrip(" ,;:.-") + ELLIPSIS


def make_snippet(raw_html: str, max_chars: int = SNIPPET_MAX_CHARS) -> str:
    """Pipeline complet : HTML → texte → sans citations ni signature → une ligne → tronqué."""
    text = strip_quotes_and_signature(html_to_text(raw_html))
    one_line = _SPACES.sub(" ", text.replace("\n", " ")).strip()
    return truncate(one_line, max_chars)
