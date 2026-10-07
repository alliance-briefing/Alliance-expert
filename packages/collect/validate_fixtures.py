"""Valide un dossier de fixtures contre le JSON Schema du contrat A (critère d'acceptation L1.0).

Trois contrôles, pour chaque journée :
  1. chaque ligne de items.jsonl respecte contracts/schemas/item.schema.json (JSON Schema, indépendant de Python) ;
  2. chaque ligne passe aussi le modèle pydantic (règles métier : URL https, propriétaire dans l'acl…) ;
  3. l'empreinte sha256 de chaque fichier correspond au manifest.json (rien n'a été modifié à la main).

Usage : uv run python -m contracts.validate_fixtures [dossier]   (défaut : contracts/fixtures/demo)
Code de sortie 0 si tout est valide, 1 sinon : utilisable tel quel en CI.
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import ValidationError

from contracts.models import Item

SCHEMA_PATH = Path(__file__).parent / "schemas" / "item.schema.json"
DEFAULT_DIR = Path(__file__).parent / "fixtures" / "demo"
MAX_REPORTED = 20


@dataclass
class Report:
    files: int = 0
    items: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def add(self, message: str) -> None:
        self.errors.append(message)


def validate_dir(root: Path, schema_path: Path = SCHEMA_PATH) -> Report:
    from jsonschema import Draft202012Validator  # dépendance de développement uniquement

    root = Path(root)
    report = Report()
    validator = Draft202012Validator(json.loads(schema_path.read_text("utf-8")))

    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        report.add(f"{root}: manifest.json absent")
        return report
    checksums: dict[str, str] = json.loads(manifest_path.read_text("utf-8"))["sha256"]

    on_disk = {
        p.relative_to(root).as_posix()
        for p in root.rglob("*")
        if p.is_file() and p.name != "manifest.json"
    }
    for missing in sorted(set(checksums) - on_disk):
        report.add(f"{missing}: listé dans le manifeste mais absent")
    for extra in sorted(on_disk - set(checksums)):
        report.add(f"{extra}: présent mais absent du manifeste")

    for rel, expected in sorted(checksums.items()):
        path = root / rel
        if not path.exists():
            continue
        data = path.read_bytes()
        report.files += 1
        if hashlib.sha256(data).hexdigest() != expected:
            report.add(
                f"{rel}: empreinte sha256 différente du manifeste (fichier modifié à la main ?)"
            )
        if not rel.endswith("items.jsonl"):
            continue
        for number, line in enumerate(data.decode("utf-8").splitlines(), start=1):
            report.items += 1
            where = f"{rel}:{number}"
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                report.add(f"{where}: JSON illisible ({exc.msg})")
                continue
            for error in validator.iter_errors(obj):
                path_str = "/".join(str(p) for p in error.absolute_path) or "(racine)"
                report.add(f"{where}: schéma, champ {path_str} : {error.message}")
            try:
                Item.model_validate(obj)
            except ValidationError as exc:
                first = exc.errors()[0]
                report.add(f"{where}: modèle, {'/'.join(map(str, first['loc']))} : {first['msg']}")
    return report


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    root = Path(args[0]) if args else DEFAULT_DIR
    report = validate_dir(root)
    if report.ok:
        print(f"OK : {report.items} Items valides dans {report.files} fichiers ({root})")
        return 0
    print(f"ÉCHEC : {len(report.errors)} erreur(s) dans {root}")
    for message in report.errors[:MAX_REPORTED]:
        print(f"  - {message}")
    if len(report.errors) > MAX_REPORTED:
        print(f"  … et {len(report.errors) - MAX_REPORTED} autre(s)")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
