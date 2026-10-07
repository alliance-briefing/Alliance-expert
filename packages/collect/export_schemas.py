"""Exporte les JSON Schema des contrats dans contracts/schemas/.

Usage : uv run python -m contracts.export_schemas
"""

from __future__ import annotations

import json
from pathlib import Path

from contracts.models import AuditEvent, Item

SCHEMAS_DIR = Path(__file__).parent / "schemas"


def export(out_dir: Path = SCHEMAS_DIR) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for name, model in (("item", Item), ("audit_event", AuditEvent)):
        path = out_dir / f"{name}.schema.json"
        schema = model.model_json_schema()
        path.write_text(json.dumps(schema, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        written.append(path)
    return written


if __name__ == "__main__":
    for written_path in export():
        print(f"écrit : {written_path}")
