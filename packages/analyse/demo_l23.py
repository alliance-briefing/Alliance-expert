"""Lance generate_brief avec le FakeLLMClient et affiche le résultat.

uv run python -m packages.analyse.demo_l23
"""

from contracts.demo_data import load_items
from packages.analyse import generate_brief
from packages.analyse.llm.client import FakeLLMClient

items = load_items("2026-10-07")
brief = generate_brief(items, "u_demo_001", "2026-10-07", client=FakeLLMClient())

for section in brief.sections:
    print(f"\n=== {section.kind.upper()} ===")
    for entry in section.entries:
        print(f"\n  [{entry.rank}] score={entry.priority_score}")
        print(f"  texte       : {entry.text}")
        print(f"  action      : {entry.recommended_action}")
        print(f"  raisons     : {entry.priority_reasons}")

if brief.warnings:
    print("\n--- warnings ---")
    for w in brief.warnings:
        print(f"  {w}")
