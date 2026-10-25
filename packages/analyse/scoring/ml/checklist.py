"""Tests de comportement des intentions apprises (méthode CheckList, Ribeiro et al., 2020).

Trois sortes de tests sur des mails fictifs (`scoring/annotations/intent_checklist.json`) :
- base : un cas simple dont la réponse est connue ;
- invariance : une transformation qui ne doit pas changer la décision (signature, fautes,
  vouvoiement, citation…) ;
- direction : une transformation qui doit faire monter ou baisser la probabilité (« pas
  besoin de répondre » fait baisser `reponse`).
Le corps passe par le même nettoyage que les vrais mails (citations et signature retirées).
Ces mails ne servent jamais à l'entraînement : ils mesurent où et pourquoi le modèle échoue.

`uv run --group embeddings python -m packages.analyse.scoring.ml.checklist`
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path

from packages.analyse.scoring.ml.intents import (
    IntentModel,
    VectorCache,
    cache_path,
    embedder_for,
    intent_text,
)
from packages.analyse.scoring.ml.sample import ANNOTATIONS
from packages.collect.core.text import normalize_whitespace, strip_quotes_and_signature

SUITE_PATH = ANNOTATIONS / "intent_checklist.json"
FEATURES = ("action", "reponse")
Predict = Callable[[list[str]], dict[str, list[float]]]


def mail_text(title: str, body: str) -> str:
    """Comme un vrai mail converti : corps sans citations ni signature."""
    clean = strip_quotes_and_signature(normalize_whitespace(body))
    return intent_text(title, clean or body)


def run(suite: dict, predict: Predict) -> list[dict]:
    """Un résultat par vérification : {type, cat, feature, ok, détail}."""
    texts: list[str] = []

    def slot(title: str, body: str) -> int:
        texts.append(mail_text(title, body))
        return len(texts) - 1

    base = [(t, slot(t["title"], t["body"])) for t in suite["base"]]
    inv = [(t, slot(*t["a"]), slot(*t["b"])) for t in suite["invariance"]]
    dirs = [(t, slot(*t["a"]), slot(*t["b"])) for t in suite["direction"]]
    p = predict(texts)
    results = []
    for test, i in base:
        for f in FEATURES:
            if f in test:
                got = p[f][i] >= 0.5
                results.append(
                    {
                        "type": "base",
                        "cat": test["cat"],
                        "feature": f,
                        "ok": got == test[f],
                        "détail": f"{test['title']} — attendu {test[f]}, p={p[f][i]:.2f}",
                    }
                )
    for test, a, b in inv:
        for f in test["features"]:
            same = (p[f][a] >= 0.5) == (p[f][b] >= 0.5)
            results.append(
                {
                    "type": "invariance",
                    "cat": test["cat"],
                    "feature": f,
                    "ok": same,
                    "détail": f"{test['a'][1][:40]} : {p[f][a]:.2f} → {p[f][b]:.2f}",
                }
            )
    for test, a, b in dirs:
        f, delta = test["feature"], p[test["feature"]][b] - p[test["feature"]][a]
        ok = delta > 0 if test["dir"] == "up" else delta < 0
        results.append(
            {
                "type": "direction",
                "cat": test["cat"],
                "feature": f,
                "ok": ok,
                "détail": f"{test['a'][1][:40]} : {p[f][a]:.2f} → {p[f][b]:.2f}",
            }
        )
    return results


def report(results: list[dict]) -> str:
    lines = []
    for kind in ("base", "invariance", "direction"):
        rows = [r for r in results if r["type"] == kind]
        for f in FEATURES:
            sub = [r for r in rows if r["feature"] == f]
            if sub:
                ok = sum(r["ok"] for r in sub)
                lines.append(f"{kind:<11} {f:<8} {ok:>3}/{len(sub):<3} réussis")
    by_cat: dict[str, list[bool]] = defaultdict(list)
    for r in results:
        by_cat[f"{r['type']} · {r['cat']}"].append(r["ok"])
    lines.append("\nPar catégorie (échecs / total) :")
    for cat, oks in sorted(by_cat.items()):
        if not all(oks):
            lines.append(f"  {oks.count(False)}/{len(oks)}  {cat}")
    lines.append("\nÉchecs :")
    lines += [f"  [{r['type']}/{r['feature']}] {r['détail']}" for r in results if not r["ok"]]
    return "\n".join(lines)


def model_predict(model: IntentModel | None = None) -> Predict:
    model = model or IntentModel.load()
    embedder = embedder_for(model.embedder_name)
    cache = VectorCache(cache_path(embedder))

    def predict(texts: list[str]) -> dict[str, list[float]]:
        probs = model.predict(cache.embed(embedder, texts), texts)
        return {f: [float(x) for x in probs[f]] for f in FEATURES}

    return predict


def main(argv: list[str]) -> int:
    suite = json.loads(Path(SUITE_PATH).read_text(encoding="utf-8"))
    print(report(run(suite, model_predict())))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
