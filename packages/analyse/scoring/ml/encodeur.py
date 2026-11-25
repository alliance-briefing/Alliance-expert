"""M6, levier 3 : MiniLM réajusté avec la fiche, sur le PC GPU.

Au levier 2, le candidat retenu est N : MiniLM gelé sur le texte normalisé, la fiche, une
régression. Ici, l'encodeur n'est plus gelé : il apprend de bout en bout, avec la fiche, trois
têtes (action, urgence, alerte). Même entrée que N (texte normalisé et fiche), mêmes découpages
de la validation dure (`comprendre.hard_validation`, apprenant remplacé), même jeu réel ; N est
refait dans la même commande pour que les écarts soient appariés. Règle écrite le 23/11 avant
la mesure (plan, §11).

    uv sync --all-packages --group gpu                                  # PC GPU seulement
    uv run python -m packages.analyse.scoring.ml.encodeur --sonde       # temps d'un apprentissage
    uv run python -m packages.analyse.scoring.ml.encodeur --levier      # N contre E, 3 graines
    uv run python -m packages.analyse.scoring.ml.encodeur --exporter    # ONNX int8 (PC GPU)
    uv run python -m packages.analyse.scoring.ml.encodeur --latence <dossier>  # portable

torch et transformers ne sont importés que par les fonctions qui en ont besoin : la mesure de
latence (`--latence`) n'utilise que onnxruntime et tokenizers, comme la production.
"""

from __future__ import annotations

import json
import math
import random
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from packages.analyse.rag.embeddings import DEFAULT_MODEL, Embedder, FastEmbedder
from packages.analyse.scoring.ml import comprendre as C
from packages.analyse.scoring.ml.annoter import SERIES
from packages.analyse.scoring.ml.intents import VectorCache, cache_path

REPORT_PATH = Path(__file__).with_name("comprendre_levier_encodeur.json")
EXPORT_REPORT_PATH = Path(__file__).with_name("comprendre_encodeur_export.json")
MODEL_DIR = Path.home() / ".cache" / "alliance-briefing" / "encodeur"
SEEDS = (0, 1, 2)  # G4
TEXTE = "normalise"  # le texte retenu au levier 2
MAX_MB = 300  # contraintes de production (plan, §6)
MAX_MS = 30.0
# La validation dure attend une grille : l'encodeur n'en a pas (recette fixée d'avance).
RECIPE_LABEL = C.Config(0.0, False)


@dataclass(frozen=True)
class Recipe:
    """La recette du réajustement, fixée avant la mesure (valeurs courantes pour un petit
    encodeur ; aucune n'est choisie sur la validation)."""

    epochs: int = 3
    batch: int = 32
    max_tokens: int = 256
    lr_encoder: float = 2e-5
    lr_heads: float = 1e-3
    weight_decay: float = 0.01
    warmup: float = 0.1  # part des pas en montée linéaire du pas d'apprentissage
    dropout: float = 0.1
    clip: float = 1.0


RECIPE = Recipe()


# --- le réseau --------------------------------------------------------------------------------


def _net_class():
    import torch
    from torch import nn

    class HybridNet(nn.Module):
        """Moyenne des jetons de l'encodeur, fiche à côté, une tête linéaire par cible."""

        def __init__(self, encoder, n_fiche: int, dropout: float) -> None:
            super().__init__()
            self.encoder = encoder
            width = encoder.config.hidden_size + n_fiche
            self.dropout = nn.Dropout(dropout)
            self.heads = nn.ModuleDict(
                {t: nn.Linear(width, len(classes)) for t, classes in C.TARGETS.items()}
            )

        def forward(self, input_ids, attention_mask, fiche):
            # token_type_ids explicites (des zéros, la valeur par défaut de BERT) : sinon l'export
            # ONNX fige la longueur de la séquence vue au traçage.
            hidden = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=torch.zeros_like(input_ids),
            )
            mask = attention_mask.unsqueeze(-1).to(hidden.last_hidden_state.dtype)
            pooled = (hidden.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1.0)
            z = self.dropout(torch.cat([pooled, fiche.to(pooled.dtype)], dim=1))
            return {t: head(z) for t, head in self.heads.items()}

    return HybridNet


def hf_builder(name: str = DEFAULT_MODEL) -> Callable[[], tuple]:
    """Le tokeniseur et l'encodeur pré-entraîné de Hugging Face (téléchargés une fois)."""

    def build():
        from transformers import AutoModel, AutoTokenizer

        return AutoTokenizer.from_pretrained(name), AutoModel.from_pretrained(name)

    return build


def device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def _seed_all(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _batches(rows: Sequence[int], size: int, order: Sequence[int] | None = None):
    rows = [rows[i] for i in order] if order is not None else list(rows)
    for begin in range(0, len(rows), size):
        yield rows[begin : begin + size]


def _tensors(tokenizer, tokens: dict, F: np.ndarray, rows: Sequence[int], where: str):
    import torch

    batch = tokenizer.pad(
        {"input_ids": [tokens["input_ids"][r] for r in rows]}, return_tensors="pt"
    )
    fiche = torch.tensor(F[list(rows)], dtype=torch.float32)
    return batch["input_ids"].to(where), batch["attention_mask"].to(where), fiche.to(where)


def train(
    build: Callable[[], tuple],
    tokens: dict,
    tokenizer,
    F: np.ndarray,
    labels: dict[str, np.ndarray],
    rows: Sequence[int],
    seed: int,
    recipe: Recipe = RECIPE,
    where: str = "cpu",
    log: Callable[[str], None] = lambda _: None,
):
    """Réajuste l'encodeur et apprend les têtes sur `rows` : somme des entropies croisées des
    trois cibles, AdamW (pas plus grand pour les têtes), montée puis descente linéaire du pas,
    précision mixte sur carte graphique. Graine fixée : mêmes données, mêmes poids (à l'arrondi
    près des calculs de la carte graphique)."""
    import torch

    _seed_all(seed)
    _, encoder = build()
    net = _net_class()(encoder, F.shape[1], recipe.dropout).to(where)
    heads = list(net.heads.parameters())
    optimizer = torch.optim.AdamW(
        [
            {"params": net.encoder.parameters(), "lr": recipe.lr_encoder},
            {"params": heads, "lr": recipe.lr_heads},
        ],
        weight_decay=recipe.weight_decay,
    )
    steps = recipe.epochs * math.ceil(len(rows) / recipe.batch)
    warm = max(1, int(recipe.warmup * steps))
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: min((s + 1) / warm, max(0.0, (steps - s) / max(1, steps - warm)))
    )
    use_amp = where == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    y = {t: torch.tensor(v, dtype=torch.long) for t, v in labels.items()}
    generator = torch.Generator().manual_seed(seed)
    net.train()
    done, start = 0, time.monotonic()
    for epoch in range(recipe.epochs):
        order = torch.randperm(len(rows), generator=generator).tolist()
        for batch in _batches(rows, recipe.batch, order):
            ids, mask, fiche = _tensors(tokenizer, tokens, F, batch, where)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                out = net(ids, mask, fiche)
                loss = sum(
                    torch.nn.functional.cross_entropy(out[t].float(), y[t][batch].to(where))
                    for t in C.TARGETS
                )
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(net.parameters(), recipe.clip)
            scaler.step(optimizer)
            scaler.update()
            schedule.step()
            done += 1
            if done % 100 == 0 or done == steps:
                elapsed = time.monotonic() - start
                log(
                    f"    époque {epoch + 1}/{recipe.epochs}, pas {done}/{steps} "
                    f"({elapsed:.0f} s, reste ≈ {elapsed / done * (steps - done):.0f} s)"
                )
    return net


def predict_proba(
    net, tokens: dict, tokenizer, F: np.ndarray, rows: Sequence[int], where: str = "cpu"
) -> dict[str, np.ndarray]:
    """Les probabilités de chaque cible pour `rows`, par lots, sans gradient."""
    import torch

    net.eval()
    out: dict[str, list] = {t: [] for t in C.TARGETS}
    with torch.no_grad():
        for batch in _batches(rows, 64):
            ids, mask, fiche = _tensors(tokenizer, tokens, F, batch, where)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=where == "cuda"):
                logits = net(ids, mask, fiche)
            for t in C.TARGETS:
                out[t].append(torch.softmax(logits[t].float(), dim=1).cpu().numpy())
    return {t: np.concatenate(v) if v else np.zeros((0, len(C.TARGETS[t]))) for t, v in out.items()}


class FoldLearner:
    """L'apprenant de la validation dure (remplace `out_of_fold`) : un réajustement par jeu
    d'entraînement, gardé pour les trois cibles du découpage 1 (un seul réseau à trois têtes),
    puis oublié (seules les probabilités restent)."""

    def __init__(self, build, tokens, tokenizer, F, labels, seed, recipe, where, log):
        self.args = (build, tokens, tokenizer, F, labels)
        self.seed, self.recipe, self.where, self.log = seed, recipe, where, log
        self.cache: dict[tuple[int, ...], dict[int, dict[str, np.ndarray]]] = {}
        self.trainings = 0

    def __call__(self, target, X, y, folds, log=None, grid=None) -> dict[str, C.OutOfFold]:
        build, tokens, tokenizer, F, labels = self.args
        for f, fold in enumerate(folds):
            key = tuple(fold.train)
            if key in self.cache:
                continue
            self.log(f"  réajustement {self.trainings + 1} (pli {f + 1}, {len(key)} cartes)")
            net = train(
                build, tokens, tokenizer, F, labels, fold.train, self.seed, self.recipe,
                self.where, self.log,
            )  # fmt: skip
            rows = sorted({n for v in fold.measured.values() for n in v})
            proba = predict_proba(net, tokens, tokenizer, F, rows, self.where)
            self.cache[key] = {n: {t: proba[t][i] for t in C.TARGETS} for i, n in enumerate(rows)}
            self.trainings += 1
            del net
            _free(self.where)
        out = {}
        for name in folds[0].measured:
            idx = [np.asarray(f.measured[name], dtype=int) for f in folds]
            rows = [self.cache[tuple(f.train)] for f in folds]
            k = len(C.TARGETS[target])
            proba = [
                np.array([r[n][target] for n in i]).reshape(len(i), k)
                for i, r in zip(idx, rows, strict=True)
            ]
            out[name] = C.OutOfFold(
                np.concatenate(idx),
                np.concatenate([np.full(len(i), f) for f, i in enumerate(idx)]),
                [np.concatenate(proba)],
            )
        return out


def _free(where: str) -> None:
    if where == "cuda":
        import torch

        torch.cuda.empty_cache()


# --- le levier --------------------------------------------------------------------------------


def run_encoder(
    examples: list[C.Example],
    F: np.ndarray,
    dev: tuple[list[dict], list[str], dict],
    seed: int,
    build: Callable[[], tuple],
    recipe: Recipe = RECIPE,
    where: str = "cpu",
    n_folds: int = C.HARD_FOLDS,
    log: Callable[[str], None] = lambda _: None,
    save: Path | None = None,
) -> tuple[dict, dict, float]:
    """Une graine de E : validation dure avec l'encodeur réajusté (17 apprentissages : 4 plis
    de métiers, 4 de gabarits, 8 d'écrivain), puis réseau appris sur tout l'entraînement et
    mesuré sur le jeu réel, enregistré dans `save` pour l'export."""
    tokenizer, _ = build()
    texts = [C.encoder_text(e.card, TEXTE) for e in examples]
    tokens = tokenizer(texts, truncation=True, max_length=recipe.max_tokens)
    labels = {t: C._codes(examples, t) for t in C.TARGETS}
    learner = FoldLearner(build, tokens, tokenizer, F, labels, seed, recipe, where, log)
    keep: dict = {}
    dummy = np.zeros((len(examples), 1))
    hard = C.hard_validation(
        examples, dummy, n_folds, log=log, keep=keep, grid=(RECIPE_LABEL,), fit_folds=learner
    )
    log("  réajustement sur tout l'entraînement")
    net = train(build, tokens, tokenizer, F, labels, range(len(examples)), seed, recipe, where, log)
    dev_cards = dev[0]
    dev_tokens = tokenizer(
        [C.encoder_text(c, TEXTE) for c in dev_cards], truncation=True, max_length=recipe.max_tokens
    )
    F_dev = np.array([C.card_features(c) for c in dev_cards]).reshape(len(dev_cards), -1)
    proba = predict_proba(net, dev_tokens, tokenizer, F_dev, range(len(dev_cards)), where)
    threshold = hard["metiers"]["cibles"]["alerte"]["seuil_rappel_095"]["seuil"]
    dev_pred = {
        t: (p[:, 1] >= threshold).astype(int) if t == "alerte" else p.argmax(axis=1)
        for t, p in proba.items()
    }
    if save is not None:
        save.mkdir(parents=True, exist_ok=True)
        import torch

        torch.save(net.state_dict(), save / "reseau.pt")
        tokenizer.save_pretrained(save)
        net.encoder.config.save_pretrained(save)
        meta = {"graine": seed, "seuil_alerte": threshold, "recette": asdict(recipe)}
        (save / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    del net
    _free(where)
    entry, preds, recall = C.variant_entry(examples, hard, keep, dev, dev_pred, "hybride", TEXTE)
    entry["apprentissages"] = learner.trainings + 1
    return entry, preds, recall


def levier_encodeur(
    corpora: dict[str, Path] = C.CORPORA,
    embedder: Embedder | None = None,
    cache: VectorCache | None = None,
    series: Path = SERIES,
    out_report: Path = REPORT_PATH,
    n_folds: int = C.HARD_FOLDS,
    seeds: Sequence[int] = SEEDS,
    build: Callable[[], tuple] | None = None,
    recipe: Recipe = RECIPE,
    where: str | None = None,
    model_dir: Path | None = MODEL_DIR,
    log: Callable[[str], None] = lambda _: None,
) -> dict:
    """Le levier 3 : N (gelé, refait ici) contre E (réajusté), 3 graines, écarts appariés.
    Règle du 23/11 : E remplace N si E bat N (`comprendre._beats`) avec chacune de ses graines,
    sous le veto des alertes ; une graine perdante arrête le calcul (même décision). Les
    contraintes de taille et de latence sont vérifiées ensuite (`--exporter`, `--latence`)."""
    build = build or hf_builder()
    where = where or device()
    embedder = embedder or FastEmbedder(DEFAULT_MODEL)
    cache = cache or VectorCache(cache_path(embedder))
    log(f"cartes des personas « train »… (apprentissage sur {where})")
    examples, _ = C.training_examples(corpora)
    F = np.array([C.card_features(e.card) for e in examples])
    dev = C._dev_set(series)
    log("N, MiniLM gelé (référence du levier)")
    texts = [C.encoder_text(e.card, TEXTE) for e in examples]
    X = C.embed_with_progress(cache, embedder, texts, log=log)
    Xd = np.hstack(
        [
            C.design(dev[0], embedder, "aucune", cache, TEXTE),
            C.design(dev[0], embedder, "seule"),
        ]
    )
    report: dict = {"recette": asdict(recipe), "texte": TEXTE, "machine": where, "variantes": {}}
    entry, n_preds, n_recall = C.run_variant(
        examples, np.hstack([X, F]), Xd, dev, "hybride", n_folds, log, C.TEXT_GRID, TEXTE
    )
    report["variantes"]["N"] = entry
    preds, recall, gaps = {"N": n_preds}, {"N": n_recall}, {}
    measures = ("metiers", *C.DECISIVE)
    wins = True
    for seed in seeds:
        name = f"E{seed}"
        log(f"{name}, MiniLM réajusté, graine {seed}")
        start = time.monotonic()
        save = model_dir / name if model_dir is not None else None
        entry, preds[name], recall[name] = run_encoder(
            examples, F, dev, seed, build, recipe, where, n_folds, log, save
        )
        report["variantes"][name] = entry | {"minutes": round((time.monotonic() - start) / 60, 1)}
        gaps[f"{name}-N"] = {
            m: C._gap(preds[name][m][0], preds[name][m][1], preds["N"][m][1]) for m in measures
        }
        if recall[name] < C.ALERT_RECALL or not C._beats([gaps[f"{name}-N"]]):
            wins = False
            break
    report["ecarts_apparies_action"] = gaps
    report["decision"] = {
        "graines_calculees": len(gaps),
        "arret_anticipe": len(gaps) < len(seeds),
        "E_bat_N": wins,
        "veto_alertes": any(r < C.ALERT_RECALL for n, r in recall.items() if n != "N"),
        "retenue": "E sous réserve des contraintes" if wins else "N",
    }
    out_report.write_text(
        json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n"
    )
    return report


def sonde(
    corpora: dict[str, Path] = C.CORPORA,
    build: Callable[[], tuple] | None = None,
    recipe: Recipe = RECIPE,
    where: str | None = None,
    log: Callable[[str], None] = print,
    n_folds: int = C.HARD_FOLDS,
) -> dict:
    """Le temps d'un réajustement d'une époque sur le premier pli de métiers, et l'estimation
    du calcul complet (17 réajustements de `recipe.epochs` époques par graine). Aucun chiffre
    de mesure n'est rendu : la sonde ne sert qu'au temps."""
    build = build or hf_builder()
    where = where or device()
    examples, _ = C.training_examples(corpora)
    tokenizer, _ = build()
    tokens = tokenizer(
        [C.encoder_text(e.card, TEXTE) for e in examples],
        truncation=True,
        max_length=recipe.max_tokens,
    )
    F = np.array([C.card_features(e.card) for e in examples])
    labels = {t: C._codes(examples, t) for t in C.TARGETS}
    held = C.metier_folds((e.metier for e in examples), n_folds)[0]
    rows = [n for n, e in enumerate(examples) if e.metier not in held]
    start = time.monotonic()
    one = Recipe(**(asdict(recipe) | {"epochs": 1}))
    train(build, tokens, tokenizer, F, labels, rows, 0, one, where, log)
    epoch = time.monotonic() - start
    # 17 réajustements par graine ; les 8 de l'écrivain croisé ne portent que sur des mails
    # d'un écrivain (environ un tiers des cartes) : compté ici comme 8 × 1/3.
    per_seed = epoch * recipe.epochs * (4 + 4 + 8 / 3 + 4 / 3)
    out = {
        "machine": where,
        "cartes": len(rows),
        "secondes_par_epoque": round(epoch, 1),
        "minutes_par_graine": round(per_seed / 60),
        "heures_pour_3_graines": round(3 * per_seed / 3600, 1),
        "tronquees": round(
            float(np.mean([len(x) >= recipe.max_tokens for x in tokens["input_ids"]])), 4
        ),
    }
    log(json.dumps(out, ensure_ascii=False))
    return out


# --- export et contraintes de production ------------------------------------------------------


def exporter(model_dir: Path = MODEL_DIR / "E0", out_dir: Path | None = None, build=None) -> Path:
    """Le réseau enregistré exporté en ONNX (entrées : jetons, masque, fiche ; sorties : les
    trois logits), puis quantifié en int8 (onnxruntime) pour tenir ≤ 300 Mo sur processeur."""
    import torch
    from onnxruntime.quantization import QuantType, quantize_dynamic
    from transformers import AutoConfig, AutoModel, AutoTokenizer

    out_dir = out_dir or model_dir / "onnx"
    out_dir.mkdir(parents=True, exist_ok=True)
    if build is None:
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        encoder = AutoModel.from_config(AutoConfig.from_pretrained(model_dir))
    else:
        tokenizer, encoder = build()
    net = _net_class()(encoder, len(C.FEATURE_NAMES), 0.0)
    net.load_state_dict(torch.load(model_dir / "reseau.pt", map_location="cpu"))
    net.eval()
    sample = tokenizer(["mail : essai\nMerci de valider le devis."], return_tensors="pt")
    fiche = torch.zeros((1, len(C.FEATURE_NAMES)))

    class Flat(torch.nn.Module):  # sorties dans un ordre fixe, pour ONNX
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, input_ids, attention_mask, fiche):
            out = self.inner(input_ids, attention_mask, fiche)
            return tuple(out[t] for t in C.TARGETS)

    full = out_dir / "reseau.onnx"
    torch.onnx.export(
        Flat(net),
        (sample["input_ids"], sample["attention_mask"], fiche),
        str(full),
        input_names=["input_ids", "attention_mask", "fiche"],
        output_names=list(C.TARGETS),
        dynamic_axes={
            "input_ids": {0: "lot", 1: "jetons"},
            "attention_mask": {0: "lot", 1: "jetons"},
            "fiche": {0: "lot"},
            **{t: {0: "lot"} for t in C.TARGETS},
        },
        opset_version=17,
        dynamo=False,
    )
    quantize_dynamic(str(full), str(out_dir / "reseau-int8.onnx"), weight_type=QuantType.QInt8)
    tokenizer.save_pretrained(out_dir)
    meta = json.loads((model_dir / "meta.json").read_text(encoding="utf-8"))
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    return out_dir


class OnnxModel:
    """Le réseau exporté, servi par onnxruntime et tokenizers seuls (comme en production)."""

    def __init__(self, folder: Path, name: str = "reseau-int8.onnx", max_tokens: int = 256):
        import onnxruntime
        from tokenizers import Tokenizer

        self.session = onnxruntime.InferenceSession(
            str(folder / name), providers=["CPUExecutionProvider"]
        )
        meta = json.loads((folder / "meta.json").read_text("utf-8"))
        self.tokenizer = Tokenizer.from_file(str(folder / "tokenizer.json"))
        self.tokenizer.enable_truncation(meta.get("recette", {}).get("max_tokens", max_tokens))
        self.threshold = meta["seuil_alerte"]

    def logits(self, cards: Sequence[dict]) -> dict[str, np.ndarray]:
        out: dict[str, list] = {t: [] for t in C.TARGETS}
        for c in cards:  # un élément à la fois : la latence mesurée est celle d'un élément
            enc = self.tokenizer.encode(C.encoder_text(c, TEXTE))
            feeds = {
                "input_ids": np.array([enc.ids], dtype=np.int64),
                "attention_mask": np.array([enc.attention_mask], dtype=np.int64),
                "fiche": C.card_features(c).astype(np.float32)[None, :],
            }
            for t, v in zip(C.TARGETS, self.session.run(None, feeds), strict=True):
                out[t].append(v[0])
        return {t: np.array(v) for t, v in out.items()}

    def predict(self, cards: Sequence[dict]) -> dict[str, np.ndarray]:
        logits = self.logits(cards)
        out = {}
        for t, z in logits.items():
            if t == "alerte":
                e = np.exp(z - z.max(axis=1, keepdims=True))
                out[t] = ((e / e.sum(axis=1, keepdims=True))[:, 1] >= self.threshold).astype(int)
            else:
                out[t] = z.argmax(axis=1)
        return out


def latence(
    folder: Path, series: Path = SERIES, out_report: Path | None = EXPORT_REPORT_PATH
) -> dict:
    """Taille, latence par élément (processeur, un élément à la fois) et accord entre le
    réseau float32 et sa version int8 sur le jeu réel : la quantification ne doit pas coûter
    de score (écart de F1 de l'action rendu avec son intervalle apparié)."""
    cards, _, truth = C._dev_set(series)
    full, small = OnnxModel(folder, "reseau.onnx"), OnnxModel(folder)
    small.predict(cards[:5])  # échauffement
    start = time.perf_counter()
    pred_small = small.predict(cards)
    ms = (time.perf_counter() - start) / len(cards) * 1000
    pred_full = full.predict(cards)
    y = truth["action"]
    low, high = C.paired_diff_ci(C._f1_action, pred_small["action"], pred_full["action"], y)
    size = (folder / "reseau-int8.onnx").stat().st_size / 1e6
    report = {
        "taille_mo": round(size, 1),
        "latence_ms_par_element": round(ms, 1),
        "f1_action_int8": round(C._f1_action(pred_small["action"], y), 4),
        "f1_action_float32": round(C._f1_action(pred_full["action"], y), 4),
        "ecart_int8_moins_float32": [round(low, 4), round(high, 4)],
        "accord_action": round(float(np.mean(pred_small["action"] == pred_full["action"])), 4),
        "contraintes_tenues": size <= MAX_MB and ms <= MAX_MS,
    }
    if out_report is not None:
        out_report.write_text(
            json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n"
        )
    return report


def main(argv: list[str]) -> int:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    if argv == ["--sonde"]:
        sonde()
        return 0
    if argv == ["--levier"]:
        report = levier_encodeur(log=lambda line: print(line, flush=True))
        print(json.dumps(report["decision"], ensure_ascii=False))
        print(f"→ {REPORT_PATH.name}")
        return 0
    if argv == ["--exporter"]:
        print(f"→ {exporter()}")
        return 0
    if len(argv) == 2 and argv[0] == "--latence":
        print(json.dumps(latence(Path(argv[1])), ensure_ascii=False))
        print(f"→ {EXPORT_REPORT_PATH.name}")
        return 0
    print("usage : encodeur --sonde | --levier | --exporter | --latence <dossier onnx>")
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
