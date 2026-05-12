"""
Evaluación 1 — Comparación de modelos KGE: métricas de entrenamiento + datos no vistos.

Sección A (rápida, sin cargar modelos):
  Lee training_comparison.json generado por phase2_kge_train.py y muestra la tabla
  de métricas del test set interno de PyKEEN (split 80/10/10 dentro del 95 %).

Sección B (requiere modelos en disco + corpus LP):
  Evalúa cada modelo cargándolo en memoria sobre LP_EVAL_CORPUS
  (data/corpus/link_prediction_eval.json), que contiene tripletas con entidades
  conocidas que NO formaron parte del split de entrenamiento de PyKEEN.
  Calcula Hit@k y MRR de forma independiente al entrenamiento.

Con ambas secciones se puede detectar sobreajuste: un modelo con alto MRR interno
(sección A) pero bajo MRR externo (sección B) está sobreajustado.

Uso:
  python src/evaluacion/eval_kge_comparacion.py               # solo sección A
  python src/evaluacion/eval_kge_comparacion.py --live        # A + B
  python src/evaluacion/eval_kge_comparacion.py --live --n-samples 200
  python src/evaluacion/eval_kge_comparacion.py --sort hit@10
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import src.config as cfg


# ---------------------------------------------------------------------------
# Sección A — métricas del entrenamiento (training_comparison.json)
# ---------------------------------------------------------------------------

def load_training_comparison() -> list[dict]:
    path = cfg.MODEL_COMPARISON_DIR / "training_comparison.json"
    if not path.exists():
        raise FileNotFoundError(
            f"No encontrado: {path}\n"
            "Ejecuta primero:\n"
            "  python src/phase2_kge_train.py --all-models"
        )
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_best() -> dict | None:
    if not cfg.BEST_MODEL_FILE.exists():
        return None
    with open(cfg.BEST_MODEL_FILE, encoding="utf-8") as f:
        return json.load(f)


_METRIC_COLS = ["mrr", "hit@1", "hit@3", "hit@10"]
_LABELS      = {"mrr": "MRR", "hit@1": "Hit@1", "hit@3": "Hit@3", "hit@10": "Hit@10"}


def _table_header(title: str, sort_by: str) -> tuple[str, str, str]:
    col_w  = 10
    name_w = 12
    sep    = "─" * (name_w + 2 + col_w * len(_METRIC_COLS) + 3 * len(_METRIC_COLS))
    header = f"  {'Modelo':<{name_w}}" + "".join(
        f"  {_LABELS[m]:>{col_w}}" for m in _METRIC_COLS
    )
    return sep, header, f"  (ordenado por {_LABELS[sort_by]} ↓)"


def print_training_table(rows: list[dict], sort_by: str, best_model: str | None) -> None:
    sep, header, sort_marker = _table_header("SECCIÓN A — Métricas test interno PyKEEN", sort_by)
    rows_sorted = sorted(rows, key=lambda r: r.get(sort_by, 0.0), reverse=True)

    print()
    print("=" * len(sep))
    print("  SECCIÓN A — Métricas del test interno PyKEEN (split 80/10/10)")
    print(sort_marker)
    print("=" * len(sep))
    print(header)
    print("  " + "─" * (len(sep) - 2))

    for rank, row in enumerate(rows_sorted, 1):
        name  = row["model"]
        star  = "★ " if name == best_model else "  "
        line  = f"{star}{name:<12}" + "".join(
            f"  {row.get(m, 0.0):>10.4f}" for m in _METRIC_COLS
        )
        if rank == 1:
            line = f"\033[1m{line}\033[0m"
        print(f"  {line}")

    print("=" * len(sep))
    if best_model:
        best_row = next((r for r in rows if r["model"] == best_model), None)
        if best_row:
            print(f"\n  ★  Mejor modelo seleccionado para inferencia: {best_model}")
            print(f"     MRR={best_row.get('mrr', 0):.4f}  "
                  f"Hit@1={best_row.get('hit@1', 0):.4f}  "
                  f"Hit@10={best_row.get('hit@10', 0):.4f}")
    print()


def print_gaps(rows: list[dict]) -> None:
    if len(rows) < 2:
        return
    best_vals = {m: max(r.get(m, 0.0) for r in rows) for m in _METRIC_COLS}
    print("  Diferencia vs. mejor modelo (Sección A):")
    for row in sorted(rows, key=lambda r: r.get("mrr", 0.0), reverse=True):
        print(f"  {row['model']:<12}" + "".join(
            f"  Δ{_LABELS[m]}={row.get(m, 0.0) - best_vals[m]:+.4f}"
            for m in _METRIC_COLS
        ))
    print()


# ---------------------------------------------------------------------------
# Sección B — evaluación en vivo sobre LP_EVAL_CORPUS (datos no vistos)
# ---------------------------------------------------------------------------

def evaluate_model_live(
    model_name: str,
    corpus: list[dict],
    top_k_values: list[int],
) -> dict:
    from src.phase3_link_prediction import load_model_by_name, predict_tails

    print(f"\n  [{model_name}] Cargando modelo ...")
    model, factory = load_model_by_name(model_name)

    max_k  = max(top_k_values)
    hits   = {k: 0 for k in top_k_values}
    rr_sum = 0.0
    per_rel: dict[str, dict] = defaultdict(lambda: {
        "n": 0,
        **{f"hit@{k}": 0 for k in top_k_values},
        "rr_sum": 0.0,
    })

    n = len(corpus)
    print(f"  [{model_name}] Evaluando {n} entradas ...")

    for i, entry in enumerate(corpus):
        if (i + 1) % 200 == 0:
            print(f"    {i+1}/{n} ...")

        preds        = predict_tails(model, factory, entry["subject"], entry["predicate"], top_k=max_k)
        pred_entities = [e for e, _ in preds]
        true_obj     = entry["object_true"]
        pred         = entry["predicate"]

        rank = (pred_entities.index(true_obj) + 1) if true_obj in pred_entities else None
        rr   = (1.0 / rank) if rank else 0.0

        for k in top_k_values:
            hit = 1 if (rank is not None and rank <= k) else 0
            hits[k] += hit
            per_rel[pred][f"hit@{k}"] += hit

        rr_sum                += rr
        per_rel[pred]["rr_sum"] += rr
        per_rel[pred]["n"]      += 1

    per_relation = {
        rel: {
            "n":   stats["n"],
            "mrr": round(stats["rr_sum"] / stats["n"], 4) if stats["n"] else 0.0,
            **{f"hit@{k}": round(stats[f"hit@{k}"] / stats["n"], 4)
               for k in top_k_values},
        }
        for rel, stats in per_rel.items()
    }

    return {
        "model":       model_name,
        "n_evaluated": n,
        "mrr":         round(rr_sum / n, 4) if n else 0.0,
        **{f"hit@{k}": round(hits[k] / n, 4) for k in top_k_values},
        "per_relation": per_relation,
    }


def run_live_evaluation(
    models: list[str],
    n_samples: int | None,
    top_k_values: list[int],
) -> list[dict]:
    if not cfg.LP_EVAL_CORPUS.exists():
        print(f"\n  [Sección B] Corpus no encontrado: {cfg.LP_EVAL_CORPUS}")
        print("  Genera el corpus con:  python src/generate_corpus.py")
        return []

    with open(cfg.LP_EVAL_CORPUS, encoding="utf-8") as f:
        corpus = json.load(f)

    if n_samples:
        corpus = corpus[:n_samples]

    print(f"\n{'='*60}")
    print("  SECCIÓN B — Evaluación en vivo (datos no vistos por PyKEEN)")
    print(f"  Corpus: {cfg.LP_EVAL_CORPUS.name}  ({len(corpus)} entradas)")
    print(f"{'='*60}")

    results = []
    for model_name in models:
        try:
            results.append(evaluate_model_live(model_name, corpus, top_k_values))
        except FileNotFoundError as e:
            print(f"  [{model_name}] Modelo no disponible — {e}")

    return results


def print_live_table(results: list[dict], sort_by: str, best_model_A: str | None) -> None:
    if not results:
        return

    rows_sorted = sorted(results, key=lambda r: r.get(sort_by, 0.0), reverse=True)
    sep, header, sort_marker = _table_header("", sort_by)

    print()
    print("=" * len(sep))
    print("  SECCIÓN B — Métricas en datos no vistos (LP_EVAL_CORPUS)")
    print(sort_marker)
    print("=" * len(sep))
    print(header)
    print("  " + "─" * (len(sep) - 2))

    best_live = rows_sorted[0]["model"] if rows_sorted else None
    for rank, row in enumerate(rows_sorted, 1):
        name = row["model"]
        star = "★ " if name == best_live else "  "
        line = f"{star}{name:<12}" + "".join(
            f"  {row.get(m, 0.0):>10.4f}" for m in _METRIC_COLS
        )
        if rank == 1:
            line = f"\033[1m{line}\033[0m"
        print(f"  {line}")

    print("=" * len(sep))
    print()


def print_overfit_analysis(train_rows: list[dict], live_results: list[dict]) -> None:
    if not live_results:
        return

    live_by_model = {r["model"]: r for r in live_results}
    train_by_model = {r["model"]: r for r in train_rows}

    common = [m for m in train_by_model if m in live_by_model]
    if not common:
        return

    col_w  = 10
    name_w = 12
    sep    = "─" * (name_w + 2 + col_w * 2 * len(_METRIC_COLS) + 6 * len(_METRIC_COLS) + 4)

    print("=" * len(sep))
    print("  ANÁLISIS DE GENERALIZACIÓN  (A = train interno, B = datos no vistos)")
    header_cols = "".join(
        f"  {'A:'+_LABELS[m]:>{col_w}}  {'B:'+_LABELS[m]:>{col_w}}"
        for m in _METRIC_COLS
    )
    print(f"  {'Modelo':<{name_w}}{header_cols}  {'Sobreajuste MRR':>16}")
    print("  " + "─" * (len(sep) - 2))

    for model_name in sorted(common, key=lambda m: train_by_model[m].get("mrr", 0.0), reverse=True):
        ta = train_by_model[model_name]
        tb = live_by_model[model_name]
        val_cols = "".join(
            f"  {ta.get(m, 0.0):>{col_w}.4f}  {tb.get(m, 0.0):>{col_w}.4f}"
            for m in _METRIC_COLS
        )
        delta_mrr = ta.get("mrr", 0.0) - tb.get("mrr", 0.0)
        flag = "  ⚠ alto" if delta_mrr > 0.1 else ("  ok" if delta_mrr < 0.02 else "")
        print(f"  {model_name:<{name_w}}{val_cols}  {delta_mrr:>+14.4f}{flag}")

    print("=" * len(sep))
    print()


# ---------------------------------------------------------------------------
# Punto de entrada
# ---------------------------------------------------------------------------

def run(
    sort_by:   str        = "mrr",
    show_gaps: bool       = True,
    live:      bool       = False,
    models:    list[str]  | None = None,
    n_samples: int        | None = None,
    top_k_values: list[int] | None = None,
) -> None:
    top_k_values = top_k_values or cfg.HIT_K_VALUES
    models       = models or cfg.KGE_MODELS

    # --- Sección A ---
    try:
        train_rows = load_training_comparison()
    except FileNotFoundError as e:
        print(f"[Error] {e}")
        return

    best      = load_best()
    best_name = best["model"] if best else None

    print_training_table(train_rows, sort_by=sort_by, best_model=best_name)
    if show_gaps:
        print_gaps(train_rows)

    # --- Sección B (opcional) ---
    live_results = []
    if live:
        live_results = run_live_evaluation(models, n_samples, top_k_values)
        print_live_table(live_results, sort_by=sort_by, best_model_A=best_name)
        print_overfit_analysis(train_rows, live_results)

    # --- Guardar JSON combinado ---
    out = cfg.MODEL_COMPARISON_DIR / "eval_kge_comparacion.json"
    cfg.MODEL_COMPARISON_DIR.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({
            "sort_by":     sort_by,
            "best_model":  best_name,
            "section_A":   train_rows,
            "section_B":   live_results,
        }, f, ensure_ascii=False, indent=2)
    print(f"  Resultados guardados en {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Comparación de modelos KGE: entrenamiento + datos no vistos")
    parser.add_argument("--sort",      default="mrr", choices=_METRIC_COLS,
                        help="Métrica de ordenación (default: mrr)")
    parser.add_argument("--no-gaps",   action="store_true",
                        help="No mostrar tabla de diferencias (Sección A)")
    parser.add_argument("--live",      action="store_true",
                        help="Ejecutar Sección B: evaluación en vivo sobre LP_EVAL_CORPUS")
    parser.add_argument("--models",    nargs="+", default=None,
                        help="Modelos a evaluar en Sección B (default: todos en KGE_MODELS)")
    parser.add_argument("--n-samples", type=int, default=None,
                        help="Nº de entradas del corpus LP a evaluar (default: todas)")
    args = parser.parse_args()
    run(
        sort_by=args.sort,
        show_gaps=not args.no_gaps,
        live=args.live,
        models=args.models,
        n_samples=args.n_samples,
    )
