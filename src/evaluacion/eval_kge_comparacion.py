"""
Evaluación 1 — Comparación de modelos KGE entrenados.

Lee los resultados guardados por phase2_kge_train.py (training_comparison.json)
y presenta una tabla comparativa con el ranking por MRR, destacando el mejor modelo.

No requiere cargar ningún modelo en memoria.

Uso:
  python src/evaluacion/eval_kge_comparacion.py
  python src/evaluacion/eval_kge_comparacion.py --sort mrr
  python src/evaluacion/eval_kge_comparacion.py --sort hit@10
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import src.config as cfg


# ---------------------------------------------------------------------------
# Carga de resultados
# ---------------------------------------------------------------------------

def load_comparison() -> list[dict]:
    """Lee training_comparison.json. Lanza FileNotFoundError si no existe."""
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


# ---------------------------------------------------------------------------
# Presentación
# ---------------------------------------------------------------------------

_METRIC_COLS = ["mrr", "hit@1", "hit@3", "hit@10"]
_LABELS      = {"mrr": "MRR", "hit@1": "Hit@1", "hit@3": "Hit@3", "hit@10": "Hit@10"}


def print_table(rows: list[dict], sort_by: str, best_model: str | None) -> None:
    rows_sorted = sorted(rows, key=lambda r: r.get(sort_by, 0.0), reverse=True)

    col_w = 10
    name_w = 12
    sep = "─" * (name_w + 2 + col_w * len(_METRIC_COLS) + 3 * len(_METRIC_COLS))

    header = f"  {'Modelo':<{name_w}}" + "".join(
        f"  {_LABELS[m]:>{col_w}}" for m in _METRIC_COLS
    )
    sort_marker = f"  (ordenado por {_LABELS[sort_by]} ↓)"

    print()
    print("=" * len(sep))
    print("  COMPARACIÓN DE MODELOS KGE")
    print(sort_marker)
    print("=" * len(sep))
    print(header)
    print("  " + "─" * (len(sep) - 2))

    for rank, row in enumerate(rows_sorted, 1):
        name = row["model"]
        star = "★ " if name == best_model else "  "
        line = f"{star}{name:<{name_w}}" + "".join(
            f"  {row.get(m, 0.0):>{col_w}.4f}" for m in _METRIC_COLS
        )
        # Destacar el mejor en cada columna (rank 1 siempre es el primero)
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
    """Muestra diferencias absolutas respecto al mejor modelo en cada métrica."""
    if len(rows) < 2:
        return
    best_vals = {m: max(r.get(m, 0.0) for r in rows) for m in _METRIC_COLS}
    print("  Diferencia vs. mejor modelo:")
    for row in sorted(rows, key=lambda r: r.get("mrr", 0.0), reverse=True):
        gaps = "".join(
            f"  {row['model']:<10} Δ{_LABELS[m]}={row.get(m, 0.0) - best_vals[m]:+.4f}"
            for m in _METRIC_COLS
        )
        print(f"  {row['model']:<12}" + "".join(
            f"  Δ{_LABELS[m]}={row.get(m, 0.0) - best_vals[m]:+.4f}"
            for m in _METRIC_COLS
        ))
    print()


# ---------------------------------------------------------------------------
# Punto de entrada
# ---------------------------------------------------------------------------

def run(sort_by: str = "mrr", show_gaps: bool = True) -> None:
    try:
        rows = load_comparison()
    except FileNotFoundError as e:
        print(f"[Error] {e}")
        return

    best = load_best()
    best_name = best["model"] if best else None

    print_table(rows, sort_by=sort_by, best_model=best_name)
    if show_gaps:
        print_gaps(rows)

    # Guardar también en JSON para uso programático
    out = cfg.MODEL_COMPARISON_DIR / "eval_kge_comparacion.json"
    cfg.MODEL_COMPARISON_DIR.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"rows": rows, "best_model": best_name, "sort_by": sort_by},
                  f, ensure_ascii=False, indent=2)
    print(f"  Resultados guardados en {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Comparación de modelos KGE entrenados")
    parser.add_argument("--sort", default="mrr",
                        choices=_METRIC_COLS,
                        help="Métrica por la que ordenar (default: mrr)")
    parser.add_argument("--no-gaps", action="store_true",
                        help="No mostrar tabla de diferencias")
    args = parser.parse_args()
    run(sort_by=args.sort, show_gaps=not args.no_gaps)
