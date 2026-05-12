"""
Evaluación 2 — KGE por sí solo: link prediction por predicado.

Carga el modelo KGE entrenado y evalúa su capacidad de link prediction
sobre las incidencias del test.tsv (el 5 % reservado para el sistema).

Para cada incidencia de test y cada propiedad que tenga rellena, llama a
predict_tails(incident_id, propiedad, top_k) y comprueba si el valor real
aparece en el top-1, top-3 o top-10 de predicciones.

Métricas reportadas:
  Hit@1, Hit@3, Hit@10, MRR
  — global y por predicado —

Los resultados se guardan en out/evaluation/eval_kge.json

Uso:
  python src/evaluacion/eval_kge.py
  python src/evaluacion/eval_kge.py --model ComplEx
  python src/evaluacion/eval_kge.py --model best --top-k 10 --n-samples 500
"""

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import src.config as cfg
from src.phase4_incident_creator import INCIDENT_PROPS

K_VALUES = [1, 3, 10]


# ---------------------------------------------------------------------------
# Carga de datos
# ---------------------------------------------------------------------------

def _load_test_incidents() -> dict[str, dict[str, list[str]]]:
    """Lee test.tsv y devuelve {incident_id: {predicate: [values]}}."""
    prop_set  = set(INCIDENT_PROPS)
    incidents: dict = {}
    with open(cfg.TEST_TSV, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 3:
                continue
            head, rel, tail = parts
            if not head.startswith("incident_") or rel not in prop_set:
                continue
            incidents.setdefault(head, {}).setdefault(rel, []).append(tail)
    return incidents


# ---------------------------------------------------------------------------
# Evaluación
# ---------------------------------------------------------------------------

def evaluate(
    model_name: str  = "best",
    top_k:      int  = 10,
    n_samples:  int  = None,
    seed:       int  = cfg.RANDOM_SEED,
) -> dict:
    from src.phase3_link_prediction import load_model_by_name, get_best_model_name, predict_tails

    if model_name.lower() == "best":
        model_name = get_best_model_name()
    model, factory = load_model_by_name(model_name)

    test_incidents = _load_test_incidents()
    if not test_incidents:
        raise FileNotFoundError(
            f"test.tsv vacío o no encontrado: {cfg.TEST_TSV}\n"
            "Ejecuta primero:  python src/phase1_triples.py"
        )

    inc_ids = list(test_incidents.keys())
    if n_samples and n_samples < len(inc_ids):
        rng = random.Random(seed)
        inc_ids = rng.sample(inc_ids, n_samples)

    print(f"\n{'='*60}")
    print(f"  Evaluación KGE — {model_name}")
    print(f"  Incidencias de test evaluadas: {len(inc_ids):,}")
    print(f"  Top-k: {top_k}")
    print(f"{'='*60}\n")

    # Acumuladores por predicado
    hits:   dict[str, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    mrr_sc: dict[str, list[float]]   = defaultdict(list)
    total:  dict[str, int]           = defaultdict(int)

    for i, inc_id in enumerate(inc_ids):
        if i % 100 == 0 and i:
            print(f"  [{i}/{len(inc_ids)}] ...")
        props = test_incidents[inc_id]
        for pred, true_vals in props.items():
            true_set = set(true_vals)
            try:
                preds = predict_tails(model, factory, inc_id, pred, top_k)
            except Exception:
                continue

            pred_entities = [e for e, _ in preds]

            for k in K_VALUES:
                if any(v in pred_entities[:k] for v in true_set):
                    hits[pred][k] += 1

            # MRR: rango del primer valor correcto
            for rank, (e, _) in enumerate(preds, 1):
                if e in true_set:
                    mrr_sc[pred].append(1.0 / rank)
                    break
            else:
                mrr_sc[pred].append(0.0)

            total[pred] += 1

    # Calcular métricas globales y por predicado
    results: dict = {
        "model":    model_name,
        "n_tested": len(inc_ids),
        "top_k":    top_k,
        "global":   {},
        "by_predicate": {},
    }

    all_hits  = {k: 0 for k in K_VALUES}
    all_mrr:  list[float] = []
    all_total = 0

    print(f"\n  {'Predicado':<28} {'N':>5}" + "".join(
        f"  {'Hit@'+str(k):>7}" for k in K_VALUES
    ) + f"  {'MRR':>7}")
    print("  " + "─" * (28 + 5 + 2 + 7 * len(K_VALUES) + 3 * len(K_VALUES) + 9))

    for pred in INCIDENT_PROPS:
        n = total.get(pred, 0)
        if n == 0:
            continue
        pred_hits = {k: hits[pred].get(k, 0) / n for k in K_VALUES}
        pred_mrr  = sum(mrr_sc[pred]) / n if mrr_sc[pred] else 0.0

        results["by_predicate"][pred] = {
            **{f"hit@{k}": round(pred_hits[k], 4) for k in K_VALUES},
            "mrr": round(pred_mrr, 4),
            "n":   n,
        }
        for k in K_VALUES:
            all_hits[k] += hits[pred].get(k, 0)
        all_mrr.extend(mrr_sc[pred])
        all_total += n

        row = f"  {pred:<28} {n:>5}" + "".join(
            f"  {pred_hits[k]:>7.4f}" for k in K_VALUES
        ) + f"  {pred_mrr:>7.4f}"
        print(row)

    if all_total:
        g_hits = {k: all_hits[k] / all_total for k in K_VALUES}
        g_mrr  = sum(all_mrr) / len(all_mrr) if all_mrr else 0.0
        results["global"] = {
            **{f"hit@{k}": round(g_hits[k], 4) for k in K_VALUES},
            "mrr":     round(g_mrr, 4),
            "n_total": all_total,
        }
        print("  " + "─" * (28 + 5 + 2 + 7 * len(K_VALUES) + 3 * len(K_VALUES) + 9))
        row_global = f"  {'GLOBAL':<28} {all_total:>5}" + "".join(
            f"  {g_hits[k]:>7.4f}" for k in K_VALUES
        ) + f"  {g_mrr:>7.4f}"
        print(f"\033[1m{row_global}\033[0m")

    print()

    # Guardar resultados
    out_dir = cfg.EVAL_DIR / "kge"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"eval_kge_{model_name.lower()}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"  Resultados guardados en {out_path}\n")

    return results


# ---------------------------------------------------------------------------
# Punto de entrada
# ---------------------------------------------------------------------------

def run(model_name="best", top_k=10, n_samples=None):
    return evaluate(model_name=model_name, top_k=top_k, n_samples=n_samples)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluación KGE por predicado sobre test.tsv")
    parser.add_argument("--model",     default="best",
                        help="Modelo KGE ('best' = mejor por MRR). Opciones: " + str(cfg.KGE_MODELS))
    parser.add_argument("--top-k",    type=int, default=10)
    parser.add_argument("--n-samples", type=int, default=None,
                        help="Nº de incidencias a evaluar (default: todas las de test.tsv)")
    args = parser.parse_args()
    run(model_name=args.model, top_k=args.top_k, n_samples=args.n_samples)
