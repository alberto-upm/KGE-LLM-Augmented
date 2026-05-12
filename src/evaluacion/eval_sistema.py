"""
Evaluación 4 — Sistema completo con el 5 % de incidencias reservado.

Evalúa el pipeline completo (Reglas → KGE + CBR) sobre las incidencias del
test.tsv que NUNCA han sido vistas por el modelo KGE ni por el CBR durante
el entrenamiento.

Protocolo leave-one-out por propiedad:
  Para cada incidencia de test y cada propiedad que tenga rellena:
    - Se toman como "conocidas" el resto de propiedades de esa incidencia.
    - Se lanza la cascada: Reglas → KGE+CBR.
    - Se comprueba si el valor real aparece en top-1, top-3 o top-5.
    - Se registra la fuente (RULE / CBR / KGE).

Métricas reportadas:
  Hit@1, Hit@3, Hit@5, MRR  — global y por predicado
  Cobertura de reglas (% predicciones que vinieron de una regla)
  Breakdown por fuente (RULE / CBR / KGE)

Los resultados se guardan en out/evaluation/sistema/eval_sistema.json

Uso:
  python src/evaluacion/eval_sistema.py
  python src/evaluacion/eval_sistema.py --model best --n-samples 300
  python src/evaluacion/eval_sistema.py --model ComplEx --top-k 5 --min-props 3
"""

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import src.config as cfg
from src.phase4_incident_creator import (
    INCIDENT_PROPS,
    _build_incidents_map_from_tsv,
    recommend_property,
)
from src.rules.rule_engine import RuleEngine

K_VALUES = [1, 3, 5]


# ---------------------------------------------------------------------------
# Carga de incidencias de test
# ---------------------------------------------------------------------------

def _load_test_incidents(min_props: int) -> dict[str, dict[str, str]]:
    """
    Lee test.tsv y devuelve {incident_id: {predicate: first_value}}.
    Solo incluye incidencias con al menos min_props propiedades rellenas.
    """
    prop_set = set(INCIDENT_PROPS)
    raw: dict = {}
    with open(cfg.TEST_TSV, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 3:
                continue
            head, rel, tail = parts
            if not head.startswith("incident_") or rel not in prop_set:
                continue
            if rel not in raw.get(head, {}):
                raw.setdefault(head, {})[rel] = tail

    return {
        inc_id: props
        for inc_id, props in raw.items()
        if len(props) >= min_props
    }


# ---------------------------------------------------------------------------
# Evaluación del sistema
# ---------------------------------------------------------------------------

def evaluate(
    model_name: str = "best",
    top_k:      int = 5,
    n_samples:  int | None = None,
    min_props:  int = 3,
    seed:       int = cfg.RANDOM_SEED,
) -> dict:
    from src.phase3_link_prediction import load_model_by_name, get_best_model_name

    if model_name.lower() == "best":
        model_name = get_best_model_name()

    print(f"\n{'='*60}")
    print(f"  Evaluación del Sistema Completo — {model_name}")
    print(f"  Cascada: Reglas → KGE + CBR")
    print(f"{'='*60}\n")

    # Cargar recursos
    print("[1/4] Cargando incidencias de test (5 % reservado) ...")
    test_incidents = _load_test_incidents(min_props)
    if not test_incidents:
        raise FileNotFoundError(
            f"test.tsv vacío o no encontrado: {cfg.TEST_TSV}\n"
            "Ejecuta primero:  python src/phase1_triples.py"
        )
    print(f"      {len(test_incidents):,} incidencias con ≥{min_props} propiedades")

    if n_samples and n_samples < len(test_incidents):
        rng = random.Random(seed)
        keys = rng.sample(list(test_incidents.keys()), n_samples)
        test_incidents = {k: test_incidents[k] for k in keys}
        print(f"      Muestra aleatoria: {len(test_incidents):,} incidencias")

    print("\n[2/4] Cargando base de conocimiento (train + valid) ...")
    # El incidents_map NO debe incluir el test: usamos solo train.tsv y valid.tsv
    incidents_map = _build_incidents_map_from_tsv()
    print(f"      {len(incidents_map):,} incidencias históricas cargadas")

    print("\n[3/4] Cargando modelo KGE y motor de reglas ...")
    model, factory = load_model_by_name(model_name)
    rule_engine = RuleEngine(cfg.RULES_PATH)
    print(f"      Motor de reglas: {rule_engine.stats()['total_rules']} reglas")

    # ---------------------------------------------------------------------------
    # Loop de evaluación
    # ---------------------------------------------------------------------------
    print(f"\n[4/4] Evaluando {len(test_incidents):,} incidencias ...")

    hits:   dict[str, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    mrr_sc: dict[str, list[float]]   = defaultdict(list)
    total:  dict[str, int]           = defaultdict(int)
    source_counts: dict[str, int]    = defaultdict(int)

    for idx, (inc_id, props) in enumerate(test_incidents.items()):
        if idx % 100 == 0 and idx:
            print(f"  [{idx}/{len(test_incidents)}] ...")

        for target_prop, true_val in props.items():
            # Propiedades conocidas = todas excepto la que se predice
            known_vals = {p: v for p, v in props.items() if p != target_prop}
            known_full = {p: known_vals.get(p) for p in INCIDENT_PROPS}

            # — CASCADA —
            rule_hint = rule_engine.infer(known_full, target_prop)
            if rule_hint:
                top_preds = [rule_hint["value"]]
                source    = "RULE"
            else:
                try:
                    recs, _ = recommend_property(
                        known_full, target_prop, incidents_map,
                        model, factory, top_k,
                    )
                except Exception:
                    recs = []
                top_preds = [e for e, _, _ in recs]
                freq      = recs[0][1] if recs else 0
                source    = "CBR" if freq > 0 else "KGE"

            source_counts[source] += 1

            for k in K_VALUES:
                if true_val in top_preds[:k]:
                    hits[target_prop][k] += 1

            # MRR
            try:
                rank = top_preds.index(true_val) + 1
                mrr_sc[target_prop].append(1.0 / rank)
            except ValueError:
                mrr_sc[target_prop].append(0.0)

            total[target_prop] += 1

    # ---------------------------------------------------------------------------
    # Calcular y presentar métricas
    # ---------------------------------------------------------------------------
    results: dict = {
        "model":       model_name,
        "n_incidents": len(test_incidents),
        "top_k":       top_k,
        "min_props":   min_props,
        "global":      {},
        "by_source":   dict(source_counts),
        "by_predicate": {},
    }

    all_hits  = {k: 0 for k in K_VALUES}
    all_mrr:  list[float] = []
    all_total = 0

    col_h = 7
    header = f"  {'Predicado':<28} {'N':>5}" + "".join(
        f"  {'Hit@'+str(k):>{col_h}}" for k in K_VALUES
    ) + f"  {'MRR':>{col_h}}"
    sep = "─" * (len(header) - 2)

    print(f"\n{header}")
    print(f"  {sep}")

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
            f"  {pred_hits[k]:>{col_h}.4f}" for k in K_VALUES
        ) + f"  {pred_mrr:>{col_h}.4f}"
        print(row)

    if all_total:
        g_hits = {k: all_hits[k] / all_total for k in K_VALUES}
        g_mrr  = sum(all_mrr) / len(all_mrr) if all_mrr else 0.0
        results["global"] = {
            **{f"hit@{k}": round(g_hits[k], 4) for k in K_VALUES},
            "mrr":     round(g_mrr, 4),
            "n_total": all_total,
        }
        print(f"  {sep}")
        row_g = f"  {'GLOBAL':<28} {all_total:>5}" + "".join(
            f"  {g_hits[k]:>{col_h}.4f}" for k in K_VALUES
        ) + f"  {g_mrr:>{col_h}.4f}"
        print(f"\033[1m{row_g}\033[0m")

    # Breakdown por fuente
    total_preds = sum(source_counts.values())
    print(f"\n  Breakdown por fuente ({total_preds} predicciones):")
    for src in ("RULE", "CBR", "KGE"):
        n   = source_counts.get(src, 0)
        pct = n / total_preds if total_preds else 0.0
        bar = "█" * int(pct * 20) + "░" * (20 - int(pct * 20))
        print(f"  {src:<6}  {bar}  {pct:.1%}  ({n:,})")

    print()

    # Guardar resultados
    out_dir  = cfg.EVAL_DIR / "sistema"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "eval_sistema.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"  Resultados guardados en {out_path}\n")

    return results


# ---------------------------------------------------------------------------
# Punto de entrada
# ---------------------------------------------------------------------------

def run(model_name="best", top_k=5, n_samples=None, min_props=3):
    return evaluate(
        model_name=model_name, top_k=top_k,
        n_samples=n_samples, min_props=min_props,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluación del sistema completo sobre el 5 % de test reservado"
    )
    parser.add_argument("--model",     default="best",
                        help="Modelo KGE ('best' = mejor por MRR)")
    parser.add_argument("--top-k",    type=int, default=5)
    parser.add_argument("--n-samples", type=int, default=None,
                        help="Nº de incidencias de test a evaluar (default: todas)")
    parser.add_argument("--min-props", type=int, default=3,
                        help="Mínimo de propiedades rellenas por incidencia (default: 3)")
    args = parser.parse_args()
    run(
        model_name=args.model,
        top_k=args.top_k,
        n_samples=args.n_samples,
        min_props=args.min_props,
    )
