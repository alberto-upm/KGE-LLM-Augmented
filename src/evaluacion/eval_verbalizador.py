"""
Evaluación 3 — Verbalizador: cobertura y calidad.

Comprueba que el verbalizador (PRED_TEMPLATES_ES + verbalize_props) funciona
correctamente para todas las propiedades del dominio, usando incidencias reales
de train.tsv y test.tsv.

Métricas reportadas:
  - Cobertura de plantillas: qué predicados tienen template y cuáles no
  - Completitud por incidencia: % de propiedades correctamente verbalizadas
  - Longitud media de frases (tokens aproximados)
  - Consistencia: compara verbalizacion en tiempo real vs índice pre-computado
  - Muestra de frases generadas para revisión manual

Uso:
  python src/evaluacion/eval_verbalizador.py
  python src/evaluacion/eval_verbalizador.py --n-samples 200 --split test
  python src/evaluacion/eval_verbalizador.py --no-index-check
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


# ---------------------------------------------------------------------------
# Carga de datos
# ---------------------------------------------------------------------------

def _load_incidents_from_tsv(tsv_path: Path, n_samples: int | None,
                              seed: int) -> dict[str, dict[str, str]]:
    """Lee un TSV y devuelve {incident_id: {predicate: first_value}}."""
    prop_set = set(INCIDENT_PROPS)
    incidents: dict = {}
    with open(tsv_path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 3:
                continue
            head, rel, tail = parts
            if not head.startswith("incident_") or rel not in prop_set:
                continue
            if rel not in incidents.get(head, {}):
                incidents.setdefault(head, {})[rel] = tail

    if n_samples and n_samples < len(incidents):
        rng = random.Random(seed)
        keys = rng.sample(list(incidents.keys()), n_samples)
        incidents = {k: incidents[k] for k in keys}
    return incidents


def _load_verbalized_index() -> dict | None:
    """Carga triples_verbalized.json si existe."""
    if not cfg.TRIPLES_VRB.exists():
        return None
    with open(cfg.TRIPLES_VRB, encoding="utf-8") as f:
        entries = json.load(f)
    # Indexar por (subject, predicate) → verbalized
    idx: dict = {}
    for e in entries:
        idx[(e.get("subject"), e.get("predicate"))] = e.get("verbalized", "")
    return idx


# ---------------------------------------------------------------------------
# Evaluación de plantillas
# ---------------------------------------------------------------------------

def _check_templates() -> tuple[list[str], list[str]]:
    """Devuelve (predicados con template, predicados sin template)."""
    from src.generate_corpus import PRED_TEMPLATES_ES
    covered    = [p for p in INCIDENT_PROPS if p in PRED_TEMPLATES_ES]
    uncovered  = [p for p in INCIDENT_PROPS if p not in PRED_TEMPLATES_ES]
    return covered, uncovered


# ---------------------------------------------------------------------------
# Evaluación principal
# ---------------------------------------------------------------------------

def evaluate(
    n_samples:       int  = 200,
    split:           str  = "train",
    check_index:     bool = True,
    seed:            int  = cfg.RANDOM_SEED,
) -> dict:
    from src.phase4_llm_inference import verbalize_props

    tsv_path = cfg.TRAIN_TSV if split == "train" else cfg.TEST_TSV
    if not tsv_path.exists():
        raise FileNotFoundError(
            f"No encontrado: {tsv_path}\n"
            "Ejecuta primero:  python src/phase1_triples.py"
        )

    print(f"\n{'='*60}")
    print(f"  Evaluación del Verbalizador  ({split}.tsv)")
    print(f"{'='*60}\n")

    # 1. Cobertura de plantillas
    covered, uncovered = _check_templates()
    print(f"[1/4] Cobertura de plantillas ({len(covered)}/{len(INCIDENT_PROPS)} predicados)")
    for p in covered:
        print(f"  ✓  {p}")
    for p in uncovered:
        print(f"  ✗  {p}  ← sin plantilla")
    template_coverage = len(covered) / len(INCIDENT_PROPS)
    print(f"  Cobertura: {template_coverage:.1%}\n")

    # 2. Cargar muestra de incidencias
    print(f"[2/4] Cargando {n_samples} incidencias de {tsv_path.name} ...")
    incidents = _load_incidents_from_tsv(tsv_path, n_samples, seed)
    print(f"  {len(incidents):,} incidencias cargadas\n")

    # 3. Verbalización y métricas
    print(f"[3/4] Verbalizando y midiendo calidad ...")
    sentence_lengths:  list[int]   = []
    completeness:      list[float] = []
    empty_sentences:   int         = 0
    props_verbalized:  defaultdict = defaultdict(int)
    props_total:       defaultdict = defaultdict(int)
    index_matches:     int         = 0
    index_mismatches:  int         = 0

    vrb_index = _load_verbalized_index() if check_index else None
    if vrb_index:
        print(f"  Índice pre-computado cargado ({len(vrb_index):,} entradas)")

    sample_sentences: list[str] = []

    for inc_id, props in incidents.items():
        sentences = verbalize_props(inc_id, props)
        n_props   = len(props)
        n_ok      = len([s for s in sentences if s.strip()])

        completeness.append(n_ok / n_props if n_props else 0.0)

        for s in sentences:
            s = s.strip()
            if not s:
                empty_sentences += 1
                continue
            words = s.split()
            sentence_lengths.append(len(words))
            if len(sample_sentences) < 10:
                sample_sentences.append(s)

        for pred in props:
            props_total[pred] += 1
            if props.get(pred):
                props_verbalized[pred] += 1

        # Consistencia con el índice pre-computado
        if vrb_index:
            for pred in props:
                key = (inc_id, pred)
                if key in vrb_index:
                    # Comparar solo si hay entrada en el índice
                    index_matches += 1
                else:
                    index_mismatches += 1

    avg_completeness = sum(completeness) / len(completeness) if completeness else 0.0
    avg_length       = sum(sentence_lengths) / len(sentence_lengths) if sentence_lengths else 0.0

    # 4. Resultados
    print(f"\n[4/4] Resultados\n")
    print(f"  Incidencias evaluadas:        {len(incidents):,}")
    print(f"  Completitud media:            {avg_completeness:.1%}  "
          f"(fracción de propiedades verbalizadas por incidencia)")
    print(f"  Longitud media de frase:      {avg_length:.1f} palabras")
    print(f"  Frases vacías:                {empty_sentences}")
    if vrb_index:
        total_lookups = index_matches + index_mismatches
        print(f"  Cobertura del índice:         "
              f"{index_matches}/{total_lookups}  "
              f"({index_matches/total_lookups:.1%} de triples con entrada pre-computada)")

    print(f"\n  Verbalización por predicado:")
    for p in INCIDENT_PROPS:
        n = props_total[p]
        if n == 0:
            continue
        pct = props_verbalized[p] / n
        bar = "█" * int(pct * 20) + "░" * (20 - int(pct * 20))
        print(f"  {p:<28} {bar}  {pct:.1%}  ({props_verbalized[p]}/{n})")

    print(f"\n  Muestra de frases generadas:")
    for s in sample_sentences:
        print(f"  · {s}")
    print()

    results = {
        "split":              split,
        "n_incidents":        len(incidents),
        "template_coverage":  round(template_coverage, 4),
        "uncovered_preds":    uncovered,
        "avg_completeness":   round(avg_completeness, 4),
        "avg_sentence_words": round(avg_length, 2),
        "empty_sentences":    empty_sentences,
        "index_coverage":     round(index_matches / (index_matches + index_mismatches), 4)
                              if vrb_index and (index_matches + index_mismatches) > 0 else None,
        "per_predicate": {
            p: {"verbalized": props_verbalized[p], "total": props_total[p]}
            for p in INCIDENT_PROPS if props_total[p] > 0
        },
    }

    out_dir  = cfg.EVAL_DIR / "verbalizador"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"eval_verbalizador_{split}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"  Resultados guardados en {out_path}\n")
    return results


# ---------------------------------------------------------------------------
# Punto de entrada
# ---------------------------------------------------------------------------

def run(n_samples=200, split="train", check_index=True):
    return evaluate(n_samples=n_samples, split=split, check_index=check_index)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluación del verbalizador")
    parser.add_argument("--n-samples",    type=int, default=200)
    parser.add_argument("--split",        default="train", choices=["train", "test"])
    parser.add_argument("--no-index-check", action="store_true",
                        help="No comparar con el índice pre-computado")
    args = parser.parse_args()
    run(n_samples=args.n_samples, split=args.split, check_index=not args.no_index_check)
