"""
Fase 2 — Entrenamiento de modelos KGE con PyKEEN.

Modelos soportados: TransE, DistMult, ComplEx, RotatE, PairRE.
Cada modelo tiene hiperparámetros propios optimizados para grafos estrella de
incidencias (muchas relaciones 1-N y N-N). Los parámetros pueden sobreescribirse
con flags de CLI.

Tras entrenar todos los modelos (--all-models), se selecciona automáticamente
el mejor por MRR y se guarda en out/evaluation/model_comparison/best_model.json
para que la inferencia lo use sin configuración manual.

Requisito previo: ejecutar phase1_triples.py para generar los TSV.

Salida por modelo (ej. ComplEx):
  out/models/complex/            (modelo PyKEEN completo)
  out/embeddings/complex/entity_embeddings.pt
  out/embeddings/complex/relation_embeddings.pt

Uso:
  python src/phase2_kge_train.py                        # ComplEx (por defecto)
  python src/phase2_kge_train.py --model RotatE
  python src/phase2_kge_train.py --all-models           # entrena los 5 y elige el mejor
  python src/phase2_kge_train.py --epochs N --dim D --device cpu|cuda
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
import src.config as cfg


# ---------------------------------------------------------------------------
# Configuración por modelo (hiperparámetros optimizados para grafos estrella)
# ---------------------------------------------------------------------------
#
# Todos los modelos usan:
#   training_loop = "sLCWA"   (más eficiente que LCWA para grafos grandes)
#   negative_sampler = "bernoulli"  (correcto para relaciones N-N y 1-N)
#   evaluator = "RankBasedEvaluator" con filtered=True
#
# Los parámetros pueden sobreescribirse con flags de CLI.
# ---------------------------------------------------------------------------

_MODEL_CONFIGS: dict[str, dict] = {
    "transe": {
        "loss":             "NSSALoss",
        "loss_kwargs":      {"margin": 9.0, "adversarial_temperature": 1.0},
        "model_kwargs":     {"scoring_fct_norm": 1},
        "negative_sampler": "bernoulli",
        "num_negs_per_pos": cfg.NEG_PER_POS,
        "epochs":           cfg.N_EPOCHS,
        "lr":               cfg.LEARNING_RATE,
        "weight_decay":     0.0,
        "eval_batch_size":  32,
    },
    "distmult": {
        "loss":             "BCEWithLogitsLoss",
        "loss_kwargs":      {},
        "model_kwargs":     {},
        "negative_sampler": "bernoulli",
        "num_negs_per_pos": cfg.NEG_PER_POS,
        "epochs":           cfg.N_EPOCHS,
        "lr":               cfg.LEARNING_RATE,
        "weight_decay":     0.0,
        "eval_batch_size":  32,
    },
    # Baseline principal: robusto, expresivo y estable en grafos estrella.
    "complex": {
        "loss":             "BCEWithLogitsLoss",
        "loss_kwargs":      {},
        "model_kwargs":     {},
        "negative_sampler": "bernoulli",
        "num_negs_per_pos": 10,
        "epochs":           200,
        "lr":               1e-3,
        "weight_decay":     1e-6,
        "eval_batch_size":  8,   # embeddings complejos ocupan el doble de RAM
    },
    # Captura patrones direccionales y composicionales en relaciones.
    "rotate": {
        "loss":             "SoftplusLoss",
        "loss_kwargs":      {},
        "model_kwargs":     {},
        "negative_sampler": "bernoulli",
        "num_negs_per_pos": 10,
        "epochs":           300,
        "lr":               5e-4,
        "weight_decay":     1e-6,
        "eval_batch_size":  32,
    },
    # Muy bueno para relaciones 1-N y N-N frecuentes en incidencias.
    "pairre": {
        "loss":             "SoftplusLoss",
        "loss_kwargs":      {},
        "model_kwargs":     {},
        "negative_sampler": "bernoulli",
        "num_negs_per_pos": 10,
        "epochs":           250,
        "lr":               1e-3,
        "weight_decay":     1e-6,
        "eval_batch_size":  32,
    },
}


# ---------------------------------------------------------------------------
# Entrenamiento de un modelo
# ---------------------------------------------------------------------------

def train(
    model_name:      str          = 'ComplEx',
    epochs:          int | None   = None,   # None → default del modelo
    dim:             int          = cfg.EMBEDDING_DIM,
    batch:           int          = cfg.BATCH_SIZE,
    lr:              float | None = None,   # None → default del modelo
    device:          str          = "cpu",
    eval_batch_size: int | None   = None,   # None → default del modelo
):
    from pykeen.pipeline import pipeline
    from pykeen.triples import TriplesFactory

    model_lower = model_name.lower()
    mcfg = _MODEL_CONFIGS.get(model_lower, _MODEL_CONFIGS["complex"])

    # Parámetros efectivos: CLI > per-model defaults
    eff_epochs    = epochs         if epochs         is not None else mcfg["epochs"]
    eff_lr        = lr             if lr             is not None else mcfg["lr"]
    eff_eval_bs   = eval_batch_size if eval_batch_size is not None else mcfg["eval_batch_size"]
    loss          = mcfg["loss"]
    loss_kwargs   = mcfg["loss_kwargs"]
    model_kwargs  = dict(embedding_dim=dim, **mcfg["model_kwargs"])
    neg_sampler   = mcfg["negative_sampler"]
    num_negs      = mcfg["num_negs_per_pos"]
    weight_decay  = mcfg["weight_decay"]

    print("=" * 60)
    print(f"FASE 2 — Entrenamiento {model_name} con PyKEEN")
    print("=" * 60)

    for tsv in (cfg.TRAIN_TSV, cfg.VALID_TSV, cfg.TEST_TSV):
        if not tsv.exists():
            raise FileNotFoundError(
                f"No encontrado: {tsv}\n"
                "Ejecuta primero:  python src/phase1_triples.py"
            )

    print(f"[1/3] Cargando splits pre-divididos por bloques de incidencia ...")
    # train.tsv tiene los bloques de incidencias del 95 % de entrenamiento
    # (≈85.5 % del total, Phase 1 ya los dividió correctamente por incidencia)
    training = TriplesFactory.from_path(cfg.TRAIN_TSV)
    # valid.tsv y test.tsv usan el vocabulario de train para que PyKEEN
    # no rechace entidades/relaciones no vistas en el entrenamiento.
    validation = TriplesFactory.from_path(
        cfg.VALID_TSV,
        entity_to_id=training.entity_to_id,
        relation_to_id=training.relation_to_id,
    )
    # test.tsv = 5 % de sistema; se usa aquí para la métrica KGE estándar.
    # La evaluación del sistema completo (CBR + Reglas + LLM) se hace aparte.
    testing = TriplesFactory.from_path(
        cfg.TEST_TSV,
        entity_to_id=training.entity_to_id,
        relation_to_id=training.relation_to_id,
    )
    print(f"      Entidades:  {training.num_entities:,}")
    print(f"      Relaciones: {training.num_relations:,}")
    print(f"      Train / Valid / Test (bloques de incidencia): "
          f"{training.num_triples:,} / {validation.num_triples:,} / {testing.num_triples:,}")

    print(f"\n[2/3] Entrenando {model_name}  "
          f"(dim={dim}, epochs={eff_epochs}, lr={eff_lr}, loss={loss}, "
          f"negs={num_negs}, device={device}, eval_batch={eff_eval_bs}) ...")

    pipeline_kwargs = dict(
        training=training,
        validation=validation,
        testing=testing,
        model=model_name,
        model_kwargs=model_kwargs,
        optimizer="Adam",
        optimizer_kwargs=dict(lr=eff_lr, weight_decay=weight_decay),
        training_loop="sLCWA",
        training_loop_kwargs=dict(automatic_memory_optimization=False),
        training_kwargs=dict(
            num_epochs=eff_epochs,
            batch_size=batch,
        ),
        loss=loss,
        loss_kwargs=loss_kwargs if loss_kwargs else None,
        negative_sampler=neg_sampler,
        negative_sampler_kwargs=dict(num_negs_per_pos=num_negs),
        evaluator="RankBasedEvaluator",
        evaluator_kwargs=dict(filtered=True),
        evaluation_kwargs=dict(batch_size=eff_eval_bs, device="cpu"),
        random_seed=cfg.RANDOM_SEED,
        device=device,
    )

    result = pipeline(**pipeline_kwargs)

    print(f"\n[3/3] Guardando modelo y embeddings ...")
    out_model_dir = cfg.model_dir(model_name)
    out_embed_dir = cfg.embed_dir(model_name)
    out_model_dir.mkdir(parents=True, exist_ok=True)
    out_embed_dir.mkdir(parents=True, exist_ok=True)

    result.save_to_directory(str(out_model_dir))
    print(f"      Modelo guardado en {out_model_dir}")

    entity_repr   = result.model.entity_representations[0]
    relation_repr = result.model.relation_representations[0]
    entity_embs   = entity_repr(indices=None).detach().cpu()
    relation_embs = relation_repr(indices=None).detach().cpu()

    torch.save(entity_embs,   cfg.entity_embeddings_path(model_name))
    torch.save(relation_embs, cfg.relation_embeddings_path(model_name))
    print(f"      Embeddings guardados en {out_embed_dir}")
    print(f"      entity_embeddings.pt  shape: {list(entity_embs.shape)}")
    print(f"      relation_embeddings.pt shape: {list(relation_embs.shape)}")

    metrics = result.metric_results.to_dict()
    hits = metrics.get("both", {}).get("realistic", {})
    print(f"\n--- Métricas en test set ({model_name}) ---")
    for k in ("hits_at_1", "hits_at_3", "hits_at_10", "mean_reciprocal_rank"):
        v = hits.get(k)
        if v is not None:
            print(f"  {k}: {v:.4f}")

    print(f"\n✓ Fase 2 completada para {model_name}.")
    return result


# ---------------------------------------------------------------------------
# Entrenamiento de todos los modelos + tabla comparativa + selección del mejor
# ---------------------------------------------------------------------------

def train_all_models(
    epochs: int | None = None,   # None → cada modelo usa sus épocas propias
    dim:    int        = cfg.EMBEDDING_DIM,
    batch:  int        = cfg.BATCH_SIZE,
    lr:     float | None = None, # None → cada modelo usa su lr propio
    device: str        = "cpu",
) -> dict:
    """
    Entrena todos los modelos en cfg.KGE_MODELS secuencialmente.
    Guarda tabla comparativa y selecciona automáticamente el mejor por MRR.
    """
    results = {}
    for model_name in cfg.KGE_MODELS:
        print(f"\n{'='*60}\nEntrenando {model_name}\n{'='*60}")
        results[model_name] = train(
            model_name=model_name,
            epochs=epochs, dim=dim, batch=batch, lr=lr, device=device,
        )
    _save_comparison_table(results)
    return results


def _save_comparison_table(results: dict) -> None:
    """Extrae métricas, guarda JSON + CSV y escribe best_model.json."""
    rows = []
    for model_name, result in results.items():
        metrics = result.metric_results.to_dict()
        hits = metrics.get("both", {}).get("realistic", {})
        rows.append({
            "model":  model_name,
            "hit@1":  round(hits.get("hits_at_1",             0.0), 4),
            "hit@3":  round(hits.get("hits_at_3",             0.0), 4),
            "hit@10": round(hits.get("hits_at_10",            0.0), 4),
            "mrr":    round(hits.get("mean_reciprocal_rank",  0.0), 4),
        })

    cfg.MODEL_COMPARISON_DIR.mkdir(parents=True, exist_ok=True)

    json_path = cfg.MODEL_COMPARISON_DIR / "training_comparison.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    csv_path = cfg.MODEL_COMPARISON_DIR / "training_comparison.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "hit@1", "hit@3", "hit@10", "mrr"])
        writer.writeheader()
        writer.writerows(rows)

    # Tabla ASCII
    print("\n" + "=" * 55)
    print(f"  {'Modelo':<12} {'Hit@1':>8} {'Hit@3':>8} {'Hit@10':>8} {'MRR':>8}")
    print("  " + "-" * 51)
    for row in rows:
        print(f"  {row['model']:<12} {row['hit@1']:>8.4f} {row['hit@3']:>8.4f} "
              f"{row['hit@10']:>8.4f} {row['mrr']:>8.4f}")
    print("=" * 55)
    print(f"\n  Tabla guardada en {csv_path}")

    # Selección automática del mejor modelo por MRR
    best_row = max(rows, key=lambda r: r["mrr"])
    with open(cfg.BEST_MODEL_FILE, "w", encoding="utf-8") as f:
        json.dump(best_row, f, ensure_ascii=False, indent=2)
    print(f"\n  ★  Mejor modelo: {best_row['model']}  "
          f"(MRR={best_row['mrr']:.4f}, Hit@10={best_row['hit@10']:.4f})")
    print(f"     Guardado en {cfg.BEST_MODEL_FILE}")


def get_best_model_name() -> str:
    """
    Lee best_model.json y devuelve el nombre del modelo con mejor MRR.
    Lanza FileNotFoundError si todavía no se ha ejecutado --all-models.
    """
    if not cfg.BEST_MODEL_FILE.exists():
        raise FileNotFoundError(
            f"No existe {cfg.BEST_MODEL_FILE}\n"
            "Ejecuta primero:  python src/phase2_kge_train.py --all-models"
        )
    with open(cfg.BEST_MODEL_FILE, encoding="utf-8") as f:
        return json.load(f)["model"]


# ---------------------------------------------------------------------------
# Punto de entrada
# ---------------------------------------------------------------------------

def run(model_name=None, epochs=None, dim=None, device=None, all_models=False):
    dim    = dim    or cfg.EMBEDDING_DIM
    device = device or cfg.DEVICE
    if all_models:
        train_all_models(epochs=epochs, dim=dim, device=device)
    else:
        train(model_name=model_name or 'ComplEx', epochs=epochs, dim=dim, device=device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrenar modelos KGE con PyKEEN")
    parser.add_argument("--model",      default="ComplEx",
                        help=f"Modelo KGE a entrenar (default: ComplEx). Opciones: {cfg.KGE_MODELS}")
    parser.add_argument("--all-models", action="store_true",
                        help=f"Entrena todos los modelos {cfg.KGE_MODELS} y elige el mejor por MRR")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Épocas (default: las del modelo). Sobreescribe el default por modelo.")
    parser.add_argument("--dim",    type=int, default=cfg.EMBEDDING_DIM,
                        help=f"Dimensión de embeddings (default: {cfg.EMBEDDING_DIM})")
    parser.add_argument("--device", default=cfg.DEVICE, choices=["cpu", "cuda"])
    args = parser.parse_args()
    run(
        model_name=args.model,
        epochs=args.epochs,
        dim=args.dim,
        device=args.device,
        all_models=args.all_models,
    )
