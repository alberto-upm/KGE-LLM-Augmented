"""
Configuración compartida del pipeline KGE + LLM.
Importar desde cualquier módulo src/phaseN_*.py para usar rutas y parámetros.
"""

from pathlib import Path

# Auto-detección de dispositivo (CUDA si está disponible, si no CPU)
try:
    import torch as _torch
    DEVICE = "cuda" if _torch.cuda.is_available() else "cpu"
except ImportError:
    DEVICE = "cpu"

# ---------------------------------------------------------------------------
# Rutas
# ---------------------------------------------------------------------------

BASE_DIR    = Path(__file__).parent.parent
DATA_DIR    = BASE_DIR / "data"
TTL_FILE    = DATA_DIR / "filtrado.ttl"

TRIPLES_DIR = DATA_DIR / "triples"
CORPUS_DIR  = DATA_DIR / "corpus"
RULES_DIR   = DATA_DIR / "reglas"
RULES_PATH  = RULES_DIR / "rules-1000-3"

OUT_DIR     = BASE_DIR / "out"
MAPS_DIR    = OUT_DIR / "maps"  # Mapas entity_to_id / relation_to_id (compartidos)
PRED_DIR    = OUT_DIR / "predictions"
EVAL_DIR    = OUT_DIR / "evaluation"

# Corpus generado por generate_corpus.py
QA_CORPUS   = CORPUS_DIR / "qa_corpus.json"
TRIPLES_VRB = CORPUS_DIR / "triples_verbalized.json"

# Splits Phase 1 (por bloques de incidencia)
TRAIN_TSV    = TRIPLES_DIR / "train.tsv"
VALID_TSV    = TRIPLES_DIR / "valid.tsv"
TEST_TSV     = TRIPLES_DIR / "test.tsv"       # 5 % sistema — nunca visto por KGE
# Split KGE test (triple-level, generado automáticamente en Phase 2)
KGE_TEST_TSV = TRIPLES_DIR / "kge_test.tsv"  # 10 % KGE — entidades conocidas

# Predicciones y evaluación
IMPLICIT_RELS_FILE  = PRED_DIR / "implicit_relations.json"
EVAL_RESULTS_FILE   = EVAL_DIR / "results.json"

# ---------------------------------------------------------------------------
# Multi-model KGE
# ---------------------------------------------------------------------------

KGE_MODELS = ['TransE', 'DistMult', 'ComplEx', 'RotatE', 'PairRE']


def model_dir(model_name: str) -> Path:
    return OUT_DIR / "models" / model_name.lower()


def embed_dir(model_name: str) -> Path:
    return OUT_DIR / "embeddings" / model_name.lower()


def entity_embeddings_path(model_name: str) -> Path:
    return embed_dir(model_name) / "entity_embeddings.pt"


def relation_embeddings_path(model_name: str) -> Path:
    return embed_dir(model_name) / "relation_embeddings.pt"


# Mapas compartidos (independientes del modelo KGE)
ENTITY_TO_ID        = MAPS_DIR / "entity_to_id.json"
RELATION_TO_ID      = MAPS_DIR / "relation_to_id.json"

# Rutas por defecto para embeddings apuntan a TransE (ahora el modelo default)
MODELS_DIR          = model_dir('transe')
EMBED_DIR           = embed_dir('transe')
ENTITY_EMBEDDINGS   = entity_embeddings_path('transe')
RELATION_EMBEDDINGS = relation_embeddings_path('transe')

# ---------------------------------------------------------------------------
# GLiNER2
# ---------------------------------------------------------------------------

GLINER_MODEL = "fastino/gliner2-base-v1"

# ---------------------------------------------------------------------------
# Corpus de evaluación link prediction (por modelo)
# ---------------------------------------------------------------------------

LP_EVAL_CORPUS       = CORPUS_DIR / "link_prediction_eval.json"
MODEL_COMPARISON_DIR = EVAL_DIR / "model_comparison"
BEST_MODEL_FILE      = MODEL_COMPARISON_DIR / "best_model.json"

# Entity-to-entity evaluation corpus
ENTITY_EVAL_CORPUS = CORPUS_DIR / "entity_to_entity_eval.json"

# Pares a evaluar: (source_prop, target_prop)
# Lectura: "dado el valor de source_prop, predice el valor de target_prop"
ENTITY_EVAL_PAIRS = [
    ("int_hasCustomer",  "hasTechnician"),        # empresa → técnico
    ("hasSupportGroup",  "hasSupportCategory"),   # grupo de soporte → categoría
    ("hasTypeInc",       "hasTechnician"),         # tipo de incidencia → técnico
    ("int_hasCustomer",  "hasSupportGroup"),       # empresa → grupo de soporte
    ("hasSupportGroup",  "hasTechnician"),         # grupo de soporte → técnico
]

# ---------------------------------------------------------------------------
# Hiperparámetros KGE (DistMult)
# ---------------------------------------------------------------------------

EMBEDDING_DIM  = 256
N_EPOCHS       = 600
BATCH_SIZE     = 2048
LEARNING_RATE  = 1e-3
NEG_PER_POS    = 128      # 128 para NSSA (escala con negativos), 50 default A100, 10 fallback CPU
RANDOM_SEED    = 42

# División en dos niveles:
#   Nivel 1 (por bloques de incidencias, Phase 1):
#     95 % → entrenamiento + validación KGE
#      5 % → test del SISTEMA COMPLETO (nunca visto por KGE ni CBR)
#   Nivel 2 (dentro del 95 %, Phase 2 KGE):
#     90 % del 95 % → entrenamiento KGE  (≈ 85.5 % del total)
#     10 % del 95 % → validación KGE     (≈  9.5 % del total)

SYSTEM_TEST_RATIO = 0.05   # bloques reservados para evaluación del sistema
KGE_VALID_SPLIT   = 0.10   # fracción del 95 % usada para validación KGE

TRAIN_RATIO = round((1 - SYSTEM_TEST_RATIO) * (1 - KGE_VALID_SPLIT), 4)  # 0.855
VALID_RATIO = round((1 - SYSTEM_TEST_RATIO) * KGE_VALID_SPLIT,       4)  # 0.095
# TEST_RATIO = SYSTEM_TEST_RATIO = 0.05 (implícito: 1 - TRAIN_RATIO - VALID_RATIO)

# ---------------------------------------------------------------------------
# LLM — vLLM (servidor local OpenAI-compatible)
#
# Arrancar con:
#   vllm serve meta-llama/Meta-Llama-3-8B-Instruct \
#       --port 8000 --dtype float16 --max-model-len 4096 \
#       --tool-call-parser llama3_json
# ---------------------------------------------------------------------------

VLLM_BASE_URL   = "http://localhost:8000/v1"
DEFAULT_MODEL   = "meta-llama/Meta-Llama-3-8B-Instruct"
MAX_NEW_TOKENS  = 128

# ---------------------------------------------------------------------------
# Evaluación
# ---------------------------------------------------------------------------

EVAL_SAMPLE_N   = 200          # nº de Q&A a evaluar en phase6
HIT_K_VALUES    = [1, 3, 10]   # valores de k para Hit@k
TOP_K_PREDICT   = 10           # top-k en link prediction (phase3)
TOP_K_SIMILAR   = 5            # incidencias similares en CBR (phase5)

# ---------------------------------------------------------------------------
# Weighted RRF (fusión KGE + CBR en recommend_property)
# ---------------------------------------------------------------------------

RRF_K   = 60    # constante de suavizado (estándar IR)
W_KGE   = 0.7   # peso del ranking KGE (link prediction)
W_CBR   = 0.3   # peso del ranking CBR (frecuencia histórica real)
