"""
exp024 — Simple RAG vs proposed Ontology grounding, LoRA fine-tuned, full test set.

Both arms share everything (base model, LoRA recipe, prompt template, decoding,
gold references, metrics); the ONLY difference is the source of the injected
context block:

  arm "rag"  : dense top-k passages retrieved from the train split
  arm "onto" : facts retrieved from the Jeju tourism ontology + temporal layer

Two extra reference arms are generated for context:
  arm "none" : no context at all (fine-tuned closed-book)
  arm "base" : no context, no fine-tuning (vanilla instruct model)

Gold reference for every metric is the dataset's own `Output` field — NOT an
ontology-derived reference — so neither arm is favoured by construction.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# ── environment ────────────────────────────────────────────────────────────
# All four 8B checkpoints, roberta-large (BERTScore) and wmt22-comet-da live here.
HF_HOME = os.environ.get("HF_HOME", "/ceph_data/tahirahmad/hf_home")
os.environ.setdefault("HF_HOME", HF_HOME)

def _load_env_token() -> None:
    """Pick HF_TOKEN out of the project .env — Llama 3.1 is a gated repo and PEFT
    emits 401 warnings on every save without it (weights themselves are cached)."""
    if os.environ.get("HF_TOKEN"):
        return
    env = ROOT / ".env"
    if not env.exists():
        return
    for line in env.read_text(encoding="utf-8").splitlines():
        if line.startswith("HF_TOKEN=") and not line.startswith("#"):
            os.environ["HF_TOKEN"] = line.split("=", 1)[1].strip().strip("'\"")
            break

_load_env_token()

# GPUs 5-9 belong to the GROUND project's AMBER evaluation jobs — do not touch.
TRAIN_GPUS = [0, 1, 2, 3]     # one model per GPU, two waves (rag then onto)
GEN_GPUS = [0, 1, 2, 3]       # one model per GPU
EVAL_GPU = 4                  # BERTScore / COMET / LLM-judge

# ── data ───────────────────────────────────────────────────────────────────
SPLITS = ROOT / "data" / "jeju_full" / "splits"
TRAIN_JSONL = SPLITS / "train.jsonl"
VAL_JSONL = SPLITS / "val.jsonl"
TEST_JSONL = SPLITS / "test.jsonl"
# exp025: 200-item gazetteer-grounded benchmark (100 multi-hop / 100 single-hop)
BENCH_JSONL = ROOT / "experiments" / "exp025_multihop" / "outputs" / "jeju_bench_200.jsonl"

OUT = ROOT / "experiments" / "exp024_rag_vs_ontology" / "outputs"
# EXP024_TAG namespaces adapters/generations/scores/reports so a pilot can run
# the whole pipeline end-to-end without colliding with the real campaign.
TAG = os.environ.get("EXP024_TAG", "")
_suffix = f"_{TAG}" if TAG else ""

CTX_DIR = OUT / "contexts"           # contexts are tag-independent (built once)
INDEX_DIR = OUT / "rag_index"
ADAPTER_DIR = Path(f"/ceph_data/tahirahmad/ontology_exp024/adapters{_suffix}")
GEN_DIR = OUT / f"generations{_suffix}"
SCORE_DIR = OUT / f"scores{_suffix}"
REPORT_DIR = OUT / f"reports{_suffix}"
LOG_DIR = OUT / f"logs{_suffix}"

# ── models under test ──────────────────────────────────────────────────────
# Scale ladder for Goal 1. The 1B/3B checkpoints use the ungated unsloth mirrors
# because meta-llama/Llama-3.2-* is gated for this account; weights are identical.
MODELS = [
    {"name": "Llama-3.2-1B",   "key": "llama3.2-1b",    "hf_id": "unsloth/Llama-3.2-1B-Instruct",           "reasoning": False},
    {"name": "Llama-3.2-3B",   "key": "llama3.2-3b",    "hf_id": "unsloth/Llama-3.2-3B-Instruct",           "reasoning": False},
    {"name": "Llama-3.1-8B",   "key": "llama3.1-8b",    "hf_id": "meta-llama/Llama-3.1-8B-Instruct",        "reasoning": False},
    {"name": "DeepSeek-R1-8B", "key": "deepseek-r1-8b", "hf_id": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B", "reasoning": True},
    {"name": "Qwen3-8B",       "key": "qwen3-8b",       "hf_id": "Qwen/Qwen3-8B",                            "reasoning": True},
    {"name": "Mistral-7B",     "key": "mistral-7b",     "hf_id": "mistralai/Mistral-7B-Instruct-v0.3",       "reasoning": False},
]
MODEL_BY_KEY = {m["key"]: m for m in MODELS}

ARMS = ["rag", "onto", "onto_c", "onto_v3", "onto_v4", "onto_v5", "onto_v6", "onto_g", "onto_v7", "onto_v8", "onto_v9", "onto_v10", "hybrid"] + ['hybrid10', 'hybrid15', 'hybrid20', 'hybrid40', 'hybrid20d']  # arms in the results table
# hybrid20d: the WBS V4/V5 context, assembled by run_variants.py's own rule and
# wrapped in the SAME prompt header that arm sees at inference ("rag"), so a
# dedicated adapter can be trained for it. The older `hybrid20` keeps its own
# header and its published exp036 numbers; this is a separate key, not a change.
# onto_v2 (dense retrieval over atomic triples) is kept in the codebase as a
# measured negative result: its gold-entity coverage was 0.091 vs v1's 0.093,
# so it was never worth generation compute. See README.
REFERENCE_ARMS = ["none", "base"]  # context-free controls
ALL_ARMS = ARMS + REFERENCE_ARMS

# Both arms are served by ONE adapter per model, fine-tuned on a 50/50 mix of RAG
# and ontology contexts over the same 36,847 examples. That keeps the compared
# weights identical, so the measured difference is attributable to the retrieved
# context alone rather than to two separately-converged adapters — and it halves
# the training bill on a shared box. ADAPTER_FOR_ARM maps arm -> adapter to load.
ADAPTER_ARM = "mixed"
ADAPTER_FOR_ARM = {"rag": ADAPTER_ARM, "onto": ADAPTER_ARM, "onto_v2": ADAPTER_ARM,
                   "onto_c": ADAPTER_ARM, "onto_v3": ADAPTER_ARM,
                   "onto_v4": ADAPTER_ARM, "onto_v5": ADAPTER_ARM,
                   "onto_v6": ADAPTER_ARM, "onto_g": ADAPTER_ARM,
                   "onto_v7": ADAPTER_ARM, "onto_v8": ADAPTER_ARM,
                   "onto_v9": ADAPTER_ARM, "onto_v10": ADAPTER_ARM,
                   "hybrid": ADAPTER_ARM,
                   **{a: ADAPTER_ARM for a in ['hybrid10', 'hybrid15', 'hybrid20', 'hybrid40']},
                   "hybrid20d": "hybrid20d",
                   "none": "none", "base": None}
MIX_SEED = 1234

# ── retrieval ──────────────────────────────────────────────────────────────
EMBED_MODEL = "BAAI/bge-base-en-v1.5"
EMBED_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
RAG_TOP_K = 5
RAG_PASSAGE_CHARS = 900       # per-passage truncation
ONTO_VERSION = "1.2.0"        # project ontology: 1,585 concepts / 9,208 relations, Jeju-clean
TEMPORAL_VERSION = "1.2.0"
ONTO_MAX_FACTS = 15
CTX_MAX_CHARS = 3000          # identical budget for every arm

# ── onto_v2: dense retrieval over verbalised triples ───────────────────────
# Diagnosis behind this arm (measured on the v1 generations, 4,589 scored items):
#   gold key-tokens present in the RAG context ........ 0.726
#   gold key-tokens present in the onto v1 context .... 0.093  (52% got ZERO)
#   gold key-tokens present ANYWHERE in ontology v1.2.0 0.677
# So the ontology holds the facts; v1's substring scan simply never surfaced
# them, and it filled only 30% of the prompt budget against RAG's 89%. onto_v2
# swaps the substring scan for the SAME dense encoder RAG uses, so the two arms
# differ only in what is indexed (triples vs passages), and fills the budget.
ONTO_V2_MAX_FACTS = 50
# onto_v3: entity cards + intent->class routing. Measured gold-entity coverage
# (boilerplate excluded): rag 0.552 | onto v1 0.141 | onto v3 0.275.
# Cleaned ontology (v1.3.0): 884 type-violating relations removed from v1.2.0.
ONTO_CLEAN_VERSION = "1.3.0"
# v4 = entity-linked 2-hop traversal; v5 = v4 plus always-on class routing.
ONTO_V4 = dict(max_facts=45, depth=2, route_always=False, route_per_class=4)
ONTO_V5 = dict(max_facts=60, depth=2, route_always=True, route_per_class=5)
# onto_c = the v1 retriever (which ranks best on multi-hop) running on the CLEANED
# ontology, i.e. the most precise retriever with the false facts removed.
ONTO_C_MAX_FACTS = 18
ONTO_V3_ENTITIES = 16
ONTO_V3_FACTS_PER_ENTITY = 3
ONTO_V2_EXPAND = True         # top-up with sibling relations of the best subjects
ONTO_V2_EXPAND_PER_SUBJ = 4

# ── prompting ──────────────────────────────────────────────────────────────
SYSTEM = (
    "You are a Jeju Island (South Korea) tourism assistant. Answer the user's "
    "question directly, naming specific places, months, seasons and dishes. "
    "Never mention that any context, facts, or ontology were provided."
)
CTX_HEADER = {
    "rag": "Retrieved reference passages:",
    "onto": "Authoritative ontology facts:",
    # same header as `onto` on purpose: the mixed adapter was tuned on that
    # wording, so onto_v2 stays in-distribution and needs no retraining.
    "onto_v2": "Authoritative ontology facts:",
    "onto_c": "Authoritative ontology facts:",
    "onto_v3": "Authoritative ontology facts:",
    "onto_v4": "Authoritative ontology facts:",
    "onto_v5": "Authoritative ontology facts:",
    "onto_v6": "Authoritative ontology facts:",
    "onto_g": "Authoritative ontology facts:",
    "onto_v7": "Authoritative ontology facts:",
    "onto_v8": "Authoritative ontology facts:",
    "onto_v9": "Authoritative ontology facts:",
    "onto_v10": "Authoritative ontology facts:",
    # The hybrid block carries its own two sub-headers, so the wrapper header is
    # neutral rather than claiming the whole block is ontology facts.
    "hybrid": "Reference material:",
    **{a: "Reference material:" for a in ['hybrid10', 'hybrid15', 'hybrid20', 'hybrid40']},
    # matches run_variants.py, which prompts the V4/V5 hybrid context as arm "rag"
    "hybrid20d": "Retrieved reference passages:",
}
USER_WITH_CTX = "{header}\n{ctx}\n\nQuestion: {q}\n\nAnswer using the information above."

# Mechanism analysis found the model holds the right fact and fails to use it on
# 21.5% of multi-hop items ("onto_retrieved_not_used") - a grounding failure, not
# a retrieval one. This variant states the expectation explicitly instead of
# relying on "use the information above".
USER_WITH_CTX_GROUNDED = (
    "{header}\n{ctx}\n\nQuestion: {q}\n\n"
    "Answer the question using the facts above. Name the specific places, dishes, "
    "months and seasons that appear in those facts rather than generic "
    "descriptions, and do not invent names that are absent from them."
)
GROUNDED_ARMS = {"onto_g"}
USER_NO_CTX = "Question: {q}"

JUDGE_SYSTEM = "You are a strict evaluator. Reply with only an integer."
JUDGE = (
    "Grade how accurately the candidate answer matches the reference (ground truth) "
    "for a Jeju tourism question.\n"
    "Question: {q}\nReference (ground truth): {ref}\nCandidate: {ans}\n\n"
    "Give an integer score from 0 to 10 for how well the candidate matches the "
    "reference's key facts (places, months, seasons, dishes, activities): 10 = fully "
    "correct and complete, 7-9 = mostly correct, 4-6 = partially correct, 1-3 = mostly "
    "wrong, 0 = wrong or missing. Reply with ONLY the integer score. Score:"
)
JUDGE_MODEL_KEY = "llama3.1-8b"   # one fixed judge for every arm and model
JUDGE_REF_CHARS = 2200
JUDGE_ANS_CHARS = 2200

# ── LoRA fine-tuning ───────────────────────────────────────────────────────
LORA = dict(r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"])
TRAIN = dict(epochs=1, lr=1e-4, warmup_ratio=0.03, micro_bs=2, grad_accum=8,
             max_len=2048, weight_decay=0.0, lr_scheduler="cosine", logging_steps=25,
             save_steps=500, seed=42)
# bf16 LoRA is ~2x faster but needs ~30 GB; QLoRA (4-bit) fits a ~21 GB slice on
# a GPU shared with another tenant. Set EXP024_4BIT=1 to force QLoRA.
LOAD_4BIT = os.environ.get("EXP024_4BIT", "0") == "1"
MIN_FREE_MIB_BF16 = 26000   # measured peak 22.1 GB at max_len=2048, micro_bs=2 (8B)
MIN_FREE_MIB_4BIT = 16000

# The 8B figure over-reserves for the small models, which then never get placed
# on a busy box. Scale the requirement with parameter count instead.
TRAIN_MIB = {"llama3.2-1b": 9000, "llama3.2-3b": 15000}
GEN_MIB = {"llama3.2-1b": 7000, "llama3.2-3b": 12000}


def train_mib(model_key: str) -> int:
    return TRAIN_MIB.get(model_key, MIN_FREE_MIB_BF16)


def gen_mib(model_key: str) -> int:
    return GEN_MIB.get(model_key, 22000)

# ── generation ─────────────────────────────────────────────────────────────
MAX_NEW_TOKENS = 512
GEN_BATCH = 16
GEN_MAX_PROMPT_TOKENS = 1536

# ── metrics ────────────────────────────────────────────────────────────────
METRICS = ["bertf1", "meteor", "comet", "keytok", "judge"]
METRIC_LABELS = {
    "bertf1": "BERT-F1", "meteor": "METEOR", "comet": "COMET",
    "keytok": "Key-token", "judge": "Response Accuracy",
}
COMPOSITE_WEIGHTS = {"bertf1": 0.25, "meteor": 0.20, "comet": 0.20, "keytok": 0.20, "judge": 0.15}
BERTSCORE_MODEL = "roberta-large"
COMET_MODEL = "Unbabel/wmt22-comet-da"

for _d in (CTX_DIR, GEN_DIR, SCORE_DIR, REPORT_DIR, LOG_DIR, INDEX_DIR):
    _d.mkdir(parents=True, exist_ok=True)
