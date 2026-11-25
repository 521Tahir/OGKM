"""Central configuration for the Ontology-Augmented LLM Benchmark app."""
from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS = Path(__file__).resolve().parent / "results"
LOGS = RESULTS / "logs"
ANSWERS = RESULTS / "generated_answers"
METRICS_DIR = RESULTS / "metrics"

ONTOLOGY_VERSION = "1.1.0"        # hybrid ontology for KG context
TEMPORAL_VERSION = "1.2.0"        # temporal layer for time-scoped facts

# model_key -> GPU (per application.txt deployment plan). GPU 9 = eval/metrics.
# Each model runs as an independent HTTP microservice on its own port + GPU,
# started before Streamlit (see run_app.sh). The app is a thin parallel client.
SERVICE_HOST = "127.0.0.1"
MODELS = [
    {"name": "Llama-3.1-8B",   "key": "llama3.1-8b",   "gpu": 5, "port": 8101, "reasoning": False},
    {"name": "DeepSeek-R1-8B", "key": "deepseek-r1-8b", "gpu": 6, "port": 8102, "reasoning": True},
    {"name": "Qwen3-8B",       "key": "qwen3-8b",       "gpu": 7, "port": 8103, "reasoning": True},
    {"name": "Mistral-7B",     "key": "mistral-7b",     "gpu": 8, "port": 8104, "reasoning": False},
]
EVAL_GPU = 9
MAX_NEW_TOKENS = 160
TEMPERATURE = 0.0

# Composite leaderboard weights (application.txt §Leaderboard)
COMPOSITE_WEIGHTS = {"bertf1": 0.25, "meteor": 0.20, "comet": 0.20, "keytok": 0.20, "judge": 0.15}

EXAMPLE_QUESTIONS = [
    "When do hydrangeas bloom in Jeju?",
    "What is the best time to visit Hallasan?",
    "When is Cheonhyehyang citrus harvested?",
    "When is galchi season?",
]

QUALITY_METRICS = ["bertf1", "meteor", "comet", "keytok", "judge"]
METRIC_LABELS = {
    "bertf1": "BERT-F1", "meteor": "METEOR", "comet": "COMET",
    "keytok": "Key-token", "judge": "Response Accuracy", "ttft": "TTFT (ms)", "tpot": "TPOT (ms)",
}

for _d in (LOGS, ANSWERS, METRICS_DIR):
    _d.mkdir(parents=True, exist_ok=True)
