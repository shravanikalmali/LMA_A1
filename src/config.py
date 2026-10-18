"""
Central configuration for the RAG Baseline-vs-Re-Ranking assignment.

Keeping every tunable knob in one place means the notebook and the individual
modules all agree on the SAME settings, which is important because the whole
point of the assignment is a *fair* comparison between two pipelines.
"""
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
PDF_PATH = DATA_DIR / "monopoly_rules.pdf"

RESULTS_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Chunking  (Phase 1)
# ---------------------------------------------------------------------------
# We chunk by *words* with overlap. ~90 words keeps a single Monopoly rule
# roughly self-contained while still being small enough that a specific
# "except when..." clause lands in its own chunk (so it can be buried at rank
# 4-10 and later rescued by the re-ranker).
CHUNK_SIZE_WORDS = 90
CHUNK_OVERLAP_WORDS = 20

# ---------------------------------------------------------------------------
# Models  (Phase 2 & 3)
# ---------------------------------------------------------------------------
# Bi-Encoder: encodes query and chunks INDEPENDENTLY -> fast, approximate.
BI_ENCODER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# Cross-Encoder: encodes (query, chunk) JOINTLY -> slow, more accurate.
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
TOP_K_RETRIEVE = 10   # Stage 1 candidates (bi-encoder)
TOP_K_CONTEXT = 3     # Chunks actually handed to the LLM (both pipelines)

# ---------------------------------------------------------------------------
# Generation (Gemini)
# ---------------------------------------------------------------------------
# Free-tier quota is PER-MODEL (20 requests/day each). We pin gemini-3.5-flash.
# If you hit a daily-quota 429, switch to another flash id (gemini-3.6-flash,
# gemini-flash-latest, ...) — each has its own separate daily bucket.
GEMINI_MODEL = "gemini-3.5-flash"
# The API key is read from the environment / a local .env file at run time.
# It is NEVER hardcoded here.  See src/generation.py
GEMINI_API_KEY_ENV = "GEMINI_API_KEY"
