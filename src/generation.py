"""
Answer generation via the Gemini API.

Crucially, BOTH pipelines use the *identical* prompt template below. The only
difference between Answer 1 and Answer 2 is which 3 chunks fill the {context}
slot. That is what makes the comparison fair: any quality difference is due to
retrieval quality, not prompt wording.

The API key is loaded from the environment or a local .env file and is never
written to disk or logged.
"""
from __future__ import annotations

import os
from typing import List, Dict

from . import config

# Load a local .env (if present) so GEMINI_API_KEY is available.
try:
    from dotenv import load_dotenv
    load_dotenv(config.PROJECT_ROOT / ".env")
except Exception:
    pass


# The single shared prompt. Grounding instructions force the model to rely on
# the retrieved context, which is exactly what surfaces retrieval failures:
# if the critical chunk is missing, a well-behaved model should say so rather
# than invent the rule.
PROMPT_TEMPLATE = """You are a precise assistant answering questions about the MONOPOLY board game rules.
Use ONLY the information in the context below. If the context does not contain
the answer, say "The provided rules do not specify this." Do not rely on outside
knowledge or popular house rules.

Context:
{context}

Question: {question}

Answer:"""


def format_context(chunks: List[Dict]) -> str:
    """Render the chosen chunks into the {context} slot, numbered."""
    return "\n\n".join(
        f"[Chunk {i}] {c['text']}" for i, c in enumerate(chunks, start=1)
    )


def build_prompt(question: str, chunks: List[Dict]) -> str:
    return PROMPT_TEMPLATE.format(context=format_context(chunks), question=question)


_model_cache = {}


def _get_model():
    """Lazily configure and cache the Gemini client."""
    import google.generativeai as genai

    key = os.environ.get(config.GEMINI_API_KEY_ENV)
    if not key:
        raise RuntimeError(
            f"{config.GEMINI_API_KEY_ENV} is not set. Put it in a .env file "
            "at the project root (see .env.example) or export it."
        )
    if config.GEMINI_MODEL not in _model_cache:
        genai.configure(api_key=key)
        _model_cache[config.GEMINI_MODEL] = genai.GenerativeModel(config.GEMINI_MODEL)
    return _model_cache[config.GEMINI_MODEL]


def generate_answer(question: str, chunks: List[Dict], max_retries: int = 5) -> str:
    """
    Generate an answer from the given context chunks using Gemini.

    The free tier allows only a few requests per minute, so we transparently
    retry on quota errors (HTTP 429), honouring the server's suggested
    retry_delay. This keeps the notebook runnable end-to-end on a free key.
    """
    import time
    from google.api_core import exceptions as gexc

    model = _get_model()
    prompt = build_prompt(question, chunks)
    for attempt in range(max_retries):
        try:
            resp = model.generate_content(prompt)
            return resp.text.strip()
        except gexc.ResourceExhausted as e:
            # Pull the server-suggested delay if present, else back off linearly.
            delay = 20 * (attempt + 1)
            for d in getattr(e, "details", []) or []:
                rd = getattr(d, "retry_delay", None)
                secs = getattr(rd, "seconds", None)
                if isinstance(secs, int) and 0 < secs <= 120:
                    delay = secs + 2
            delay = min(delay, 65)       # never sleep longer than ~a minute
            if attempt == max_retries - 1:
                raise
            print(f"  [rate-limited; waiting {delay}s then retrying...]", flush=True)
            time.sleep(delay)
