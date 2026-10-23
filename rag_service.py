"""Question + retrieved context -> grounded JSON answer from Gemini.

Anti-hallucination measures:
- The prompt only contains retrieved sources, each labelled [S1], [S2], ...
- Gemini must cite the labels it used; pages/sections shown to the user are taken from OUR
  retrieved sources for those labels, never from page numbers the model invents.
- Gemini must say found=false when the sources don't contain the answer.
- JSON output is enforced by the API (response_schema), not by asking nicely.
"""
import re
import threading
import time

from google import genai
from google.genai import errors
from google.genai import types
from pydantic import BaseModel, Field

from config import DOCUMENTS, settings
from retrieval.search import retrieve

PROFILE_HINTS = {
    "narrative": "The sources are passages of a narrative report.",
    "hierarchical": ("The sources are full sections of a long report; each is labelled with its chapter/section path. "
                     "Prefer the section whose path matches the question and mention it when helpful."),
    "structured": ("Some sources are tables or charts in Markdown. Read values carefully, keep units and years exactly "
                   "as written, and do not do arithmetic unless asked. Values prefixed with '~' were estimated from a "
                   "chart, so say they are approximate. If a footnote qualifies a number, mention the qualification."),
}

NOT_FOUND = "I couldn't find this in the document."


class LLMAnswer(BaseModel):
    found: bool = Field(description="false if the sources do not contain the answer")
    answer: str = Field(description="answer with inline citations like [S1]; or the not-found message")
    cited_sources: list[str] = Field(description="labels like 'S1' that the answer actually uses")
    confidence: float = Field(description="0.0-1.0, how fully the sources support the answer")


_client: genai.Client | None = None
_client_lock = threading.Lock()


def gemini() -> genai.Client:
    # one shared client, created under a lock so parallel threads never build (and close) duplicates
    global _client
    with _client_lock:
        if _client is None:
            # retries 408/429/5xx with exponential backoff ("model overloaded" 503s are common)
            retry = types.HttpRetryOptions(attempts=3, initial_delay=2, max_delay=10)
            _client = genai.Client(api_key=settings.google_api_key,
                                   http_options=types.HttpOptions(retry_options=retry))
        return _client


_cooldown: dict[str, float] = {}  # model -> time until which we skip it (it hit its rate limit)


def _retry_delay(e: errors.APIError, default: float) -> float:
    """Google tells us how long to wait in the 429 body ("retryDelay": "23s")."""
    match = re.search(r"retryDelay['\"]?:\s*['\"](\d+(?:\.\d+)?)s", str(e))
    return min(float(match.group(1)) + 1, 20) if match else default


def generate(contents, config: types.GenerateContentConfig, model: str | None = None, max_retries: int = 3):
    """Call Gemini with graceful degradation:
    - if the preferred model is overloaded / rate-limited / gone, fall back to the default model,
      and skip the rate-limited one for 5 minutes instead of hitting it again on every call;
    - if the default model is rate-limited too, wait as long as Google asks and retry (free tier = requests/minute).
    """
    candidates = list(dict.fromkeys([model or settings.gemini_model, settings.gemini_model]))
    candidates = [m for m in candidates if _cooldown.get(m, 0) < time.time()] or [settings.gemini_model]
    for i, m in enumerate(candidates):
        last = i == len(candidates) - 1
        for attempt in range(max_retries + 1 if last else 1):
            try:
                return gemini().models.generate_content(model=m, contents=contents, config=config)
            except (errors.ServerError, errors.ClientError) as e:
                code = getattr(e, "code", None)
                if code not in (404, 429, 500, 503):
                    raise
                if code == 429 and m != settings.gemini_model:
                    _cooldown[m] = time.time() + 300
                if not last:
                    print(f"WARNING: {m} failed ({code}); falling back to {candidates[i + 1]}")
                    break
                if attempt == max_retries or code == 404:
                    raise
                wait = _retry_delay(e, 15 * (attempt + 1))
                print(f"WARNING: {m} failed ({code}); waiting {wait:.0f}s before retry {attempt + 1}/{max_retries}")
                time.sleep(wait)


def build_prompt(question: str, doc_id: str, units: list[dict]) -> str:
    doc = DOCUMENTS[doc_id]
    sources = "\n\n".join(
        f"[{u['sid']}] (page {', '.join(map(str, u['pages']))}"
        f"{' | section: ' + u['section'] if u['section'] else ''} | type: {u['type']})\n{u['context']}"
        for u in units
    )
    return f"""You answer questions about the document "{doc['name']}" using ONLY the sources below.
{PROFILE_HINTS[doc['profile']]}

Rules:
- Use only facts stated in the sources. No outside knowledge, no guessing.
- Put a citation like [S2] after every sentence that uses a source.
- If the sources do not contain the answer, set found=false, answer exactly "{NOT_FOUND}", cited_sources=[], confidence=0.
- If the sources only partly answer it, answer that part and say what is missing; lower the confidence.
- Be concise and specific: numbers, names, years.
- When the sources show directly related context for the figure asked about (a revision vs. an earlier forecast,
  a comparison, a total it belongs to), include it in one short clause.

SOURCES:
{sources}

QUESTION: {question}"""


def answer(question: str, doc_id: str, include_context: bool = False) -> dict:
    t0 = time.time()
    units = retrieve(question, doc_id)
    retrieval_ms = round((time.time() - t0) * 1000)

    if not units:
        return {"answer": "This document has not been indexed yet.", "found": False, "confidence_score": 0.0,
                "source_pages": [], "sources": [], "token_count": 0,
                "timings": {"retrieval_ms": retrieval_ms, "llm_ms": 0}}

    sources = [
        {"sid": u["sid"], "cited": False, "page": u["page"], "pages": u["pages"], "section": u["section"],
         "type": u["type"], "score": u["score"], "text": u["matched_text"],
         **({"context": u["context"]} if include_context else {})}
        for u in units
    ]
    if units[0]["score"] < settings.min_relevance:
        # nothing in the document is even remotely about this -> don't let the LLM improvise
        return {"answer": NOT_FOUND, "found": False, "confidence_score": 0.0, "source_pages": [],
                "cited_sources": [], "sources": sources, "token_count": 0,
                "timings": {"retrieval_ms": retrieval_ms, "llm_ms": 0}}

    t1 = time.time()
    resp = generate(
        build_prompt(question, doc_id, units),
        types.GenerateContentConfig(temperature=0.1, response_mime_type="application/json",
                                    response_schema=LLMAnswer),
        max_retries=1,
    )
    llm_ms = round((time.time() - t1) * 1000)
    usage = resp.usage_metadata
    token_count = (usage.total_token_count or 0) if usage else 0

    parsed: LLMAnswer | None = resp.parsed if isinstance(resp.parsed, LLMAnswer) else None
    if parsed is None:
        try:
            parsed = LLMAnswer.model_validate_json(resp.text or "")
        except Exception:
            parsed = LLMAnswer(found=False, answer="The model did not return a valid answer. Please retry.",
                               cited_sources=[], confidence=0.0)

    by_sid = {u["sid"]: u for u in units}
    cited = [s.strip("[] ") for s in parsed.cited_sources if s.strip("[] ") in by_sid]
    if not parsed.found:
        cited = []
    pages = sorted({pg for s in cited for pg in by_sid[s]["pages"]})

    return {
        "answer": parsed.answer,
        "found": parsed.found,
        "confidence_score": round(max(0.0, min(1.0, parsed.confidence)) if parsed.found else 0.0, 2),
        "source_pages": pages,
        "cited_sources": cited,
        "sources": [{**s, "cited": s["sid"] in cited} for s in sources],
        "token_count": token_count,
        "timings": {"retrieval_ms": retrieval_ms, "llm_ms": llm_ms},
    }
