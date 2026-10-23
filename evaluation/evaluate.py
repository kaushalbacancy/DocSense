"""Run the sample questions and score the answers.

    python -m evaluation.evaluate --doc all

For each question we record:
- correctness  : does the answer match the expected answer? (LLM judge, 0-1). Skipped if no expected answer.
- faithfulness : is every claim backed by the retrieved sources? (LLM judge, 0-1) -> hallucination check
- page_hit     : did we cite the expected page? (only if expected_pages is given)
- latency and tokens
"""
import argparse
import json
import time

from google.genai import types
from pydantic import BaseModel

from config import BASE_DIR, DOCUMENTS
from rag_service import answer, generate

QUESTIONS_FILE = BASE_DIR / "evaluation" / "questions.json"
RESULTS_DIR = BASE_DIR / "evaluation" / "results"


class Judgement(BaseModel):
    correctness: float
    faithfulness: float
    reason: str


def load_questions() -> dict[str, list[dict]]:
    if not QUESTIONS_FILE.exists():
        return {}
    return json.loads(QUESTIONS_FILE.read_text())


def judge(question: str, expected: str | None, result: dict) -> Judgement:
    # the judge must see exactly what the answering model saw (for hierarchical docs: the parent sections)
    sources = "\n\n".join(f"[{s['sid']}] {s.get('context', s['text'])}" for s in result["sources"])
    prompt = f"""You are grading a RAG system's answer.

QUESTION: {question}
EXPECTED ANSWER: {expected or "(not provided - set correctness to -1)"}
SYSTEM ANSWER: {result['answer']}
RETRIEVED SOURCES:
{sources}

Score:
- correctness (0-1): does the system answer contain the key facts of the expected answer? Wording may differ.
  If no expected answer is provided, use -1.
- faithfulness (0-1): fraction of the system answer's claims that are supported by the retrieved sources.
  A correct "not found in the document" answer counts as fully faithful (1).
- reason: one short sentence."""
    resp = generate(prompt, types.GenerateContentConfig(temperature=0, response_mime_type="application/json",
                                                        response_schema=Judgement))
    return resp.parsed if isinstance(resp.parsed, Judgement) else Judgement.model_validate_json(resp.text)


def evaluate_one(doc_id: str, q: dict) -> dict:
    t = time.time()
    try:
        result = answer(q["question"], doc_id, include_context=True)
        j = judge(q["question"], q.get("expected"), result)
        error = None
    except Exception as e:
        result, j, error = {"answer": "", "source_pages": [], "token_count": 0, "found": False}, None, str(e)
    expected_pages = q.get("expected_pages") or []
    return {
        "kind": q.get("kind", "fact"),
        "question": q["question"],
        "expected": q.get("expected"),
        "answer": result["answer"],
        "found": result.get("found"),
        "source_pages": result["source_pages"],
        "page_hit": (bool(set(expected_pages) & set(result["source_pages"])) if expected_pages else None),
        "correctness": (j.correctness if j and j.correctness >= 0 else None),
        "faithfulness": j.faithfulness if j else None,
        "reason": j.reason if j else error,
        "tokens": result["token_count"],
        "seconds": round(time.time() - t, 1),
    }


def _avg(values):
    values = [v for v in values if v is not None]
    return round(sum(values) / len(values), 3) if values else None


def run(doc_id: str) -> dict:
    questions = load_questions().get(doc_id, [])
    # sequential on purpose: the Gemini free tier allows only a few requests per minute
    rows = [evaluate_one(doc_id, q) for q in questions]
    summary = {
        "doc_id": doc_id,
        "questions": len(rows),
        "correctness": _avg([r["correctness"] for r in rows]),
        "faithfulness": _avg([r["faithfulness"] for r in rows]),
        "page_hit_rate": _avg([float(r["page_hit"]) for r in rows if r["page_hit"] is not None]),
        "avg_seconds": _avg([r["seconds"] for r in rows]),
        "avg_tokens": _avg([r["tokens"] for r in rows]),
        "by_kind": {k: _avg([r["correctness"] for r in rows if r["kind"] == k])
                    for k in dict.fromkeys(r["kind"] for r in rows)},
    }
    out = {"summary": summary, "rows": rows}
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / f"{doc_id}-{time.strftime('%Y%m%d-%H%M%S')}.json").write_text(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--doc", default="all")
    args = ap.parse_args()
    for d in (DOCUMENTS if args.doc == "all" else [args.doc]):
        res = run(d)
        print(json.dumps(res["summary"], indent=2))
        for r in res["rows"]:
            print(f"  - [{r['kind']}] {r['question'][:70]:70} corr={r['correctness']} faith={r['faithfulness']} "
                  f"pages={r['source_pages']} hit={r['page_hit']}")
