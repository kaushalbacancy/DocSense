"""Run the offline pipeline for one or all documents.

    python -m ingestion.ingest --doc all
    python -m ingestion.ingest --doc imac --reparse     # ignore the cached parse

Parsing is the slow / paid step, so its output is cached in data/parsed/<doc>.json
(the same idea as output_v2.json in the Medicare project). Chunking and indexing are cheap to redo.
"""
import argparse
import json
import time
from collections import Counter

from config import CHUNKS_DIR, DATA_DIR, DOCUMENTS, PARSED_DIR
from ingestion.chunkers import chunk
import pymupdf

from ingestion.parsers import parse, repair_pages, verify_footnotes
from retrieval import store


def ingest(doc_id: str, reparse: bool = False, progress=print, step=None) -> dict:
    doc = DOCUMENTS[doc_id]
    pdf = DATA_DIR / doc["file"]
    if not pdf.exists():
        raise FileNotFoundError(f"Put the PDF at {pdf}")
    PARSED_DIR.mkdir(parents=True, exist_ok=True)
    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
    started = time.time()
    step = step or (lambda stage, done, total: None)

    parsed_path = PARSED_DIR / f"{doc_id}.json"
    if parsed_path.exists() and not reparse:
        progress(f"[{doc_id}] using cached parse {parsed_path.name}")
        parsed = json.loads(parsed_path.read_text())
        if parsed.get("parser") in ("pymupdf", "gemini_vision"):
            progress(f"[{doc_id}] running parse quality check on cached parse")
            step("Checking parse quality", 0, 1)
            parsed = repair_pages(pdf, parsed, step)
            parsed_path.write_text(json.dumps(parsed, indent=2, ensure_ascii=False))
        # cheap and idempotent: re-verify vision footnotes in parses cached before the check existed
        pdf_doc = pymupdf.open(pdf)
        for p in parsed["pages"]:
            if p.get("source") == "vision" or parsed.get("parser") == "gemini_vision":
                p["md"] = verify_footnotes(p["md"], pdf_doc[p["page"] - 1].get_text())
    else:
        progress(f"[{doc_id}] parsing with {doc['parser']} (this is the slow step)")
        parsed = parse(pdf, doc["parser"], step)
        parsed_path.write_text(json.dumps(parsed, indent=2, ensure_ascii=False))

    progress(f"[{doc_id}] chunking with '{doc['profile']}' strategy")
    step("Splitting into chunks", 0, 1)
    result = chunk(doc_id, doc["profile"], parsed)
    chunks, parents = result["chunks"], result["parents"]
    stats = {
        "pages": len(parsed["pages"]),
        "parser": parsed.get("parser", doc["parser"]),
        "profile": doc["profile"],
        "chunks": len(chunks),
        "parents": len(parents),
        "types": dict(Counter(c["type"] for c in chunks)),
        "avg_words": round(sum(len(c["content"].split()) for c in chunks) / max(len(chunks), 1)),
    }
    (CHUNKS_DIR / f"{doc_id}.json").write_text(
        json.dumps({"stats": stats, "chunks": chunks, "parents": parents}, indent=2, ensure_ascii=False))
    store.parents.cache_clear()

    progress(f"[{doc_id}] embedding + indexing {len(chunks)} chunks into Qdrant")
    store.index_chunks(doc_id, chunks, step=step)
    stats["seconds"] = round(time.time() - started, 1)
    progress(f"[{doc_id}] done: {stats}")
    return stats


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--doc", default="all", help=f"one of {list(DOCUMENTS)} or 'all'")
    ap.add_argument("--reparse", action="store_true")
    args = ap.parse_args()
    for d in (DOCUMENTS if args.doc == "all" else [args.doc]):
        try:
            ingest(d, reparse=args.reparse)
        except FileNotFoundError as e:
            print(f"SKIP {d}: {e}")
