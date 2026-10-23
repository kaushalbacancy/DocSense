"""Question -> best context.

1. Hybrid search inside one document: dense (meaning) + BM25 (exact words), merged with RRF.
2. Rerank the ~20 candidates with a cross-encoder, which reads question and chunk together
   and is much more accurate than vector similarity alone. Keep the best few.
3. Hierarchical docs: swap each small child chunk for its parent section (small-to-big),
   so the LLM sees the whole surrounding argument, not a fragment.
4. Respect a context budget so long documents never blow up the prompt.
"""
import math

from qdrant_client import models

from config import DOCUMENTS, settings
from retrieval import store
from retrieval.models import embed_query, rerank


def _sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))


def hybrid_candidates(question: str, doc_id: str) -> list[models.ScoredPoint]:
    dense, sparse = embed_query(question)
    flt = store.doc_filter(doc_id)
    return store.client().query_points(
        collection_name=settings.collection_name,
        prefetch=[
            models.Prefetch(query=dense, using="dense", filter=flt, limit=settings.prefetch_limit),
            models.Prefetch(query=sparse, using="bm25", filter=flt, limit=settings.prefetch_limit),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=settings.fusion_limit,
        with_payload=True,
    ).points


def retrieve(question: str, doc_id: str, top_k: int | None = None) -> list[dict]:
    top_k = top_k or settings.final_top_k
    candidates = hybrid_candidates(question, doc_id)
    if not candidates:
        return []

    scores = rerank(question, [p.payload["content"] for p in candidates])
    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)

    hierarchical = DOCUMENTS[doc_id]["profile"] == "hierarchical"
    parent_map = store.parents(doc_id) if hierarchical else {}

    units, seen_parents, budget = [], set(), settings.max_context_chars
    for point, score in ranked:
        if len(units) >= top_k:
            break
        p = point.payload
        unit = {
            "chunk_id": p["chunk_id"],
            "score": round(_sigmoid(score), 3),
            "page": p["page"],
            "pages": p.get("pages", [p["page"]]),
            "section": " > ".join(p.get("section_path") or []),
            "type": p.get("type", "text"),
            "matched_text": p["content"],
            "context": p["content"],
        }
        parent = parent_map.get(p.get("parent_id"))
        if parent:
            if parent["parent_id"] in seen_parents:  # two children of one section -> send the section once
                continue
            seen_parents.add(parent["parent_id"])
            unit["context"] = parent["content"]
            unit["pages"] = parent["pages"]
        if len(unit["context"]) > budget:
            if units:
                break
            unit["context"] = unit["context"][:budget]
        budget -= len(unit["context"])
        units.append(unit)

    for i, u in enumerate(units, 1):
        u["sid"] = f"S{i}"
    return units
