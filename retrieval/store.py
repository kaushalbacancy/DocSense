"""Qdrant: one collection for all documents, each point tagged with doc_id.

Every point has two vectors (dense meaning + BM25 keywords) and the chunk as payload.
Point ids are derived from chunk_id, so re-ingesting a document overwrites instead of duplicating.
"""
import json
import uuid
from functools import lru_cache

from qdrant_client import QdrantClient, models

from config import CHUNKS_DIR, settings
from retrieval.models import dense_size, embed_passages


@lru_cache
def client() -> QdrantClient:
    return QdrantClient(url=settings.qdrant_url, timeout=60, check_compatibility=False)


def ensure_collection():
    c = client()
    if c.collection_exists(settings.collection_name):
        return
    c.create_collection(
        collection_name=settings.collection_name,
        vectors_config={"dense": models.VectorParams(size=dense_size(), distance=models.Distance.COSINE)},
        sparse_vectors_config={"bm25": models.SparseVectorParams(modifier=models.Modifier.IDF)},
    )
    c.create_payload_index(settings.collection_name, "doc_id", models.PayloadSchemaType.KEYWORD)
    c.create_payload_index(settings.collection_name, "type", models.PayloadSchemaType.KEYWORD)


def doc_filter(doc_id: str) -> models.Filter:
    return models.Filter(must=[models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id))])


def index_chunks(doc_id: str, chunks: list[dict], batch: int = 64, step=None):
    ensure_collection()
    c = client()
    c.delete(settings.collection_name, points_selector=models.FilterSelector(filter=doc_filter(doc_id)))
    for start in range(0, len(chunks), batch):
        part = chunks[start:start + batch]
        dense, sparse = embed_passages([ch["embed_text"] for ch in part])
        c.upsert(settings.collection_name, points=[
            models.PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, ch["chunk_id"])),
                vector={"dense": d, "bm25": s},
                payload={k: v for k, v in ch.items() if k != "embed_text"},
            )
            for ch, d, s in zip(part, dense, sparse)
        ])
        if step:
            step("Embedding and saving to the vector DB", min(start + batch, len(chunks)), len(chunks))


def count(doc_id: str) -> int:
    try:
        if not client().collection_exists(settings.collection_name):
            return 0
        return client().count(settings.collection_name, count_filter=doc_filter(doc_id), exact=True).count
    except Exception:
        return 0


@lru_cache(maxsize=8)
def parents(doc_id: str) -> dict[str, dict]:
    """Parent sections (hierarchical docs) live next to the chunks file, keyed by parent_id."""
    path = CHUNKS_DIR / f"{doc_id}.json"
    if not path.exists():
        return {}
    return {p["parent_id"]: p for p in json.loads(path.read_text()).get("parents", [])}
