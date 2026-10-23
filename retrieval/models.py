"""Local models (all run on CPU through FastEmbed / ONNX, no API key needed). Loaded once, lazily."""
from functools import lru_cache

from fastembed import SparseTextEmbedding, TextEmbedding
from fastembed.rerank.cross_encoder import TextCrossEncoder
from qdrant_client import models

from config import settings


@lru_cache
def dense_model() -> TextEmbedding:
    print(f"INFO: loading dense model {settings.dense_model}")
    return TextEmbedding(settings.dense_model)


@lru_cache
def sparse_model() -> SparseTextEmbedding:
    print(f"INFO: loading sparse model {settings.sparse_model}")
    return SparseTextEmbedding(settings.sparse_model)


@lru_cache
def reranker() -> TextCrossEncoder:
    print(f"INFO: loading reranker {settings.rerank_model}")
    return TextCrossEncoder(settings.rerank_model)


@lru_cache
def dense_size() -> int:
    return len(next(iter(dense_model().embed(["size probe"]))))


def embed_passages(texts: list[str]) -> tuple[list[list[float]], list[models.SparseVector]]:
    dense = [v.tolist() for v in dense_model().passage_embed(texts, batch_size=32)]
    sparse = [models.SparseVector(indices=s.indices.tolist(), values=s.values.tolist())
              for s in sparse_model().passage_embed(texts, batch_size=32)]
    return dense, sparse


def embed_query(text: str) -> tuple[list[float], models.SparseVector]:
    # query_embed adds the model's query instruction (bge needs it) and BM25 query weighting
    dense = next(iter(dense_model().query_embed(text))).tolist()
    s = next(iter(sparse_model().query_embed(text)))
    return dense, models.SparseVector(indices=s.indices.tolist(), values=s.values.tolist())


def rerank(query: str, documents: list[str]) -> list[float]:
    return list(reranker().rerank(query, documents))
