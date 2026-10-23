from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
PARSED_DIR = DATA_DIR / "parsed"
CHUNKS_DIR = DATA_DIR / "chunks"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=BASE_DIR / ".env", extra="ignore")

    google_api_key: str
    gemini_model: str = "gemini-flash-lite-latest"   # answers + evaluation judge (fast)
    vision_model: str = "gemini-flash-latest"        # reading charts/tables from page images (more accurate)
    llama_cloud_api_key: str | None = None

    qdrant_url: str = "http://localhost:6333"
    collection_name: str = "advanced_rag"

    dense_model: str = "BAAI/bge-small-en-v1.5"
    sparse_model: str = "Qdrant/bm25"
    rerank_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"

    prefetch_limit: int = 30      # candidates from each of dense / sparse search
    fusion_limit: int = 20        # candidates kept after RRF, sent to the reranker
    final_top_k: int = 6          # chunks kept after reranking
    max_context_chars: int = 24000
    min_relevance: float = 0.02   # if even the best chunk scores below this, answer "not found" without calling the LLM


settings = Settings()


# Each document gets its own parsing + chunking strategy ("profile").
#   narrative    -> plain text, paragraph-aware recursive chunks with overlap
#   hierarchical -> long report, chapter/section tree, small child chunks + parent sections
#   structured   -> tables / charts / footnotes, each element kept whole and typed
DOCUMENTS = {
    "gpo": {
        "name": "GPO Annual Report",
        "file": "GPO-Anual-report.pdf",
        "profile": "narrative",
        "parser": "pymupdf",
        "description": "FY2018 narrative report + financial statements. Baseline RAG.",
    },
    "worldbank": {
        "name": "World Bank Global Economic Prospects",
        "file": "GEP-Jun-2026.pdf",
        "profile": "hierarchical",
        "parser": "pymupdf",
        "description": "June 2026, 200 pages, chapters & boxes. Hierarchical chunking.",
    },
    "imac": {
        "name": "Apple iMac Environmental Report",
        "file": "iMac_PER_Oct2024.pdf",
        "profile": "structured",
        "parser": "gemini_vision",
        "description": "Charts, tables & footnotes. Element-aware chunking.",
    },
}
