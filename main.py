import json
import threading
import time

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from config import BASE_DIR, CHUNKS_DIR, DATA_DIR, DOCUMENTS
from evaluation import evaluate
from ingestion.ingest import ingest
from rag_service import answer
from retrieval import store

app = FastAPI(title="DocSense API", version="2.0.0",
              description="One RAG pipeline, three document-specific strategies.")


@app.on_event("startup")
def warm_up():
    # load embedding, BM25 and reranker models now, so the first question doesn't pay ~20s of loading
    from retrieval.models import dense_model, reranker, sparse_model
    dense_model(), sparse_model(), reranker()
    print("INFO: models loaded, ready for questions")


app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")

jobs: dict[str, dict] = {d: {"state": "idle", "log": []} for d in DOCUMENTS}
jobs_lock = threading.Lock()


class QueryRequest(BaseModel):
    question: str = Field(min_length=2, max_length=1000)
    doc_id: str


def _check_doc(doc_id: str):
    if doc_id not in DOCUMENTS:
        raise HTTPException(404, f"Unknown document '{doc_id}'. Use one of {list(DOCUMENTS)}")


@app.get("/")
def index():
    return FileResponse(BASE_DIR / "static" / "index.html")


@app.get("/api/documents")
def documents():
    out = []
    for doc_id, d in DOCUMENTS.items():
        chunks_file = CHUNKS_DIR / f"{doc_id}.json"
        stats = json.loads(chunks_file.read_text())["stats"] if chunks_file.exists() else None
        out.append({
            "id": doc_id, **d,
            "file_present": (DATA_DIR / d["file"]).exists(),
            "indexed_chunks": store.count(doc_id),
            "stats": stats,
            "job": jobs[doc_id],
        })
    return out


# Plain `def` (not async): embedding, reranking and Gemini calls are blocking,
# so FastAPI runs them in its thread pool instead of freezing the event loop.
@app.post("/api/query")
def query(req: QueryRequest):
    _check_doc(req.doc_id)
    if jobs[req.doc_id]["state"] == "running":
        raise HTTPException(409, "This document is being updated right now. Try again when it shows Ready.")
    try:
        return answer(req.question.strip(), req.doc_id)
    except Exception as e:
        print(f"ERROR: query failed: {e!r}")
        if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
            raise HTTPException(429, "Gemini free-tier limit reached (requests per minute). Wait ~1 minute and ask again.")
        raise HTTPException(500, "Something went wrong while answering. Check the server log.")


@app.post("/api/documents/{doc_id}/upload")
def upload(doc_id: str, file: UploadFile = File(...)):
    _check_doc(doc_id)
    if jobs[doc_id]["state"] == "running":
        raise HTTPException(409, "Wait until the current processing finishes.")
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(400, "Please upload a PDF")
    DATA_DIR.mkdir(exist_ok=True)
    data = file.file.read()
    if not data.startswith(b"%PDF"):
        raise HTTPException(400, "That file isn't a valid PDF.")
    (DATA_DIR / DOCUMENTS[doc_id]["file"]).write_bytes(data)
    return {"ok": True}


def _friendly_error(e: Exception) -> str:
    text = str(e)
    if "429" in text or "RESOURCE_EXHAUSTED" in text:
        return "Gemini free-tier limit reached. Wait a minute and try again."
    if "503" in text or "UNAVAILABLE" in text:
        return "Gemini is overloaded right now. Try again in a few minutes."
    if "Connection refused" in text or "qdrant" in text.lower():
        return "Can't reach the vector database. Is Qdrant running? (docker compose up -d)"
    if isinstance(e, FileNotFoundError):
        return "The PDF file is missing. Upload it first."
    return f"Unexpected error: {text[:200]}"


@app.post("/api/documents/{doc_id}/ingest")
def start_ingest(doc_id: str, reparse: bool = False):
    _check_doc(doc_id)
    with jobs_lock:
        if jobs[doc_id]["state"] == "running":
            raise HTTPException(409, "This document is already being processed.")
        job = jobs[doc_id] = {"state": "running", "action": "reparse" if reparse else "reindex", "log": [],
                              "stage": "Starting", "done": 0, "total": 0,
                              "started": time.time(), "stage_started": time.time()}

    def log(msg):
        print(f"INFO: {msg}")
        job["log"] = (job["log"] + [msg])[-20:]

    def step(stage, done, total):
        if stage != job["stage"]:
            job["stage_started"] = time.time()
        job.update(stage=stage, done=done, total=total)

    def run():
        try:
            job["stats"] = ingest(doc_id, reparse=reparse, progress=log, step=step)
            job["state"] = "done"
        except Exception as e:
            log(f"ERROR: {e!r}")
            job["error"] = _friendly_error(e)
            job["state"] = "error"
        finally:
            job["finished"] = time.time()

    threading.Thread(target=run, daemon=True).start()
    return job


@app.get("/api/eval/questions")
def eval_questions():
    q = evaluate.load_questions()
    return {d: q.get(d, []) for d in DOCUMENTS}


@app.post("/api/eval/{doc_id}")
def run_eval(doc_id: str):
    _check_doc(doc_id)
    return evaluate.run(doc_id)
