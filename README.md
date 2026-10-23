# DocSense: one pipeline, three document strategies

DocSense is a question-answering system for three very different PDFs. All three use the same overall pipeline, but each document gets its own **parsing, chunking and retrieval strategy**. Answers come only from the document, with page citations, and the system says "not found" when the document doesn't contain the answer.

| Document | File (`data/`) | Structure | Strategy |
|---|---|---|---|
| GPO Annual Report FY2018 (82 pages) | `GPO-Anual-report.pdf` | Narrative text plus financial statements | **narrative**: paragraph-aware chunks of about 350 words with a 60-word overlap; tables kept whole |
| World Bank Global Economic Prospects, June 2026 (200 pages) | `GEP-Jun-2026.pdf` | Chapters, boxes, regional sections | **hierarchical**: search small chunks, then send the LLM their whole parent section; section paths come from the PDF outline |
| Apple iMac Environmental Report (12 pages) | `iMac_PER_Oct2024.pdf` | Charts, tables, endnotes | **structured**: every page read by Gemini Vision; tables and charts become their own chunks; endnotes are linked to the text that cites them |

---

## 1. Quick start

### Requirements
- Python 3.12
- Docker Desktop (running)
- A Google Gemini API key ([aistudio.google.com](https://aistudio.google.com))

### First-time setup
```bash
cd DocSense

python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env        # then open .env and set GOOGLE_API_KEY
```
`requirements.txt` pins `cryptography<49`. Newer versions have no prebuilt wheel for Intel Macs, so pip would try to compile them, and that fails without OpenSSL and `pkg-config`.

### Start
```bash
cd DocSense
source .venv/bin/activate

docker compose up -d                 # 1. vector database (Qdrant) on port 6333
uvicorn main:app --port 8000         # 2. web app
```
Wait for `INFO: models loaded, ready for questions`, then open **http://localhost:8000**.

If the documents are already indexed in your local Qdrant, you can start asking questions right away. On a fresh clone, see **Fresh machine or empty database** below.

### Stop
```bash
# in the uvicorn terminal: Ctrl + C
docker compose down                  # stops Qdrant; the indexed data is kept
```

### Fresh machine or empty database
If Qdrant has no data, for example on a new machine, index the documents once:
```bash
python -m ingestion.ingest --doc all
```
Or use **Documents → Process** in the UI.

`data/parsed/` and `data/chunks/` are git-ignored, so a fresh clone has no saved parses. The first run parses every PDF from scratch, which calls Gemini Vision and takes several minutes. Later runs reuse the saved parses.

---

## 2. Using the app

### Chat tab
1. **Pick a document** from the 3 cards at the top.
2. Type a question, or click a suggested question.
3. Every answer shows:
   - **[S1], [S2] citations**: click one to jump to the source chunk.
   - **🎯 Confidence**: how fully the sources support the answer.
   - **📄 Pages** and **📂 Section**: where the answer comes from. These come from the retrieved sources, never from the model's own guess.
   - **📊 / 📈 / 🔖**: shown when the answer came from a table, chart or footnote.
   - **Tokens and time**: search time on your Mac vs. Gemini time.
   - **Show sources**: the exact chunks that were retrieved and cited.
4. If the document doesn't contain the answer, you get **"I couldn't find this in the document."** with 0% confidence.

### Documents tab
You only need this tab when a PDF or the processing code changes.

| Button | What it does | Time | Use it when |
|---|---|---|---|
| **Replace PDF** | Uploads a new PDF for that document, then re-parses it automatically | minutes | A new or corrected version of the report |
| **Re-index** | Re-chunks and re-embeds the **saved** parse. No AI calls. | ~10–100 s | You changed chunking code or chunk sizes, or the database is empty |
| **Re-parse PDF** | Reads the whole PDF again (AI vision for tables, charts and scanned pages), then re-indexes | several minutes | You replaced the PDF, or changed the parser or Vision prompt |

While a job runs, the card shows the **current stage, a progress bar, pages done / total and time remaining**. Questions about that document are paused until it shows **Ready**; the other documents keep working.

---

## 3. How it works

```
OFFLINE (once per document)
PDF ─▶ Parse ─▶ Quality check ─▶ Chunk ─▶ Embed ─▶ Qdrant
        │            │              └ per-document strategy
        │            └ pages that lost text (complex tables) or have no text layer (scans)
        │              are re-read with Gemini Vision; invented footnotes are removed
        └ PyMuPDF (text + tables, 4 parallel processes); iMac: Gemini Vision for every page

ONLINE (every question)
Question ─▶ hybrid search inside the chosen document (dense meaning + BM25 keywords, merged with RRF)
         ─▶ cross-encoder reranker: top 20 → best 6
         ─▶ World Bank only: small chunk → its full parent section
         ─▶ relevance gate: nothing relevant → "not found" without calling the LLM
         ─▶ Gemini answers ONLY from the sources, citing [S1], [S2]…
         ─▶ pages and sections taken from the cited sources
```

### Why each piece exists
| Piece | Problem it solves |
|---|---|
| **Hybrid search** | Dense vectors match meaning ("emissions" ≈ "carbon footprint"). BM25 matches exact terms ("FY 2018", "Box 1.2", "kg CO2e"). |
| **Reranker** | Vector search compares two separate embeddings. A cross-encoder reads the question and the chunk together, so it ranks far more accurately. |
| **Small-to-big chunking** (World Bank) | Small chunks give precise matches; the parent section gives the LLM the full context. |
| **Tables kept whole** | A row without its header is meaningless. For embedding, tables are rewritten as `Year: 2023; Value: 397` sentences. |
| **Charts → data** (Apple) | Gemini Vision turns each chart into a description plus a values table; estimated values are marked `~`. |
| **Endnote linking** (Apple) | "Reduced emissions by 20%⁷" travels together with endnote 7, which explains what the 20% is measured against. |
| **Parse quality check** | Plain text extraction silently dropped up to 99% of some table pages (World Bank forecasts, GPO balance sheets). Each page is compared with the PDF's own text layer, and bad pages are re-read with Vision. |
| **Footnote verification** | Gemini Vision sometimes *invents* footnotes. Each one is checked word-by-word against the PDF text, and invented ones are dropped. |
| **Text normalization** | The World Bank PDF stores "fi" as a single ligature glyph and drops "Th" ("Te economy"), which breaks keyword search; both are fixed. |

---

## 4. Qdrant (vector database)

Qdrant runs in Docker (`docker-compose.yaml`, image `qdrant/qdrant:v1.15.4`, container `advanced-rag-qdrant`).

| | |
|---|---|
| REST API | http://localhost:6333 |
| **Dashboard** (browse collections and points) | **http://localhost:6333/dashboard** |
| gRPC | port 6334 |
| Data on disk | Docker volume `advanced-rag_qdrant-storage` (survives `docker compose down` and restarts) |

### What is stored
One collection, **`advanced_rag`**, holds all 3 documents (about 980 points). Each point is one chunk:

| Part | Content |
|---|---|
| Vector `dense` | 384-dim `bge-small-en-v1.5` embedding, cosine distance |
| Sparse vector `bm25` | BM25 keyword weights (IDF modifier on) |
| Payload | `doc_id`, `chunk_id`, `order`, `page`, `pages`, `section_path`, `type` (text / table / chart / footnote), `content`, `title`, `parent_id`, `footnotes` |
| Payload indexes | `doc_id` and `type` (keyword), used to filter searches to one document |

- Point IDs are derived from `chunk_id` (UUID5), so re-indexing a document **replaces** its points instead of duplicating them.
- Each search is filtered by `doc_id`, so one document's answers never use another document's chunks.
- **Parent sections** (World Bank) are not in Qdrant. They're in `data/chunks/worldbank.json`, looked up by `parent_id` at question time.

### Useful commands
```bash
# is it running? how many points per document?
curl -s localhost:6333/collections/advanced_rag | python3 -m json.tool | grep points_count
curl -s -X POST localhost:6333/collections/advanced_rag/points/count \
  -H 'Content-Type: application/json' \
  -d '{"filter":{"must":[{"key":"doc_id","match":{"value":"worldbank"}}]},"exact":true}'

# logs
docker logs advanced-rag-qdrant --tail 50
```

### Backup and share (snapshot)
A snapshot lets someone else use the indexed data **without re-processing the PDFs**, the same idea as `qudrent-snapshot/` in the Medicare project.
```bash
# create a snapshot (file saved inside the container)
curl -s -X POST localhost:6333/collections/advanced_rag/snapshots
# list snapshots and download one
curl -s localhost:6333/collections/advanced_rag/snapshots
curl -o advanced_rag.snapshot localhost:6333/collections/advanced_rag/snapshots/<snapshot-name>
```
To restore: open the **Dashboard → Collections → Upload snapshot**, choose the file and name the collection `advanced_rag`.
Also copy `data/chunks/` and `data/parsed/` to the other machine, since the World Bank parent sections live there.

### Reset
```bash
curl -X DELETE localhost:6333/collections/advanced_rag     # delete the collection only
python -m ingestion.ingest --doc all                       # rebuild from the saved parses (no AI calls)

docker compose down -v                                     # ⚠️ deletes the volume and all Qdrant data
```

---

## 5. Configuration (`.env`)

| Key | Default | Purpose |
|---|---|---|
| `GOOGLE_API_KEY` | (required) | Gemini |
| `GEMINI_MODEL` | `gemini-flash-lite-latest` | Writes answers (fast) |
| `VISION_MODEL` | `gemini-flash-latest` | Reads charts and tables from page images; falls back to `GEMINI_MODEL` when busy or rate-limited |
| `QDRANT_URL` | `http://localhost:6333` | Vector database |
| `COLLECTION_NAME` | `advanced_rag` | One collection; points are tagged with `doc_id` |
| `LLAMA_CLOUD_API_KEY` | (none) | Optional, only if a document's parser is set to `llamaparse` |

Retrieval settings (`config.py`): `prefetch_limit=30`, `fusion_limit=20`, `final_top_k=6`, `max_context_chars=24000`, `min_relevance=0.02`.
The document list and each document's strategy are in `DOCUMENTS` in `config.py`.

Local models (CPU, no API key; downloaded on first run):
- Dense: `BAAI/bge-small-en-v1.5`
- Sparse: `Qdrant/bm25`
- Reranker: `Xenova/ms-marco-MiniLM-L-6-v2`

---

## 6. Command line

```bash
python -m ingestion.ingest --doc all              # process all documents (reuses saved parses)
python -m ingestion.ingest --doc worldbank        # one document: gpo | worldbank | imac
python -m ingestion.ingest --doc imac --reparse   # parse the PDF again from scratch
python -m evaluation.evaluate --doc all           # run the test questions (LLM judge)
```

**Evaluation** (command line only): `evaluation/questions.json` holds 33 test questions with expected answers and pages. Results are saved to `evaluation/results/`. The last run scored correctness of GPO 100%, World Bank 96% and iMac 98%, with 100% faithfulness and 100% page hit rate.

---

## 7. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Answers take 10–15 s | Gemini time (shown as `LLM …ms` on each answer). Search is only about 1.5 s. Free-tier Gemini is slower and varies call to call. | Use a paid or billing-enabled key. Fewer sources (`final_top_k`) also help. |
| "Gemini free-tier limit reached" | Too many requests per minute (429) | Wait about 1 minute and ask again |
| Page loads but shows no documents | Server not running, or a JavaScript error | Check that `uvicorn` is running; open the browser console (Cmd + Option + J) |
| "Can't reach the vector database" | Qdrant isn't running | `docker compose up -d` |
| Documents show "Not processed yet" | Empty Qdrant (new machine, volume deleted) | Documents → **Process**, or `python -m ingestion.ingest --doc all` |
| First start is slow | Local models download once (~200 MB) | Wait; later starts are fast |
| Qdrant version warning | Client 1.19 vs server 1.15 | Harmless; the features used here work on both |

---

## 8. Project layout

```
DocSense/
├── config.py                 settings + DOCUMENTS registry (strategy per document)
├── main.py                   FastAPI: chat, documents, upload, ingest jobs with progress
├── rag_service.py            prompt, Gemini call (fallback + rate-limit handling), grounded answer
├── ingestion/
│   ├── parsers.py            PDF → page-wise markdown; quality check; Vision; footnote verification
│   ├── chunkers.py           narrative / hierarchical / structured chunking, TOC section paths
│   └── ingest.py             parse → chunk → embed → Qdrant (caches parses)
├── retrieval/
│   ├── models.py             dense, BM25, reranker (FastEmbed, local CPU)
│   ├── store.py              Qdrant collection, indexing, parent-section store
│   └── search.py             hybrid search → rerank → parent expansion → context budget
├── evaluation/
│   ├── questions.json        33 test questions with expected answers and pages
│   ├── evaluate.py           LLM-judge scoring (correctness, faithfulness, page hits)
│   └── results/              evaluation run outputs (git-ignored)
├── static/index.html         UI (Chat, Documents)
├── data/
│   ├── *.pdf                 the 3 source documents
│   ├── parsed/<doc>.json     saved parse (slow step, cached; git-ignored)
│   └── chunks/<doc>.json     chunks + parent sections + stats (git-ignored)
├── docker-compose.yaml       Qdrant v1.15.4
├── requirements.txt
├── .gitignore
└── .env / .env.example       .env is git-ignored; copy it from .env.example
```
`gpo`, `worldbank` and `imac` are the document IDs used in file names, in `--doc` and in the `doc_id` payload field.

---

## 9. Security note
- `.env` holds your Gemini key and is git-ignored. **Never commit it.**
- The key currently in `.env` was copied from the Medicare project, and that key is exposed in the Medicare notebook (`Chunk.ipynb`). **Revoke it in Google AI Studio and put a new key in `.env`.**
