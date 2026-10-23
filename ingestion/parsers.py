"""PDF -> page-wise markdown.

Every parser returns the same shape so the chunkers don't care where the text came from:
    {"pages": [{"page": 1, "md": "..."}], "toc": [[level, title, page], ...]}

- pymupdf       : fast local text + table extraction. Good for narrative and long reports.
- gemini_vision : renders each page as an image and lets Gemini transcribe it, including
                  tables, charts (read into data tables) and footnotes. Good for visual docs.
- llamaparse    : LlamaParse premium (same as the Medicare project). Needs LLAMA_CLOUD_API_KEY.
"""
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pymupdf

from config import settings

VISION_PROMPT = """You are converting one page of a PDF report into clean Markdown for a search index.

Rules:
1. Transcribe ALL meaningful text. Use #, ##, ### for headings exactly as they appear in the visual hierarchy.
2. Every table -> wrap it like this:
<table title="table title or a short description">
| col | col |
|---|---|
| ... | ... |
</table>
3. Every chart, graph, diagram or infographic -> wrap it like this:
<chart title="chart title or a short description">
One or two sentences saying what the chart shows and the main takeaway.
| label | value |
|---|---|
| ... | ... |
</chart>
   Read the values from labels on the chart. If you have to estimate a value from the axis, prefix it with "~".
4. Footnote/endnote markers (small superscript numbers or symbols like *, †) -> write them inline as [^1], [^2], ...
   Write every footnote found on the page at the very end as:
[^1]: footnote text
   If the page is an "Endnotes"/"Notes" page, write EVERY numbered note in that same [^n]: format.
   Only write a [^n]: line if that note's text is actually printed on THIS page. Never invent or summarize
   note text; a marker whose note is printed elsewhere stays as an inline [^n] only.
5. Skip page headers, page numbers, running footers and purely decorative images.
6. Output only the Markdown. No code fences, no commentary."""


def _noop(stage: str, done: int, total: int):
    pass


def _md_batch(path: str, page_numbers: list[int]) -> list[dict]:
    """Runs in a worker process: page-wise markdown for a slice of the PDF."""
    import pymupdf4llm

    doc = pymupdf.open(path)
    try:
        items = pymupdf4llm.to_markdown(doc, pages=[n - 1 for n in page_numbers], page_chunks=True, show_progress=False)
        return [{"page": n, "md": it.get("text", "")} for n, it in zip(page_numbers, items)]
    except Exception as e:  # pymupdf4llm can choke on odd layouts; plain text is a safe fallback
        print(f"WARNING: pymupdf4llm failed on pages {page_numbers[0]}-{page_numbers[-1]} ({e}); using plain text")
        return [{"page": n, "md": doc[n - 1].get_text("text")} for n in page_numbers]


def parse_pymupdf(path: Path, step=_noop) -> dict:
    from concurrent.futures import ProcessPoolExecutor, as_completed

    doc = pymupdf.open(path)
    toc = doc.get_toc(simple=True)  # [[level, title, page], ...] from the PDF outline
    numbers = list(range(1, doc.page_count + 1))
    batches = [numbers[i:i + 10] for i in range(0, len(numbers), 10)]
    pages, done = [], 0
    step("Reading text and tables", 0, len(numbers))
    # table detection is CPU-heavy, so slices of 10 pages run in parallel processes (~3-4x faster)
    with ProcessPoolExecutor(max_workers=4) as pool:
        for fut in as_completed([pool.submit(_md_batch, str(path), b) for b in batches]):
            result = fut.result()
            pages += result
            done += len(result)
            step("Reading text and tables", done, len(numbers))
    pages.sort(key=lambda p: p["page"])
    return repair_pages(path, {"pages": pages, "toc": toc}, step)


def _word_count(md: str) -> int:
    return len(re.sub(r"[|*#_\-]", " ", md).split())


def repair_pages(path: Path, parsed: dict, step=_noop) -> dict:
    """Parse quality check: compare every page with the PDF's raw text layer and re-read bad pages with vision.

    - no text layer but has images -> scanned page / full-page chart / cover
    - lost >30% of the raw words    -> a complex table the text extractor dropped (forecast tables, balance sheets)
    """
    doc = pymupdf.open(path)
    bad = {}
    for p in parsed["pages"]:
        page = doc[p["page"] - 1]
        raw, got = len(page.get_text().split()), _word_count(p["md"])
        if p.get("source") == "vision" and (raw < 80 or got >= 0.5 * raw):
            continue  # already re-read and complete
        if got < 25 and page.get_images():
            bad[p["page"]] = "no text layer"
        elif raw > 80 and got < 0.7 * raw:
            bad[p["page"]] = f"lost {100 - round(100 * got / raw)}% of text"
    if bad:
        print(f"INFO: quality check -> {len(bad)} pages re-read with Gemini vision: {bad}")
        vision = _vision_pages(doc, list(bad), "Fixing tables and image pages with AI vision", step)
        for p in parsed["pages"]:
            if p["page"] in vision:
                p["md"], p["source"], p["repair_reason"] = vision[p["page"]]["md"], vision[p["page"]]["source"], bad[p["page"]]
    parsed["quality_checked"] = True
    return parsed


def verify_footnotes(md: str, raw_text: str) -> str:
    """Vision models sometimes invent footnote text. If the PDF has a text layer, keep a [^n]: line only
    when most of its words really appear on the page."""
    raw_words = set(re.findall(r"[a-z0-9]+", raw_text.lower()))
    if len(raw_words) < 25:  # scanned page: nothing to verify against
        return md
    kept = []
    for line in md.splitlines():
        note = re.match(r"^\[\^[^\]]+\]:\s*(.*)$", line.strip())
        if note:
            words = re.findall(r"[a-z0-9]+", note.group(1).lower())
            if words and sum(w in raw_words for w in words) / len(words) < 0.8:
                print(f"WARNING: dropped footnote not found in the PDF text: {line.strip()[:80]}")
                continue
        kept.append(line)
    return "\n".join(kept)


def _vision_page(doc: pymupdf.Document, page_no: int) -> dict:
    """One page -> markdown via Gemini vision, checked against the PDF's own text layer:
    incomplete output (e.g. an empty reply) is retried, and if it stays incomplete we keep the plain text
    rather than silently losing the page. Invented footnotes are removed."""
    from google.genai import types
    from rag_service import generate

    page = doc[page_no - 1]
    png, raw = page.get_pixmap(dpi=150).tobytes("png"), page.get_text()
    raw_words = len(raw.split())
    best = ""
    for attempt in range(3):
        try:
            resp = generate([types.Part.from_bytes(data=png, mime_type="image/png"), VISION_PROMPT],
                            types.GenerateContentConfig(temperature=0), model=settings.vision_model)
            md = (resp.text or "").strip()
            if _word_count(md) > _word_count(best):
                best = md
            if raw_words < 80 or _word_count(md) >= 0.5 * raw_words:
                break
            print(f"WARNING: page {page_no} vision output incomplete ({_word_count(md)} of ~{raw_words} words); retrying")
        except Exception as e:
            print(f"WARNING: page {page_no} attempt {attempt + 1} failed ({e}); retrying")
        time.sleep(3 * (attempt + 1))
    if raw_words >= 80 and _word_count(best) < 0.5 * raw_words:
        print(f"WARNING: page {page_no} vision stayed incomplete; keeping the PDF text layer instead")
        return {"page": page_no, "md": raw, "source": "text-fallback"}
    if not best and raw_words < 80:
        best = raw
    return {"page": page_no, "md": verify_footnotes(best, raw), "source": "vision"}


def _vision_pages(doc, page_numbers: list[int], label: str, step=_noop) -> dict[int, dict]:
    from concurrent.futures import as_completed

    out = {}
    step(label, 0, len(page_numbers))
    with ThreadPoolExecutor(max_workers=4) as pool:
        for fut in as_completed([pool.submit(_vision_page, doc, n) for n in page_numbers]):
            v = fut.result()
            out[v["page"]] = v
            step(label, len(out), len(page_numbers))
    return out


def parse_gemini_vision(path: Path, step=_noop) -> dict:
    doc = pymupdf.open(path)
    toc = doc.get_toc(simple=True)
    vision = _vision_pages(doc, list(range(1, doc.page_count + 1)), "Reading pages with AI vision", step)
    return {"pages": [vision[n] for n in sorted(vision)], "toc": toc}


def parse_llamaparse(path: Path, step=_noop) -> dict:
    from llama_parse import LlamaParse  # pip install llama-parse

    parser = LlamaParse(api_key=settings.llama_cloud_api_key, result_type="markdown", premium_mode=True)
    result = parser.get_json_result(str(path))
    pages = [{"page": p["page"], "md": p.get("md", "")} for p in result[0]["pages"]]
    return {"pages": pages, "toc": pymupdf.open(path).get_toc(simple=True)}


PARSERS = {
    "pymupdf": parse_pymupdf,
    "gemini_vision": parse_gemini_vision,
    "llamaparse": parse_llamaparse,
}


def parse(path: Path, parser: str, step=_noop) -> dict:
    if parser == "llamaparse" and not settings.llama_cloud_api_key:
        print("INFO: no LLAMA_CLOUD_API_KEY set, using gemini_vision instead of llamaparse")
        parser = "gemini_vision"
    print(f"INFO: parsing {path.name} with {parser}")
    result = PARSERS[parser](path, step)
    result["parser"] = parser
    return result
