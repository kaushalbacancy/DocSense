"""Page-wise markdown -> chunks. One strategy per document profile.

Step 1 (shared): iter_blocks() walks every page and yields typed blocks
        (paragraph / table / chart / footnote) together with the section path they sit under.
Step 2 (per profile):
    narrative    -> merge paragraphs into ~350-word chunks with overlap
    hierarchical -> parent = a section window (~1200 words), children = ~180-word pieces.
                    We search the small children (precise) but give the LLM the parent (full context).
    structured   -> every table / chart / footnote is its own typed chunk, never split mid-way;
                    footnotes are also attached to the chunks that reference them.
"""
import re
import unicodedata
from collections import Counter, defaultdict

HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
OPEN_TAG_RE = re.compile(r'^<(table|chart)(?:\s+title="([^"]*)")?\s*>$', re.I)
FOOTNOTE_RE = re.compile(r"^\[\^([^\]]+)\]:\s*(.*)$")
FOOTNOTE_REF_RE = re.compile(r"\[\^([^\]]+)\]")
PAGE_NUMBER_RE = re.compile(r"^\d{1,4}$")


# PDFs with broken fonts drop the "Th" ligature ("Te global economy", "Tese economies")
TH_FIX_RE = re.compile(r"\bT(e|is|ese|ey|ere|at|ose|us|eir)\b")


def normalize_text(md: str) -> str:
    """NFKC turns ligatures like 'ﬁ' into 'fi' so BM25 matches 'fiscal'; then repair dropped 'Th'."""
    md = unicodedata.normalize("NFKC", md)
    return TH_FIX_RE.sub(lambda m: "Th" + m.group(1), md)


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower().replace("*", ""))


def _clean_heading(text: str) -> str:
    text = re.sub(r"\s*--\s*", " ", text.replace("*", "").replace("_", " "))
    return re.sub(r"\s+", " ", text).strip(" -")


def _word_count(text: str) -> int:
    return len(text.split())


def _split_sentences(text: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"(])", text) if s.strip()]


# ---------------------------------------------------------------- step 1: blocks
def _running_lines(pages: list[dict]) -> dict[int, set[int]]:
    """Find page headers/footers: lines near the top/bottom of a page that repeat on many pages
    (ignoring page numbers). Returns {page: {line indexes to drop}}."""
    def norm(line):
        return re.sub(r"[\d*#_\s|]", "", line.lower())

    edges, counts = {}, Counter()
    for p in pages:
        lines = (p.get("md") or "").splitlines()
        idx = [i for i, l in enumerate(lines) if l.strip()]
        edge = [(i, norm(lines[i])) for i in idx[:3] + idx[-3:] if not lines[i].strip().startswith("|")]
        edges[p["page"]] = edge
        counts.update({n for _, n in edge if len(n) > 3})
    threshold = max(3, int(0.15 * len(pages)))
    repeated = {n for n, c in counts.items() if c >= threshold}
    return {page: {i for i, n in edge if n in repeated} for page, edge in edges.items()}


def iter_blocks(pages: list[dict], toc: list | None = None):
    """Yield {"page", "path", "kind", "text", "title"?, "marker"?} in reading order.

    Section path comes from the PDF outline (TOC) when it exists, which is far more reliable
    than guessing from font sizes. Markdown headings inside a section are appended below it.
    """
    toc = [t for t in (toc or []) if t[2] > 0]
    use_toc = len(toc) >= 3
    toc_by_page = defaultdict(list)
    for level, title, page in toc:
        # outlines often list boxes and "-- sub-entries" at chapter level; nest them under the chapter
        if "--" in title or re.match(r"(?i)\s*(box|annex)\b", title):
            level += 1
        toc_by_page[page].append((level, _clean_heading(title)))

    running = _running_lines(pages)
    toc_stack: list[tuple[int, str]] = []
    md_stack: list[tuple[int, str]] = []

    def apply_toc(level, title):
        nonlocal toc_stack, md_stack
        toc_stack = [t for t in toc_stack if t[0] < level] + [(level, title)]
        md_stack = []

    def path():
        return [t for _, t in toc_stack] + [t for _, t in md_stack]

    def toc_match(pending, text):
        n = _norm(_clean_heading(text))
        if len(n) < 4 or len(text) > 200:
            return None
        return next((t for t in pending if _norm(t[1]) and (
            _norm(t[1]) == n or (len(n) >= 8 and (_norm(t[1]).startswith(n) or n.startswith(_norm(t[1])))))), None)

    def apply_upto(pending, match):
        for t in pending[: pending.index(match) + 1]:  # every TOC entry up to the matched one starts here
            apply_toc(*t)
        return pending[pending.index(match) + 1:]

    for page_data in sorted(pages, key=lambda p: p["page"]):
        page = page_data["page"]
        pending = list(toc_by_page.get(page, [])) if use_toc else []
        drop = running.get(page, set())
        lines = normalize_text(page_data.get("md") or "").splitlines()
        para: list[str] = []
        i = 0

        # TOC entries whose title we can't find as a line on this page start at the top of the page
        # (chapters and boxes usually do); the ones we can find start exactly at that line.
        findable = [t for t in pending if any(toc_match([t], l.strip()) for l in lines)]
        for t in [t for t in pending if t not in findable]:
            apply_toc(*t)
        pending = findable

        def flush_para():
            if para:
                text = " ".join(para).strip()
                if text:
                    yield {"page": page, "path": path(), "kind": "paragraph", "text": text}
                para.clear()

        while i < len(lines):
            line = lines[i].strip()

            open_tag = OPEN_TAG_RE.match(line)
            if open_tag:
                yield from flush_para()
                kind, title = open_tag.group(1).lower(), (open_tag.group(2) or "").strip()
                body = []
                i += 1
                while i < len(lines) and not lines[i].strip().lower().startswith(f"</{kind}"):
                    body.append(lines[i].rstrip())
                    i += 1
                yield {"page": page, "path": path(), "kind": kind, "title": title,
                       "text": "\n".join(body).strip()}
                i += 1
                continue

            if line.startswith("|"):  # raw markdown table (pymupdf4llm / llamaparse output)
                title = para[-1] if para and len(para[-1]) < 150 else ""
                yield from flush_para()
                body = []
                while i < len(lines) and lines[i].strip().startswith("|"):
                    body.append(lines[i].strip())
                    i += 1
                yield {"page": page, "path": path(), "kind": "table", "title": title, "text": "\n".join(body)}
                continue

            footnote = FOOTNOTE_RE.match(line)
            if footnote:
                yield from flush_para()
                marker, body = footnote.group(1), [footnote.group(2)]
                i += 1
                while i < len(lines) and lines[i].strip() and not FOOTNOTE_RE.match(lines[i].strip()):
                    body.append(lines[i].strip())
                    i += 1
                yield {"page": page, "path": path(), "kind": "footnote", "marker": marker,
                       "text": " ".join(body).strip()}
                continue

            heading = HEADING_RE.match(line)
            if heading:
                yield from flush_para()
                level, title = len(heading.group(1)), _clean_heading(heading.group(2))
                match = toc_match(pending, title)
                if match:
                    pending = apply_upto(pending, match)
                elif title and i not in drop:
                    md_stack = [t for t in md_stack if t[0] < level] + [(level, title)]
                i += 1
                continue

            match = toc_match(pending, line) if pending and line else None
            if match:  # a title in bold/plain text, e.g. "**BOX 1.1 How much will AI affect global growth?**"
                yield from flush_para()
                pending = apply_upto(pending, match)
            elif i in drop:
                pass
            elif not line:
                yield from flush_para()
            elif not PAGE_NUMBER_RE.match(line.replace("*", "")):
                para.append(line)
            i += 1

        yield from flush_para()


# ---------------------------------------------------------------- helpers
def _path_str(path: list[str]) -> str:
    return " > ".join(path)


def _split_long(text: str, max_words: int) -> list[str]:
    """Split a long paragraph on sentence boundaries."""
    if _word_count(text) <= max_words:
        return [text]
    pieces, current = [], []
    for sentence in _split_sentences(text):
        if current and _word_count(" ".join(current + [sentence])) > max_words:
            pieces.append(" ".join(current))
            current = []
        current.append(sentence)
    if current:
        pieces.append(" ".join(current))
    # a single giant "sentence" (lists, bad OCR) still needs cutting
    out = []
    for p in pieces:
        words = p.split()
        out += [" ".join(words[k:k + max_words]) for k in range(0, len(words), max_words)]
    return out


def _table_rows_as_text(table_md: str, max_rows: int = 60) -> str:
    """'| Year | CO2 |' rows -> 'Year: 2023; CO2: 397 kg'. Embeddings understand sentences far better than pipes."""
    rows = [r for r in table_md.splitlines() if r.strip().startswith("|") and not re.match(r"^\|[\s:|-]+\|$", r.strip())]
    if not rows:
        return table_md
    cells = [[c.replace("<br>", " ").replace("**", "").strip() for c in r.strip().strip("|").split("|")] for r in rows]
    header, body = cells[0], cells[1:]
    if not body:
        return "; ".join(header)
    lines = []
    for row in body[:max_rows]:
        pairs = [f"{h}: {v}" for h, v in zip(header, row) if v]
        lines.append("; ".join(pairs))
    return "\n".join(lines)


def _split_table(table_md: str, max_rows: int = 40) -> list[str]:
    """Split very long tables by rows, repeating the header on each part."""
    lines = table_md.splitlines()
    if len(lines) <= max_rows + 2:
        return [table_md]
    header, rows = lines[:2], lines[2:]
    return ["\n".join(header + rows[k:k + max_rows]) for k in range(0, len(rows), max_rows)]


def _make_chunk(doc_id, order, block_pages, path, kind, content, embed_text, **extra):
    return {
        "chunk_id": f"{doc_id}-{order:05d}",
        "doc_id": doc_id,
        "order": order,
        "page": min(block_pages),
        "pages": sorted(set(block_pages)),
        "section_path": path,
        "type": kind,
        "content": content,
        "embed_text": embed_text,
        **extra,
    }


def _element_chunks(doc_id, order, block):
    """Tables and charts become their own chunks (never cut through the middle of one)."""
    path, title = block["path"], block.get("title", "")
    header = f"Section: {_path_str(path)}\n" if path else ""
    label = "Table" if block["kind"] == "table" else "Chart"
    parts = _split_table(block["text"]) if block["kind"] == "table" else [block["text"]]
    chunks = []
    for n, part in enumerate(parts):
        suffix = f" (part {n + 1}/{len(parts)})" if len(parts) > 1 else ""
        content = f"{label}: {title}{suffix}\n{part}" if title else f"{label}{suffix}\n{part}"
        rows = _table_rows_as_text(part) if block["kind"] == "table" else part
        embed = f"{header}{label}: {title}\n{rows}"
        chunks.append(_make_chunk(doc_id, order + n, [block["page"]], path, block["kind"], content, embed, title=title))
    return chunks


# ---------------------------------------------------------------- narrative
def chunk_narrative(doc_id, parsed, max_words=350, overlap_words=60):
    chunks, buf, buf_pages, buf_path = [], [], [], None
    order = 0

    def flush():
        nonlocal buf, buf_pages, order
        if not buf_pages:  # only overlap text left, nothing new
            return
        text = "\n\n".join(buf).strip()
        header = f"Section: {_path_str(buf_path)}\n" if buf_path else ""
        chunks.append(_make_chunk(doc_id, order, buf_pages, buf_path or [], "text", text, header + text))
        order += 1
        # carry the tail forward as overlap so a fact split across the boundary is still findable
        tail = " ".join(text.split()[-overlap_words:])
        buf, buf_pages = [tail], []

    for block in iter_blocks(parsed["pages"], parsed.get("toc")):
        if block["kind"] in ("table", "chart"):
            flush()
            buf = []
            new = _element_chunks(doc_id, order, block)
            chunks += new
            order += len(new)
            continue
        if block["path"] != buf_path:  # never mix two sections in one chunk
            flush()
            buf, buf_path = [], block["path"]
        for piece in _split_long(block["text"], max_words):
            if _word_count("\n\n".join(buf + [piece])) > max_words and buf_pages:
                flush()
            buf.append(piece)
            buf_pages.append(block["page"])
    flush()
    return {"chunks": chunks, "parents": []}


# ---------------------------------------------------------------- hierarchical
def chunk_hierarchical(doc_id, parsed, parent_words=1200, child_words=180):
    """Small-to-big: children are indexed, parents are what the LLM reads."""
    chunks, parents = [], []
    order = 0
    sections: list[tuple[list, list[dict]]] = []
    for block in iter_blocks(parsed["pages"], parsed.get("toc")):
        if not sections or sections[-1][0] != block["path"]:
            sections.append((block["path"], []))
        sections[-1][1].append(block)

    for path, blocks in sections:
        # cut each section into parent windows on block boundaries
        windows, current = [], []
        for block in blocks:
            if current and _word_count(" ".join(b["text"] for b in current + [block])) > parent_words:
                windows.append(current)
                current = []
            current.append(block)
        if current:
            windows.append(current)

        for window in windows:
            parent_id = f"{doc_id}-p{len(parents):04d}"
            parent_text = "\n\n".join(
                (f"Table: {b.get('title', '')}\n{b['text']}" if b["kind"] == "table" else b["text"]) for b in window)
            parents.append({
                "parent_id": parent_id,
                "doc_id": doc_id,
                "section_path": path,
                "pages": sorted({b["page"] for b in window}),
                "content": parent_text,
            })
            header = f"Section: {_path_str(path)}\n" if path else ""
            buf, buf_pages = [], []

            def flush_child():
                nonlocal order
                if buf:
                    text = "\n\n".join(buf)
                    chunks.append(_make_chunk(doc_id, order, buf_pages, path, "text", text,
                                              header + text, parent_id=parent_id))
                    order += 1
                    buf.clear()
                    buf_pages.clear()

            for block in window:
                if block["kind"] in ("table", "chart", "footnote"):
                    flush_child()
                    new = (_element_chunks(doc_id, order, block) if block["kind"] != "footnote" else
                           [_make_chunk(doc_id, order, [block["page"]], path, "footnote", block["text"],
                                        header + block["text"])])
                    for c in new:
                        c["parent_id"] = parent_id
                    chunks += new
                    order += len(new)
                    continue
                for piece in _split_long(block["text"], child_words):
                    if buf and _word_count("\n\n".join(buf + [piece])) > child_words:
                        flush_child()
                    buf.append(piece)
                    buf_pages.append(block["page"])
            flush_child()
    return {"chunks": chunks, "parents": parents}


# ---------------------------------------------------------------- structured
def chunk_structured(doc_id, parsed, max_words=250):
    blocks = list(iter_blocks(parsed["pages"], parsed.get("toc")))

    # collect notes per page AND document-wide: some reports put footnotes at the bottom of the page,
    # others (like Apple's) collect numbered endnotes at the back of the document
    footnotes, endnotes = defaultdict(dict), {}
    for b in blocks:
        if b["kind"] == "footnote":
            footnotes[b["page"]][b["marker"]] = b["text"]
            endnotes.setdefault(b["marker"], b["text"])

    chunks, order = [], 0
    buf, buf_pages, buf_path = [], [], None

    def flush():
        nonlocal buf, buf_pages, order
        text = "\n\n".join(buf).strip()
        if text:
            header = f"Section: {_path_str(buf_path)}\n" if buf_path else ""
            chunks.append(_make_chunk(doc_id, order, buf_pages, buf_path or [], "text", text, header + text))
            order += 1
        buf, buf_pages = [], []

    for block in blocks:
        if block["kind"] in ("table", "chart"):
            flush()
            new = _element_chunks(doc_id, order, block)
            chunks += new
            order += len(new)
        elif block["kind"] == "footnote":
            flush()
            header = f"Section: {_path_str(block['path'])}\n" if block["path"] else ""
            text = f"Footnote [^{block['marker']}]: {block['text']}"
            chunks.append(_make_chunk(doc_id, order, [block["page"]], block["path"], "footnote",
                                      text, header + text))
            order += 1
        else:
            if block["path"] != buf_path:
                flush()
                buf_path = block["path"]
            for piece in _split_long(block["text"], max_words):
                if buf and _word_count("\n\n".join(buf + [piece])) > max_words:
                    flush()
                buf.append(piece)
                buf_pages.append(block["page"])
    flush()

    # attach referenced footnotes so "397 kg[^1]" travels together with "[^1]: based on 4-year use"
    for c in chunks:
        if c["type"] == "footnote":
            continue
        refs = dict.fromkeys(FOOTNOTE_REF_RE.findall(c["content"]))
        notes = []
        for m in refs:
            text = next((footnotes[p][m] for p in c["pages"] if m in footnotes[p]), endnotes.get(m))
            if text:
                notes.append(f"[^{m}]: {text}")
        if notes:
            c["footnotes"] = notes
            c["content"] += "\n\nFootnotes:\n" + "\n".join(notes)
            c["embed_text"] += "\n" + "\n".join(notes)
    return {"chunks": chunks, "parents": []}


CHUNKERS = {
    "narrative": chunk_narrative,
    "hierarchical": chunk_hierarchical,
    "structured": chunk_structured,
}


def chunk(doc_id: str, profile: str, parsed: dict) -> dict:
    result = CHUNKERS[profile](doc_id, parsed)
    # never index empty chunks or 1-3 word fragments (stray labels, leftover headers)
    result["chunks"] = [c for c in result["chunks"]
                        if c["content"].strip() and (c["type"] != "text" or _word_count(c["content"]) >= 4)]
    return result
