"""Ingestion: PDF/TXT/MD -> pages -> cleaned lines -> clause segments -> chunks.

Design (easy to explain in an interview):

1. Extract text page by page with pypdf, so every line keeps its page number.
2. Clean: drop running headers/footers (lines repeated at the top/bottom of many
   pages), bare page numbers, and table-of-contents pages.
3. Detect numbered headings/clauses with a regex ("4", "4.2", "4.2.1 ...") plus
   heuristics that separate a *heading* ("4.2 Risk Management Framework") from
   a *clause* ("4.2.1 The FI should establish ..."), and reject false positives
   such as wrapped numbers ("2.5 per cent"), dates and footnotes.
4. Section chunking: one chunk per clause, carrying doc title, clause ID,
   nearest heading title, page range and access level. Over-long clauses are
   split into parts that share the clause ID.
   Fixed-size chunking (Phase 3 baseline): character windows with overlap; each
   window records which clause IDs it overlaps (`covers`) so recall@k still works.

The folder decides access level: data/raw/public/* -> "public",
data/raw/internal/* -> "internal".
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from .config import ACCESS_LEVELS
from .schemas import Chunk

SUPPORTED_SUFFIXES = {".pdf", ".txt", ".md"}
PAGE_MARKER = "---page---"  # page break marker for .txt/.md files (form feed \f also works)

# "4", "4.2", "4.2.1", "12.5" followed by text. Top-level number limited to 2 digits
# so years ("2021 ...") are never treated as section numbers.
NUM_RE = re.compile(r"^(?P<num>\d{1,2}(?:\.\d{1,3}){0,4})\.?\s+(?P<rest>\S.*)$")
ANNEX_RE = re.compile(r"^(?P<kind>annex|appendix|schedule)\s+(?P<id>[A-Z0-9]{1,3})\b[\s:.\-–—]*(?P<rest>.*)$", re.I)
# Table-of-contents entries end in dot leaders and/or a page number.
TOC_RE = re.compile(r"(\.{3,}|…+|\s\.\s\.|\s{2,})\s*\d{1,3}\s*$")
PAGE_NUM_RE = re.compile(r"^(page\s*)?\d{1,4}(\s*(of|/)\s*\d{1,4})?$", re.I)
MONTH_RE = re.compile(
    r"^(january|february|march|april|may|june|july|august|september|october|november|december)\b", re.I
)
SMALL_WORDS = {"a", "an", "and", "as", "at", "by", "for", "in", "of", "on", "or", "the", "to", "with", "vs", "via"}
# Lowercase verbs that mark a sentence, not a heading ("The PDPA establishes ...", "... number is a ...").
SENTENCE_VERBS = {"is", "are", "was", "were", "be", "been", "should", "must", "may", "will", "shall", "can",
                  "could", "would", "has", "have", "had", "does", "do", "applies", "apply", "provides", "sets",
                  "establishes", "refers", "refer", "clarify", "clarifies", "includes", "include", "means"}
# Marks a line whose paragraph number was drawn as an image (see _mark_image_numbers).
MISSING_NUMBER = "\x00"
_MARKER_RE = re.compile(r"^ [A-Z“\"(‘]")


def _image_count(page) -> int:
    xobjects = page.get("/Resources", {}).get("/XObject") or {}
    return sum(1 for x in xobjects.values() if x.get_object().get("/Subtype") == "/Image")


def _mark_image_numbers(text: str, n_images: int) -> str | None:
    """Some PDFs (e.g. the PDPC Key Concepts guidelines, pages 2-89) draw paragraph numbers as
    small images. pypdf then emits each such line with a leading space where the number was.
    If a page has exactly as many of those lines as images, tag them so the segmenter can
    reconstruct the numbers; otherwise return None."""
    lines = text.splitlines()
    markers = [i for i, ln in enumerate(lines) if _MARKER_RE.match(ln)]
    if not markers or len(markers) != n_images:
        return None
    for i in markers:
        lines[i] = MISSING_NUMBER + lines[i].strip()
    return "\n".join(lines)


# --------------------------------------------------------------------------- pages


def load_pages(path: Path) -> list[str]:
    """Return the text of each page (index 0 = page 1)."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _load_pdf_pages(path)
    if suffix in {".txt", ".md"}:
        text = path.read_text(encoding="utf-8")
        text = text.replace(f"\n{PAGE_MARKER}\n", "\f")
        return text.split("\f")
    raise ValueError(f"Unsupported file type: {path}")


def _load_pdf_pages(path: Path) -> list[str]:
    """Per page: PDFium text (fewer broken words) unless the page uses image paragraph
    numbers, where pypdf's output lets us find and reconstruct them."""
    import logging

    from pypdf import PdfReader

    logging.getLogger("pypdf").setLevel(logging.ERROR)  # font warnings are noise here
    reader = PdfReader(str(path))
    try:
        import pypdfium2 as pdfium

        doc = pdfium.PdfDocument(str(path))
    except Exception:  # pypdfium2 not installed or can't open the file: pypdf only
        doc = None
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        marked = _mark_image_numbers(text, _image_count(page))
        if marked is not None:
            pages.append(marked)
        elif doc is not None:
            pages.append(doc[i].get_textpage().get_text_range().replace("\r", ""))
        else:
            pages.append(text)
    return pages


def _norm(line: str) -> str:
    return re.sub(r"\d+", "#", line.strip().lower())


def clean_pages(pages: list[str], edge_lines: int = 3) -> list[list[str]]:
    """Split pages into lines, removing headers/footers, page numbers and TOC pages."""
    split = [[ln.rstrip() for ln in p.splitlines()] for p in pages]
    split = [[ln for ln in p if ln.strip()] for p in split]

    # Running headers/footers: the same (digit-normalised) line near the top or
    # bottom of at least half the pages (and at least 3 pages).
    edge_counts: Counter[str] = Counter()
    for lines in split:
        edges = {_norm(ln) for ln in lines[:edge_lines] + lines[-edge_lines:]}
        edge_counts.update(edges)
    threshold = max(3, len(split) // 2)
    repeated = {k for k, c in edge_counts.items() if c >= threshold}

    cleaned: list[list[str]] = []
    for lines in split:
        n = len(lines)
        out = []
        for i, ln in enumerate(lines):
            at_edge = i < edge_lines or i >= n - edge_lines
            if at_edge and (_norm(ln) in repeated or PAGE_NUM_RE.match(ln.strip())):
                continue
            out.append(ln)
        if _is_toc_page(out):
            out = []
        cleaned.append(out)
    return cleaned


def _is_toc_page(lines: list[str]) -> bool:
    if not lines:
        return False
    head = " ".join(lines[:3]).lower()
    toc_lines = sum(1 for ln in lines if TOC_RE.search(ln))
    return toc_lines >= 4 and (toc_lines / len(lines) >= 0.4 or "contents" in head)


# ----------------------------------------------------------------- heading detection


def _title_like(text: str) -> bool:
    """Heading-style text: ALL CAPS or Title Case, short, no sentence punctuation."""
    text = text.strip()
    if not text or len(text) > 120 or text[-1] in ".;:,":
        return False
    words = re.findall(r"[A-Za-z][A-Za-z'’\-]*", text)
    if not words or len(words) > 16:
        return False
    letters = "".join(words)
    if letters.isupper():
        return True
    if any(w in SENTENCE_VERBS for w in words):
        return False
    content = [w for w in words if w.lower() not in SMALL_WORDS]
    if not content:
        return False
    capitalised = sum(1 for w in content if w[0].isupper())
    return capitalised / len(content) >= 0.6


@dataclass
class _Candidate:
    num: str
    rest: str
    kind: str  # "heading" | "clause"

    @property
    def parts(self) -> tuple[int, ...]:
        return tuple(int(p) for p in self.num.split(".")) if self.num[0].isdigit() else ()


def classify_line(line: str, next_line: str | None = None) -> _Candidate | None:
    """Decide whether a line starts a heading, a clause, or neither."""
    s = line.strip()
    m = ANNEX_RE.match(s)
    if m and (not m.group("rest") or _title_like(m.group("rest"))) and not TOC_RE.search(s):
        ident = f"{m.group('kind').title()} {m.group('id').upper()}"
        return _Candidate(ident, m.group("rest").strip() or ident, "heading")

    m = NUM_RE.match(s)
    if not m:
        return None
    num, rest = m.group("num"), m.group("rest").strip()
    if TOC_RE.search(rest):
        return None  # table-of-contents entry
    if not (rest[0].isupper() or rest[0] in "(\"“‘'"):
        return None  # wrapped number in running text, e.g. "2.5 per cent"
    if MONTH_RE.match(rest):
        return None  # a date, e.g. "1 January 2021"
    depth = num.count(".") + 1

    next_is_numbered = bool(next_line and NUM_RE.match(next_line.strip()))
    if _title_like(rest) or (len(rest) < 60 and rest[-1] not in ".;:," and next_is_numbered):
        return _Candidate(num, rest, "heading")
    if depth == 1:
        return None  # a bare "5 The ..." in sentence case is usually a footnote/list item
    return _Candidate(num, rest, "clause")


def _plausible_next(prev: tuple[int, ...] | None, cur: tuple[int, ...]) -> bool:
    """Reject numbers that jump implausibly (footnotes, figures, stray numbers)."""
    if not cur or prev is None or not prev:
        return True
    if cur[0] == prev[0]:
        return True
    if prev[0] < cur[0] <= prev[0] + 2:
        return True
    return cur == (1,) or cur[:2] == (1, 1)  # numbering restart (e.g. after annexes)


# ------------------------------------------------------------------------ segments


@dataclass
class Segment:
    section_id: str
    section_title: str
    lines: list[tuple[str, int]] = field(default_factory=list)  # (line text, page number)

    @property
    def text(self) -> str:
        return "\n".join(t for t, _ in self.lines).strip()


def clean_title(title: str) -> str:
    """Strip footnote markers glued to a heading: 'The Accountability Obligation82' -> '... Obligation'."""
    return re.sub(r"(?<=[A-Za-z”’)])\d{1,3}$", "", title.strip()).strip()


def _reconstruct_number(rest: str, prev: tuple[int, ...] | None) -> _Candidate:
    """Give an image-numbered line its number back. Short lines without sentence punctuation
    are chapter headings ("Definitions and related matters" -> next chapter); everything else
    is the next paragraph in the current chapter. Typed numbers elsewhere re-sync the count."""
    chapter = prev[0] if prev else 0
    para = prev[1] if prev and len(prev) > 1 else 0
    if len(rest) < 60 and rest[-1] not in ".:;,":
        return _Candidate(str(chapter + 1), rest, "heading")
    return _Candidate(f"{chapter}.{para + 1}", rest, "clause")


def segment_document(pages: list[list[str]], doc_title: str) -> list[Segment]:
    """Turn cleaned pages into clause-level segments."""
    flat: list[tuple[str, int]] = [(ln, pno + 1) for pno, lines in enumerate(pages) for ln in lines]
    headings: dict[str, str] = {}  # section number -> heading title
    segments: list[Segment] = []
    current = Segment("preamble", doc_title)
    prev_parts: tuple[int, ...] | None = None

    def nearest_title(num: str) -> str:
        parts = num.split(".")
        for cut in range(len(parts) - 1, 0, -1):
            key = ".".join(parts[:cut])
            if key in headings:
                return headings[key]
        return headings.get(num, doc_title)

    for i, (line, page) in enumerate(flat):
        if line.startswith(MISSING_NUMBER):  # number was drawn as an image: reconstruct it
            cand = _reconstruct_number(line[1:].strip(), prev_parts)
            line = f"{cand.num} {cand.rest}"
        else:
            nxt = flat[i + 1][0] if i + 1 < len(flat) else None
            nxt = nxt[1:] if nxt and nxt.startswith(MISSING_NUMBER) else nxt
            cand = classify_line(line, nxt)
            if cand and cand.parts and not _plausible_next(prev_parts, cand.parts):
                cand = None
        if cand is None:
            current.lines.append((line, page))
            continue

        segments.append(current)
        if cand.parts:
            prev_parts = cand.parts
        if cand.kind == "heading":
            cand.rest = clean_title(cand.rest)
            headings[cand.num] = cand.rest
            current = Segment(cand.num, cand.rest, [(line.strip(), page)])
        else:
            current = Segment(cand.num, nearest_title(cand.num), [(line.strip(), page)])
    segments.append(current)

    # Drop heading-only segments (their title is carried by the clauses below)
    # and a tiny or empty preamble.
    out = []
    for seg in segments:
        body_lines = seg.lines[1:] if seg.section_id != "preamble" else seg.lines
        body = " ".join(t for t, _ in body_lines).strip()
        if seg.section_id == "preamble" and len(body) < 80:
            continue
        if seg.section_id != "preamble" and not body and seg.section_id in headings:
            continue
        if seg.lines:
            out.append(seg)
    return out


def numbering_is_unreliable(segments: list[Segment], n_pages: int, min_clauses: int = 8,
                            max_share: float = 0.3) -> bool:
    """True for documents that don't use numbered paragraphs (e.g. designed brochures): no
    numbered segments at all, or - in a document of 6+ pages - too few of them, or one
    'section' holding a large share of all the text."""
    numbered = [s for s in segments if s.section_id != "preamble"]
    if not numbered:
        return True
    if n_pages < 6:
        return False
    total = sum(len(s.text) for s in segments) or 1
    biggest = max(len(s.text) for s in numbered)
    return len(numbered) < min_clauses or biggest / total > max_share


def page_segments(pages: list[list[str]], doc_title: str) -> list[Segment]:
    """Fallback: one segment per page, cited as 'p12', titled by the page's first heading-like
    line (or the last one seen). Honest and checkable when there are no clause numbers."""
    segments, title = [], doc_title
    for pno, lines in enumerate(pages, start=1):
        if not lines:
            continue
        heads, run = [], []
        for ln in lines[:6] + [""]:  # join runs of ALL-CAPS lines: "TRUSTED" + "DEVELOPMENT AND" + ...
            text = ln.strip()
            if text and text.isupper() and not re.search(r"\d", text):
                run.append(text)
                continue
            if run:
                heads.append(" ".join(run))
                run = []
            if 3 < len(text) < 70 and _title_like(text) and not re.search(r"\d", text):
                heads.append(text)
        heads = [h for h in heads if 3 < len(h) < 90]
        if heads:
            text = max(heads, key=len)  # the full heading, not a fragment like "TRUSTED"
            title = clean_title(text.title() if text.isupper() else text)
        segments.append(Segment(f"p{pno}", title, [(ln, pno) for ln in lines]))
    return segments


# -------------------------------------------------------------------------- chunks


def _split_long(text: str, max_chars: int) -> list[str]:
    """Split text into <= max_chars parts on paragraph, then sentence, boundaries."""
    if len(text) <= max_chars:
        return [text]
    pieces = re.split(r"(?<=[.;:])\s+(?=[A-Z(])|\n(?=\([a-z0-9]{1,4}\))", text)
    parts, buf = [], ""
    for piece in pieces:
        if buf and len(buf) + len(piece) + 1 > max_chars:
            parts.append(buf.strip())
            buf = ""
        while len(piece) > max_chars:  # a single enormous sentence: hard split
            parts.append(piece[:max_chars])
            piece = piece[max_chars:]
        buf = f"{buf} {piece}" if buf else piece
    if buf.strip():
        parts.append(buf.strip())
    return parts


def section_chunks(
    segments: list[Segment], doc_id: str, doc_title: str, access_level: str, max_chars: int = 2400
) -> list[Chunk]:
    chunks: list[Chunk] = []
    seen: Counter[str] = Counter()
    for seg in segments:
        seen[seg.section_id] += 1
        base = f"{doc_id}::{seg.section_id}"
        if seen[seg.section_id] > 1:  # duplicate numbering (e.g. restarted in an annex)
            base += f"~{seen[seg.section_id]}"
        pages = [p for _, p in seg.lines]
        parts = _split_long(seg.text, max_chars)
        for n, part in enumerate(parts, start=1):
            chunks.append(
                Chunk(
                    chunk_id=base if len(parts) == 1 else f"{base}#{n}",
                    doc_id=doc_id,
                    doc_title=doc_title,
                    section_id=seg.section_id,
                    section_title=seg.section_title,
                    page_start=min(pages),
                    page_end=max(pages),
                    access_level=access_level,
                    text=part,
                    covers=[seg.section_id],
                )
            )
    return chunks


def fixed_chunks(
    segments: list[Segment], doc_id: str, doc_title: str, access_level: str, size: int = 1200, overlap: int = 200
) -> list[Chunk]:
    """Fixed-size character windows (word-aligned) over the whole document."""
    # Flatten to one string, remembering for each character offset its page and segment.
    text_parts, spans = [], []  # spans: (start, end, page, segment index)
    pos = 0
    for si, seg in enumerate(segments):
        for line, page in seg.lines:
            text_parts.append(line)
            spans.append((pos, pos + len(line), page, si))
            pos += len(line) + 1
    full = "\n".join(text_parts)
    chunks: list[Chunk] = []
    start, n = 0, 0
    step = max(1, size - overlap)
    while start < len(full):
        end = min(len(full), start + size)
        if end < len(full):  # don't cut a word in half
            space = full.rfind(" ", start + step, end)
            end = space if space > start else end
        window = [s for s in spans if s[1] > start and s[0] < end]
        if window and full[start:end].strip():
            n += 1
            seg_ids = list(dict.fromkeys(segments[s[3]].section_id for s in window))
            first = segments[window[0][3]]
            chunks.append(
                Chunk(
                    chunk_id=f"{doc_id}::w{n:04d}",
                    doc_id=doc_id,
                    doc_title=doc_title,
                    section_id=first.section_id,
                    section_title=first.section_title,
                    page_start=min(s[2] for s in window),
                    page_end=max(s[2] for s in window),
                    access_level=access_level,
                    text=full[start:end].strip(),
                    covers=seg_ids,
                )
            )
        if end >= len(full):
            break
        start = max(end - overlap, start + 1)
    return chunks


# ----------------------------------------------------------------------- documents


def load_catalog(data_dir: Path) -> dict[str, str]:
    """Optional data/raw/catalog.json: {"file-stem": "Human readable title"}."""
    path = data_dir / "catalog.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def doc_title_for(path: Path, pages: list[str], catalog: dict[str, str]) -> str:
    if path.stem in catalog:
        return catalog[path.stem]
    if path.suffix.lower() in {".md", ".txt"} and pages:
        for line in pages[0].splitlines():
            if line.startswith("# "):
                return line[2:].strip()
    if path.suffix.lower() == ".pdf":
        try:
            from pypdf import PdfReader

            title = (PdfReader(str(path)).metadata or {}).get("/Title")
            if title and len(str(title).strip()) > 5:
                return str(title).strip()
        except Exception:  # metadata is best-effort
            pass
    return path.stem.replace("-", " ").replace("_", " ").title()


def access_level_for(path: Path, data_dir: Path) -> str:
    """The first folder under data_dir decides the access level. Unknown -> internal (fail closed)."""
    try:
        top = path.relative_to(data_dir).parts[0]
    except (ValueError, IndexError):
        return "internal"
    return top if top in ACCESS_LEVELS else "internal"


def ingest_file(
    path: Path,
    data_dir: Path,
    strategy: str = "section",
    catalog: dict[str, str] | None = None,
    max_chars: int = 2400,
    fixed_size: int = 1200,
    fixed_overlap: int = 200,
) -> list[Chunk]:
    raw_pages = load_pages(path)
    title = doc_title_for(path, raw_pages, catalog or {})
    pages = clean_pages(raw_pages)
    if path.suffix.lower() == ".md" and pages and pages[0] and pages[0][0].startswith("# "):
        pages[0] = pages[0][1:]  # the markdown title line is metadata, not content
    segments = segment_document(pages, title)
    if path.suffix.lower() == ".pdf" and numbering_is_unreliable(segments, len(pages)):
        segments = page_segments(pages, title)
    doc_id = path.stem
    level = access_level_for(path, data_dir)
    if strategy == "fixed":
        return fixed_chunks(segments, doc_id, title, level, fixed_size, fixed_overlap)
    if strategy == "section":
        return section_chunks(segments, doc_id, title, level, max_chars)
    raise ValueError(f"Unknown chunking strategy: {strategy!r}")


def discover_files(data_dir: Path) -> list[Path]:
    return sorted(
        p
        for p in data_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES and p.name != "README.md"
    )


def ingest_directory(data_dir: Path, strategy: str = "section", **kwargs) -> list[Chunk]:
    catalog = load_catalog(data_dir)
    chunks: list[Chunk] = []
    for path in discover_files(data_dir):
        chunks.extend(ingest_file(path, data_dir, strategy=strategy, catalog=catalog, **kwargs))
    return chunks


def build_index(settings, embedder=None):
    """Ingest settings.data_dir, embed (if an embedder is given) and save to settings.index_dir."""
    from .retrieve import index_text
    from .store import IndexData, save_index

    chunks = ingest_directory(
        settings.data_dir,
        strategy=settings.chunking,
        max_chars=settings.max_chunk_chars,
        fixed_size=settings.fixed_chunk_chars,
        fixed_overlap=settings.fixed_overlap_chars,
    )
    embeddings = embedder.embed([index_text(c) for c in chunks]) if (embedder and chunks) else None
    meta = {"chunking": settings.chunking, "embedder": embedder.name if embedder else None,
            "documents": sorted({c.doc_id for c in chunks})}
    data = IndexData(chunks=chunks, embeddings=embeddings, meta=meta)
    save_index(settings.index_dir, data)
    return data
