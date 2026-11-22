from pathlib import Path

from pdf_helper import make_pdf

from complibot.ingest import (
    _split_long,
    access_level_for,
    classify_line,
    clean_pages,
    ingest_directory,
    ingest_file,
)

FIX = Path(__file__).parent / "fixtures" / "raw"
PUBLIC_DOC = FIX / "public" / "fake-data-protection-guidelines.txt"


def by_id(chunks):
    return {c.chunk_id.split("::", 1)[1]: c for c in chunks}


def test_section_chunking_splits_on_clause_numbers():
    chunks = by_id(ingest_file(PUBLIC_DOC, FIX))
    assert {"2.1", "2.2", "2.3", "3.1.1", "3.1.2", "4.1", "4.2"} <= set(chunks)
    assert chunks["2.3"].text.startswith("2.3 An individual may withdraw consent")
    assert "10 business days" in chunks["2.3"].text
    assert "2.2" not in chunks["2.3"].text  # no bleed between clauses


def test_chunks_carry_metadata():
    c = by_id(ingest_file(PUBLIC_DOC, FIX))["3.1.1"]
    assert c.doc_id == "fake-data-protection-guidelines"
    assert c.section_title == "Security Arrangements"  # nearest heading, not the top-level one
    assert c.page_start == 4 and c.page_end == 4
    assert c.access_level == "public"
    assert c.covers == ["3.1.1"]


def test_headers_footers_and_toc_are_removed():
    chunks = ingest_file(PUBLIC_DOC, FIX)
    text = "\n".join(c.text for c in chunks)
    assert "FAKE TEST DOCUMENT" not in text  # running header
    assert "Page 3" not in text  # footer
    assert "......" not in text  # table of contents page dropped


def test_wrapped_numbers_are_not_clauses():
    chunks = by_id(ingest_file(PUBLIC_DOC, FIX))
    assert "2.5" not in chunks  # "2.5 per cent" is running text
    assert "2.5 per cent" in chunks["3.1.2"].text


def test_classify_line_heuristics():
    assert classify_line("4.2 Risk Management Framework").kind == "heading"
    assert classify_line("4 TECHNOLOGY RISK GOVERNANCE").kind == "heading"
    assert classify_line("4.2.1 The FI should establish a framework to manage risk.").kind == "clause"
    assert classify_line("2.5 per cent of revenue") is None
    assert classify_line("1 January 2021") is None
    assert classify_line("3.1 Role of the Board .............. 7") is None  # TOC entry
    assert classify_line("12 See the Act for details.") is None  # footnote-like
    assert classify_line("Annex A Glossary").num == "Annex A"


def test_access_level_comes_from_folder(tmp_path):
    data = tmp_path / "raw"
    assert access_level_for(data / "public" / "x.pdf", data) == "public"
    assert access_level_for(data / "internal" / "x.pdf", data) == "internal"
    assert access_level_for(data / "misc" / "x.pdf", data) == "internal"  # fail closed


def test_internal_doc_and_annex():
    chunks = by_id(ingest_file(FIX / "internal" / "fake-tech-risk-notice.txt", FIX))
    assert chunks["5.1.1"].access_level == "internal"
    assert chunks["5.1.1"].section_title == "Recovery Objectives"
    assert "Annex A" in chunks
    assert chunks["5.2.2"].page_start == 3


def test_pdf_ingestion(tmp_path):
    data = tmp_path / "raw"
    (data / "internal").mkdir(parents=True)
    header = "FAKE TEST PDF - NOT A REAL REGULATION"
    make_pdf(
        data / "internal" / "fake-trm.pdf",
        [
            [header, "4 TECHNOLOGY RISK FRAMEWORK", "4.1 Risk Management Framework",
             "4.1.1 The FI should establish a framework to manage",
             "technology risk across the organisation.", "Page 1"],
            [header, "4.1.2 The framework should be reviewed annually.", "Page 2"],
            [header, "4.1.3 Senior management should approve the framework.", "Page 3"],
        ],
        title="Fake Technology Risk Guidelines",
    )
    chunks = by_id(ingest_file(data / "internal" / "fake-trm.pdf", data))
    assert set(chunks) == {"4.1.1", "4.1.2", "4.1.3"}
    c = chunks["4.1.1"]
    assert c.doc_title == "Fake Technology Risk Guidelines"  # from PDF metadata
    assert "technology risk across the organisation" in c.text  # continuation line kept
    assert chunks["4.1.2"].page_start == 2
    assert header not in c.text


def test_long_clause_is_split_into_parts():
    text = "4.1.1 " + " ".join(f"Sentence number {i} is here." for i in range(200))
    parts = _split_long(text, 500)
    assert len(parts) > 1 and all(len(p) <= 500 for p in parts)
    assert "".join(parts).replace(" ", "") == text.replace(" ", "")


def test_fixed_chunking_tracks_covered_clauses():
    chunks = ingest_directory(FIX, strategy="fixed", fixed_size=300, fixed_overlap=50)
    assert all(c.chunk_id.split("::")[1].startswith("w") for c in chunks)
    assert all(len(c.text) <= 300 for c in chunks)
    covered = {s for c in chunks for s in c.covers}
    assert {"2.1", "3.1.1", "5.1.1"} <= covered
    assert any(len(c.covers) > 1 for c in chunks)  # windows span clause boundaries


def test_clean_pages_drops_page_numbers():
    pages = ["Header\nBody text one\n1", "Header\nBody text two\n2", "Header\nBody text three\n3"]
    assert clean_pages(pages) == [["Body text one"], ["Body text two"], ["Body text three"]]


def test_chunk_ids_are_unique():
    chunks = ingest_directory(FIX)
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))
