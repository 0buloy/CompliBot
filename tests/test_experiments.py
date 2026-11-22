import json

import pytest

from complibot.config import Settings
from complibot.experiments import apply_changes, comparison_markdown, paired_delta, run_experiments, update_readme
from complibot.llm import FakeProvider
from complibot.rerank import LexicalReranker, LLMReranker, get_reranker
from complibot.rewrite import LLMRewriter, RulesRewriter, get_rewriter


def test_rules_rewriter_expands_acronyms_and_synonyms():
    out = RulesRewriter().rewrite("What RTO applies to a client's backup site?")
    assert len(out) == 1
    assert "recovery time objective" in out[0] and "customer" in out[0] and "alternate recovery site" in out[0]
    assert RulesRewriter().rewrite("consent obligation") == []


def test_llm_rewriter_parses_and_fails_open():
    ok = LLMRewriter(FakeProvider(lambda s, u, sc: json.dumps({"queries": ["alternate recovery site", "q"]})))
    assert ok.rewrite("q") == ["alternate recovery site"]  # original query is not duplicated
    assert len(ok.usage) == 1
    bad = LLMRewriter(FakeProvider(lambda s, u, sc: "nonsense"))
    assert bad.rewrite("q") == []


def test_rewriter_improves_paraphrase_retrieval(retriever):
    q = "Should client info on laptops be encrypted?"
    plain = retriever.search(q, role="public")
    retriever.rewriter = RulesRewriter()
    expanded = retriever.search(q, role="public")
    assert expanded[0].chunk.section_id == "3.1.2"
    assert len(expanded) >= len(plain)


def test_lexical_reranker_prefers_full_coverage(retriever):
    cands = retriever.search("encryption personal data portable devices laptops", role="public", top_k=5)
    reranked = LexicalReranker().rerank("encryption personal data portable devices laptops", list(reversed(cands)))
    assert reranked[0].chunk.section_id == "3.1.2"


def test_llm_reranker_orders_by_grade_and_fails_open(retriever):
    cands = retriever.search("personal data", role="public", top_k=3)
    grades = {"grades": [{"id": "C3", "grade": 3}, {"id": "C1", "grade": 1}, {"id": "C2", "grade": 0}]}
    rr = LLMReranker(FakeProvider(lambda s, u, sc: json.dumps(grades)))
    out = rr.rerank("q", cands)
    assert [o.chunk.chunk_id for o in out] == [cands[i].chunk.chunk_id for i in (2, 0, 1)]
    broken = LLMReranker(FakeProvider(lambda s, u, sc: "oops"))
    assert [o.chunk.chunk_id for o in broken.rerank("q", cands)] == [c.chunk.chunk_id for c in cands]


def test_reranker_sees_a_wider_pool_than_top_k(retriever):
    seen = {}

    class Spy:
        name = "spy"

        def rerank(self, q, cands):
            seen["n"] = len(cands)
            return cands

    retriever.reranker = Spy()
    assert len(retriever.search("personal data organisation", role="internal", top_k=2)) == 2
    assert seen["n"] > 2


def test_factories_use_offline_stand_ins_for_fake_llm():
    assert isinstance(get_rewriter("llm", FakeProvider()), RulesRewriter)
    assert isinstance(get_reranker("llm", FakeProvider()), LexicalReranker)
    assert get_rewriter("none") is None and get_reranker("none") is None
    with pytest.raises(ValueError):
        get_reranker("magic")


def test_paired_delta():
    a = [{"id": str(i), "rep": 0, "status": "answered", "m": 0.0} for i in range(10)]
    b = [{"id": str(i), "rep": 0, "status": "answered", "m": 1.0 if i < 8 else 0.0} for i in range(10)]
    d = paired_delta(a, b, "m")
    assert d["delta"] == pytest.approx(0.8) and d["significant"]
    assert paired_delta(a, a, "m")["significant"] is False


def test_apply_changes_rejects_unknown_settings():
    assert apply_changes(Settings(), {"provider": "fake"}).llm_provider == "fake"
    with pytest.raises(ValueError):
        apply_changes(Settings(), {"chunkng": "fixed"})


def test_run_experiments_end_to_end(tmp_path):
    from conftest import FIXTURE_DATA

    golden = tmp_path / "g.jsonl"
    golden.write_text("\n".join(json.dumps(x) for x in [
        {"id": "a", "question": "How many business days to stop using data after consent is withdrawn?",
         "answerable": True, "role": "public", "verified": True, "reference_answer": "10 business days.",
         "source_clauses": [{"doc_id": "fake-data-protection-guidelines", "section_id": "2.3"}]},
        {"id": "b", "question": "What is the telco fine?", "answerable": False, "role": "public", "verified": True},
    ]))
    config = {"golden": str(golden), "baseline": {"provider": "fake"},
              "experiments": [{"name": "Fixed", "changes": {"chunking": "fixed"}},
                              {"name": "Off", "changes": {}, "enabled": False}]}
    report = run_experiments(config, tmp_path / "out", base_settings=Settings(data_dir=FIXTURE_DATA))
    assert [e["name"] for e in report["experiments"]] == ["Baseline", "Fixed"]
    md = (tmp_path / "out" / "comparison.md").read_text()
    assert "| Baseline | section |" in md and "Paired difference" in md
    readme = tmp_path / "README.md"
    readme.write_text("x\n<!-- experiments:start -->\nSTALE-TABLE\n<!-- experiments:end -->\ny\n")
    assert update_readme(readme, comparison_markdown(report))
    text = readme.read_text()
    assert "STALE-TABLE" not in text and "| Fixed |" in text and text.endswith("y\n")
