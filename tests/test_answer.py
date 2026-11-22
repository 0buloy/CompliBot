import json

from complibot.answer import IDK, answer_question, validate_citations
from complibot.llm import FakeProvider, LLMError


def responder_returning(payload):
    return FakeProvider(lambda s, u, schema: payload if isinstance(payload, str) else json.dumps(payload))


def test_answer_has_valid_clause_citation(bot):
    r = bot.ask("How many business days to stop using personal data after consent is withdrawn?")
    assert r.status == "answered" and r.supported and not r.refused
    assert r.citations[0].section_id == "2.3"
    assert r.citations[0].doc_title == "Fake Data Protection Guidelines"
    assert "10 business days" in r.answer
    assert r.citations[0].chunk_id in r.retrieved_ids


def test_unanswerable_question_is_refused(bot):
    r = bot.ask("What is the maximum fine for a breach of the cybersecurity code of practice for telcos?")
    assert r.refused and r.status == "refused" and r.citations == []


def test_no_retrieval_hits_skips_the_llm(retriever):
    llm = FakeProvider()
    r = answer_question("zzzz qqqq", retriever, llm)
    assert r.refused and r.answer == IDK
    assert llm.calls == []  # no tokens spent


def test_public_user_cannot_get_internal_answer(bot):
    r = bot.ask("What recovery time objective applies to critical systems?", role="public")
    assert all("tech-risk" not in cid for cid in r.retrieved_ids)
    assert not any("tech-risk" in c.chunk_id for c in r.citations)


def test_internal_user_gets_internal_answer(bot):
    r = bot.ask("What recovery time objective applies to each critical system?", role="internal")
    assert r.status == "answered"
    assert r.citations[0].section_id == "5.1.1"


def test_hallucinated_citation_marks_answer_unsupported(retriever):
    llm = responder_returning({"answer": "Consent is required [S1] and fines are $1m [S9].",
                               "citations": ["S1", "S9"], "refused": False})
    r = answer_question("When is consent needed for personal data?", retriever, llm)
    assert r.status == "unsupported" and not r.supported
    assert r.invalid_citations == ["S9"]
    assert len(r.citations) == 1  # the valid one is still shown


def test_answer_without_citations_is_unsupported(retriever):
    llm = responder_returning({"answer": "Consent is required.", "citations": [], "refused": False})
    r = answer_question("When is consent needed for personal data?", retriever, llm)
    assert r.status == "unsupported" and r.citations == []


def test_inline_marker_must_also_be_valid(retriever):
    llm = responder_returning({"answer": "See [S1] and [S7].", "citations": ["S1"], "refused": False})
    r = answer_question("When is consent needed for personal data?", retriever, llm)
    assert r.invalid_citations == ["S7"]


def test_full_chunk_id_citation_is_accepted(retriever):
    cid = "fake-data-protection-guidelines::2.1"
    llm = responder_returning({"answer": "Consent first.", "citations": [cid], "refused": False})
    r = answer_question("When is consent needed for collecting personal data?", retriever, llm)
    assert r.status == "answered" and r.citations[0].chunk_id == cid


def test_malformed_json_is_an_error_not_a_crash(retriever):
    r = answer_question("When is consent needed for personal data?", retriever, responder_returning("not json"))
    assert r.status == "error" and "parse" in r.error


def test_fenced_json_is_parsed(retriever):
    payload = '```json\n{"answer": "Consent first [S1].", "citations": ["S1"], "refused": false}\n```'
    r = answer_question("When is consent needed for personal data?", retriever, responder_returning(payload))
    assert r.status == "answered"


def test_llm_failure_is_handled(retriever):
    def boom(s, u, schema):
        raise LLMError("rate limited", retryable=True)

    r = answer_question("When is consent needed for personal data?", retriever, FakeProvider(boom))
    assert r.status == "error" and r.error == "rate limited"


def test_prompt_contains_sources_and_question(retriever):
    llm = FakeProvider()
    answer_question("When is consent needed for personal data?", retriever, llm)
    system, user = llm.calls[0]
    assert "ONLY from the sources" in system
    assert '<source id="S1"' in user and "<question>" in user


def test_validate_citations_dedupes(retriever):
    hits = retriever.search("consent personal data", role="public")
    labels = {f"S{i}": h for i, h in enumerate(hits, start=1)}
    valid, invalid = validate_citations(["S1", "[S1]", " S1 "], "as stated [S1]", labels)
    assert len(valid) == 1 and invalid == []
