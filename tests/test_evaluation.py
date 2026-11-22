import json

import pytest

from complibot.answer import CompliBot
from complibot.evaluation import (
    GoldenItem,
    check_thresholds,
    format_table,
    load_golden,
    retrieval_metrics,
    run_eval,
    summarize,
)
from complibot.judge import Judge, heuristic_judge_responder
from complibot.llm import FakeProvider
from complibot.schemas import Chunk, Retrieved

DOC = "fake-data-protection-guidelines"
TRM = "fake-tech-risk-notice"


def item(i, q, secs=(), role="public", ref="", doc=DOC):
    return GoldenItem(id=f"t{i}", question=q, role=role, answerable=bool(secs),
                      source_clauses=[{"doc_id": doc, "section_id": s} for s in secs], reference_answer=ref,
                      tags=["t"], verified=True)


GOLDEN = [
    item(1, "How many business days to stop using data after consent is withdrawn?", ["2.3"],
         ref="Within 10 business days of receiving the withdrawal notice."),
    item(2, "Should personal data on laptops and USB drives be encrypted?", ["3.1.2"],
         ref="Yes, encryption should be applied to personal data on portable devices."),
    item(3, "What recovery time objective applies to each critical system?", ["5.1.1"], role="internal",
         ref="Not more than 4 hours.", doc=TRM),
    item(4, "What recovery time objective applies to each critical system?", role="public"),  # role-gated
    item(5, "What is the fine for breaching the telco cybersecurity code?"),  # out of corpus
]


def ret(doc, sec, covers=None):
    return Retrieved(Chunk(f"{doc}::{sec}", doc, "T", sec, "", 1, 1, "public", "x", covers or [sec]), 1.0)


def test_retrieval_metrics_recall_and_mrr():
    gold = {(DOC, "2.1"), (DOC, "2.2")}
    retrieved = [ret(DOC, "4.1"), ret(DOC, "2.2"), ret(DOC, "3.1")]
    m = retrieval_metrics(gold, retrieved)
    assert m["recall@1"] == 0 and m["recall@3"] == 0.5 and m["hit@3"] == 1.0
    assert m["mrr"] == pytest.approx(0.5)


def test_fixed_chunks_count_via_covers():
    gold = {(DOC, "2.1"), (DOC, "2.2")}
    window = Retrieved(Chunk(f"{DOC}::w0001", DOC, "T", "2.1", "", 1, 1, "public", "x", ["2.1", "2.2"]), 1.0)
    assert retrieval_metrics(gold, [window])["recall@1"] == 1.0


def test_rag_eval_end_to_end(bot):
    rows = run_eval(GOLDEN, bot, Judge(FakeProvider(heuristic_judge_responder)))
    by = {r["id"]: r for r in rows}
    assert by["t1"]["recall@1"] == 1.0 and by["t1"]["refusal_correct"]
    assert by["t1"]["faithfulness"] == 1.0 and by["t1"]["correctness"] == 1.0
    assert by["t4"]["refused"] and not any(TRM in c for c in by["t4"]["retrieved"])  # internal doc hidden
    assert by["t3"]["citations"] == [f"{TRM}::5.1.1"]
    s = summarize(rows)
    assert s["n_answerable"] == 3 and s["n_unanswerable"] == 2
    assert s["recall@5"]["mean"] == 1.0
    assert "recall@5" in format_table(s)


def test_oracle_and_null_baselines_bracket_the_metrics(bot):
    oracle = summarize(run_eval(GOLDEN, bot, None, system="oracle"))
    null = summarize(run_eval(GOLDEN, bot, None, system="null"))
    assert oracle["recall@1"]["mean"] == 1.0 and oracle["refusal_accuracy"]["mean"] == 1.0
    assert oracle["citation_precision"]["mean"] == 1.0
    assert null["recall@10"]["mean"] == 0.0 and null["false_refusal_rate"]["mean"] == 1.0
    assert null["unsafe_answer_rate"]["mean"] == 0.0


def test_llm_errors_are_excluded_not_scored(retriever, settings):
    from complibot.llm import LLMError

    def boom(s, u, schema):
        raise LLMError("down")

    bot = CompliBot(settings, retriever, FakeProvider(boom))
    rows = run_eval(GOLDEN[:2], bot, None)
    s = summarize(rows)
    assert s["n_errors"] == 2 and s["refusal_accuracy"]["n"] == 0


def test_judge_fails_known_negatives():
    judge = Judge(FakeProvider(heuristic_judge_responder))
    src = [("S1", "An organisation must stop using the data within 10 business days.")]
    assert judge.faithfulness("q", "", src).score == 0.0
    assert judge.faithfulness("q", "The fine is $1 million per breach.", src).score == 0.0
    assert judge.faithfulness("q", "Stop using the data within 10 business days [S1].", []).score == 0.0
    idk = "I don't know based on the provided documents."
    assert judge.correctness("q", idk, "Within 10 business days.").score == 0.0
    assert judge.correctness("q", "", "Within 10 business days.").score == 0.0


def test_judge_error_is_reported_not_scored():
    judge = Judge(FakeProvider(lambda s, u, schema: "garbage"))
    res = judge.correctness("q", "an answer", "a reference")
    assert res.score is None and res.error


def test_thresholds():
    s = {"recall@5": {"mean": 0.7}, "faithfulness": {"mean": None}}
    assert check_thresholds(s, {"recall@5": 0.6}) == []
    assert len(check_thresholds(s, {"recall@5": 0.8, "faithfulness": 0.5})) == 2


def test_load_golden_validates(tmp_path):
    p = tmp_path / "g.jsonl"
    p.write_text(json.dumps({"id": "a", "question": "q", "answerable": True, "verified": True}) + "\n")
    with pytest.raises(ValueError, match="no source_clauses"):
        load_golden(p)
    p.write_text("\n".join(json.dumps({"id": "a", "question": "q", "answerable": False, "verified": v})
                           for v in (True, True)))
    with pytest.raises(ValueError, match="duplicate"):
        load_golden(p)


def test_shipped_golden_sets_load():
    demo = load_golden(__import__("pathlib").Path("eval/golden/acme_demo.jsonl"))
    assert len(demo) >= 20 and any(not i.answerable for i in demo)
    drafts = load_golden(__import__("pathlib").Path("eval/golden/sg_regulations.draft.jsonl"),
                         include_unverified=True)
    assert all(not i.verified for i in drafts)  # drafts are never scored until labelled


def test_label_items_scripted(retriever):
    from complibot.labeling import label_items

    items = [
        {"id": "x1", "question": "How long to stop using data after consent is withdrawn?", "role": "public"},
        {"id": "x2", "question": "What is the telco fine?", "role": "public"},
        {"id": "x3", "question": "skip me", "role": "public"},
    ]
    answers = iter(["1", "Within 10 business days.", "u", "s"])
    done = label_items(items, retriever, ask=lambda prompt: next(answers), show=lambda *_: None)
    assert [d["id"] for d in done] == ["x1", "x2"]
    assert done[0]["source_clauses"] == [{"doc_id": DOC, "section_id": "2.3"}] and done[0]["verified"]
    assert done[1]["answerable"] is False


def test_generate_candidates(index):
    from complibot.labeling import fake_generator_provider, generate_candidates

    items = generate_candidates(index.chunks, fake_generator_provider(), per_doc=2)
    assert items and all(not i["verified"] for i in items)
    assert {i["answerable"] for i in items} == {True, False}
    assert all(i["source_clauses"] for i in items if i["answerable"])
