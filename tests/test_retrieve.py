import pytest

from complibot.retrieve import rrf_fuse, tokenize


def ids(results):
    return [r.chunk.chunk_id.split("::")[1] for r in results]


def test_tokenize_keeps_clause_numbers_and_stems():
    toks = tokenize("What does clause 4.2.1 say about Obligations?")
    assert "4.2.1" in toks and "obligation" in toks and "what" not in toks


def test_rrf_fusion_rewards_agreement():
    scores = rrf_fuse([[1, 2, 3], [3, 1, 4]], k=60)
    assert max(scores, key=scores.get) == 1  # rank 1 + rank 2 beats rank 3 + rank 1
    assert scores[1] == pytest.approx(1 / 61 + 1 / 62)
    assert scores[4] == pytest.approx(1 / 63) and scores[4] < scores[2]


def test_bm25_finds_the_right_clause(retriever):
    res = retriever.search("How long to stop using data after consent is withdrawn?", role="public")
    assert ids(res)[0] == "2.3"
    assert res[0].bm25_rank == 1 and res[0].dense_rank is None


def test_public_role_cannot_see_internal_documents(retriever):
    res = retriever.search("recovery time objective for critical systems", role="public", top_k=10)
    assert all(r.chunk.access_level == "public" for r in res)
    assert "5.1.1" not in ids(res)


def test_internal_role_sees_internal_documents(retriever):
    res = retriever.search("recovery time objective for critical systems", role="internal")
    assert ids(res)[0] == "5.1.1"


def test_unknown_role_gets_least_privilege(retriever):
    res = retriever.search("recovery time objective", role="admin", top_k=10)
    assert all(r.chunk.access_level == "public" for r in res)


def test_no_match_returns_empty(retriever):
    assert retriever.search("zzzz qqqq", role="internal") == []
    assert retriever.search("   ", role="internal") == []


def test_top_k_is_respected(retriever):
    assert len(retriever.search("personal data organisation", role="internal", top_k=2)) == 2


def test_hybrid_search_uses_both_retrievers(hybrid_retriever):
    res = hybrid_retriever.search("encryption of personal data on laptops", role="public")
    assert ids(res)[0] == "3.1.2"
    assert res[0].bm25_rank is not None and res[0].dense_rank is not None


def test_dense_index_mismatch_is_rejected(index, settings):
    from complibot.embeddings import HashEmbedder
    from complibot.retrieve import HybridRetriever

    with pytest.raises(ValueError, match="no embeddings"):
        HybridRetriever(index, settings, embedder=HashEmbedder())
