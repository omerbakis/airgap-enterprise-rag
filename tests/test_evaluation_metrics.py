from local_rag.evaluation.metrics import (
    dedupe_preserve_order,
    mean,
    ndcg_at_k,
    precision_at_k,
    reciprocal_rank,
    recall_at_k,
)


def test_precision_at_k_counts_hits_in_top_k():
    retrieved = ["a.md", "b.md", "c.md", "d.md"]
    relevant = {"a.md", "c.md"}
    assert precision_at_k(retrieved, relevant, k=2) == 0.5
    assert precision_at_k(retrieved, relevant, k=4) == 0.5


def test_recall_at_k_full_when_all_relevant_found():
    retrieved = ["x.md", "a.md", "c.md"]
    relevant = {"a.md", "c.md"}
    assert recall_at_k(retrieved, relevant, k=3) == 1.0
    assert recall_at_k(retrieved, relevant, k=1) == 0.0


def test_reciprocal_rank_of_first_relevant_hit():
    assert reciprocal_rank(["x.md", "a.md", "b.md"], {"a.md"}) == 0.5
    assert reciprocal_rank(["a.md"], {"a.md"}) == 1.0
    assert reciprocal_rank(["x.md", "y.md"], {"a.md"}) == 0.0


def test_ndcg_perfect_ranking_is_one():
    retrieved = ["a.md", "b.md", "c.md"]
    relevant = {"a.md"}
    assert ndcg_at_k(retrieved, relevant, k=3) == 1.0


def test_ndcg_worse_ranking_scores_lower_than_perfect():
    relevant = {"c.md"}
    perfect = ndcg_at_k(["c.md", "a.md", "b.md"], relevant, k=3)
    worse = ndcg_at_k(["a.md", "b.md", "c.md"], relevant, k=3)
    assert worse < perfect


def test_mean_of_empty_list_is_zero():
    assert mean([]) == 0.0
    assert mean([1.0, 2.0, 3.0]) == 2.0


def test_dedupe_preserve_order_keeps_first_occurrence():
    assert dedupe_preserve_order(["a.md", "a.md", "b.md", "a.md", "c.md"]) == ["a.md", "b.md", "c.md"]


def test_recall_stays_bounded_when_retrieved_has_duplicate_chunks_from_same_doc():
    """Chunk düzeyinde retrieved listesi aynı dokümandan birden çok kez
    içerebilir; dedupe edilmeden recall_at_k'ye verilirse recall 1.0'ı
    aşabilir (gerçek bir eval koşumunda yakalanan hata) — dedupe ile düzelir."""
    retrieved_chunks = ["izin.md", "izin.md", "izin.md", "diger.md", "izin.md"]
    relevant = {"izin.md"}
    buggy_recall = recall_at_k(retrieved_chunks, relevant, k=5)
    assert buggy_recall > 1.0  # hatanın kendisini de belgeler

    fixed_recall = recall_at_k(dedupe_preserve_order(retrieved_chunks), relevant, k=5)
    assert fixed_recall == 1.0
