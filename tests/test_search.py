from local_rag.retrieval.search import hybrid_candidates, reciprocal_rank_fusion
from local_rag.storage import db
from tests.fakes import FakeEmbeddingProvider


class _QueryTrackingEmbeddingProvider(FakeEmbeddingProvider):
    """embed_query çağrılarını kaydeder — hybrid_candidates'ın sorgu embedding'i
    için asimetrik metodu (bkz. embeddings/base.EmbeddingProvider.embed_query)
    kullandığını doğrulamak için."""

    def __init__(self, dimension: int = 64):
        super().__init__(dimension=dimension)
        self.embed_query_calls: list[str] = []

    def embed_query(self, text: str) -> list[float]:
        self.embed_query_calls.append(text)
        return self.embed_one(text)


def test_reciprocal_rank_fusion_favors_items_ranked_high_in_both_lists():
    dense = [1, 2, 3, 100]
    keyword = [3, 1, 2, 200]
    fused = reciprocal_rank_fusion([dense, keyword])
    # 1, 2, 3 her iki listede de üst sıralarda -> ilk 3 bunlar olmalı.
    assert set(fused[:3]) == {1, 2, 3}
    # Yalnızca tek bir listede ve en sonda görünen id'ler en altta kalmalı.
    assert set(fused[3:]) == {100, 200}


def test_reciprocal_rank_fusion_includes_ids_missing_from_one_list():
    fused = reciprocal_rank_fusion([[1, 2], [3]])
    assert set(fused) == {1, 2, 3}


def test_hybrid_candidates_combines_dense_and_keyword_hits(tmp_path):
    embedder = FakeEmbeddingProvider(dimension=64)
    conn = db.get_connection(tmp_path / "index.db", embedding_dimension=embedder.dimension)
    db.insert_document(
        conn,
        document_id="doc-1",
        filename="politika.md",
        title="politika",
        file_type=".md",
        content_hash="h1",
        language="tr",
        ingested_at="2026-07-20T00:00:00Z",
    )

    semantic_text = "çalışanların yıllık dinlenme hakkı kıdeme göre artar"
    exact_code_text = "PRC-7 kodlu prosedür izin onay akışını tanımlar"
    unrelated_text = "sunucu bakım penceresi cumartesi gecesi planlanmıştır"

    for idx, text in enumerate([semantic_text, exact_code_text, unrelated_text]):
        db.insert_chunk(
            conn,
            document_id="doc-1",
            chunk_index=idx,
            section_path="Bölüm",
            chunk_type="text",
            text=text,
            embedding=embedder.embed_one(text),
        )

    # "PRC-7" tam kelime eşleşmesi yalnızca BM25/FTS5 ile bulunur (dense embedding
    # bu basit fake'te kod benzeri tek kelimeyi anlamsal olarak ayırt etmez);
    # sorgu hem semantik hem tam-eşleşme unsuru içeriyor.
    results = hybrid_candidates(conn, embedder, "PRC-7 izin onay süreci nedir", top_n=5)
    texts = [r.text for r in results]
    assert exact_code_text in texts
    assert unrelated_text not in texts or texts.index(exact_code_text) < texts.index(unrelated_text)


def test_hybrid_candidates_embeds_query_via_embed_query(tmp_path):
    embedder = _QueryTrackingEmbeddingProvider(dimension=32)
    conn = db.get_connection(tmp_path / "index.db", embedding_dimension=embedder.dimension)
    db.insert_document(
        conn,
        document_id="doc-1",
        filename="politika.md",
        title="politika",
        file_type=".md",
        content_hash="h1",
        language="tr",
        ingested_at="2026-07-20T00:00:00Z",
    )
    db.insert_chunk(
        conn,
        document_id="doc-1",
        chunk_index=0,
        section_path="Bölüm",
        chunk_type="text",
        text="örnek içerik",
        embedding=embedder.embed_one("örnek içerik"),
    )

    hybrid_candidates(conn, embedder, "örnek sorgu", top_n=5)

    assert embedder.embed_query_calls == ["örnek sorgu"]
