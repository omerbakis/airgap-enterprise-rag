from local_rag.storage import db
from local_rag.storage.db import SearchFilters
from tests.fakes import FakeEmbeddingProvider


def _make_conn(tmp_path, dim=16):
    return db.get_connection(tmp_path / "index.db", embedding_dimension=dim)


def _insert_doc(conn, document_id, filename, file_type=".txt", content_hash=None):
    db.insert_document(
        conn,
        document_id=document_id,
        filename=filename,
        title=filename,
        file_type=file_type,
        content_hash=content_hash or f"hash-{document_id}",
        language="tr",
        ingested_at="2026-07-20T00:00:00Z",
    )


def test_insert_and_dense_search_returns_nearest_first(tmp_path):
    # Hash çakışmalarını azaltmak için görece yüksek bir boyut kullanılıyor;
    # gerçek embedding modelleri için bu bir kaygı değildir, yalnızca bu
    # basit hash tabanlı fake'in test kararlılığı için gerekli.
    embedder = FakeEmbeddingProvider(dimension=256)
    conn = _make_conn(tmp_path, dim=embedder.dimension)
    _insert_doc(conn, "doc-1", "izin_politikasi.docx", file_type=".docx")

    texts = {
        "chunk-izin": "Yıllık izin süresi kıdeme göre değişir çalışan izin",
        "chunk-network": "Ağ güvenliği için VPN kullanımı zorunludur firewall",
        "chunk-giderler": "Seyahat giderleri onay sürecinden geçer fatura",
    }
    ids = {}
    for key, text in texts.items():
        embedding = embedder.embed_one(text)
        chunk_id = db.insert_chunk(
            conn,
            document_id="doc-1",
            chunk_index=len(ids),
            section_path="Test Bölümü",
            chunk_type="text",
            text=text,
            embedding=embedding,
        )
        ids[key] = chunk_id

    query_embedding = embedder.embed_one("çalışanın yıllık izin hakkı kaç gün")
    results = db.dense_search(conn, query_embedding, top_k=2)

    assert len(results) == 2
    assert results[0].chunk_id == ids["chunk-izin"]
    assert results[0].source_filename == "izin_politikasi.docx"
    assert results[0].distance <= results[1].distance


def test_get_document_by_hash_detects_duplicates(tmp_path):
    conn = _make_conn(tmp_path)
    _insert_doc(conn, "doc-1", "a.txt", content_hash="abc123")
    assert db.get_document_by_hash(conn, "abc123") is not None
    assert db.get_document_by_hash(conn, "does-not-exist") is None


def test_delete_document_removes_chunks_vectors_and_fts_rows(tmp_path):
    embedder = FakeEmbeddingProvider(dimension=8)
    conn = _make_conn(tmp_path, dim=embedder.dimension)
    _insert_doc(conn, "doc-1", "a.txt", content_hash="abc123")
    chunk_id = db.insert_chunk(
        conn,
        document_id="doc-1",
        chunk_index=0,
        section_path="X",
        chunk_type="text",
        text="silinecek benzersizkelime chunk",
        embedding=embedder.embed_one("silinecek benzersizkelime chunk"),
    )
    assert db.keyword_search_ids(conn, "benzersizkelime", top_k=5) == [chunk_id]

    db.delete_document(conn, "doc-1")
    assert conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM vec_chunks").fetchone()[0] == 0
    assert db.get_document_by_hash(conn, "abc123") is None
    assert db.keyword_search_ids(conn, "benzersizkelime", top_k=5) == []


def test_keyword_search_finds_exact_term_dense_would_miss(tmp_path):
    """BM25, tam eşleşen kod/kısaltmaları (örn. bir madde numarasını) yakalamalı
    — bu tam da hybrid search'ün MVP'nin dense-only aramasına kattığı şey."""
    embedder = FakeEmbeddingProvider(dimension=32)
    conn = _make_conn(tmp_path, dim=embedder.dimension)
    _insert_doc(conn, "doc-1", "prosedur.docx", file_type=".docx")

    chunk_id = db.insert_chunk(
        conn,
        document_id="doc-1",
        chunk_index=0,
        section_path="Prosedürler",
        chunk_type="text",
        text="IK-2024-17 numaralı prosedür onay sürecini tanımlar",
        embedding=embedder.embed_one("IK-2024-17 numaralı prosedür onay sürecini tanımlar"),
    )
    db.insert_chunk(
        conn,
        document_id="doc-1",
        chunk_index=1,
        section_path="Diğer",
        chunk_type="text",
        text="alakasız başka bir konu hakkında metin",
        embedding=embedder.embed_one("alakasız başka bir konu hakkında metin"),
    )

    results = db.keyword_search_ids(conn, "IK-2024-17", top_k=5)
    assert results and results[0] == chunk_id


def test_search_filters_restrict_to_matching_file_type(tmp_path):
    embedder = FakeEmbeddingProvider(dimension=32)
    conn = _make_conn(tmp_path, dim=embedder.dimension)
    _insert_doc(conn, "doc-pdf", "a.pdf", file_type=".pdf", content_hash="h1")
    _insert_doc(conn, "doc-docx", "b.docx", file_type=".docx", content_hash="h2")

    for doc_id in ("doc-pdf", "doc-docx"):
        db.insert_chunk(
            conn,
            document_id=doc_id,
            chunk_index=0,
            section_path="X",
            chunk_type="text",
            text="ortak kelime tekrar eden içerik",
            embedding=embedder.embed_one("ortak kelime tekrar eden içerik"),
        )

    filters = SearchFilters(file_types=[".pdf"])
    query_embedding = embedder.embed_one("ortak kelime tekrar eden içerik")
    dense_ids = db.dense_search_ids(conn, query_embedding, top_k=5, filters=filters)
    keyword_ids = db.keyword_search_ids(conn, "ortak kelime", top_k=5, filters=filters)

    chunks = db.get_chunks_by_ids(conn, dense_ids + keyword_ids)
    assert all(chunks[cid].document_id == "doc-pdf" for cid in dense_ids)
    assert all(chunks[cid].document_id == "doc-pdf" for cid in keyword_ids)


def test_search_filters_restrict_to_matching_section(tmp_path):
    embedder = FakeEmbeddingProvider(dimension=32)
    conn = _make_conn(tmp_path, dim=embedder.dimension)
    _insert_doc(conn, "doc-1", "politika.md", file_type=".md", content_hash="h1")

    sections = {
        "Güvenlik > Uzaktan Erişim": "ortak kelime vpn mfa zorunlu",
        "İnsan Kaynakları > İzin": "ortak kelime izin süresi kıdem",
    }
    for section, text in sections.items():
        db.insert_chunk(
            conn,
            document_id="doc-1",
            chunk_index=0,
            section_path=section,
            chunk_type="text",
            text=text,
            embedding=embedder.embed_one(text),
        )

    filters = SearchFilters(section_contains="Güvenlik")
    keyword_ids = db.keyword_search_ids(conn, "ortak kelime", top_k=5, filters=filters)
    chunks = db.get_chunks_by_ids(conn, keyword_ids)

    assert keyword_ids, "Filtreyle eşleşen bir chunk bekleniyordu"
    assert all("Güvenlik" in chunks[cid].section_path for cid in keyword_ids)


def test_search_filters_restrict_to_matching_classification_and_department(tmp_path):
    embedder = FakeEmbeddingProvider(dimension=16)
    conn = _make_conn(tmp_path, dim=embedder.dimension)
    db.insert_document(
        conn,
        document_id="doc-genel",
        filename="genel.md",
        title="genel",
        file_type=".md",
        content_hash="h-genel",
        language="tr",
        ingested_at="2026-07-20T00:00:00Z",
        classification="genel",
        department="Genel",
    )
    db.insert_document(
        conn,
        document_id="doc-gizli",
        filename="gizli.md",
        title="gizli",
        file_type=".md",
        content_hash="h-gizli",
        language="tr",
        ingested_at="2026-07-20T00:00:00Z",
        classification="gizli",
        department="Yönetim",
    )
    for doc_id in ("doc-genel", "doc-gizli"):
        db.insert_chunk(
            conn,
            document_id=doc_id,
            chunk_index=0,
            section_path="X",
            chunk_type="text",
            text="ortak kelime tekrar eden içerik",
            embedding=embedder.embed_one("ortak kelime tekrar eden içerik"),
        )

    filters = SearchFilters(classifications=["genel"])
    ids = db.keyword_search_ids(conn, "ortak kelime", top_k=5, filters=filters)
    chunks = db.get_chunks_by_ids(conn, ids)
    assert ids and all(chunks[cid].document_id == "doc-genel" for cid in ids)


def test_department_filter_gates_only_confidential_documents(tmp_path):
    """RBAC: 'genel' (herkese açık) dokümanlar departman kısıtından muaftır;
    departman filtresi yalnızca 'gizli' dokümanları bölümler.

    Regresyon: genel/IK bir izin politikası, gizli görme yetkisi departmanla
    sınırlı bir rolden (ör. bt_uzmani: dept=BT,Genel) GİZLENMEMELİ. Aksi halde
    departman-kısıtsız 'calisan', departman-kısıtlı 'bt_uzmani'den daha çok
    doküman görür — ayrıcalık ters dönmesi (bkz. security/rbac.py)."""
    embedder = FakeEmbeddingProvider(dimension=16)
    conn = _make_conn(tmp_path, dim=embedder.dimension)
    for doc_id, cls, dept in [
        ("genel-ik", "genel", "IK"),
        ("gizli-ik", "gizli", "IK"),
        ("genel-bt", "genel", "BT"),
        ("gizli-bt", "gizli", "BT"),
    ]:
        db.insert_document(
            conn,
            document_id=doc_id,
            filename=f"{doc_id}.md",
            title=doc_id,
            file_type=".md",
            content_hash=f"h-{doc_id}",
            language="tr",
            ingested_at="2026-07-20T00:00:00Z",
            classification=cls,
            department=dept,
        )
        db.insert_chunk(
            conn,
            document_id=doc_id,
            chunk_index=0,
            section_path="X",
            chunk_type="text",
            text="paylasilan ortak anahtar kelime",
            embedding=embedder.embed_one("paylasilan ortak anahtar kelime"),
        )

    # bt_uzmani benzeri: gizli görebilir ama yalnızca BT/Genel departmanlarında.
    filters = SearchFilters(classifications=["genel", "gizli"], departments=["BT", "Genel"])
    ids = db.keyword_search_ids(conn, "paylasilan anahtar", top_k=10, filters=filters)
    visible = {db.get_chunks_by_ids(conn, ids)[cid].document_id for cid in ids}

    assert "genel-ik" in visible  # herkese açık → departmandan bağımsız (asıl düzeltme)
    assert "genel-bt" in visible
    assert "gizli-bt" in visible
    assert "gizli-ik" not in visible  # gizli + yetkisiz departman → gizli kalır


def test_audit_chain_detects_valid_and_tampered_history(tmp_path):
    conn = _make_conn(tmp_path)
    db.insert_audit_entry(conn, role="calisan", query="soru 1", retrieved_chunk_ids=[1, 2], confident=True, answer="cevap 1")
    db.insert_audit_entry(conn, role="yonetici", query="soru 2", retrieved_chunk_ids=[], confident=False, answer="cevap 2")
    assert db.verify_audit_chain(conn) is True

    # Bir satırı normal API dışından (doğrudan SQL ile) değiştir — zincir kopmalı.
    conn.execute("UPDATE audit_log SET answer = 'değiştirilmiş cevap' WHERE id = 1")
    conn.commit()
    assert db.verify_audit_chain(conn) is False


def test_list_audit_entries_returns_most_recent_first(tmp_path):
    conn = _make_conn(tmp_path)
    db.insert_audit_entry(conn, role="calisan", query="ilk", retrieved_chunk_ids=[], confident=True, answer="a")
    db.insert_audit_entry(conn, role="calisan", query="ikinci", retrieved_chunk_ids=[], confident=True, answer="b")
    entries = db.list_audit_entries(conn, limit=10)
    assert entries[0]["query"] == "ikinci"
    assert entries[1]["query"] == "ilk"


def test_dense_candidate_k_capped_and_warns_when_corpus_exceeds_cap(tmp_path, monkeypatch, caplog):
    # RBAC dense-filter, aday havuzunu _DENSE_FILTER_CANDIDATE_CAP'te sabitler;
    # toplam chunk sayısı bunu aşarsa artık gözlemlenebilir bir uyarı loglanmalı
    # (eskiden tamamen sessizdi — bkz. storage/db.py modül üstü yorum).
    monkeypatch.setattr(db, "_DENSE_FILTER_CANDIDATE_CAP", 2)
    embedder = FakeEmbeddingProvider(dimension=16)
    conn = _make_conn(tmp_path, dim=embedder.dimension)
    _insert_doc(conn, "doc-1", "belge.md", file_type=".md", content_hash="h1")
    for i in range(4):
        db.insert_chunk(
            conn,
            document_id="doc-1",
            chunk_index=i,
            section_path="Bölüm",
            chunk_type="text",
            text=f"içerik parçası {i}",
            embedding=embedder.embed_one(f"içerik parçası {i}"),
        )

    filters = SearchFilters(file_types=[".md"])
    with caplog.at_level("WARNING"):
        candidate_k = db._dense_candidate_k(conn, top_k=1, filters=filters)

    assert candidate_k == 2
    assert any("_DENSE_FILTER_CANDIDATE_CAP" in record.message for record in caplog.records)


def test_dense_candidate_k_no_warning_when_within_cap(tmp_path, caplog):
    embedder = FakeEmbeddingProvider(dimension=16)
    conn = _make_conn(tmp_path, dim=embedder.dimension)
    _insert_doc(conn, "doc-1", "belge.md", file_type=".md", content_hash="h1")
    db.insert_chunk(
        conn,
        document_id="doc-1",
        chunk_index=0,
        section_path="Bölüm",
        chunk_type="text",
        text="tek chunk",
        embedding=embedder.embed_one("tek chunk"),
    )

    filters = SearchFilters(file_types=[".md"])
    with caplog.at_level("WARNING"):
        candidate_k = db._dense_candidate_k(conn, top_k=1, filters=filters)

    assert candidate_k == 1
    assert caplog.records == []
