"""SQLCipher (şifreli SQLite) + sqlite-vec + FTS5 depolama katmanı.

Tek bir `.db` dosyasında dört tablo yaşar: documents (doküman metadata'sı,
classification/department dahil — bkz. RBAC), chunks (chunk metni +
metadata), vec_chunks (sqlite-vec vec0 sanal tablosu, dense KNN) ve
chunks_fts (FTS5 sanal tablosu, BM25 keyword arama) — ikincisi chunks
tablosuna trigger'larla senkron tutulur. Hybrid arama, bu iki indeksin
sonuçlarını RRF ile birleştirir (bkz. retrieval/search.py).

Dosya, `sqlcipher3` (SQLCipher'ın Python DB-API 2.0 bağlayıcısı — stdlib
`sqlite3` ile aynı arayüz, drop-in) üzerinden AES-256 ile şifreli açılır;
anahtar `security/keys.py` tarafından yönetilir. `sqlcipher3.Row`,
stdlib `sqlite3.Row` ile ikame edilemez (farklı C tipleri) — bu yüzden
tüm bağlantılarda `sqlcipher3.Row` kullanılmalıdır. Kodun geri kalanında
(pipeline.py, retrieval/search.py, testler) yer alan `sqlite3.Connection`/
`sqlite3.Row` tip belirteçleri, DB-API 2.0 arayüzünü belgeleyen yapısal bir
referanstır — çalışma zamanındaki gerçek nesneler buradan üretilen
sqlcipher3 nesneleridir.

Not: Şifresiz (eski) bir `.db` dosyası bu sürümle uyumlu değildir (SQLCipher
formatı farklıdır) — böyle bir `data/index.db` varsa silinip yeniden ingest
edilmelidir.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import sqlite3  # yalnızca tip belirteçleri için (bkz. modül docstring'i)
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import sqlcipher3
import sqlite_vec

from local_rag.config import EMBEDDING_DIMENSION, PUBLIC_CLASSIFICATION
from local_rag.security.keys import get_or_create_db_key

# Filtre aktifken (bkz. _dense_candidate_k) taranacak azami aday sayısı. Gerçek
# sqlite-vec gerçek "partition key" pre-filtering desteklemiyor; bunun yerine
# KNN'i geniş bir aday havuzuyla (candidate_k) çalıştırıp SONRA SQL WHERE ile
# RBAC filtresi uyguluyoruz (post-filter). Bu, YALNIZCA toplam chunk sayısı
# _DENSE_FILTER_CANDIDATE_CAP'in altında kaldığı sürece tam doğrudur (o zaman
# candidate_k=total olur, yani TÜM chunk'lar adaydır, filtre hiçbir chunk'ı
# kaybetmez). Toplam bunu aşarsa candidate_k CAP'te sabitlenir — yalnızca ham
# (RBAC'tan bağımsız) mesafeye göre en yakın CAP kadar aday filtrelenir; bu
# adayların DIŞINDA kalan ama filtreye uyan bir chunk sonuçlardan sessizce
# düşebilir (bkz. _dense_candidate_k'daki runtime uyarısı). Çok daha büyük
# ölçekte (>1M chunk) gerçek partition-key desteğine geçilmelidir.
_DENSE_FILTER_CANDIDATE_CAP = 2000

_logger = logging.getLogger(__name__)


def get_connection(
    db_path: Path,
    embedding_dimension: int = EMBEDDING_DIMENSION,
    encryption_key: str | None = None,
) -> sqlcipher3.Connection:
    """`encryption_key` verilmezse `LOCAL_RAG_DB_KEY` ortam değişkeni, o da
    yoksa `db_path` ile aynı dizindeki `.dbkey` dosyası kullanılır (bkz.
    security/keys.py). Testler genelde bu varsayılanı, izole tmp_path
    dizininde otomatik üretilen bir anahtarla kullanır."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    key = encryption_key or get_or_create_db_key(db_path.parent / ".dbkey")

    # check_same_thread=False: Streamlit gibi çok thread'li sunucularda, cache_resource
    # ile önbelleklenen bir pipeline'ın bağlantısı oluşturulduğu thread dışında da
    # (örn. bir sonraki script rerun'unda) kullanılabilir olmalı. Eşzamanlı erişim
    # RagPipeline'daki kilit (bkz. pipeline.py) ile serileştirilir.
    conn = sqlcipher3.connect(str(db_path), check_same_thread=False)
    # PRAGMA key parametre bağlama (?) desteklemez; anahtar kendi ürettiğimiz
    # bir secret olduğundan (hex ya da operatör tarafından sağlanan passphrase),
    # tek tırnak kaçışlanarak literal olarak gömülür.
    escaped_key = key.replace("'", "''")
    conn.execute(f"PRAGMA key = '{escaped_key}'")
    conn.row_factory = sqlcipher3.Row
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    _init_schema(conn, embedding_dimension)
    return conn


def _init_schema(conn: sqlite3.Connection, embedding_dimension: int) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS documents (
            id TEXT PRIMARY KEY,
            filename TEXT NOT NULL,
            title TEXT,
            file_type TEXT NOT NULL,
            classification TEXT NOT NULL DEFAULT 'genel',
            department TEXT NOT NULL DEFAULT 'Genel',
            content_hash TEXT NOT NULL,
            language TEXT,
            ingested_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
            chunk_index INTEGER NOT NULL,
            section_path TEXT,
            chunk_type TEXT NOT NULL,
            page_start INTEGER,
            page_end INTEGER,
            text TEXT NOT NULL,
            injection_flag TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks(document_id);

        CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
            text, content='chunks', content_rowid='id'
        );

        CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
            INSERT INTO chunks_fts(rowid, text) VALUES (new.id, new.text);
        END;

        CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
            INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES ('delete', old.id, old.text);
        END;

        CREATE TABLE IF NOT EXISTS audit_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            role TEXT NOT NULL,
            query TEXT NOT NULL,
            retrieved_chunk_ids TEXT NOT NULL,
            confident INTEGER NOT NULL,
            answer TEXT NOT NULL,
            prev_hash TEXT NOT NULL,
            row_hash TEXT NOT NULL
        );
        """
    )
    conn.execute(
        f"""
        CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks USING vec0(
            chunk_id INTEGER PRIMARY KEY,
            embedding FLOAT[{embedding_dimension}]
        )
        """
    )
    conn.commit()


def get_document_by_hash(conn: sqlite3.Connection, content_hash: str) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM documents WHERE content_hash = ?", (content_hash,)).fetchone()


def list_documents(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    """UI'da dosya türü/doküman filtresi seçenekleri doldurmak için kullanılır."""
    return conn.execute(
        "SELECT id, filename, file_type, classification, department, language FROM documents ORDER BY filename"
    ).fetchall()


def delete_document(conn: sqlite3.Connection, document_id: str) -> None:
    chunk_ids = [row["id"] for row in conn.execute("SELECT id FROM chunks WHERE document_id = ?", (document_id,))]
    for chunk_id in chunk_ids:
        conn.execute("DELETE FROM vec_chunks WHERE chunk_id = ?", (chunk_id,))
    conn.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
    conn.execute("DELETE FROM documents WHERE id = ?", (document_id,))
    conn.commit()


def insert_document(
    conn: sqlite3.Connection,
    document_id: str,
    filename: str,
    title: str,
    file_type: str,
    content_hash: str,
    language: str,
    ingested_at: str,
    classification: str = "genel",
    department: str = "Genel",
) -> None:
    conn.execute(
        """
        INSERT INTO documents
            (id, filename, title, file_type, classification, department, content_hash, language, ingested_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (document_id, filename, title, file_type, classification, department, content_hash, language, ingested_at),
    )
    conn.commit()


def insert_chunk(
    conn: sqlite3.Connection,
    document_id: str,
    chunk_index: int,
    section_path: str,
    chunk_type: str,
    text: str,
    embedding: list[float],
    page_start: int | None = None,
    page_end: int | None = None,
    injection_flag: str | None = None,
) -> int:
    cursor = conn.execute(
        """
        INSERT INTO chunks
            (document_id, chunk_index, section_path, chunk_type, page_start, page_end, text, injection_flag)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (document_id, chunk_index, section_path, chunk_type, page_start, page_end, text, injection_flag),
    )
    chunk_id = cursor.lastrowid
    conn.execute(
        "INSERT INTO vec_chunks (chunk_id, embedding) VALUES (?, ?)",
        (chunk_id, sqlite_vec.serialize_float32(embedding)),
    )
    conn.commit()
    return chunk_id


def list_flagged_chunks(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    """Prompt-injection taramasında şüpheli bulunan chunk'ları (insan
    incelemesi kuyruğu) döner (bkz. security/injection.py)."""
    return conn.execute(
        """
        SELECT c.id AS chunk_id, c.injection_flag AS injection_flag, c.text AS text,
               c.section_path AS section_path, d.filename AS source_filename
        FROM chunks c
        JOIN documents d ON d.id = c.document_id
        WHERE c.injection_flag IS NOT NULL
        ORDER BY c.id
        """
    ).fetchall()


@dataclass
class RetrievedChunk:
    chunk_id: int
    text: str
    section_path: str
    chunk_type: str
    document_id: str
    source_filename: str
    distance: float | None = None
    page_start: int | None = None
    page_end: int | None = None


@dataclass
class SearchFilters:
    document_ids: list[str] | None = None
    file_types: list[str] | None = None
    section_contains: str | None = None
    """Bölüm (section_path) içinde geçen bir alt dize — ör. 'Güvenlik' verilirse
    yalnızca section_path'i bu ifadeyi içeren chunk'larda arama yapılır
    (dosya türü/bölüm filtreleri)."""
    classifications: list[str] | None = None
    departments: list[str] | None = None
    """RBAC tarafından doldurulur (bkz. security/rbac.py, pipeline.answer) —
    kullanıcının rolüne göre izin verilen classification/department değerleri.
    UI'nın seçtiği file_types/section_contains ile birlikte AND'lenir; yani
    kullanıcı filtreleri rolün izin verdiği kümeyi asla genişletemez, yalnızca
    daraltabilir."""

    @property
    def is_empty(self) -> bool:
        return not any(
            [
                self.document_ids,
                self.file_types,
                self.section_contains,
                self.classifications,
                self.departments,
            ]
        )


def _filter_clause(filters: SearchFilters | None) -> tuple[str, list]:
    if filters is None or filters.is_empty:
        return "", []
    clauses = []
    params: list = []
    if filters.document_ids:
        clauses.append(f"c.document_id IN ({','.join('?' * len(filters.document_ids))})")
        params.extend(filters.document_ids)
    if filters.file_types:
        clauses.append(f"d.file_type IN ({','.join('?' * len(filters.file_types))})")
        params.extend(filters.file_types)
    if filters.section_contains:
        clauses.append("c.section_path LIKE ?")
        params.append(f"%{filters.section_contains}%")
    if filters.classifications:
        clauses.append(f"d.classification IN ({','.join('?' * len(filters.classifications))})")
        params.extend(filters.classifications)
    if filters.departments:
        # Departman kısıtı YALNIZCA gizli (genel olmayan) dokümanlara uygulanır:
        # "genel" sınıflandırmalı dokümanlar herkese açıktır ve departmandan
        # bağımsız görünür. Aksi halde bir departmanın herkese açık dokümanı
        # (ör. genel/IK izin politikası) diğer departman rollerinden gizlenir; o
        # zaman departman-kısıtsız "calisan" rolü, departman-kısıtlı "bt_uzmani"
        # rolünden DAHA çok doküman görür — ayrıcalık ters dönmesi (bkz. rbac.py).
        dept_placeholders = ",".join("?" * len(filters.departments))
        clauses.append(f"(d.classification = ? OR d.department IN ({dept_placeholders}))")
        params.append(PUBLIC_CLASSIFICATION)
        params.extend(filters.departments)
    return " AND " + " AND ".join(clauses), params


def _dense_candidate_k(conn: sqlite3.Connection, top_k: int, filters: SearchFilters | None) -> int:
    if filters is None or filters.is_empty:
        return top_k
    total = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    if total > _DENSE_FILTER_CANDIDATE_CAP:
        _logger.warning(
            "RBAC dense-filter: toplam chunk sayısı (%d) _DENSE_FILTER_CANDIDATE_CAP'i "
            "(%d) aşıyor — yalnızca ham mesafeye göre en yakın %d aday RBAC filtresinden "
            "geçirilecek; bu adayların dışında kalan ama filtreye uyan bir chunk sonuçlardan "
            "sessizce düşebilir.",
            total, _DENSE_FILTER_CANDIDATE_CAP, _DENSE_FILTER_CANDIDATE_CAP,
        )
    return min(max(total, 1), _DENSE_FILTER_CANDIDATE_CAP)


def dense_search_ids(
    conn: sqlite3.Connection,
    query_embedding: list[float],
    top_k: int,
    filters: SearchFilters | None = None,
) -> list[int]:
    """En yakın chunk_id'leri (en yakından en uzağa) döner."""
    candidate_k = _dense_candidate_k(conn, top_k, filters)
    where, params = _filter_clause(filters)
    rows = conn.execute(
        f"""
        SELECT c.id AS chunk_id
        FROM vec_chunks v
        JOIN chunks c ON c.id = v.chunk_id
        JOIN documents d ON d.id = c.document_id
        WHERE v.embedding MATCH ? AND v.k = ?{where}
        ORDER BY v.distance
        LIMIT ?
        """,
        (sqlite_vec.serialize_float32(query_embedding), candidate_k, *params, top_k),
    ).fetchall()
    return [row["chunk_id"] for row in rows]


_FTS5_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def _build_fts5_match_query(text: str) -> str | None:
    """Serbest metin sorgusunu güvenli bir FTS5 MATCH ifadesine çevirir.

    Her kelime çift tırnak içine alınır (FTS5'in `-`, `:`, `(` gibi operatör
    karakterlerini literal olarak ele alması için) ve OR ile birleştirilir —
    amaç, dense aramanın tamamlayıcısı olacak geniş bir recall sağlamaktır."""
    tokens = _FTS5_TOKEN_RE.findall(text)
    if not tokens:
        return None
    escaped = [t.replace('"', '""') for t in tokens]
    return " OR ".join(f'"{t}"' for t in escaped)


def keyword_search_ids(
    conn: sqlite3.Connection,
    query_text: str,
    top_k: int,
    filters: SearchFilters | None = None,
) -> list[int]:
    """BM25'e göre en iyi eşleşen chunk_id'leri (en iyiden en kötüye) döner."""
    match_query = _build_fts5_match_query(query_text)
    if match_query is None:
        return []
    where, params = _filter_clause(filters)
    # bm25() FTS5 yardımcı fonksiyonu, FROM'daki alias'ı değil yalnızca gerçek
    # sanal tablo adını kabul eder (doğrulandı: alias ile "no such column"
    # hatası verir) — bu yüzden burada "f" değil "chunks_fts" kullanılır.
    rows = conn.execute(
        f"""
        SELECT c.id AS chunk_id
        FROM chunks_fts f
        JOIN chunks c ON c.id = f.rowid
        JOIN documents d ON d.id = c.document_id
        WHERE f.text MATCH ?{where}
        ORDER BY bm25(chunks_fts)
        LIMIT ?
        """,
        (match_query, *params, top_k),
    ).fetchall()
    return [row["chunk_id"] for row in rows]


def get_chunks_by_ids(conn: sqlite3.Connection, chunk_ids: list[int]) -> dict[int, RetrievedChunk]:
    if not chunk_ids:
        return {}
    placeholders = ",".join("?" * len(chunk_ids))
    rows = conn.execute(
        f"""
        SELECT
            c.id AS chunk_id, c.text AS text, c.section_path AS section_path,
            c.chunk_type AS chunk_type, c.document_id AS document_id,
            c.page_start AS page_start, c.page_end AS page_end,
            d.filename AS source_filename
        FROM chunks c
        JOIN documents d ON d.id = c.document_id
        WHERE c.id IN ({placeholders})
        """,
        chunk_ids,
    ).fetchall()
    return {
        row["chunk_id"]: RetrievedChunk(
            chunk_id=row["chunk_id"],
            text=row["text"],
            section_path=row["section_path"],
            chunk_type=row["chunk_type"],
            document_id=row["document_id"],
            source_filename=row["source_filename"],
            page_start=row["page_start"],
            page_end=row["page_end"],
        )
        for row in rows
    }


def dense_search(
    conn: sqlite3.Connection,
    query_embedding: list[float],
    top_k: int,
    filters: SearchFilters | None = None,
) -> list[RetrievedChunk]:
    """Dense-only arama (mesafe dahil). Erken bir API ile geriye dönük uyumluluk
    ve testler için korunur; uygulama artık hybrid_search kullanır (bkz. retrieval/search.py)."""
    candidate_k = _dense_candidate_k(conn, top_k, filters)
    where, params = _filter_clause(filters)
    rows = conn.execute(
        f"""
        SELECT
            c.id AS chunk_id, c.text AS text, c.section_path AS section_path,
            c.chunk_type AS chunk_type, c.document_id AS document_id,
            c.page_start AS page_start, c.page_end AS page_end,
            d.filename AS source_filename, v.distance AS distance
        FROM vec_chunks v
        JOIN chunks c ON c.id = v.chunk_id
        JOIN documents d ON d.id = c.document_id
        WHERE v.embedding MATCH ? AND v.k = ?{where}
        ORDER BY v.distance
        LIMIT ?
        """,
        (sqlite_vec.serialize_float32(query_embedding), candidate_k, *params, top_k),
    ).fetchall()
    return [
        RetrievedChunk(
            chunk_id=row["chunk_id"],
            text=row["text"],
            section_path=row["section_path"],
            chunk_type=row["chunk_type"],
            document_id=row["document_id"],
            source_filename=row["source_filename"],
            distance=row["distance"],
            page_start=row["page_start"],
            page_end=row["page_end"],
        )
        for row in rows
    ]


# --------------------------------------------------------------------------
# Audit log (hash-chained, append-only)
# --------------------------------------------------------------------------
# Her satır bir öncekinin hash'ini içerir (blockchain'deki gibi basit bir
# hash-chain); bu modülde UPDATE/DELETE için hiçbir fonksiyon YOKTUR — tabloya
# yazmanın tek yolu insert_audit_entry'dir. verify_audit_chain(), zinciri baştan
# sona yeniden hesaplayarak sonradan değiştirilmiş/araya silinmiş bir satır
# olup olmadığını tespit eder (tamper-evidence, tamper-proof değil — dosyaya
# tam erişimi olan biri conn.execute ile tabloyu manuel olarak da bozabilir;
# asıl garanti "normal API üzerinden sessiz değişiklik yapılamaz" olmasıdır).

_AUDIT_GENESIS_HASH = "0" * 64


def _compute_audit_hash(
    prev_hash: str, ts: str, role: str, query: str, chunk_ids_json: str, confident: int, answer: str
) -> str:
    payload = "|".join([prev_hash, ts, role, query, chunk_ids_json, str(confident), answer])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def insert_audit_entry(
    conn: sqlite3.Connection,
    role: str,
    query: str,
    retrieved_chunk_ids: list[int],
    confident: bool,
    answer: str,
) -> int:
    prev_row = conn.execute("SELECT row_hash FROM audit_log ORDER BY id DESC LIMIT 1").fetchone()
    prev_hash = prev_row["row_hash"] if prev_row else _AUDIT_GENESIS_HASH
    ts = datetime.now(timezone.utc).isoformat()
    chunk_ids_json = json.dumps(retrieved_chunk_ids)
    row_hash = _compute_audit_hash(prev_hash, ts, role, query, chunk_ids_json, int(confident), answer)
    cursor = conn.execute(
        """
        INSERT INTO audit_log (ts, role, query, retrieved_chunk_ids, confident, answer, prev_hash, row_hash)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (ts, role, query, chunk_ids_json, int(confident), answer, prev_hash, row_hash),
    )
    conn.commit()
    return cursor.lastrowid


def list_audit_entries(conn: sqlite3.Connection, limit: int = 100) -> list[sqlite3.Row]:
    return conn.execute("SELECT * FROM audit_log ORDER BY id DESC LIMIT ?", (limit,)).fetchall()


def verify_audit_chain(conn: sqlite3.Connection) -> bool:
    """Zincirdeki her satırın hash'ini yeniden hesaplayıp kayıtlı hash'le
    karşılaştırır. Herhangi bir satır sonradan değiştirilmiş veya araya bir
    satır silinmişse zincir kopar ve False döner."""
    rows = conn.execute(
        "SELECT ts, role, query, retrieved_chunk_ids, confident, answer, prev_hash, row_hash "
        "FROM audit_log ORDER BY id"
    ).fetchall()
    expected_prev = _AUDIT_GENESIS_HASH
    for row in rows:
        if row["prev_hash"] != expected_prev:
            return False
        recomputed = _compute_audit_hash(
            row["prev_hash"],
            row["ts"],
            row["role"],
            row["query"],
            row["retrieved_chunk_ids"],
            row["confident"],
            row["answer"],
        )
        if recomputed != row["row_hash"]:
            return False
        expected_prev = row["row_hash"]
    return True
