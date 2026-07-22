import os

import sqlcipher3

from local_rag.security.keys import DB_KEY_ENV_VAR, get_or_create_db_key
from local_rag.storage import db


def test_get_or_create_db_key_persists_across_calls(tmp_path):
    key_path = tmp_path / ".dbkey"
    first = get_or_create_db_key(key_path)
    second = get_or_create_db_key(key_path)
    assert first == second
    assert key_path.exists()


def test_get_or_create_db_key_prefers_env_var(tmp_path, monkeypatch):
    monkeypatch.setenv(DB_KEY_ENV_VAR, "env-provided-key")
    key_path = tmp_path / ".dbkey"
    key = get_or_create_db_key(key_path)
    assert key == "env-provided-key"
    assert not key_path.exists()  # env değişkeni varsa dosyaya yazılmaz


def test_database_file_is_actually_encrypted(tmp_path):
    db_path = tmp_path / "index.db"
    conn = db.get_connection(db_path, embedding_dimension=8, encryption_key="correct-key")
    db.insert_document(
        conn,
        document_id="doc-1",
        filename="a.txt",
        title="a",
        file_type=".txt",
        content_hash="h1",
        language="tr",
        ingested_at="2026-07-20T00:00:00Z",
    )
    conn.close()

    # Doğru anahtarla tekrar açılabilmeli.
    reopened = db.get_connection(db_path, embedding_dimension=8, encryption_key="correct-key")
    assert db.get_document_by_hash(reopened, "h1") is not None
    reopened.close()

    # Yanlış anahtarla okunamamalı (şifreleme gerçekten uygulanmış).
    wrong = sqlcipher3.connect(str(db_path))
    wrong.execute("PRAGMA key = 'wrong-key'")
    try:
        wrong.execute("SELECT * FROM documents").fetchall()
        assert False, "Yanlış anahtarla okuma başarılı oldu — şifreleme çalışmıyor!"
    except sqlcipher3.dbapi2.DatabaseError:
        pass
    finally:
        wrong.close()

    # Dosyanın kendisi düz metin SQLite header'ı ("SQLite format 3") İÇERMEMELİ.
    raw_bytes = db_path.read_bytes()[:16]
    assert b"SQLite format 3" not in raw_bytes
