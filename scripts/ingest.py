"""CLI: bir dizindeki dokümanları ayrıştırır, chunk'lar ve SQLite'a embed eder.

Kullanım:
    .venv/Scripts/python.exe scripts/ingest.py --docs data/documents --db data/index.db
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from local_rag.config import DEFAULT_DB_PATH, DEFAULT_DOCS_DIR  # noqa: E402
from local_rag.embeddings.foundry import FoundryEmbeddingProvider  # noqa: E402
from local_rag.pipeline import RagPipeline  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Dokümanları RAG index'ine ekler.")
    parser.add_argument("--docs", type=Path, default=DEFAULT_DOCS_DIR, help="Doküman dizini")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH, help="SQLite veritabanı yolu")
    args = parser.parse_args()

    if not args.docs.exists():
        parser.error(f"Doküman dizini bulunamadı: {args.docs}")

    print("Foundry Local'a bağlanılıyor (embedding modeli)...")
    embedder = FoundryEmbeddingProvider()

    pipeline = RagPipeline(embedder=embedder, db_path=args.db)
    try:
        print(f"'{args.docs}' taranıyor...")
        for line in pipeline.ingest_path(args.docs):
            print(line)
    finally:
        pipeline.close()


if __name__ == "__main__":
    main()
