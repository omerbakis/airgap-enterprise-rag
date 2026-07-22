"""Hybrid retrieval: dense (sqlite-vec) + BM25 (FTS5) + Reciprocal Rank Fusion.
Reranking pipeline.py'de, RerankerProvider ile ayrı bir adım olarak yapılır."""

from __future__ import annotations

import sqlite3

from local_rag.config import RRF_K, TOP_N_CANDIDATES
from local_rag.embeddings.base import EmbeddingProvider
from local_rag.storage.db import (
    RetrievedChunk,
    SearchFilters,
    dense_search_ids,
    get_chunks_by_ids,
    keyword_search_ids,
)


def reciprocal_rank_fusion(rankings: list[list[int]], k: int = RRF_K) -> list[int]:
    """Birden çok sıralı chunk_id listesini tek bir RRF-sıralı listeye birleştirir.

    score(id) = sum over rankings içeren her listede 1/(k + rank); id bir
    listede yoksa o listeden katkı almaz. k, ilk sıralardaki farkın etkisini
    yumuşatan standart bir sabittir (yaygın varsayılan: 60)."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores, key=lambda cid: scores[cid], reverse=True)


def hybrid_candidates(
    conn: sqlite3.Connection,
    embedder: EmbeddingProvider,
    query: str,
    top_n: int = TOP_N_CANDIDATES,
    filters: SearchFilters | None = None,
) -> list[RetrievedChunk]:
    """Dense + BM25 sonuçlarını RRF ile birleştirip en iyi top_n aday chunk'ı
    döner (henüz reranked/nihai top-k değil — bkz. pipeline.answer)."""
    query_embedding = embedder.embed_query(query)
    dense_ids = dense_search_ids(conn, query_embedding, top_n, filters)
    keyword_ids = keyword_search_ids(conn, query, top_n, filters)
    fused_ids = reciprocal_rank_fusion([dense_ids, keyword_ids])[:top_n]

    chunks_by_id = get_chunks_by_ids(conn, fused_ids)
    return [chunks_by_id[cid] for cid in fused_ids if cid in chunks_by_id]
