"""RERANK_SCORE_THRESHOLD'u TR eval setiyle KALİBRE eder (bkz. config.py'deki
eşik notu). Foundry Local + reranker gerektirir; tamamen offline çalışır.

Her soru için en iyi reranker skorunu (retrieval + rerank, LLM ÇAĞIRMADAN — hızlı)
hesaplar ve tipe göre gruplar. Amaç: answerable soruları geçiren ama cevapsız/RBAC
sorularını reddeden bir eşik bulmak. bge-reranker-v2-m3'ün skor dağılımı bu
korpusta bimodaldir (gerçek eşleşmeler ile eşleşmeyenler arasında geniş bir
boşluk), bu yüzden ikisini temiz ayıran bir eşik seçilebilir.

Kullanım:
    .venv/Scripts/python.exe eval/calibrate_threshold.py
    .venv/Scripts/python.exe eval/calibrate_threshold.py --ids q01,q05
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from local_rag.config import (  # noqa: E402
    DEFAULT_DB_PATH,
    EMBEDDING_DIMENSION,
    RERANK_SCORE_THRESHOLD,
    TOP_N_CANDIDATES,
)
from local_rag.embeddings.foundry import FoundryEmbeddingProvider  # noqa: E402
from local_rag.pipeline import _apply_rbac  # noqa: E402
from local_rag.reranking.bge import BgeRerankerProvider  # noqa: E402
from local_rag.retrieval.search import hybrid_candidates  # noqa: E402
from local_rag.storage import db  # noqa: E402

DATASET_PATH = Path(__file__).parent / "dataset_tr.json"
SWEEP = [0.15, 0.10, 0.07, 0.05, 0.03, 0.02, 0.015, 0.01]


def _best_score(conn, embedder, reranker, query: str, role: str) -> float | None:
    """Retrieval + rerank sonrası en iyi (top-1) skoru döner; aday yoksa None."""
    filters = _apply_rbac(role or "calisan", None)
    candidates = hybrid_candidates(conn, embedder, query, top_n=TOP_N_CANDIDATES, filters=filters)
    if not candidates:
        return None
    scores = reranker.score(query, [c.text for c in candidates])
    return max(scores)  # pipeline eşiği sıralama sonrası top-1'e uygular = max skor


def main() -> None:
    parser = argparse.ArgumentParser(description="Reranker eşiğini TR eval setiyle kalibre eder.")
    parser.add_argument("--ids", type=str, default=None, help="Virgülle ayrılmış soru id listesi")
    args = parser.parse_args()

    questions = json.loads(DATASET_PATH.read_text(encoding="utf-8"))["questions"]
    if args.ids:
        wanted = set(args.ids.split(","))
        questions = [q for q in questions if q["id"] in wanted]

    print("Modeller yükleniyor (Foundry embedder + bge-reranker)...", flush=True)
    conn = db.get_connection(DEFAULT_DB_PATH, embedding_dimension=EMBEDDING_DIMENSION)
    embedder = FoundryEmbeddingProvider()
    reranker = BgeRerankerProvider()

    answerable: list[float] = []
    unanswerable: list[float] = []
    print(f"\nMevcut eşik = {RERANK_SCORE_THRESHOLD}\n" + "=" * 60)
    for q in questions:
        best = _best_score(conn, embedder, reranker, q["query"], q.get("role", "calisan"))
        label = "None(aday yok)" if best is None else f"{best:.3f}"
        print(f"  {q['id']:<8} {q['type']:<22} best={label:<14} «{q['query'][:44]}»")
        if q["type"] == "answerable" and best is not None:
            answerable.append(best)
        elif q["type"] in ("unanswerable_absent", "unanswerable_rbac"):
            unanswerable.append(0.0 if best is None else best)

    print("\n" + "=" * 60)
    print(f"answerable skorları   : {sorted(round(x, 3) for x in answerable)}")
    print(f"unanswerable skorları : {sorted(round(x, 3) for x in unanswerable)}")
    print("\nEŞİK TARAMASI (answerable geçmeli, unanswerable reddedilmeli):")
    for t in SWEEP:
        ans_pass = sum(1 for x in answerable if x >= t)
        una_reject = sum(1 for x in unanswerable if x < t)
        print(f"  eşik={t:<6} answerable geçen={ans_pass}/{len(answerable)}  unanswerable reddedilen={una_reject}/{len(unanswerable)}")

    if answerable and unanswerable:
        lo, hi = max(unanswerable), min(answerable)
        if hi > lo:
            print(f"\nÖneri: answerable min={hi:.3f} > unanswerable max={lo:.3f} — temiz ayrım.")
            print(f"       maximin orta nokta ≈ {round((lo + hi) / 2, 3)} (her iki sınıfa da marj bırakır).")
        else:
            print(f"\nUYARI: skorlar örtüşüyor (answerable min={hi:.3f} <= unanswerable max={lo:.3f}); "
                  "tek eşik ikisini temiz ayıramaz.")

    conn.close()


if __name__ == "__main__":
    main()
