"""TR eval setini gerçek pipeline üzerinden çalıştırıp bir rapor üretir.
Foundry Local + reranker gerektirir; tamamen offline çalışır.

Değerlendirme, hafif ve deterministik metriklere dayanır (precision@k, recall@k,
MRR, nDCG + grounding/injection kontrolleri). Ağır LLM-judge değerlendirmesi
(Ragas) değerlendirilip bilinçli olarak projeden çıkarıldı — CPU'da ~9dk/soru
sürmesi ve ~27 paketlik bir bağımlılık zinciri getirmesi nedeniyle.

Kullanım:
    .venv/Scripts/python.exe eval/run_eval.py                # tüm eval seti
    .venv/Scripts/python.exe eval/run_eval.py --ids q01,q05  # yalnızca belirli sorular
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from local_rag.config import DEFAULT_DB_PATH, TOP_K  # noqa: E402
from local_rag.embeddings.foundry import FoundryEmbeddingProvider  # noqa: E402
from local_rag.evaluation.metrics import (  # noqa: E402
    dedupe_preserve_order,
    mean,
    ndcg_at_k,
    precision_at_k,
    reciprocal_rank,
    recall_at_k,
)
from local_rag.llm.foundry import FoundryChatProvider  # noqa: E402
from local_rag.pipeline import RagPipeline, SourceRef  # noqa: E402
from local_rag.reranking.bge import BgeRerankerProvider  # noqa: E402

DATASET_PATH = Path(__file__).parent / "dataset_tr.json"
REPORT_PATH = Path(__file__).parent / "last_eval_report.json"

# pipeline._append_source_legend'in eklediği koşulsuz legend'ı ayırt etmek için:
# legend'daki [Sn] etiketleri her zaman doğru kaynağı gösterir (programatik
# olarak üretilir), bu yüzden "modelin KENDİLİĞİNDEN doğru atıf yaptı mı"
# sorusunu yanıtlamak için legend'dan ÖNCEKİ kısım incelenmeli — aksi halde
# citation_correct her zaman triviyal olarak doğru çıkar (retrieval_correct'ın
# bir kopyası olur, gerçek bir sinyal taşımaz).
_LEGEND_MARKER = "\n\n**Kaynaklar:**"
_CITATION_TAG_RE = re.compile(r"\[S(\d+)\]")


def load_dataset() -> list[dict]:
    return json.loads(DATASET_PATH.read_text(encoding="utf-8"))["questions"]


def evaluate_answerable(
    row: dict, q: dict, retrieved_filenames: list[str], answer: str, sources: list[SourceRef]
) -> None:
    # retrieved_filenames chunk düzeyindedir (aynı dokümandan birden çok chunk
    # gelebilir); precision/recall doküman düzeyinde tanımlı olduğundan önce
    # tekilleştirilir — aksi halde recall 1.0'ı aşabilir (bkz. test_evaluation_metrics.py).
    retrieved_docs = dedupe_preserve_order(retrieved_filenames)
    relevant = {q["expected_source_filename"]}
    k = max(len(retrieved_docs), 1)
    row["precision"] = precision_at_k(retrieved_docs, relevant, k)
    row["recall"] = recall_at_k(retrieved_docs, relevant, k)
    row["reciprocal_rank"] = reciprocal_rank(retrieved_docs, relevant)
    row["ndcg"] = ndcg_at_k(retrieved_docs, relevant, k)
    row["retrieval_correct"] = q["expected_source_filename"] in retrieved_docs
    keywords = q.get("expected_keywords", [])
    row["keyword_match"] = any(kw.lower() in answer.lower() for kw in keywords) if keywords else None

    inline_answer = answer.split(_LEGEND_MARKER, 1)[0]
    cited_indices = {int(m) for m in _CITATION_TAG_RE.findall(inline_answer)}
    cited_filenames = {sources[i - 1].filename for i in cited_indices if 1 <= i <= len(sources)}
    row["citation_present"] = bool(cited_indices)
    row["citation_correct"] = (q["expected_source_filename"] in cited_filenames) if cited_indices else False


def main() -> None:
    parser = argparse.ArgumentParser(description="TR eval setini çalıştırır.")
    parser.add_argument("--ids", type=str, default=None, help="Virgülle ayrılmış soru id listesi (ör. q01,q05,llm01)")
    parser.add_argument(
        "--top-k", type=int, default=TOP_K,
        help=f"LLM bağlamına giden nihai chunk sayısı (config.TOP_K={TOP_K}). Deneysel karşılaştırma için (ör. hız/kalite denemesi) config.py'yi değiştirmeden farklı bir değer denemeyi sağlar.",
    )
    args = parser.parse_args()
    print(f"TOP_K = {args.top_k}" + (" (varsayılan)" if args.top_k == TOP_K else f" (config varsayılanı {TOP_K}'ten override)"))

    embedder = FoundryEmbeddingProvider()
    reranker = BgeRerankerProvider()
    llm = FoundryChatProvider()
    pipeline = RagPipeline(embedder=embedder, reranker=reranker, llm=llm, db_path=DEFAULT_DB_PATH)

    questions = load_dataset()
    if args.ids:
        wanted = set(args.ids.split(","))
        questions = [q for q in questions if q["id"] in wanted]
    results: list[dict] = []
    retrieval_scores: dict[str, list[float]] = {"precision": [], "recall": [], "rr": [], "ndcg": []}
    latencies: list[float] = []

    for q in questions:
        print(f"... {q['id']} pipeline.answer() çağrılıyor", flush=True)
        result = pipeline.answer(q["query"], role=q.get("role", "calisan"), top_k=args.top_k)
        print(f"... {q['id']} pipeline.answer() bitti ({result.latency_seconds:.1f}s)", flush=True)
        retrieved_filenames = [s.filename for s in result.sources]
        row: dict = {
            "id": q["id"],
            "type": q["type"],
            "confident": result.confident,
            "latency_seconds": result.latency_seconds,
            "answer": result.answer,  # denetlenebilirlik: rapor tek başına neyin test edildiğini göstersin
        }
        if result.latency_seconds is not None:
            latencies.append(result.latency_seconds)

        if q["type"] == "answerable":
            evaluate_answerable(row, q, retrieved_filenames, result.answer, result.sources)
            row["confident_as_expected"] = result.confident is True
            # confident=True olması tek başına "OK" için yeterli değildir — cevap
            # metninin beklenen içeriği gerçekten taşıması da gerekir. keyword_match
            # daha önce hesaplanıyor ama hiçbir yerde kontrol edilmiyordu; bu yüzden
            # "confident_as_expected"e dayanan eski durum, retrieval doğru+confident
            # olsa bile LLM'in ürettiği metin yanlışsa bunu maskeliyordu (gerçek vaka:
            # q17, "4 milyon" beklenirken bazı koşumlarda "4 milyendir" üretildi).
            row["status_ok"] = row["confident_as_expected"] and (row["keyword_match"] is not False)
            retrieval_scores["precision"].append(row["precision"])
            retrieval_scores["recall"].append(row["recall"])
            retrieval_scores["rr"].append(row["reciprocal_rank"])
            retrieval_scores["ndcg"].append(row["ndcg"])
        elif q["type"] in ("unanswerable_absent", "unanswerable_rbac"):
            row["confident_as_expected"] = result.confident is False
            row["status_ok"] = row["confident_as_expected"]
        elif q["type"] == "injection_resistance":
            leaked_phrases = q.get("expected_should_not_contain", [])
            row["injection_resisted"] = not any(p.lower() in result.answer.lower() for p in leaked_phrases)
            row["status_ok"] = row["injection_resisted"]
        else:
            row["status_ok"] = True

        results.append(row)
        status = "OK" if row["status_ok"] else "FAIL"
        print(f"[{status}] {q['id']} ({q['type']}) confident={result.confident} ({result.latency_seconds:.1f}s)")

    print("\n=== ÖZET (yalnızca 'answerable' sorular) ===")
    print(f"Mean Precision: {mean(retrieval_scores['precision']):.2f}")
    print(f"Mean Recall:    {mean(retrieval_scores['recall']):.2f}")
    print(f"MRR:            {mean(retrieval_scores['rr']):.2f}")
    print(f"nDCG:           {mean(retrieval_scores['ndcg']):.2f}")

    keyword_checks = [r["keyword_match"] for r in results if r.get("keyword_match") is not None]
    if keyword_checks:
        print(f"Anahtar kelime doğruluğu (cevap metni bekleneni içeriyor mu): {sum(keyword_checks)}/{len(keyword_checks)}")

    citation_checks = [r["citation_correct"] for r in results if "citation_correct" in r]
    if citation_checks:
        present = sum(1 for r in results if r.get("citation_present"))
        print(
            f"Kendiliğinden doğru atıf ([Sn] doğru kaynağı gösteriyor mu): "
            f"{sum(citation_checks)}/{len(citation_checks)} (model hiç etiket kullandı: {present}/{len(citation_checks)})"
        )

    confident_checks = [r["confident_as_expected"] for r in results if "confident_as_expected" in r]
    if confident_checks:
        print(f"Grounding doğruluğu (confident beklenen gibi mi): {sum(confident_checks)}/{len(confident_checks)}")

    overall_checks = [r["status_ok"] for r in results if "status_ok" in r]
    if overall_checks:
        print(f"TOPLAM durum (confident + cevap içeriği doğruluğu birlikte): {sum(overall_checks)}/{len(overall_checks)}")

    injection_checks = [r["injection_resisted"] for r in results if "injection_resisted" in r]
    if injection_checks:
        print(f"Injection direnci: {sum(injection_checks)}/{len(injection_checks)}")

    if latencies:
        print("\n=== YANIT SÜRESİ (tüm sorular, saniye) ===")
        print(f"Ortalama: {statistics.mean(latencies):.1f}s")
        print(f"Medyan:   {statistics.median(latencies):.1f}s")
        print(f"Min/Max:  {min(latencies):.1f}s / {max(latencies):.1f}s")

    # top_k override edilmişse resmi (config varsayılanı TOP_K ile koşulmuş)
    # last_eval_report.json'ın üzerine yazmayız — dokümanlarda referans verilen
    # temel koşum bu dosyadır (bkz. docs/DEMO_SENARYOLARI.md).
    report_path = REPORT_PATH if args.top_k == TOP_K else REPORT_PATH.with_stem(f"{REPORT_PATH.stem}_topk{args.top_k}")
    report_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nDetaylı rapor: {report_path}")

    pipeline.close()


if __name__ == "__main__":
    main()
