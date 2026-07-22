"""Retrieval değerlendirme metrikleri.

Deterministik, LLM'e ihtiyaç duymayan metrikler — eval veri setindeki her
soru için "hangi doküman(lar) alakalı" bilgisiyle, gerçek retrieval
sonucunu (sıralı dosya adı listesi) karşılaştırır. Generation kalitesi
ise deterministik grounding kontrolleriyle ölçülür (confident/RBAC-red/
cevapsız davranışı, zorunlu kaynak künyesi, kaba bağlam-örtüşmesi sinyali);
ağır LLM-judge değerlendirmesi (Ragas) değerlendirilip bilinçli olarak
projeden çıkarıldı."""

from __future__ import annotations

import math


def dedupe_preserve_order(items: list[str]) -> list[str]:
    """Bir sıralı chunk listesini, ilk görülme sırasını koruyarak benzersiz
    doküman listesine indirger. precision_at_k/recall_at_k gibi metrikler tek
    bir 'relevant' öğe varsayımıyla (döküman düzeyinde) çalışır; retrieved
    listesi chunk düzeyinde olup aynı dokümandan birden çok chunk içerebilir
    — bu fonksiyon olmadan recall 1.0'ı aşabilir (bkz. eval/run_eval.py)."""
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if k <= 0:
        return 0.0
    top_k = retrieved[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for item in top_k if item in relevant)
    return hits / len(top_k)


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    hits = sum(1 for item in retrieved[:k] if item in relevant)
    return hits / len(relevant)


def reciprocal_rank(retrieved: list[str], relevant: set[str]) -> float:
    for rank, item in enumerate(retrieved, start=1):
        if item in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """İkili (binary) alaka derecesiyle nDCG@k."""
    top_k = retrieved[:k]
    dcg = sum(1.0 / math.log2(rank + 1) for rank, item in enumerate(top_k, start=1) if item in relevant)
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0
