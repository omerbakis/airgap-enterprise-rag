"""CLI: bge-reranker-v2-m3'ü bir kez yükleyip HuggingFace yerel cache'ine
(`~/.cache/huggingface`) indirir — air-gapped hazırlığın "reranker modelini
önbelleğe al" adımı için deterministik bir alternatif (bkz.
docs/AIRGAP_KURULUM.md 1.3). İnternet gerektirir; sonraki çalıştırmalarda
zaten var olan cache'i kullanır (yeniden indirmez).

Kullanım:
    .venv/Scripts/python.exe scripts/prepare_reranker_cache.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from local_rag.config import RERANKER_MODEL_NAME  # noqa: E402
from local_rag.reranking.bge import BgeRerankerProvider  # noqa: E402


def main() -> None:
    print(f"'{RERANKER_MODEL_NAME}' indiriliyor/yükleniyor...")
    reranker = BgeRerankerProvider()

    print("Model yüklendi, bir örnek sorguyla doğrulanıyor...")
    scores = reranker.score("örnek sorgu", ["alakalı bir örnek metin", "alakasız bir başka metin"])
    assert len(scores) == 2 and all(0.0 <= s <= 1.0 for s in scores)

    print(f"Reranker modeli başarıyla önbelleğe alındı ve doğrulandı: {scores}")
    print("USB transferi için ~/.cache/huggingface/ klasörünü kopyalayın (bkz. docs/AIRGAP_KURULUM.md).")


if __name__ == "__main__":
    main()
