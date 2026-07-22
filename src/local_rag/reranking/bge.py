from __future__ import annotations

import os

# Air-gapped garanti: HuggingFace kütüphanelerini offline'a zorla. Reranker
# modeli ilk indirmeden sonra YALNIZCA yerel cache'den (~/.cache/huggingface)
# yüklenir, ağa hiç çıkmaz. Bu satırlar `sentence_transformers`/`huggingface_hub`
# import edilmeden ÖNCE gelmelidir; aksi halde huggingface_hub, modeli
# kullanmadan önce Hub'a "güncel mi?" kontrolü yapar ve ağ yoksa (ör. wifi
# kapalı) `Cannot send a request, as the client has been closed` ile çöker.
# setdefault kullanıldığı için ilk indirme sırasında HF_HUB_OFFLINE=0 ile
# geçici olarak override edilebilir (bkz. docs/AIRGAP_KURULUM.md).
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from sentence_transformers import CrossEncoder  # noqa: E402

from local_rag.config import RERANKER_MODEL_NAME  # noqa: E402
from local_rag.reranking.base import RerankerProvider  # noqa: E402


class BgeRerankerProvider(RerankerProvider):
    """bge-reranker-v2-m3'ü yerel bir sentence-transformers CrossEncoder süreci
    olarak çalıştırır (Foundry Local dışında).

    Model, `num_labels=1` ile sequence-classification olarak eğitildiği için
    CrossEncoder varsayılan olarak Sigmoid aktivasyonu uygular — `.predict()`
    çağrısı zaten 0-1 aralığında skor döner, ekstra normalize gerekmez."""

    def __init__(self, model_name: str = RERANKER_MODEL_NAME, max_length: int = 512):
        self._model = CrossEncoder(model_name, max_length=max_length)

    def score(self, query: str, documents: list[str]) -> list[float]:
        if not documents:
            return []
        pairs = [[query, doc] for doc in documents]
        scores = self._model.predict(pairs)
        return [float(s) for s in scores]
