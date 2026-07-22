"""EmbeddingProvider arayüzü.

Foundry Local dışına (örn. bağımsız ONNX/sentence-transformers ile BGE-M3'e)
geçilebilmesi için embedding üretimi bu arayüz arkasında soyutlanır."""

from __future__ import annotations

from abc import ABC, abstractmethod


class EmbeddingProvider(ABC):
    @property
    @abstractmethod
    def dimension(self) -> int: ...

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Metin listesini embedding vektörlerine çevirir (doküman/chunk tarafı)."""

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]

    def embed_query(self, text: str) -> list[float]:
        """Bir arama sorgusunu embed eder. Varsayılan davranış `embed_one` ile
        aynıdır (simetrik: sorgu ve doküman aynı şekilde embed edilir); bazı
        modeller (ör. Qwen3-Embedding) asimetrik kullanımda — sorguya bir görev
        talimatı önekiyle, dokümana ise ham metinle — daha iyi retrieval kalitesi
        verir. Bu tür bir provider, bu metodu override edip yalnızca sorgu
        tarafına talimat ekleyebilir (bkz. embeddings/foundry.py)."""
        return self.embed_one(text)
