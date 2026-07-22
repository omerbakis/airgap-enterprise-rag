from __future__ import annotations

import math

from local_rag import foundry_client
from local_rag.config import EMBEDDING_MODEL_ALIAS
from local_rag.embeddings.base import EmbeddingProvider


def _l2_normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vector))
    return [v / norm for v in vector] if norm > 0 else vector


# Qwen3-Embedding, resmi model kartında belgelenen "Instruct: {görev}\nQuery:
# {sorgu}" formatıyla ASİMETRİK kullanıldığında (yalnızca sorgu tarafına görev
# talimatı eklenir, doküman tarafı ham kalır) daha iyi retrieval kalitesi verir
# — talimat İngilizce (modelin resmi örneklerinde kullanılan dil), sorgunun
# kendisi TR/EN karışık olabilir; çok dilli embedding modellerinde talimat
# dilinin sorgu diliyle eşleşmesi gerekmez.
_QUERY_INSTRUCTION = "Instruct: Given a question, retrieve relevant passages that answer the question\nQuery: {query}"


class FoundryEmbeddingProvider(EmbeddingProvider):
    """Qwen3-Embedding-0.6B'yi Foundry Local üzerinden native çalıştırır."""

    def __init__(self, alias: str = EMBEDDING_MODEL_ALIAS, dimension: int | None = None) -> None:
        self._connection = foundry_client.connect(alias)
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            self._dimension = len(self.embed_one("boyut tespiti için örnek metin"))
        return self._dimension

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        response = self._connection.client.embeddings.create(
            model=self._connection.model_id,
            input=texts,
        )
        # API sıralamayı korur, ama sağlam olmak için index'e göre sıralıyoruz.
        ordered = sorted(response.data, key=lambda item: item.index)
        # vec_chunks (bkz. storage/db.py) sqlite-vec'in VARSAYILAN L2 mesafesini
        # kullanır (distance_metric=cosine belirtilmez); L2 sıralaması yalnızca
        # TÜM vektörler unit-norm ise cosine sıralamasıyla aynıdır (L2² = 2 - 2·cos_sim,
        # cosine'in monoton bir dönüşümü). Foundry Local'ın Qwen3-Embedding çıktısı
        # ölçüldüğünde zaten unit-norm (‖v‖≈1.000000) ama bu, örtük ve doğrulanmamış
        # bir varsayımdı — burada açıkça normalize ederek hem bu invaryantı koda
        # sabitliyoruz hem de ileride Foundry Local güncellemesi/model swap bu
        # davranışı sessizce değiştirirse arama sonuçlarının bozulmasını engelliyoruz.
        return [_l2_normalize(item.embedding) for item in ordered]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_one(_QUERY_INSTRUCTION.format(query=text))
