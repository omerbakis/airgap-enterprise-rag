"""Foundry Local gerektirmeyen testler için sahte (fake) provider'lar."""

from __future__ import annotations

import hashlib
import math

from local_rag.embeddings.base import EmbeddingProvider
from local_rag.llm.base import LLMProvider
from local_rag.reranking.base import RerankerProvider


class FakeEmbeddingProvider(EmbeddingProvider):
    """Kelime-hash tabanlı, deterministik, L2-normalize bag-of-words embedding.

    Ortak kelimeleri paylaşan metinler birbirine yakın vektörler üretir, bu da
    dense KNN sıralamasını anlamlı şekilde test etmeyi sağlar."""

    def __init__(self, dimension: int = 16):
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(t) for t in texts]

    def _embed_one(self, text: str) -> list[float]:
        vec = [0.0] * self._dimension
        for word in text.lower().split():
            idx = int(hashlib.md5(word.encode("utf-8")).hexdigest(), 16) % self._dimension
            vec[idx] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]


class FakeLLMProvider(LLMProvider):
    def __init__(self, canned_answer: str = "FAKE_ANSWER"):
        self.canned_answer = canned_answer
        self.last_system_prompt: str | None = None
        self.last_user_prompt: str | None = None

    def chat(self, system_prompt: str, user_prompt: str) -> str:
        self.last_system_prompt = system_prompt
        self.last_user_prompt = user_prompt
        return self.canned_answer

    def chat_stream(self, system_prompt: str, user_prompt: str):
        self.last_system_prompt = system_prompt
        self.last_user_prompt = user_prompt
        # Kelime kelime akıt; parçalar birleşince tam olarak canned_answer'a eşit.
        for i, word in enumerate(self.canned_answer.split(" ")):
            yield word if i == 0 else " " + word


class FakeRerankerProvider(RerankerProvider):
    """Kelime-örtüşme oranına dayalı sahte reranker: score = ortak kelime
    sayısı / sorgu kelime sayısı — 0-1 aralığında, gerçek reranker'la aynı
    ölçekte (bkz. config.RERANK_SCORE_THRESHOLD), eşikleme testlerini
    Foundry Local/sentence-transformers olmadan deterministik kılar."""

    def score(self, query: str, documents: list[str]) -> list[float]:
        query_words = set(query.lower().split())
        if not query_words:
            return [0.0 for _ in documents]
        scores = []
        for doc in documents:
            overlap = len(query_words & set(doc.lower().split()))
            scores.append(overlap / len(query_words))
        return scores
