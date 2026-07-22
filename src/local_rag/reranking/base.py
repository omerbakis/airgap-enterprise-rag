"""RerankerProvider arayüzü."""

from __future__ import annotations

from abc import ABC, abstractmethod


class RerankerProvider(ABC):
    @abstractmethod
    def score(self, query: str, documents: list[str]) -> list[float]:
        """`documents` ile aynı sırada skorlar döner. Yüksek skor = daha alakalı.
        Skorlar 0-1 aralığına normalize edilmiş kabul edilir (bkz. config.RERANK_SCORE_THRESHOLD)."""
