from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator


class LLMProvider(ABC):
    @abstractmethod
    def chat(self, system_prompt: str, user_prompt: str) -> str:
        """Sistem + kullanıcı promptu verilir, model cevabı döner."""

    def chat_stream(self, system_prompt: str, user_prompt: str) -> Iterator[str]:
        """Cevabı parça parça (token/delta) üretir — UI'da canlı akış için.

        Varsayılan implementasyon streaming desteklemeyen provider'lar için
        tüm cevabı tek parça olarak verir; gerçek streaming yapan provider'lar
        (bkz. FoundryChatProvider) bunu override eder."""
        yield self.chat(system_prompt, user_prompt)
