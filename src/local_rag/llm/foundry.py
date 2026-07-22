from __future__ import annotations

from collections.abc import Iterator

from local_rag import foundry_client
from local_rag.config import CHAT_MODEL_ALIAS
from local_rag.llm.base import LLMProvider


class FoundryChatProvider(LLMProvider):
    """Foundry Local üzerinde çalışan chat modeliyle (örn. Qwen2.5-7B/14B-Instruct
    veya Phi-4-mini) cevap üretir."""

    def __init__(self, alias: str = CHAT_MODEL_ALIAS, temperature: float = 0.1) -> None:
        self._connection = foundry_client.connect(alias)
        self._temperature = temperature

    def _messages(self, system_prompt: str, user_prompt: str) -> list[dict]:
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def chat(self, system_prompt: str, user_prompt: str) -> str:
        response = self._connection.client.chat.completions.create(
            model=self._connection.model_id,
            temperature=self._temperature,
            messages=self._messages(system_prompt, user_prompt),
        )
        return response.choices[0].message.content or ""

    def chat_stream(self, system_prompt: str, user_prompt: str) -> Iterator[str]:
        """Foundry Local OpenAI-uyumlu endpoint'inden token'ları geldikçe akıtır
        (stream=True). CPU'da üretim yavaş olduğundan (~dakikalar), bu, kullanıcıya
        cevabı bir bütün olarak beklemek yerine yazılırken göstermeyi sağlar."""
        stream = self._connection.client.chat.completions.create(
            model=self._connection.model_id,
            temperature=self._temperature,
            messages=self._messages(system_prompt, user_prompt),
            stream=True,
        )
        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta
