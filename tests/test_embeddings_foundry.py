"""FoundryEmbeddingProvider'ın savunmacı L2 normalizasyonu: vec_chunks
(bkz. storage/db.py) sqlite-vec'in varsayılan L2 mesafesini kullanır, bu da
yalnızca TÜM vektörler unit-norm ise cosine sıralamasıyla eşdeğerdir. Bu test
Foundry Local gerektirmez — yalnızca saf `_l2_normalize` fonksiyonunu test eder."""

import math

from local_rag.embeddings.foundry import FoundryEmbeddingProvider, _l2_normalize


class _FakeItem:
    def __init__(self, index, embedding):
        self.index = index
        self.embedding = embedding


class _FakeEmbeddingsAPI:
    def __init__(self):
        self.last_input: list[str] | None = None

    def create(self, model, input):  # noqa: A002 - OpenAI SDK'nın kendi imzasıyla eşleşir
        self.last_input = input
        return type("Resp", (), {"data": [_FakeItem(i, [1.0, 0.0]) for i in range(len(input))]})()


def _make_provider_with_fake_client():
    # __init__ gerçek Foundry Local'a bağlanır (foundry_client.connect); bu saf
    # birim testte buna gerek yok — nesneyi __init__'i atlayarak oluşturup
    # _connection'ı sahte bir OpenAI-uyumlu client ile dolduruyoruz.
    provider = object.__new__(FoundryEmbeddingProvider)
    fake_embeddings_api = _FakeEmbeddingsAPI()
    fake_client = type("Client", (), {"embeddings": fake_embeddings_api})()
    provider._connection = type("Conn", (), {"client": fake_client, "model_id": "fake-model"})()
    provider._dimension = None
    return provider, fake_embeddings_api


def test_embed_query_adds_qwen3_instruction_prefix():
    provider, fake_api = _make_provider_with_fake_client()
    provider.embed_query("mazeret izni kaç gün kullanılabilir")
    assert fake_api.last_input == [
        "Instruct: Given a question, retrieve relevant passages that answer the "
        "question\nQuery: mazeret izni kaç gün kullanılabilir"
    ]


def test_embed_does_not_add_instruction_prefix():
    # Doküman/chunk tarafı ham kalmalı — yalnızca sorgu tarafı asimetrik olarak
    # talimatlanır (Qwen3-Embedding'in önerilen kullanımı, bkz. embeddings/foundry.py).
    provider, fake_api = _make_provider_with_fake_client()
    provider.embed(["Bölüm Başlığı\nYıllık izin süresi kıdeme göre değişir."])
    assert fake_api.last_input == ["Bölüm Başlığı\nYıllık izin süresi kıdeme göre değişir."]


def test_l2_normalize_produces_unit_vector():
    vector = [3.0, 4.0]  # ||v|| = 5
    normalized = _l2_normalize(vector)
    norm = math.sqrt(sum(v * v for v in normalized))
    assert abs(norm - 1.0) < 1e-9
    assert normalized == [0.6, 0.8]


def test_l2_normalize_is_idempotent_on_already_unit_vector():
    vector = [1.0, 0.0, 0.0]
    assert _l2_normalize(vector) == [1.0, 0.0, 0.0]


def test_l2_normalize_leaves_zero_vector_unchanged():
    # Bölme hatası olmamalı; sıfır vektörü normalize etmenin matematiksel bir
    # anlamı yok, olduğu gibi bırakmak en güvenli davranıştır.
    assert _l2_normalize([0.0, 0.0, 0.0]) == [0.0, 0.0, 0.0]
