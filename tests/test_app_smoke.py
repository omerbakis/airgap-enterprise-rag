"""app.py (Streamlit arayüzü) için hızlı render smoke testi.

Gerçek Streamlit test runtime'ında (AppTest) uygulamayı sahte provider'lar +
geçici boş bir DB ile çalıştırır ve varsayılan sayfanın (Asistan) hiçbir Python
istisnası olmadan render olduğunu doğrular. Foundry Local GEREKMEZ — böylece
UI kırılmaları (import hatası, f-string/HTML kaçış hatası, sidebar/model seçici
regresyonları) Foundry'siz hızlı test paketinde yakalanır.

Not: `answer_streaming` çağrılmaz (ilk yüklemede sorgu yoktur), bu yüzden LLM
üretimi tetiklenmez; test yalnızca render yolunu kapsar.
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_PATH = PROJECT_ROOT / "app.py"


def test_app_boots_and_renders_default_page(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest

    import streamlit as st

    import local_rag.config as config
    import local_rag.embeddings.foundry as emb_mod
    import local_rag.foundry_client as fc_mod
    import local_rag.llm.foundry as llm_mod
    import local_rag.reranking.bge as rr_mod
    from tests.fakes import FakeEmbeddingProvider, FakeLLMProvider, FakeRerankerProvider

    # Foundry'ye hiç bağlanmadan: sahte provider'lar + geçici DB + sabit katalog.
    # AppTest, app.py'yi her run'da yeniden exec ettiği için bu yamalar
    # uygulamanın `from ... import ...` bağlamalarına yansır.
    monkeypatch.setattr(config, "DEFAULT_DB_PATH", tmp_path / "smoke.db")
    monkeypatch.setattr(emb_mod, "FoundryEmbeddingProvider", lambda *a, **k: FakeEmbeddingProvider(dimension=16))
    monkeypatch.setattr(llm_mod, "FoundryChatProvider", lambda *a, **k: FakeLLMProvider())
    monkeypatch.setattr(rr_mod, "BgeRerankerProvider", lambda *a, **k: FakeRerankerProvider())
    monkeypatch.setattr(fc_mod, "cached_aliases", lambda: {"qwen2.5-7b"})

    # Önceki testlerden sızmış cache olmasın (app st.cache_resource/cache_data kullanır).
    st.cache_resource.clear()
    st.cache_data.clear()

    at = AppTest.from_file(str(APP_PATH), default_timeout=90)
    at.run()

    assert not at.exception, f"app render istisna verdi: {[e.value for e in at.exception]}"
    # Marka başlığı ve sidebar seçicileri (rol + chat modeli) render olmalı.
    assert any("Kurumsal Doküman Asistanı" in m.value for m in at.markdown), "marka başlığı yok"
    assert len(at.sidebar.selectbox) >= 3, "sidebar'da rol + chat modeli + top-k seçici bekleniyordu"
