"""Kurumsal Doküman Asistanı — Streamlit arayüzü.

Üç görünümlü (Asistan / Genel Bakış / Güvenlik & Denetim) tasarım-dili
tutarlı bir arayüz: hybrid search + reranking + RBAC + hash-chained audit
log, tamamen offline (Foundry Local + SQLCipher). Görsel sistem tek bir CSS
değişken paletinden (:root --lr-*) beslenir.

Çalıştırma:
    .venv/Scripts/python.exe -m streamlit run app.py
"""

from __future__ import annotations

import html
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from local_rag.config import (  # noqa: E402
    CHAT_MODEL_ALIAS,
    CHAT_MODEL_CHOICES,
    DEFAULT_DB_PATH,
    DEFAULT_DOCS_DIR,
    TOP_K,
    TOP_K_CHOICES,
)
from local_rag.embeddings.foundry import FoundryEmbeddingProvider  # noqa: E402
from local_rag.foundry_client import (  # noqa: E402
    FoundryLocalNotAvailable,
    FoundryModelNotFound,
    cached_aliases,
)
from local_rag.llm.foundry import FoundryChatProvider  # noqa: E402
from local_rag.pipeline import RagPipeline  # noqa: E402
from local_rag.reranking.bge import BgeRerankerProvider  # noqa: E402
from local_rag.security.rbac import DEFAULT_ROLE, ROLES  # noqa: E402
from local_rag.storage.db import SearchFilters  # noqa: E402

st.set_page_config(
    page_title="Kurumsal Doküman Asistanı",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==========================================================================
# Görsel tasarım sistemi
# --------------------------------------------------------------------------
# Tek bir renk/aralık/köşe paleti (:root --lr-*) tüm bileşenleri besler:
# kartlar (bordered container), KPI kartları, durum rozetleri, sohbet
# balonları ve marka başlığı hep aynı değişkenleri kullanır. Durum renkleri
# (good/warning/critical) her zaman ikon + metinle birlikte gösterilir —
# renk tek başına anlam taşımaz.
#
# GÜVENLİK: doküman içeriğinden gelen her metin (dosya adı, bölüm adı, tespit
# edilen desen) HTML'e gömülmeden önce _escape() ile kaçışlanır — aksi halde
# kötü niyetli bir doküman başlığı arayüze HTML/script enjekte edebilir
# (prompt-injection savunmasının bir parçası).
# ==========================================================================
_STYLE = """
<style>
:root {
    --lr-bg:            #fbfbfa;
    --lr-surface:       #ffffff;
    --lr-panel:         #f4f3f0;
    --lr-border:        rgba(17,17,15,0.09);
    --lr-border-strong: rgba(17,17,15,0.16);
    --lr-text:          #1b1b18;
    --lr-muted:         #6d6c67;
    --lr-faint:         #96938d;
    --lr-accent:        #2a78d6;
    --lr-accent-deep:   #1f5cab;
    --lr-accent-soft:   rgba(42,120,214,0.08);
    --lr-good:          #0a8a0a;
    --lr-good-soft:     rgba(12,163,12,0.10);
    --lr-warn:          #8a5a00;
    --lr-warn-soft:     rgba(250,178,25,0.16);
    --lr-crit:          #d03b3b;
    --lr-crit-soft:     rgba(208,59,59,0.10);
    --lr-radius:        14px;
    --lr-radius-sm:     9px;
    --lr-shadow:        0 1px 2px rgba(16,15,14,0.04), 0 2px 8px rgba(16,15,14,0.05);
}

/* --- Streamlit varsayılan kromunu sadeleştir ------------------------- */
[data-testid="stDecoration"],
[data-testid="stAppDeployButton"],
[data-testid="stStatusWidget"],
footer { display: none !important; }
[data-testid="stHeader"] { background: transparent; }

.stApp { background: var(--lr-bg); }
html, body, [class*="css"] { -webkit-font-smoothing: antialiased; text-rendering: optimizeLegibility; }
.block-container { padding-top: 1.4rem; padding-bottom: 3.5rem; max-width: 1180px; }

/* --- Tipografi yardımcıları ------------------------------------------ */
.lr-eyebrow {
    font-size: 0.72rem; font-weight: 700; letter-spacing: 0.09em;
    text-transform: uppercase; color: var(--lr-faint); margin: 0 0 0.5rem;
}
.lr-page-title { font-size: 1.5rem; font-weight: 700; letter-spacing: -0.01em; color: var(--lr-text); margin: 0 0 0.2rem; }
.lr-page-sub   { font-size: 0.9rem; color: var(--lr-muted); margin: 0 0 1.35rem; max-width: 62ch; }

/* --- Marka başlığı ---------------------------------------------------- */
.lr-brand {
    display: flex; align-items: center; gap: 15px;
    padding: 2px 0 18px; margin-bottom: 20px;
    border-bottom: 1px solid var(--lr-border);
}
.lr-logo {
    width: 46px; height: 46px; border-radius: 13px; flex: 0 0 auto;
    background: linear-gradient(140deg, var(--lr-accent), var(--lr-accent-deep));
    color: #fff; display: flex; align-items: center; justify-content: center;
    font-size: 22px; box-shadow: 0 6px 16px rgba(31,92,171,0.30);
}
.lr-brand-txt   { display: flex; flex-direction: column; gap: 2px; min-width: 0; }
.lr-brand-title { font-size: 1.1rem; font-weight: 700; color: var(--lr-text); letter-spacing: -0.01em; line-height: 1.15; }
.lr-brand-sub   { font-size: 0.79rem; color: var(--lr-muted); }
.lr-brand-status { margin-left: auto; display: flex; gap: 8px; flex-wrap: wrap; justify-content: flex-end; }

/* --- Pill (marka başlığı durum etiketleri) --------------------------- */
.lr-pill {
    display: inline-flex; align-items: center; gap: 7px;
    padding: 5px 12px; border-radius: 999px; font-size: 0.75rem; font-weight: 600;
    border: 1px solid var(--lr-border); background: var(--lr-surface);
    color: var(--lr-muted); white-space: nowrap;
}
.lr-pill .dot { width: 7px; height: 7px; border-radius: 50%; background: var(--lr-good); box-shadow: 0 0 0 3px var(--lr-good-soft); }
.lr-pill.is-crit { color: var(--lr-crit); border-color: rgba(208,59,59,0.30); }
.lr-pill.is-crit .dot { background: var(--lr-crit); box-shadow: 0 0 0 3px var(--lr-crit-soft); }

/* --- Durum rozetleri -------------------------------------------------- */
.status-badge {
    display: inline-flex; align-items: center; gap: 5px;
    padding: 3px 11px; border-radius: 999px; font-size: 0.78rem; font-weight: 600;
    line-height: 1.7; border: 1px solid transparent; white-space: nowrap;
}
.status-good     { background: var(--lr-good-soft); color: var(--lr-good); border-color: rgba(12,163,12,0.26); }
.status-warning  { background: var(--lr-warn-soft); color: var(--lr-warn);  border-color: rgba(250,178,25,0.40); }
.status-critical { background: var(--lr-crit-soft); color: var(--lr-crit);  border-color: rgba(208,59,59,0.28); }

/* --- Kaynak chip'leri ------------------------------------------------- */
.source-chip {
    display: inline-block; background: var(--lr-panel); border: 1px solid var(--lr-border);
    border-radius: 8px; padding: 3px 10px; margin: 6px 6px 0 0;
    font-size: 0.77rem; color: #4a4945;
}

/* --- Anahtar/değer satırları (sidebar + kartlar) --------------------- */
.lr-kv { display: flex; justify-content: space-between; gap: 12px; align-items: center; font-size: 0.82rem; padding: 6px 0; border-bottom: 1px dashed var(--lr-border); }
.lr-kv:last-child { border-bottom: none; }
.lr-kv .k { color: var(--lr-muted); }
.lr-kv .v { color: var(--lr-text); font-weight: 600; text-align: right; }

/* --- KPI kart içeriği ------------------------------------------------- */
.lr-kpi-value { font-size: 1.85rem; font-weight: 700; color: var(--lr-text); line-height: 1.1; letter-spacing: -0.01em; }
.lr-kpi-sub   { font-size: 0.78rem; color: var(--lr-muted); margin-top: 5px; }

/* --- Bordered container'ı karta dönüştür ----------------------------- */
[data-testid="stVerticalBlockBorderWrapper"] {
    background: var(--lr-surface);
    border: 1px solid var(--lr-border) !important;
    border-radius: var(--lr-radius) !important;
    box-shadow: var(--lr-shadow);
}

/* --- st.metric kart görünümü ----------------------------------------- */
[data-testid="stMetric"] {
    background: var(--lr-surface); border: 1px solid var(--lr-border);
    border-radius: var(--lr-radius); padding: 14px 16px; box-shadow: var(--lr-shadow);
}
[data-testid="stMetricLabel"] p { font-size: 0.78rem !important; font-weight: 600; color: var(--lr-muted); letter-spacing: 0.02em; }
[data-testid="stMetricValue"] { font-size: 1.75rem; font-weight: 700; color: var(--lr-text); }

/* --- Butonlar --------------------------------------------------------- */
.stButton > button, .stDownloadButton > button {
    border-radius: var(--lr-radius-sm); font-weight: 600; border: 1px solid var(--lr-border-strong);
    transition: border-color .12s ease, background .12s ease;
}
.stButton > button[kind="primary"] { border: none; box-shadow: 0 3px 10px rgba(31,92,171,0.26); }
.stButton > button[kind="primary"]:hover { background: var(--lr-accent-deep); }

/* --- Sohbet balonları ------------------------------------------------- */
[data-testid="stChatMessage"] {
    background: var(--lr-surface); border: 1px solid var(--lr-border);
    border-radius: var(--lr-radius); padding: 14px 18px; box-shadow: var(--lr-shadow);
}
[data-testid="stChatInput"] textarea { font-size: 0.95rem; }

/* --- Dataframe -------------------------------------------------------- */
[data-testid="stDataFrame"] { border-radius: var(--lr-radius); overflow: hidden; border: 1px solid var(--lr-border); }

/* --- Sidebar ---------------------------------------------------------- */
section[data-testid="stSidebar"] { background: var(--lr-panel); border-right: 1px solid var(--lr-border); }
section[data-testid="stSidebar"] .block-container { padding-top: 1rem; }

/* Sidebar bölümleri daraltılabilir (st.expander) — kalabalığı azaltmak için;
   daraltıldığında yalnızca başlık satırı görünür kalır. */
section[data-testid="stSidebar"] [data-testid="stExpander"] {
    background: var(--lr-surface); border: 1px solid var(--lr-border) !important;
    border-radius: var(--lr-radius-sm); margin-bottom: 10px; box-shadow: var(--lr-shadow);
}
section[data-testid="stSidebar"] [data-testid="stExpander"] summary {
    font-size: 0.84rem; font-weight: 600; padding: 8px 10px;
}

/* --- Ayraç ------------------------------------------------------------ */
hr { margin: 1.1rem 0; border-color: var(--lr-border); }
</style>
"""
st.markdown(_STYLE, unsafe_allow_html=True)


# --------------------------------------------------------------------------
# HTML yardımcıları — hepsi kullanıcı/doküman kaynaklı metni escape eder.
# --------------------------------------------------------------------------
def _escape(text: str) -> str:
    return html.escape(str(text), quote=True)


def _badge(label: str, kind: str, icon: str) -> str:
    return f'<span class="status-badge status-{kind}">{icon} {_escape(label)}</span>'


def _chip(text: str) -> str:
    return f'<span class="source-chip">{_escape(text)}</span>'


def _pill(label: str, kind: str = "good", icon_dot: bool = True) -> str:
    dot = '<span class="dot"></span>' if icon_dot else ""
    cls = "lr-pill" + (" is-crit" if kind == "crit" else "")
    return f'<span class="{cls}">{dot}{_escape(label)}</span>'


def _eyebrow(text: str) -> str:
    return f'<div class="lr-eyebrow">{_escape(text)}</div>'


def _kv(key: str, value_html: str) -> str:
    """value_html güvenilir (rozet/escape edilmiş) HTML olmalıdır."""
    return f'<div class="lr-kv"><span class="k">{_escape(key)}</span><span class="v">{value_html}</span></div>'


def _kpi_card(col, label: str, value_html: str, sub: str | None = None) -> None:
    with col.container(border=True):
        html_block = _eyebrow(label) + f'<div class="lr-kpi-value">{value_html}</div>'
        if sub:
            html_block += f'<div class="lr-kpi-sub">{_escape(sub)}</div>'
        st.markdown(html_block, unsafe_allow_html=True)


def _page_head(title: str, subtitle: str) -> None:
    st.markdown(
        f'<div class="lr-page-title">{_escape(title)}</div>'
        f'<div class="lr-page-sub">{_escape(subtitle)}</div>',
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------
# Pipeline + anlık sistem durumu
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner="Chat modeli Foundry Local'a yükleniyor…")
def load_llm(alias: str) -> FoundryChatProvider:
    """Alias başına önbelleğe alınır — model değiştirmek embedder/reranker'ı
    yeniden yüklemeden yalnızca yeni LLM'i bağlar (yerelde yoksa indirir)."""
    return FoundryChatProvider(alias=alias)


@st.cache_resource(show_spinner="Modeller yükleniyor (Foundry Local + reranker, ilk çalıştırmada uzun sürebilir)...")
def load_pipeline() -> RagPipeline:
    embedder = FoundryEmbeddingProvider()
    reranker = BgeRerankerProvider()
    return RagPipeline(embedder=embedder, reranker=reranker, llm=load_llm(CHAT_MODEL_ALIAS), db_path=DEFAULT_DB_PATH)


@st.cache_data(ttl=30, show_spinner=False)
def _cached_aliases() -> set[str]:
    """Yerelde indirilmiş model alias'ları (30 sn önbellekli — indirme sonrası tazelenir)."""
    return cached_aliases()


def load_snapshot(pipeline: RagPipeline) -> dict:
    """Her yeniden çalıştırmada tazelenen hafif sistem görüntüsü (ucuz SQLite
    okumaları). Sayfalar bu tek görüntüyü paylaşır."""
    return {
        "documents": pipeline.list_documents(),
        "audit": pipeline.list_audit_entries(limit=200),
        "flagged": pipeline.list_flagged_chunks(),
        "chain_ok": pipeline.verify_audit_log(),
    }


def _chunk_count(raw: str | None) -> int:
    if not raw:
        return 0
    try:
        return len(json.loads(raw))
    except (ValueError, TypeError):
        return 0


# --------------------------------------------------------------------------
# Marka başlığı (her sayfanın üstünde)
# --------------------------------------------------------------------------
def render_brand_header(chain_ok: bool) -> None:
    if chain_ok:
        audit_pill = _pill("Audit zinciri doğrulandı", "good")
    else:
        audit_pill = _pill("Audit zinciri bozuk", "crit")
    st.markdown(
        f"""
        <div class="lr-brand">
            <div class="lr-logo">🔒</div>
            <div class="lr-brand-txt">
                <div class="lr-brand-title">Kurumsal Doküman Asistanı</div>
                <div class="lr-brand-sub">Air-gapped RAG · Foundry Local · SQLCipher</div>
            </div>
            <div class="lr-brand-status">
                {_pill("Çevrimdışı çalışıyor", "good")}
                {_pill("Şifreli depolama", "good")}
                {audit_pill}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------
# Sidebar — oturum kimliği (rol) + sistem durumu (her sayfada ortak)
# --------------------------------------------------------------------------
def render_sidebar(snapshot: dict) -> tuple[str, int]:
    with st.sidebar:
        with st.expander("👤 Oturum", expanded=True):
            st.caption("Gerçek kimlik doğrulama yok — RBAC'ı göstermek için bir persona seçicidir.")
            role_names = sorted(ROLES)
            selected = st.selectbox("Aktif rol", role_names, index=role_names.index(DEFAULT_ROLE))
            role = ROLES[selected]
            depts = ", ".join(role.allowed_departments) if role.allowed_departments else "tümü"
            st.markdown(
                _kv("Gizlilik erişimi", _escape(", ".join(role.allowed_classifications)))
                + _kv("Departman erişimi", _escape(depts)),
                unsafe_allow_html=True,
            )

        with st.expander("🤖 Chat Modeli", expanded=True):
            cached = _cached_aliases()
            aliases = list(CHAT_MODEL_CHOICES)
            model_alias = st.selectbox(
                "Model (hız / kalite dengesi)",
                aliases,
                index=aliases.index(CHAT_MODEL_ALIAS) if CHAT_MODEL_ALIAS in aliases else 0,
                format_func=lambda a: CHAT_MODEL_CHOICES[a] + ("  ✓" if a in cached else "  ⬇"),
            )
            st.caption("✓ yerelde yüklü · ⬇ ilk seçimde indirilir (ağ gerekir). Küçük model → daha hızlı ilk token.")
            try:
                PIPELINE.llm = load_llm(model_alias)
            except (FoundryModelNotFound, FoundryLocalNotAvailable, RuntimeError) as exc:
                st.error(f"'{model_alias}' yüklenemedi (indirilmemiş olabilir): {_escape(str(exc))}")
                PIPELINE.llm = load_llm(CHAT_MODEL_ALIAS)

        with st.expander("🎚️ Yanıt Derinliği", expanded=True):
            top_k_values = list(TOP_K_CHOICES)
            selected_top_k = st.selectbox(
                "Kaynak sayısı (top-k)",
                top_k_values,
                index=top_k_values.index(TOP_K) if TOP_K in top_k_values else 0,
                format_func=lambda k: TOP_K_CHOICES[k],
            )
            st.caption(
                "LLM bağlamına giden kaynak chunk sayısı — hızı ve yanıt kalitesini "
                "etkiler. Daha az kaynak = daha hızlı yanıt; bu korpusta kalite "
                "kaybı ölçülmedi (bkz. docs/DEMO_SENARYOLARI.md)."
            )

        with st.expander("📊 Sistem Durumu", expanded=True):
            chain_badge = (
                _badge("Doğrulandı", "good", "✓") if snapshot["chain_ok"] else _badge("Bozuk", "critical", "✕")
            )
            flagged = snapshot["flagged"]
            queue_badge = (
                _badge(f"{len(flagged)} şüpheli", "warning", "⚠") if flagged else _badge("Temiz", "good", "✓")
            )
            st.markdown(
                _kv("Foundry Local", _badge("Bağlı", "good", "●"))
                + _kv("Depolama", _badge("AES-256", "good", "🔐"))
                + _kv("Audit zinciri", chain_badge)
                + _kv("İnceleme kuyruğu", queue_badge)
                + _kv("İndeks", f'{len(snapshot["documents"])} doküman'),
                unsafe_allow_html=True,
            )
            st.caption("Tamamen offline · dış ağ çağrısı yok")
    return selected, selected_top_k


# ==========================================================================
# Sayfa: Asistan (sohbet)
# ==========================================================================
def page_chat() -> None:
    role = SELECTED_ROLE
    top_k = SELECTED_TOP_K
    documents = SNAPSHOT["documents"]
    role_obj = ROLES[role]
    depts = ", ".join(role_obj.allowed_departments) if role_obj.allowed_departments else "tümü"

    _page_head(
        "Asistan",
        "Dokümanlarınıza soru sorun. Yanıtlar yalnızca indekslenmiş içerikten üretilir, "
        "her yanıt kaynak künyesiyle gelir ve seçili rolün erişemediği içerik asla bağlama girmez.",
    )

    scope_col, filter_col = st.columns([4, 1], vertical_alignment="center")
    with scope_col:
        st.markdown(
            _pill(f"Rol: {role}", "good")
            + " "
            + f'<span class="lr-pill" style="color:var(--lr-muted)">Erişim: {_escape(", ".join(role_obj.allowed_classifications))} · {_escape(depts)}</span>',
            unsafe_allow_html=True,
        )
    with filter_col:
        with st.popover("🔍 Filtreler", width="stretch"):
            file_types = sorted({d["file_type"] for d in documents})
            selected_types = st.multiselect("Dosya türü", file_types, placeholder="Tümü")
            section_query = st.text_input("Bölüm adında geçsin", placeholder="örn. Güvenlik")
            st.caption("Filtreler role EK kısıtlama getirir; rolün yasakladığını asla açmaz.")

    filters = SearchFilters(file_types=selected_types or None, section_contains=section_query or None)

    if "history" not in st.session_state:
        st.session_state.history = []

    # chat_input alta sabitlenir; pending_query her zaman pop edilir ki
    # kullanıcı yazdığında bekleyen bir örnek soru geride kalmasın.
    typed = st.chat_input("Dokümanlar hakkında bir soru sorun...")
    pending = st.session_state.pop("pending_query", None)
    query = typed or pending

    # Boş durum: örnek sorular (yalnızca hiç mesaj ve işlenecek soru yokken)
    if not st.session_state.history and not query:
        with st.container(border=True):
            st.markdown(_eyebrow("Başlamak için"), unsafe_allow_html=True)
            st.caption("Bir örnek soruyla deneyin ya da aşağıdan kendi sorunuzu yazın:")
            examples = [
                "Onaylanan kredilerde borç/gelir oranı en fazla yüzde kaç olabilir?",
                "Şüpheli işlem raporu MASAK'a kaç iş günü içinde bildirilmelidir?",
                "Çalışanlar haftada en fazla kaç gün uzaktan çalışabilir?",
            ]
            for col, ex in zip(st.columns(len(examples)), examples):
                if col.button(ex, width="stretch", key=f"ex_{examples.index(ex)}"):
                    st.session_state.pending_query = ex
                    st.rerun()

    for entry in st.session_state.history:
        avatar = "🧑‍💼" if entry["chat_role"] == "user" else "🔒"
        with st.chat_message(entry["chat_role"], avatar=avatar):
            st.markdown(entry["content"])
            if entry.get("badge"):
                st.markdown(entry["badge"], unsafe_allow_html=True)
            if entry.get("sources"):
                st.markdown("".join(_chip(s) for s in entry["sources"]), unsafe_allow_html=True)

    if not query:
        return

    st.session_state.history.append({"chat_role": "user", "content": query})
    with st.chat_message("user", avatar="🧑‍💼"):
        st.markdown(query)

    with st.chat_message("assistant", avatar="🔒"):
        placeholder = st.empty()
        placeholder.markdown("_Aranıyor, yeniden sıralanıyor…_")
        streamed = ""
        result = None
        for item in PIPELINE.answer_streaming(query, role=role, filters=filters, top_k=top_k):
            if isinstance(item, str):
                streamed += item
                placeholder.markdown(streamed + " ▌")  # canlı yazım imleci
            else:
                result = item  # en son öğe: finalize edilmiş AnswerResult

        badge_html = ""
        if not result.confident:
            placeholder.empty()
            st.warning(result.answer)
            badge_html = _badge("Düşük güven", "warning", "⚠")
        else:
            placeholder.markdown(result.answer)  # nihai metin (kaynak künyesi dahil), imleç kalkar
            if result.faithfulness is not None:
                pct = result.faithfulness * 100
                if result.faithfulness >= 0.7:
                    badge_html = _badge(f"Bağlam örtüşmesi %{pct:.0f}", "good", "✓")
                elif result.faithfulness >= 0.4:
                    badge_html = _badge(f"Bağlam örtüşmesi %{pct:.0f}", "warning", "⚠")
                else:
                    badge_html = _badge(f"Bağlam örtüşmesi %{pct:.0f}", "critical", "✕")

        meta_bits = []
        if badge_html:
            meta_bits.append(badge_html)
        if result.latency_seconds is not None:
            meta_bits.append(_badge(f"{result.latency_seconds:.1f}s", "good", "⏱"))
        if meta_bits:
            st.markdown(" ".join(meta_bits), unsafe_allow_html=True)

        source_labels = [f"S{i + 1} · {s.filename} · {s.section_path}" for i, s in enumerate(result.sources)]
        if source_labels:
            st.markdown("".join(_chip(label) for label in source_labels), unsafe_allow_html=True)

    st.session_state.history.append(
        {
            "chat_role": "assistant",
            "content": result.answer,
            "badge": " ".join(meta_bits) if meta_bits else "",
            "sources": source_labels,
        }
    )


# ==========================================================================
# Sayfa: Genel Bakış (dashboard)
# ==========================================================================
def page_dashboard() -> None:
    documents = SNAPSHOT["documents"]
    audit = SNAPSHOT["audit"]
    flagged = SNAPSHOT["flagged"]
    chain_ok = SNAPSHOT["chain_ok"]

    _page_head("Genel Bakış", "İndeks, kullanım ve sistem sağlığı özeti.")

    c1, c2, c3, c4 = st.columns(4)
    _kpi_card(c1, "İndekslenmiş doküman", str(len(documents)), f'{len({d["file_type"] for d in documents})} farklı tür')
    _kpi_card(c2, "Kayıtlı sorgu", str(len(audit)), "hash-chained audit log")
    _kpi_card(
        c3, "Audit zinciri",
        _badge("Doğrulandı", "good", "✓") if chain_ok else _badge("Bozuk", "critical", "✕"),
        "bütünlük kontrolü",
    )
    _kpi_card(
        c4, "İnceleme kuyruğu",
        _badge(f"{len(flagged)} şüpheli", "warning", "⚠") if flagged else _badge("Temiz", "good", "✓"),
        "injection taraması",
    )

    st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)
    inv_col, act_col = st.columns([3, 2], gap="large")

    with inv_col:
        with st.container(border=True):
            st.markdown(_eyebrow("Doküman envanteri"), unsafe_allow_html=True)
            if documents:
                inv_df = pd.DataFrame(
                    [
                        {
                            "Dosya": d["filename"],
                            "Tür": d["file_type"],
                            "Gizlilik": d["classification"],
                            "Departman": d["department"],
                            "Dil": d["language"],
                        }
                        for d in documents
                    ]
                )
                st.dataframe(inv_df, hide_index=True, width="stretch", height=320)
            else:
                st.caption("Henüz doküman indekslenmedi. Sağdaki panelden dizini tarayın.")

    with act_col:
        with st.container(border=True):
            st.markdown(_eyebrow("Doküman işleme"), unsafe_allow_html=True)
            st.caption(f"Kaynak dizin: `{DEFAULT_DOCS_DIR}`")
            if st.button("🔄 Dizini yeniden tara ve indeksle", width="stretch", type="primary"):
                with st.spinner("Dokümanlar işleniyor..."):
                    log = PIPELINE.ingest_path(DEFAULT_DOCS_DIR)
                st.success(f"{len(log)} dosya tarandı.")
                with st.expander("İşlem günlüğü"):
                    for line in log:
                        st.text(line)
            st.caption(
                "Değişmemiş dosyalar içerik hash'iyle atlanır (idempotent). "
                "`data/documents/_metadata.json` ile dosya bazlı gizlilik/departman atanır."
            )

        with st.container(border=True):
            st.markdown(_eyebrow("Erişim matrisi (RBAC)"), unsafe_allow_html=True)
            matrix_df = pd.DataFrame(
                [
                    {
                        "Rol": name,
                        "Gizlilik": ", ".join(r.allowed_classifications),
                        "Departman": ", ".join(r.allowed_departments) if r.allowed_departments else "tümü",
                    }
                    for name, r in sorted(ROLES.items())
                ]
            )
            st.dataframe(matrix_df, hide_index=True, width="stretch")


# ==========================================================================
# Sayfa: Güvenlik & Denetim
# ==========================================================================
def page_audit() -> None:
    audit = SNAPSHOT["audit"]
    flagged = SNAPSHOT["flagged"]
    chain_ok = SNAPSHOT["chain_ok"]

    _page_head(
        "Güvenlik & Denetim",
        "Her sorgu, kullanılan rol ve getirilen chunk'larla birlikte kurcalamaya-dayanıklı "
        "bir hash zincirine yazılır. Şüpheli içerik insan incelemesi için ayrılır.",
    )

    s1, s2, s3, s4 = st.columns(4)
    _kpi_card(s1, "Şifreleme", _badge("AES-256", "good", "🔐"), "SQLCipher, dosya düzeyi")
    _kpi_card(s2, "Erişim kontrolü", _badge("Retrieval-time", "good", "✓"), "UI değil, SQL WHERE")
    _kpi_card(
        s3, "Injection taraması",
        _badge(f"{len(flagged)} şüpheli", "warning", "⚠") if flagged else _badge("Temiz", "good", "✓"),
        "ingestion-zamanı",
    )
    _kpi_card(
        s4, "Audit zinciri",
        _badge("Doğrulandı", "good", "✓") if chain_ok else _badge("Bozuk", "critical", "✕"),
        f"{len(audit)} kayıt",
    )

    st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown(_eyebrow("Denetim kaydı (audit log)"), unsafe_allow_html=True)
        if chain_ok:
            st.success("Zincir bütünlüğü doğrulandı — hiçbir kayıt değiştirilmemiş.", icon="✅")
        else:
            st.error("Zincir BOZUK — kayıtlar sonradan değiştirilmiş olabilir.", icon="🚫")
        if audit:
            audit_df = pd.DataFrame(
                [
                    {
                        "Zaman": e["ts"][:19].replace("T", " "),
                        "Rol": e["role"],
                        "Sorgu": e["query"][:70] + ("…" if len(e["query"]) > 70 else ""),
                        "Chunk": _chunk_count(e["retrieved_chunk_ids"]),
                        "Sonuç": "✓ Cevaplandı" if e["confident"] else "— Bulunamadı / reddedildi",
                    }
                    for e in audit
                ]
            )
            st.dataframe(audit_df, hide_index=True, width="stretch", height=340)
        else:
            st.caption("Henüz sorgu kaydı yok.")

    with st.container(border=True):
        st.markdown(_eyebrow(f"İnceleme kuyruğu · {len(flagged)} şüpheli chunk"), unsafe_allow_html=True)
        st.caption(
            "Prompt-injection taramasında şüpheli desen bulunan chunk'lar. İçerik bağlama girer "
            "ama sistem promptu bunları veri olarak işler, talimat olarak değil."
        )
        if flagged:
            flagged_df = pd.DataFrame(
                [
                    {
                        "Dosya": item["source_filename"],
                        "Bölüm": item["section_path"],
                        "Tespit edilen desen": item["injection_flag"],
                    }
                    for item in flagged
                ]
            )
            st.dataframe(flagged_df, hide_index=True, width="stretch")
        else:
            st.success("Şüpheli içerik tespit edilmedi.", icon="✅")


# ==========================================================================
# Uygulama akışı
# ==========================================================================
try:
    PIPELINE = load_pipeline()
except (FoundryLocalNotAvailable, FoundryModelNotFound) as exc:
    render_brand_header(chain_ok=True)
    st.error(str(exc))
    st.stop()

SNAPSHOT = load_snapshot(PIPELINE)

pages = st.navigation(
    [
        st.Page(page_chat, title="Asistan", icon="💬", url_path="asistan", default=True),
        st.Page(page_dashboard, title="Genel Bakış", icon="📊", url_path="genel-bakis"),
        st.Page(page_audit, title="Güvenlik & Denetim", icon="🛡️", url_path="guvenlik-denetim"),
    ]
)

SELECTED_ROLE, SELECTED_TOP_K = render_sidebar(SNAPSHOT)
render_brand_header(SNAPSHOT["chain_ok"])
pages.run()
