"""Uçtan uca RAG orkestrasyonu: ingestion + hybrid retrieval + rerank + cevap üretimi.

Katmanlar arası bağımlılık: pipeline yalnızca EmbeddingProvider/RerankerProvider/
LLMProvider arayüzlerine bağımlıdır (bkz. embeddings/base.py, reranking/base.py,
llm/base.py) — Foundry Local veya bge-reranker dışında bir implementasyona
geçmek bu dosyayı değiştirmeyi gerektirmez.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import threading
import time
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from local_rag.config import (
    DEFAULT_DB_PATH,
    RERANK_SCORE_THRESHOLD,
    SYSTEM_PROMPT,
    TOP_K,
    TOP_N_CANDIDATES,
)
from local_rag.embeddings.base import EmbeddingProvider
from local_rag.ingestion.chunker import chunk_blocks
from local_rag.ingestion.parsers import SUPPORTED_EXTENSIONS, parse_file
from local_rag.lang import detect_language
from local_rag.llm.base import LLMProvider
from local_rag.reranking.base import RerankerProvider
from local_rag.retrieval.search import hybrid_candidates
from local_rag.security.injection import scan_for_injection
from local_rag.security.rbac import DEFAULT_ROLE, get_role
from local_rag.storage import db
from local_rag.storage.db import RetrievedChunk, SearchFilters

METADATA_SIDECAR_FILENAME = "_metadata.json"
"""docs_dir altında aranan opsiyonel dosya: {"dosya.ext": {"classification":
"gizli", "department": "Yönetim"}, ...} — belirtilmeyen dosyalar varsayılan
("genel"/"Genel") ile ingest edilir."""


@dataclass
class SourceRef:
    filename: str
    section_path: str


@dataclass
class AnswerResult:
    answer: str
    sources: list[SourceRef]
    confident: bool = True
    """False: reranker skoru eşiğin altında kaldığı için LLM hiç çağrılmadı,
    kademeli bir 'emin değilim' yanıtı döndürüldü."""
    faithfulness: float | None = None
    """Cevaptaki içerik kelimelerinin bağlamda ne kadarının geçtiğini ölçen
    kaba, deterministik bir sinyal (0-1) — yalnızca confident=True iken
    doldurulur. Rigorous bir NLI/faithfulness kontrolü DEĞİLDİR; UI'da bu
    yüzden 'bağlam örtüşmesi' olarak dürüstçe etiketlenir (bkz. app.py)."""
    latency_seconds: float | None = None
    """answer() çağrısının toplam süresi (retrieval+rerank+generation dahil,
    embedding'i de kapsar) — performans raporlaması için (bkz. eval/run_eval.py,
    docs/DEMO_SENARYOLARI.md)."""


@dataclass
class _Prepared:
    """`_prepare()`'ın çıktısı: ya güvensiz bir durumun hazır fallback sonucu,
    ya da LLM üretimi için gereken bağlam (fallback=None). answer() ve
    answer_streaming() bu ara temsili paylaşır."""

    fallback: AnswerResult | None
    context: str | None = None
    user_prompt: str | None = None
    sources: list[SourceRef] | None = None
    chunk_ids: list[int] | None = None


def _append_source_legend(answer_text: str, sources: list[SourceRef]) -> str:
    """Cevaptaki [S1]/[S2] etiketlerinin hangi gerçek dokümana karşılık geldiğini
    çözen numaralı kaynak listesini KOŞULSUZ ekler. Eski davranış (yalnızca
    model hiç kaynak göstermemişse footer eklemek) dosya adının cevap metninde
    aynı yazımla geçip geçmediğine bakan kırılgan bir substring eşleşmesine
    dayanıyordu — model bir dosya adını yanlış yazdığında bile tetiklenmesi
    gerekiyordu ve gerçek bir vakada tam da bunu yaptı. [S1] gibi bir etiket tek başına
    okuyucu için anlamsız olduğundan (dosya adı değil), model doğru
    etiketlese de etiketlemese de bir legend gereklidir; bu da substring
    eşleşmesine ihtiyacı ortadan kaldırır."""
    if not sources:
        return answer_text
    legend = " · ".join(f"[S{i + 1}] {s.filename} ({s.section_path})" for i, s in enumerate(sources))
    return f"{answer_text}\n\n**Kaynaklar:** {legend}"


def _select_diverse_top_k(
    ranked: list[tuple[RetrievedChunk, float]], top_k: int
) -> list[tuple[RetrievedChunk, float]]:
    """Rerank sonrası top-k seçiminde tek bir dokümanın tüm slotları kapmasını
    önleyen hafif bir çeşitlilik koruması (basitleştirilmiş MMR). NovaBank
    korpusunda gerçek retrieval örneklemesinde kümelenme gözlenmedi, yani
    bu şu an aktif bir hatayı düzeltmiyor —
    daha büyük/daha az çeşitli bir korpusta ortaya çıkabilecek bir riske karşı
    önleyici, düşük riskli bir korumadır: önce her dokümandan en fazla
    per_doc_cap chunk alınır (skor sırası korunarak), kalan slotlar (varsa)
    elenen chunk'larla skor sırasıyla doldurulur. Böylece hiçbir zaman daha
    düşük skorlu bir chunk, cap'in ALTINDA kalan daha yüksek skorlu bir
    chunk'ın yerine geçmez ve sonuç seti her zaman tam top_k (ya da daha az
    aday varsa mevcut tüm adaylar) uzunluğundadır — doğal olarak zaten çeşitli
    olan sonuçlarda (bu korpusta olduğu gibi) davranışı değiştirmez."""
    if top_k <= 0:
        return []
    per_doc_cap = max(2, -(-top_k * 3 // 5))  # ceil(top_k * 0.6), en az 2
    selected: list[tuple[RetrievedChunk, float]] = []
    deferred: list[tuple[RetrievedChunk, float]] = []
    per_doc_count: dict[str, int] = {}
    for chunk, score in ranked:
        count = per_doc_count.get(chunk.source_filename, 0)
        if count < per_doc_cap:
            selected.append((chunk, score))
            per_doc_count[chunk.source_filename] = count + 1
        else:
            deferred.append((chunk, score))
        if len(selected) >= top_k:
            break
    for chunk, score in deferred:
        if len(selected) >= top_k:
            break
        selected.append((chunk, score))
    return selected


_WORD_RE = re.compile(r"\w+", re.UNICODE)


def _estimate_faithfulness(answer_text: str, context_text: str) -> float:
    """Cevabın (>3 harfli) içerik kelimelerinden kaçının bağlamda geçtiğinin
    oranı — ucuz, anlık bir sinyal. Cevap boşsa 1.0 döner (ölçülecek bir şey yok)."""
    answer_words = {w for w in _WORD_RE.findall(answer_text.lower()) if len(w) > 3}
    if not answer_words:
        return 1.0
    context_words = set(_WORD_RE.findall(context_text.lower()))
    return len(answer_words & context_words) / len(answer_words)


class RagPipeline:
    def __init__(
        self,
        embedder: EmbeddingProvider,
        reranker: RerankerProvider | None = None,
        llm: LLMProvider | None = None,
        db_path: Path = DEFAULT_DB_PATH,
    ):
        """`reranker`/`llm`, yalnızca `answer()` çağrılırken gereklidir — salt
        ingestion (bkz. scripts/ingest.py) için bunları yüklemek gereksiz
        kaynak tüketimine yol açar, bu yüzden opsiyoneldirler."""
        self.embedder = embedder
        self.reranker = reranker
        self.llm = llm
        self.conn = db.get_connection(db_path, embedding_dimension=embedder.dimension)
        # RagPipeline, Streamlit'in st.cache_resource'u sayesinde birden çok
        # thread/oturum arasında paylaşılan tek bir nesne olabilir; tek sqlite3
        # bağlantısına eşzamanlı erişimi bu kilit serileştirir.
        self._db_lock = threading.Lock()

    def close(self) -> None:
        self.conn.close()

    def list_documents(self) -> list[sqlite3.Row]:
        with self._db_lock:
            return db.list_documents(self.conn)

    def ingest_path(self, docs_dir: Path) -> list[str]:
        """docs_dir altındaki desteklenen tüm dosyaları işler; içerik hash'i
        değişmemiş dosyaları atlar (idempotent re-ingestion). docs_dir'de bir
        `_metadata.json` varsa, dosya bazlı classification/department
        atamaları oradan okunur (bkz. METADATA_SIDECAR_FILENAME)."""
        metadata_path = docs_dir / METADATA_SIDECAR_FILENAME
        metadata_by_filename: dict[str, dict] = {}
        if metadata_path.exists():
            metadata_by_filename = json.loads(metadata_path.read_text(encoding="utf-8"))

        log: list[str] = []
        for path in sorted(docs_dir.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue
            entry = metadata_by_filename.get(path.name, {})
            log.append(
                self.ingest_file(
                    path,
                    classification=entry.get("classification", "genel"),
                    department=entry.get("department", "Genel"),
                )
            )
        return log

    def ingest_file(self, path: Path, classification: str = "genel", department: str = "Genel") -> str:
        content_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        with self._db_lock:
            already_exists = db.get_document_by_hash(self.conn, content_hash) is not None
        if already_exists:
            return f"[atlandı, değişmemiş] {path.name}"

        blocks = parse_file(path)
        if not blocks:
            return f"[boş/ayrıştırılamadı] {path.name}"
        chunks = chunk_blocks(blocks)
        if not chunks:
            return f"[chunk üretilemedi] {path.name}"

        document_id = str(uuid.uuid4())
        sample_text = " ".join(c.text for c in chunks[:5])
        # Embed edilen metne section_path öneki eklenir (yalnızca embedding
        # GİRDİSİ için — db.insert_chunk'a aşağıda hâlâ ham chunk.text
        # gönderilir, FTS5/LLM bağlamı/UI etkilenmez). Amaç: aynı belgede
        # birbirine benzeyen ama farklı bölümlere ait chunk'ları (ör. iki ayrı
        # tablodaki benzer sayısal değerler) embedding uzayında ayırt etmeye
        # yardımcı olacak konu sinyali eklemek.
        embeddings = self.embedder.embed([f"{c.section_path}\n{c.text}" for c in chunks])

        with self._db_lock:
            db.insert_document(
                self.conn,
                document_id=document_id,
                filename=path.name,
                title=path.stem,
                file_type=path.suffix.lower(),
                classification=classification,
                department=department,
                content_hash=content_hash,
                language=detect_language(sample_text),
                ingested_at=datetime.now(timezone.utc).isoformat(),
            )
            flagged_count = 0
            for chunk, embedding in zip(chunks, embeddings):
                matches = scan_for_injection(chunk.text)
                injection_flag = ",".join(matches) if matches else None
                if injection_flag:
                    flagged_count += 1
                db.insert_chunk(
                    self.conn,
                    document_id=document_id,
                    chunk_index=chunk.chunk_index,
                    section_path=chunk.section_path,
                    chunk_type=chunk.chunk_type,
                    text=chunk.text,
                    embedding=embedding,
                    page_start=chunk.page_start,
                    page_end=chunk.page_end,
                    injection_flag=injection_flag,
                )
        flag_note = f", {flagged_count} chunk şüpheli — bkz. list_flagged_chunks()" if flagged_count else ""
        return f"[eklendi] {path.name} ({len(chunks)} chunk{flag_note})"

    def list_flagged_chunks(self) -> list[sqlite3.Row]:
        """Prompt-injection taramasında şüpheli bulunan chunk'ların insan
        incelemesi kuyruğu (bkz. security/injection.py)."""
        with self._db_lock:
            return db.list_flagged_chunks(self.conn)

    def _require_generation(self) -> None:
        if self.llm is None or self.reranker is None:
            raise RuntimeError(
                "RagPipeline bir RerankerProvider/LLMProvider olmadan başlatıldı; "
                "answer()/answer_streaming() için ikisi de gerekli."
            )

    def _prepare(
        self, query: str, role: str, top_k: int, top_n: int, filters: SearchFilters | None, start: float
    ) -> _Prepared:
        """Retrieval + rerank + eşik kontrolü — LLM'i ÇAĞIRMAZ. Güvensizse
        (aday yok ya da en iyi skor eşik altında) audit'i yazıp hazır bir
        fallback AnswerResult'lı _Prepared döner; güvenliyse üretim için gereken
        bağlamı döner."""
        rbac_filters = _apply_rbac(role, filters)
        with self._db_lock:
            candidates = hybrid_candidates(self.conn, self.embedder, query, top_n=top_n, filters=rbac_filters)
        if not candidates:
            result = AnswerResult(answer="Bu bilgiyi elimdeki dokümanlarda bulamadım.", sources=[], confident=False)
            result.latency_seconds = time.perf_counter() - start
            self._log_audit(role, query, [], result)
            return _Prepared(fallback=result)

        scores = self.reranker.score(query, [c.text for c in candidates])
        ranked = sorted(zip(candidates, scores), key=lambda pair: pair[1], reverse=True)
        top = _select_diverse_top_k(ranked, top_k)
        best_chunk, best_score = top[0]
        if best_score < RERANK_SCORE_THRESHOLD:
            pointer = SourceRef(filename=best_chunk.source_filename, section_path=best_chunk.section_path)
            result = AnswerResult(
                answer=(
                    "Kesin bir cevap bulamadım. En yakın ilgili bölüm: "
                    f"{pointer.filename} ({pointer.section_path}). "
                    "Sorunuzu farklı ifade etmeyi deneyebilirsiniz."
                ),
                sources=[pointer],
                confident=False,
            )
            result.latency_seconds = time.perf_counter() - start
            self._log_audit(role, query, [best_chunk.chunk_id], result)
            return _Prepared(fallback=result)

        top_chunks = [chunk for chunk, _ in top]
        # Dosya adı bilerek BAĞLAM'a verilmez (yalnızca [Sn] etiketi) — model bir
        # dosya adını yanlış yazıp halüsinasyon üretemesin diye (bkz. _append_source_legend).
        # section_path yine de verilir: modelin konuyu ayırt etmesine yardımcı
        # olan faydalı bir sinyaldir ve bir dosya adı gibi kelimesi kelimesine
        # kopyalanıp atıf olarak kullanılma riski taşımaz.
        context = "\n\n---\n\n".join(
            f"[S{i + 1}] ({chunk.section_path})\n{chunk.text}" for i, chunk in enumerate(top_chunks)
        )
        return _Prepared(
            fallback=None,
            context=context,
            user_prompt=f"BAĞLAM:\n{context}\n\nSORU: {query}",
            sources=[SourceRef(filename=c.source_filename, section_path=c.section_path) for c in top_chunks],
            chunk_ids=[c.chunk_id for c in top_chunks],
        )

    def _finalize(self, answer_text: str, prep: _Prepared, role: str, query: str, start: float) -> AnswerResult:
        """LLM cevabı elde edildikten sonraki ortak son işlemler: faithfulness
        tahmini, zorunlu kaynak künyesi, audit log ve gecikme. answer() ile
        answer_streaming() bunu paylaşır."""
        faithfulness = _estimate_faithfulness(answer_text, prep.context)
        answer_text = _append_source_legend(answer_text, prep.sources)
        result = AnswerResult(answer=answer_text, sources=prep.sources, confident=True, faithfulness=faithfulness)
        result.latency_seconds = time.perf_counter() - start
        self._log_audit(role, query, prep.chunk_ids, result)
        return result

    def answer(
        self,
        query: str,
        role: str = DEFAULT_ROLE,
        top_k: int = TOP_K,
        top_n: int = TOP_N_CANDIDATES,
        filters: SearchFilters | None = None,
    ) -> AnswerResult:
        """`role`, RBAC filtresini belirler ve retrieval sorgusuna girer —
        yani yetkisiz chunk'lar LLM bağlamına asla ulaşmaz, yalnızca UI'da
        gizlenmez. Her çağrı, sonucu
        `role`, `query`, kullanılan chunk_id'ler ve cevapla birlikte
        hash-chained audit log'a yazar. Canlı akış için bkz. answer_streaming()."""
        self._require_generation()
        start = time.perf_counter()
        prep = self._prepare(query, role, top_k, top_n, filters, start)
        if prep.fallback is not None:
            return prep.fallback
        answer_text = self.llm.chat(SYSTEM_PROMPT, prep.user_prompt)
        return self._finalize(answer_text, prep, role, query, start)

    def answer_streaming(
        self,
        query: str,
        role: str = DEFAULT_ROLE,
        top_k: int = TOP_K,
        top_n: int = TOP_N_CANDIDATES,
        filters: SearchFilters | None = None,
    ) -> Iterator[str | AnswerResult]:
        """answer()'ın streaming varyantı. Güvenli (confident) durumda: cevap
        parçalarını (str) geldikçe yield eder, ardından finalize edilmiş
        AnswerResult'ı EN SON öğe olarak yield eder. Güvensiz (fallback)
        durumda: hiç str parça yok, yalnızca fallback AnswerResult yield edilir.
        Çağıran (UI) str parçaları canlı gösterir ve str-olmayan tek öğeyi nihai
        sonuç olarak alır. Not: kaynak legend'ı her zaman sona eklendiğinden
        nihai `AnswerResult.answer`, akıtılan ham metinden her zaman daha uzundur
        — UI son metni result.answer'dan almalıdır."""
        self._require_generation()
        start = time.perf_counter()
        prep = self._prepare(query, role, top_k, top_n, filters, start)
        if prep.fallback is not None:
            yield prep.fallback
            return
        collected: list[str] = []
        for token in self.llm.chat_stream(SYSTEM_PROMPT, prep.user_prompt):
            collected.append(token)
            yield token
        yield self._finalize("".join(collected), prep, role, query, start)

    def _log_audit(self, role: str, query: str, chunk_ids: list[int], result: AnswerResult) -> None:
        with self._db_lock:
            db.insert_audit_entry(
                self.conn,
                role=role,
                query=query,
                retrieved_chunk_ids=chunk_ids,
                confident=result.confident,
                answer=result.answer,
            )

    def list_audit_entries(self, limit: int = 100) -> list[sqlite3.Row]:
        with self._db_lock:
            return db.list_audit_entries(self.conn, limit=limit)

    def verify_audit_log(self) -> bool:
        with self._db_lock:
            return db.verify_audit_chain(self.conn)


def _apply_rbac(role_name: str, filters: SearchFilters | None) -> SearchFilters:
    role = get_role(role_name)
    base = filters or SearchFilters()
    return SearchFilters(
        document_ids=base.document_ids,
        file_types=base.file_types,
        section_contains=base.section_contains,
        classifications=list(role.allowed_classifications),
        departments=list(role.allowed_departments) if role.allowed_departments else None,
    )
