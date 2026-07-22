"""Section-aware + recursive fallback chunking.

Strateji:
  1. Başlık (heading) blokları bir bölüm yığını (section stack) oluşturur;
     her chunk, o anki tam bölüm yolunu (section_path) metadata olarak taşır.
  2. Ardışık paragraflar TARGET_CHUNK_WORDS hedefine göre paketlenir; bir sonraki
     paragraf eklendiğinde limiti aşacaksa chunk kapatılır ve overlap kadar
     kelime bir sonraki chunk'a taşınır.
  3. Tek bir paragraf MAX_CHUNK_WORDS'ü aşarsa, cümle/boşluk sınırlarına göre
     recursive olarak bölünür (asla kelime ortasından değil).
  4. Tablo ve liste (prosedür) blokları asla paragraf akışıyla birleştirilmez;
     atomic (bölünmeden tek chunk) olarak eklenir; yalnızca
     MAX_ATOMIC_CHUNK_WORDS aşılırsa madde/satır sınırından bölünür.
"""

from __future__ import annotations

from dataclasses import dataclass

from local_rag.config import (
    CHUNK_OVERLAP_WORDS,
    MAX_ATOMIC_CHUNK_WORDS,
    MAX_CHUNK_WORDS,
    TARGET_CHUNK_WORDS,
)
from local_rag.ingestion.parsers import Block

_RECURSIVE_SEPARATORS = ["\n\n", "\n", ". ", " "]


@dataclass
class Chunk:
    text: str
    section_path: str
    chunk_type: str  # "text" | "table" | "list"
    chunk_index: int
    page_start: int | None = None
    page_end: int | None = None


def _word_count(text: str) -> int:
    return len(text.split())


def _recursive_split(text: str, max_words: int) -> list[str]:
    if _word_count(text) <= max_words:
        return [text]
    for sep in _RECURSIVE_SEPARATORS:
        if sep in text:
            parts = [p for p in text.split(sep) if p.strip()]
            if len(parts) > 1:
                pieces: list[str] = []
                buffer = ""
                for part in parts:
                    candidate = f"{buffer}{sep}{part}" if buffer else part
                    if _word_count(candidate) <= max_words:
                        buffer = candidate
                    else:
                        if buffer:
                            pieces.append(buffer)
                        buffer = part
                if buffer:
                    pieces.append(buffer)
                # Hâlâ limiti aşan parça varsa bir sonraki (daha ince) ayırıcıyla devam et.
                result: list[str] = []
                for piece in pieces:
                    if _word_count(piece) > max_words:
                        result.extend(_recursive_split(piece, max_words))
                    else:
                        result.append(piece)
                return result
    # Hiçbir ayırıcı bulunamadı (tek uzun kelime dizisi) — olduğu gibi bırak.
    return [text]


class _SectionState:
    def __init__(self) -> None:
        self.stack: list[tuple[int, str]] = []  # (level, title)

    def update(self, level: int, title: str) -> None:
        while self.stack and self.stack[-1][0] >= level:
            self.stack.pop()
        self.stack.append((level, title))

    @property
    def path(self) -> str:
        return " > ".join(title for _, title in self.stack) or "(Genel)"


def chunk_blocks(blocks: list[Block]) -> list[Chunk]:
    section = _SectionState()
    chunks: list[Chunk] = []
    chunk_index = 0

    # Paragraf akışı için tampon (rolling buffer).
    buffer_words: list[str] = []
    buffer_pages: list[int] = []

    def flush_text_buffer():
        nonlocal chunk_index, buffer_words, buffer_pages
        if not buffer_words:
            return
        text = " ".join(buffer_words)
        for piece in _recursive_split(text, MAX_CHUNK_WORDS):
            chunks.append(
                Chunk(
                    text=piece.strip(),
                    section_path=section.path,
                    chunk_type="text",
                    chunk_index=chunk_index,
                    page_start=buffer_pages[0] if buffer_pages else None,
                    page_end=buffer_pages[-1] if buffer_pages else None,
                )
            )
            chunk_index += 1
        buffer_words = []
        buffer_pages = []

    pending_list_items: list[str] = []

    def flush_list_buffer():
        nonlocal chunk_index, pending_list_items
        if not pending_list_items:
            return
        for piece in _pack_atomic_items(pending_list_items):
            chunks.append(
                Chunk(
                    text=piece,
                    section_path=section.path,
                    chunk_type="list",
                    chunk_index=chunk_index,
                )
            )
            chunk_index += 1
        pending_list_items = []

    for block in blocks:
        if block.type == "heading":
            flush_text_buffer()
            flush_list_buffer()
            section.update(block.level or 1, block.text)
            continue

        if block.type == "table":
            flush_text_buffer()
            flush_list_buffer()
            chunks.append(
                Chunk(
                    text=block.text,
                    section_path=section.path,
                    chunk_type="table",
                    chunk_index=chunk_index,
                    page_start=block.page,
                    page_end=block.page,
                )
            )
            chunk_index += 1
            continue

        if block.type == "list_item":
            flush_text_buffer()
            pending_list_items.append(block.text)
            continue

        # paragraph
        flush_list_buffer()
        candidate_words = buffer_words + block.text.split()
        if buffer_words and len(candidate_words) > TARGET_CHUNK_WORDS:
            # overlap, flush_text_buffer() buffer_words'ü sıfırlamadan ÖNCE okunmalı.
            overlap = buffer_words[-CHUNK_OVERLAP_WORDS:]
            flush_text_buffer()
            buffer_words = list(overlap)
        buffer_words.extend(block.text.split())
        if block.page is not None:
            buffer_pages.append(block.page)

    flush_text_buffer()
    flush_list_buffer()
    return chunks


def _pack_atomic_items(items: list[str]) -> list[str]:
    """Liste/prosedür maddelerini, madde sınırından bölerek atomic chunk'lara paketler."""
    pieces: list[str] = []
    buffer: list[str] = []
    buffer_words = 0
    for item in items:
        item_words = _word_count(item)
        if buffer and buffer_words + item_words > MAX_ATOMIC_CHUNK_WORDS:
            pieces.append("\n".join(f"- {i}" for i in buffer))
            buffer = []
            buffer_words = 0
        buffer.append(item)
        buffer_words += item_words
    if buffer:
        pieces.append("\n".join(f"- {i}" for i in buffer))
    return pieces
