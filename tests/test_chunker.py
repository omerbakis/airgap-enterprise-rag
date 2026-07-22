from local_rag.config import CHUNK_OVERLAP_WORDS, MAX_ATOMIC_CHUNK_WORDS
from local_rag.ingestion.chunker import chunk_blocks
from local_rag.ingestion.parsers import Block


def test_section_path_tracks_heading_hierarchy():
    blocks = [
        Block(type="heading", level=1, text="İzin Politikası"),
        Block(type="heading", level=2, text="Yıllık İzin"),
        Block(type="paragraph", text="Yıllık izin süresi kıdeme göre değişir."),
        Block(type="heading", level=2, text="Mazeret İzni"),
        Block(type="paragraph", text="Mazeret izni yılda 5 gün ile sınırlıdır."),
    ]
    chunks = chunk_blocks(blocks)
    assert [c.section_path for c in chunks] == [
        "İzin Politikası > Yıllık İzin",
        "İzin Politikası > Mazeret İzni",
    ]


def test_sibling_heading_replaces_previous_subsection():
    blocks = [
        Block(type="heading", level=1, text="Bölüm A"),
        Block(type="heading", level=2, text="Alt A.1"),
        Block(type="heading", level=1, text="Bölüm B"),
        Block(type="paragraph", text="Bölüm B içeriği."),
    ]
    chunks = chunk_blocks(blocks)
    assert chunks[0].section_path == "Bölüm B"


def test_table_is_atomic_and_not_merged_with_paragraphs():
    blocks = [
        Block(type="heading", level=1, text="Tablo Bölümü"),
        Block(type="paragraph", text="Aşağıdaki tabloyu inceleyin."),
        Block(type="table", text="| A | B |\n| --- | --- |\n| 1 | 2 |"),
        Block(type="paragraph", text="Tablodan sonraki paragraf."),
    ]
    chunks = chunk_blocks(blocks)
    types = [c.chunk_type for c in chunks]
    assert "table" in types
    table_chunk = next(c for c in chunks if c.chunk_type == "table")
    assert table_chunk.text == "| A | B |\n| --- | --- |\n| 1 | 2 |"


def test_list_items_exceeding_max_atomic_words_split_at_item_boundary():
    # Reranker (bge-reranker-v2-m3, max_length=512) sınırını sessizce aşmamak
    # için MAX_ATOMIC_CHUNK_WORDS aşıldığında chunk madde sınırından bölünmeli
    # (bkz. config.py'deki gerekçe) — asla bir madde ortasından değil.
    item_words = 40
    item_count = 10  # 400 kelime > MAX_ATOMIC_CHUNK_WORDS (250)
    blocks = [
        Block(type="heading", level=1, text="Prosedür"),
        *[
            Block(type="list_item", text=" ".join(f"madde{i}_kelime{w}" for w in range(item_words)))
            for i in range(item_count)
        ],
    ]
    chunks = chunk_blocks(blocks)
    list_chunks = [c for c in chunks if c.chunk_type == "list"]

    assert len(list_chunks) > 1, "400 kelimelik liste tek bir atomic chunk'a sığmamalı"
    for chunk in list_chunks:
        # "- " madde işaretleri ayrı bir "kelime" olarak sayılmasın diye çıkarılır;
        # test ettiğimiz gerçek kısıt madde İÇERİĞİNİN kelime sayısıdır.
        content_words = [w for w in chunk.text.split() if w != "-"]
        assert len(content_words) <= MAX_ATOMIC_CHUNK_WORDS
    # Hiçbir madde ortadan bölünmemeli: her "madde{i}_kelime{w}" token'ı bir
    # bütün olarak (madde/satır sınırından bölünmeden) bir chunk'ın içinde bulunmalı.
    all_words = set(" ".join(c.text for c in list_chunks).split())
    expected_words = {f"madde{i}_kelime{w}" for i in range(item_count) for w in range(item_words)}
    assert expected_words <= all_words


def test_list_items_are_grouped_into_single_atomic_chunk():
    blocks = [
        Block(type="heading", level=1, text="Prosedür"),
        Block(type="list_item", text="Adım 1: Formu doldur."),
        Block(type="list_item", text="Adım 2: Onaya gönder."),
        Block(type="list_item", text="Adım 3: Arşivle."),
    ]
    chunks = chunk_blocks(blocks)
    list_chunks = [c for c in chunks if c.chunk_type == "list"]
    assert len(list_chunks) == 1
    assert "Adım 1" in list_chunks[0].text
    assert "Adım 3" in list_chunks[0].text


def test_long_paragraph_is_recursively_split_without_breaking_words():
    long_paragraph = ". ".join(f"Bu cümle numara {i} içindir" for i in range(200))
    blocks = [Block(type="paragraph", text=long_paragraph)]
    chunks = chunk_blocks(blocks)
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk.text.split()) <= 520  # MAX_CHUNK_WORDS'e küçük bir tolerans
    # Hiçbir chunk, orijinal metindeki bir kelimeyi ortadan bölmemeli.
    rejoined_words = set(" ".join(c.text for c in chunks).split())
    original_words = set(long_paragraph.split())
    assert rejoined_words <= original_words


def test_consecutive_paragraphs_overlap_between_chunks():
    # Her paragrafın kelime dağarcığı BENZERSİZ (p{p}_w{i}) — ortak kelimeler
    # yalnızca overlap mekanizması gerçekten çalışıyorsa ortaya çıkabilir; ortak
    # bir kelime havuzu kullanan eski test, buffer_words'ün flush_text_buffer()
    # tarafından sıfırlandıktan SONRA okunduğu (dolayısıyla overlap'in her zaman
    # boş olduğu) bir bug'ı maskeliyordu.
    words_per_paragraph = 80
    paragraphs = [
        Block(type="paragraph", text=" ".join(f"p{p}_w{i}" for i in range(words_per_paragraph)))
        for p in range(6)
    ]
    chunks = chunk_blocks(paragraphs)
    assert len(chunks) >= 2

    first_words = chunks[0].text.split()
    second_words = chunks[1].text.split()

    expected_overlap = first_words[-CHUNK_OVERLAP_WORDS:]
    assert expected_overlap, "İlk chunk beklenenden kısa"
    assert second_words[: len(expected_overlap)] == expected_overlap, (
        "İkinci chunk, ilk chunk'ın son CHUNK_OVERLAP_WORDS kelimesiyle başlamalı"
    )
