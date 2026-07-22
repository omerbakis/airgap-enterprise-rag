from local_rag.lang import detect_language


def test_detects_turkish_text():
    assert detect_language("Yıllık izin süresi kıdeme göre değişir çalışan hakları düzenlenir.") == "tr"


def test_detects_english_text():
    assert detect_language("The annual leave duration varies based on seniority and employee rights.") == "en"


def test_empty_or_whitespace_text_returns_unknown():
    assert detect_language("") == "unknown"
    assert detect_language("   ") == "unknown"


def test_is_deterministic_across_repeated_calls():
    # DetectorFactory.seed sabitlenmemiş olsaydı bu, langdetect'in iç rastgele
    # örneklemesi yüzünden çalıştırmalar arası farklı sonuç verebilirdi.
    text = "Şirket personelinin yıllık izin hakları ve kullanım koşulları bu belgede düzenlenmektedir."
    results = {detect_language(text) for _ in range(10)}
    assert results == {"tr"}
