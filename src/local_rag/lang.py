"""Doküman dili tespiti — yalnızca bilgilendirici metadata amaçlıdır (UI'daki
"Dil" sütunu); retrieval, RBAC veya herhangi bir karar mekanizmasını etkilemez."""

from __future__ import annotations

from langdetect import DetectorFactory, LangDetectException, detect

# langdetect'in iç algoritması varsayılan olarak rastgele örnekleme kullanır;
# sabit bir seed olmadan aynı metin farklı çalıştırmalarda farklı sonuç
# verebilir. Aynı dokümanın her ingest'te aynı dil etiketini alması için
# modül yüklenirken bir kez sabitlenir.
DetectorFactory.seed = 0


def detect_language(text: str) -> str:
    """ISO 639-1 dil kodu (örn. "tr", "en") veya metin çok kısa/boşsa "unknown" döner."""
    try:
        return detect(text)
    except LangDetectException:
        return "unknown"
