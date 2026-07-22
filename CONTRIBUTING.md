# Katkıda Bulunma

Bu proje bir portfolyo/demo projesidir, ancak issue ve PR'lara açığız — hata
bildirimleri, öneri ve küçük iyileştirmeler memnuniyetle karşılanır.

## Geliştirme Ortamı

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
```

Birim testleri Foundry Local **gerektirmez** (sahte/fake provider'lar kullanır):

```bash
.venv/Scripts/python.exe -m pytest -v
```

Foundry Local gerektiren gerçek entegrasyon/eval testleri için bkz. [README.md](./README.md#test)
ve [docs/AIRGAP_KURULUM.md](./docs/AIRGAP_KURULUM.md).

## PR Göndermeden Önce

1. `pytest -v` çalıştırıp tüm testlerin geçtiğinden emin olun.
2. Yeni bir davranış eklediyseniz, mevcut desenleri takip eden bir test ekleyin
   (`tests/fakes.py`'deki sahte provider'lar gerçek Foundry Local olmadan
   deterministik test yazmayı sağlar).
3. Değişikliğinizin gerekçesini PR açıklamasında kısaca belirtin — bu proje,
   mimari kararların *neden* alındığını belgelemeyi önemser; aynı disiplini
   katkılarda da tercih ederiz.

## Kod Stili

- Python: `from __future__ import annotations` + tip belirteçleri (mevcut kod
  tabanındaki desenle tutarlı).
- Yorumlar yalnızca *neden*i açıklar (non-obvious kısıtlar, tasarım kararları);
  *ne* yaptığını açıklayan yorumlar yazılmaz — kod zaten okunabilir olmalı.
- Kod tabanı Türkçe yorum/docstring kullanır; katkılarınızda da bu tutarlılığı
  koruyabilirsiniz (İngilizce de kabul edilir, karışık olması sorun değildir).

## Sorular

Büyük bir değişiklik düşünüyorsanız (yeni bir provider, mimari bir değişiklik),
önce bir issue açıp yaklaşımınızı tartışmanız önerilir — bu, PR aşamasında
sürpriz olmasını önler.
