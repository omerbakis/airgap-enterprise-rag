# Air-Gapped Kurulum Rehberi

Bu rehber, Local-RAG'ı gerçek bir air-gapped (internete hiç bağlanmayan) makinede
çalıştırmak için iki aşamalı süreci anlatır: **(1)** internete bağlı bir makinede
gerekli her şeyin bir kez indirilmesi, **(2)** bunların USB ile air-gapped makineye
taşınıp orada offline kurulumu.

## Adım 1 — İnternete Bağlı Makinede Hazırlık

### 1.1 Python paketleri

```powershell
.\scripts\prepare_offline_wheels.ps1
```

`offline_wheels/` klasörünü doldurur (requirements.txt'teki her paketin wheel'i).

### 1.2 Foundry Local modelleri

```powershell
foundry model download qwen3-embedding-0.6b
foundry model download qwen2.5-7b
foundry cache location
```

Son komut, model dosyalarının bulunduğu klasörü gösterir (tipik olarak
`%USERPROFILE%\.foundry\cache\models`). Bu klasörün tamamı USB'ye kopyalanacak.

### 1.3 Reranker modeli (bge-reranker-v2-m3)

```powershell
.venv\Scripts\python.exe scripts\prepare_reranker_cache.py
```

Bu, `sentence-transformers` modelini HuggingFace Hub'dan indirip
`%USERPROFILE%\.cache\huggingface\` altına önbelleğe alır ve bir örnek
sorguyla gerçekten çalıştığını doğrular — "uygulamayı bir kez çalıştırıp
soru sorduğunuzdan emin olun" gibi elle takip edilmesi gereken bir adıma
göre daha deterministiktir. Beklenen çıktı:

```
Reranker modeli başarıyla önbelleğe alındı ve doğrulandı: [0.15..., 0.00...]
```

Bu klasörün tamamının (`%USERPROFILE%\.cache\huggingface\`) USB'ye
kopyalanması yeterlidir, ayrı bir adım gerekmez.

### 1.4 Foundry Local uygulamasının kendisi

Air-gapped makinede Foundry Local henüz kurulu değilse, offline installer'ı
edinin (bkz. [Microsoft Learn — Foundry Local offline kurulum](https://learn.microsoft.com/azure/foundry-local/)).
Foundry Local zaten "bağlı makinede indirip USB ile disconnected cache'e
taşıma" senaryosunu resmi olarak destekler.

## Adım 2 — USB ile Taşınacaklar

| Kaynak (bağlı makine) | Hedef (air-gapped makine) |
|---|---|
| `offline_wheels/` | Proje kök dizini |
| `%USERPROFILE%\.foundry\cache\models\` | Aynı yol (veya `foundry cache cd <yeni-yol>` ile özel bir konum) |
| `%USERPROFILE%\.cache\huggingface\` | Aynı yol |
| Foundry Local offline installer | — |
| Bu proje deposunun tamamı (`Local-RAG/`) | — |

## Adım 3 — Air-Gapped Makinede Kurulum

```powershell
# 1) Foundry Local'ı offline installer ile kurun, ardından model cache'i yerleştirin
#    (USB'den kopyalanan klasör foundry cache location çıktısıyla aynı yola gitmeli)

# 2) Python ortamı
python -m venv .venv
.venv\Scripts\python.exe -m pip install --no-index --find-links offline_wheels -r requirements.txt

# 3) HuggingFace cache'i yerleştirin (USB'den %USERPROFILE%\.cache\huggingface\ konumuna kopyalayın)
#    ve tamamen offline çalıştığından emin olmak için:
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"

# 4) Doğrulama
.venv\Scripts\python.exe -m pytest -q
foundry server status
.venv\Scripts\python.exe -m streamlit run app.py
```

`HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1`, sentence-transformers/transformers
kütüphanesinin, modeli kullanmadan önce Hub'a "güncel mi?" kontrolü yapmasını
engeller — bu kontrol, ağ yokken (ör. wifi kapalı) sessizce beklemek yerine
`Cannot send a request, as the client has been closed` gibi bir hatayla ANINDA
çöker; offline zorlandığında ise doğrudan yerel önbellek kullanılır.

**Bu iki değişken artık kod düzeyinde varsayılan olarak açıktır:** reranker'ı
yükleyen modül (`src/local_rag/reranking/bge.py`), `sentence_transformers` import
edilmeden önce `os.environ.setdefault(...)` ile ikisini de `1` yapar. Yani
manuel ayarlamaya gerek kalmadan sistem tümüyle offline çalışır. Yalnızca
**ilk kez model indirirken** (henüz önbellek yokken) geçici olarak
`HF_HUB_OFFLINE=0` ayarlayıp indirme yapılmalı, sonra tekrar kapatılabilir.
`setdefault` kullanıldığı için bu override çalışır (ortamdaki değer koda üstün gelir).

**Foundry Local model indirme yolu için ayrı bir anahtar gerekir.** Yukarıdaki
iki değişken yalnızca reranker/HuggingFace tarafını kapatır; `foundry_client.connect()`
(chat + embedding modelleri) varsayılan olarak eksik bir alias'ı sessizce
`foundry model download` ile indirmeye çalışır — bu air-gapped bir ortamda
"model adı yanlış yazıldı/unutuldu" gibi bir hatayı, anlaşılmaz bir ağ zaman
aşımına çevirebilir. Bunu engellemek için:

```powershell
$env:LOCAL_RAG_STRICT_OFFLINE = "1"
```

Bu ayarlıyken, yerelde önceden indirilmemiş bir alias için `connect()` ağa
çıkmayı hiç denemez, bunun yerine anında `FoundryModelNotFound` fırlatır
(hata mesajı hangi `foundry model download` komutunun önceden çalıştırılması
gerektiğini söyler). Air-gapped dağıtımlarda bu değişkenin kalıcı olarak
(ör. kullanıcı ortam değişkenlerinde) açık tutulması önerilir.

## Adım 4 — Ağ İzolasyonunu Uygulama

```powershell
# Yönetici PowerShell'de:
.\scripts\airgap_firewall_block.ps1
```

Bu, Python ve Foundry Local çalıştırılabilir dosyalarının yalnızca localhost'a
(127.0.0.1) bağlanabilmesini, başka hiçbir yere çıkamamasını sağlayan Windows
Defender Firewall kuralları ekler (bkz. script içindeki açıklama — kural
sırası değil kapsam/specificity önemlidir). Geri almak için:
`scripts/airgap_firewall_unblock.ps1`.

## Doğrulama Kontrol Listesi

- [ ] `pytest -q` tüm testleri internetsiz ortamda geçiyor (zaten Foundry Local
      gerektirmiyor, ama air-gapped makinede de çalıştığını doğrulayın)
      — beklenen çıktı: `NN passed` (hata/skip yok)
- [ ] `foundry server status` → `Ready`
      — beklenen çıktı: `State  Ready`
- [ ] Streamlit arayüzü açılıyor, bir soru sorulduğunda cevap alınıyor
      — beklenen: cevabın altında `[S1]` gibi bir atıf + kaynak legend'ı görünür
- [ ] Ağ kablosunu çıkarıp/Wi-Fi'ı kapatıp aynı soruyu tekrar sorduğunuzda
      davranış DEĞİŞMİYOR (zaten ağ çağrısı yapılmıyordu, bu sadece görsel kanıt)
- [ ] Firewall kuralları aktifken `Test-NetConnection example.com -Port 443`
      başarısız oluyor, ama uygulama çalışmaya devam ediyor
      — beklenen çıktı: `TcpTestSucceeded : False`

## Sık Karşılaşılan Hatalar

| Belirti | Olası neden | Çözüm |
|---|---|---|
| `foundry model list` boş/alias bulunamadı | Model hiç indirilmemiş veya alias sürüm arası değişmiş | `foundry model list --output json` ile gerçek alias'ları görün, `config.py`'deki `CHAT_MODEL_ALIAS`/`EMBEDDING_MODEL_ALIAS` ile karşılaştırın |
| `FoundryModelNotFound: ... strict-offline modu etkin` | `LOCAL_RAG_STRICT_OFFLINE=1` ayarlı ve alias yerelde yok | Hata mesajındaki `foundry model download <alias>` komutunu bağlı bir makinede çalıştırıp cache'i USB ile taşıyın |
| `sqlcipher3.dbapi2.OperationalError` / DB açılamıyor | Şifresiz (eski) bir `data/index.db` kalıntısı veya yanlış `LOCAL_RAG_DB_KEY` | `data/index.db`'yi silip yeniden `scripts/ingest.py` çalıştırın; anahtarın değişmediğinden emin olun |
| `Cannot send a request, as the client has been closed` | `HF_HUB_OFFLINE`/`TRANSFORMERS_OFFLINE` kapalıyken reranker cache'i yok | Önce `scripts/prepare_reranker_cache.py`'yi (internetliyken) çalıştırın |
| Firewall sonrası Streamlit hiç açılmıyor | Kural, localhost'u da bloke etmiş olabilir | `scripts/airgap_firewall_unblock.ps1` ile geri alıp script içindeki kapsam/specificity notunu gözden geçirin |
| Streamlit açılıyor ama soru sorulunca uzun süre yanıt gelmiyor | Foundry Local servisi `Ready` değil | `foundry server status` kontrol edin, gerekirse `foundry server start` |
