# Müşteri Veri Sınıflandırma Standardı

## Amaç

Bu standart, NovaBank'ta işlenen tüm verilerin hassasiyetine göre nasıl
sınıflandırılacağını ve her sınıfın nasıl korunacağını tanımlar. Her çalışan,
ürettiği ve işlediği veriyi bu standarda göre etiketlemekle yükümlüdür.

## Sınıflandırma Seviyeleri

| Seviye | Tanım | Örnekler |
| --- | --- | --- |
| Herkese Açık | Kamuya açıklanmış bilgi | Faiz oranları, kampanya duyuruları |
| Dahili | Yalnızca çalışanlara açık | Organizasyon şeması, iç duyurular |
| Gizli | Hassas iş bilgisi | Kredi risk modelleri, strateji belgeleri |
| Kısıtlı | Yasal koruma altındaki müşteri verisi | TCKN, kart numarası, hesap bakiyesi |

## İşleme Kuralları

**Kısıtlı** ve **Gizli** veriler hem beklemede (at rest) hem de aktarımda
(in transit) şifrelenmek zorundadır. Kısıtlı veriye erişim yalnızca "bilmesi
gereken" ilkesine göre verilir ve her erişim denetim (audit) log'una
kaydedilir.

## Kart Verisi

Kart numarası ve hesap bakiyesi gibi Kısıtlı veriler ekranlarda maskelenerek
gösterilir (örn. son 4 hane). Kart doğrulama kodu (CVV) hiçbir koşulda
saklanmaz.

## Veri Sızıntısı

Kısıtlı veri içeren bir sızıntıdan şüphelenilmesi durumunda, olay 24 saat
içinde güvenlik ekibine bildirilmelidir; süreç, Olay Müdahale prosedürüyle
aynı akışı izler.
