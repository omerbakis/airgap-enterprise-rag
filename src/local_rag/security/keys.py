"""Veritabanı şifreleme anahtarı yönetimi.

Üretim ortamında anahtar, işletim sistemi kimlik bilgisi deposu veya bir
secret manager üzerinden `LOCAL_RAG_DB_KEY` ortam değişkenine enjekte
edilmelidir. Buradaki dosya-tabanlı fallback yalnızca geliştirme/demo
kolaylığı içindir; anahtar dosyası ve veritabanı aynı diskte durduğundan,
diskin tamamı çalınırsa şifreleme tek başına yeterli bir savunma değildir —
üretimde OS-seviyesi disk şifrelemesiyle (BitLocker/LUKS) birlikte kullanılmalıdır.

Üretimde `LOCAL_RAG_DB_KEY`'i nereden okuyacağınızı da düşünmelisiniz:
ortam değişkenleri (`.env` dosyaları dahil) genellikle process listesinden
veya yanlışlıkla commit edilen dosyalardan sızabilir. Windows'ta OS'in
kimlik bilgisi deposunu (Credential Manager) kullanan `keyring` paketi
(`pip install keyring`) air-gapped bir ortamda bile çalışır (yalnızca
yerel OS API'sini kullanır, ağ gerektirmez) ve anahtarı düz metin olarak
hiçbir yerde bırakmaz — bu projeye dahil edilmedi (opsiyonel bir bağımlılık
eklemek gerekirdi) ama üretime geçerken değerlendirilmesi önerilir:
`LOCAL_RAG_DB_KEY = keyring.get_password("local-rag", "db-key")`.
"""

from __future__ import annotations

import os
import secrets
import stat
from pathlib import Path

DB_KEY_ENV_VAR = "LOCAL_RAG_DB_KEY"


def get_or_create_db_key(key_path: Path) -> str:
    """Önce `LOCAL_RAG_DB_KEY` ortam değişkenine bakar; yoksa `key_path`'te
    daha önce üretilmiş bir anahtar arar; o da yoksa kriptografik olarak
    güvenli yeni bir anahtar üretip dosyaya yazar (tek seferlik, sonraki
    çalıştırmalarda aynı anahtar tekrar kullanılır)."""
    env_key = os.environ.get(DB_KEY_ENV_VAR)
    if env_key:
        return env_key

    if key_path.exists():
        return key_path.read_text(encoding="utf-8").strip()

    key = secrets.token_hex(32)
    key_path.parent.mkdir(parents=True, exist_ok=True)
    key_path.write_text(key, encoding="utf-8")
    try:
        # Windows'ta POSIX chmod bit'leri ACL'yi değiştirmez (yalnızca
        # salt-okunur bayrağını etkiler); gerçek erişim kısıtlaması için
        # `icacls` gerekir. Bu, en azından yanlışlıkla üzerine yazmaya karşı
        # ucuz bir ek önlemdir — asıl güvenlik sınırı LOCAL_RAG_DB_KEY'dir.
        os.chmod(key_path, stat.S_IRUSR | stat.S_IWUSR)
    except (NotImplementedError, OSError):
        pass
    return key
