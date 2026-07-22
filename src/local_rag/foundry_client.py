"""Foundry Local bağlantısı için ortak bootstrap mantığı.

embeddings/foundry.py ve llm/foundry.py, model indirme/yükleme ve OpenAI-uyumlu
istemci oluşturma mantığını tekrarlamamak için bu modülü kullanır.

Uyumluluk notu: Foundry Local CLI 0.10+ hem `foundry service ...` komutlarını
`foundry server ...` olarak yeniden adlandırdı hem de model indirme/yükleme/katalog
REST uç noktalarını (/foundry/list, /openai/download, /openai/load/...) değiştirdi.
PyPI'deki foundry-local-sdk (bu proje geliştirilirken en güncel sürüm: 0.5.1) hâlâ eski
arayüzü kullanıyor; bu yüzden FoundryLocalManager'ın servis keşfi ve
katalog/indirme/yükleme metotları bu CLI sürümüyle 404 ile başarısız oluyor
(doğrulama: `foundry --help` çıktısı `service` yerine `server` komut grubunu, ve
çalışan sunucuda `/foundry/list` yerine yalnızca `/v1/models`in yanıt verdiğini
gösteriyor). Bu modül: (1) servis keşfini SDK'nın beklediği arayüze
monkeypatch'ler, (2) model indirme/yükleme/id çözümlemesini doğrudan `foundry`
CLI'sini (`--output json`) çağırarak yapar. SDK güncellendiğinde bu dosya
sadeleştirilebilir.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass

import foundry_local.api as _foundry_api
import foundry_local.service as _foundry_service
from foundry_local import FoundryLocalManager
from openai import OpenAI

_SERVICE_URI_RE = re.compile(r"http://(?:[a-zA-Z0-9.-]+|\d{1,3}(\.\d{1,3}){3}):\d+")

STRICT_OFFLINE_ENV_VAR = "LOCAL_RAG_STRICT_OFFLINE"
"""Bu değişken "1"/"true"/"yes" ise connect(), yerelde henüz indirilmemiş bir
alias için ASLA 'foundry model download' çağırmaz (ağa çıkan tek adım budur;
'model load' zaten inen ağırlıkları yerelde servis eder) — bunun yerine hemen
FoundryModelNotFound fırlatır. Air-gapped bir ortamda "model unutuldu, ilk
kullanımda sessizce internete çıkmaya çalışsın" senaryosunu engellemek için;
varsayılan (değişken tanımsız) davranış DEĞİŞMEDİ (eskiden olduğu gibi eksik
model otomatik indirilmeye çalışılır) — bu yalnızca air-gapped dağıtımlarda
bilinçli olarak açılması gereken bir güvenlik anahtarıdır (bkz.
docs/AIRGAP_KURULUM.md)."""


class FoundryLocalNotAvailable(RuntimeError):
    """Foundry Local çalışma zamanı kurulu/erişilebilir değil."""


class FoundryModelNotFound(RuntimeError):
    """İstenen alias, Foundry Local kataloğunda bulunamadı."""


@dataclass
class FoundryConnection:
    manager: FoundryLocalManager
    client: OpenAI
    model_id: str
    alias: str


def _get_service_uri_compat() -> str | None:
    with subprocess.Popen(["foundry", "server", "status"], stdout=subprocess.PIPE, stderr=subprocess.PIPE) as proc:
        stdout, _ = proc.communicate()
        match = _SERVICE_URI_RE.search(stdout.decode())
        return match.group(0) if match else None


def _start_service_compat() -> str | None:
    if (uri := _get_service_uri_compat()) is not None:
        return uri
    with subprocess.Popen(["foundry", "server", "start"]):
        for _ in range(50):
            if (uri := _get_service_uri_compat()) is not None:
                return uri
            time.sleep(0.2)
    return None


# api.py, service.py'den bu isimleri `from ... import` ile aldığı için her iki
# modülün de global ad alanı ayrı ayrı yamalanmalı.
_foundry_service.get_service_uri = _get_service_uri_compat
_foundry_service.start_service = _start_service_compat
_foundry_api.get_service_uri = _get_service_uri_compat
_foundry_api.start_service = _start_service_compat


def _run_cli(*args: str) -> str:
    # encoding UTF-8 olarak sabitlenir: `foundry` CLI çıktısı (ilerleme çubukları,
    # kutu-çizim karakterleri) Windows'un yerel kod sayfasıyla (örn. Türkçe
    # Windows'ta cp1254) her zaman çözülemez; text=True + varsayılan encoding
    # bunu UnicodeDecodeError ile sessizce bir arka plan thread'inde patlatır.
    result = subprocess.run(
        ["foundry", *args], capture_output=True, text=True, encoding="utf-8", errors="replace"
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"'foundry {' '.join(args)}' başarısız oldu: {(result.stderr or result.stdout).strip()}"
        )
    return result.stdout


def _list_catalog() -> list[dict]:
    return json.loads(_run_cli("model", "list", "--output", "json"))["models"]


def cached_aliases() -> set[str]:
    """Foundry kataloğunda yerelde İNDİRİLMİŞ (cached=true) alias'lar — UI'da
    hangi modellerin offline hazır olduğunu göstermek için. Foundry erişilemezse
    boş küme döner (UI kırılmaz)."""
    try:
        return {m["alias"] for m in _list_catalog() if m.get("cached")}
    except Exception:  # noqa: BLE001
        return set()


def strict_offline_enabled() -> bool:
    return os.environ.get(STRICT_OFFLINE_ENV_VAR, "").strip().lower() in {"1", "true", "yes"}


def connect(alias: str) -> FoundryConnection:
    """Foundry Local servisini başlatır, verilen alias'ı indirip yükler.

    Foundry Local uygulaması makinede kurulu değilse veya alias katalogda
    yoksa, kullanıcıya ne yapması gerektiğini söyleyen açık bir hata fırlatır.
    """
    if shutil.which("foundry") is None:
        raise FoundryLocalNotAvailable(
            "Foundry Local çalışma zamanına ulaşılamadı ('foundry' PATH üzerinde "
            "bulunamadı). Kurulum için: winget install --id Microsoft.FoundryLocal -e "
            "(bkz. https://learn.microsoft.com/azure/foundry-local/get-started)."
        )

    manager = FoundryLocalManager(bootstrap=False)
    manager.start_service()

    catalog = _list_catalog()
    matches = [m for m in catalog if m["alias"].lower() == alias.lower()]
    if not matches:
        available = sorted({m["alias"] for m in catalog})
        raise FoundryModelNotFound(
            f"'{alias}' alias'ı Foundry Local kataloğunda bulunamadı. "
            f"Kullanılabilir alias'lardan biri config.py içinde ayarlanmalı: {available}"
        )

    if strict_offline_enabled() and not matches[0].get("cached"):
        raise FoundryModelNotFound(
            f"'{alias}' yerelde önceden indirilmemiş ve {STRICT_OFFLINE_ENV_VAR}=1 ile "
            "strict-offline modu etkin — air-gapped ortamda örtük bir ağ çağrısı "
            "(model indirme) engellendi. Modeli önceden 'foundry model download "
            f"{alias}' ile ya da USB/disconnected-cache transferiyle indirin "
            "(bkz. docs/AIRGAP_KURULUM.md), sonra tekrar deneyin."
        )

    if not strict_offline_enabled():
        _run_cli("model", "download", alias)
    _run_cli("model", "load", alias)

    # Yüklendikten sonra servis tarafında (/v1/models) sunulan gerçek model id'sini
    # (":<versiyon>" eki olmadan) katalogdan tekrar okuyoruz.
    served = next(m for m in _list_catalog() if m["alias"].lower() == alias.lower())
    model_id = served["id"].split(":")[0]

    client = OpenAI(base_url=manager.endpoint, api_key=manager.api_key)
    return FoundryConnection(manager=manager, client=client, model_id=model_id, alias=alias)
