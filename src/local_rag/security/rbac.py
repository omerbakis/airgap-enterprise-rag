"""Basit rol tabanlı erişim kontrolü (RBAC).

Kurumsal SSO/AD entegrasyonu bu projenin kapsamı dışındadır — rol, gerçek bir
kimlik doğrulamadan değil, çağıran
tarafın (Streamlit sidebar'ındaki persona seçici, CLI argümanı) seçtiği bir
isimden gelir. Asıl güvenlik garantisi rolün NASIL seçildiğinde değil, seçilen
rolün retrieval sorgusuna nasıl uygulandığındadır: bkz. pipeline.answer() —
filtre SQL WHERE'e girer, yetkisiz chunk hiçbir zaman LLM bağlamına ulaşmaz.

Erişim modeli (iki eksen):
  1. classification — "genel" (herkese açık) veya "gizli" (sınıflandırılmış).
     Rol yalnızca `allowed_classifications` içindeki seviyeleri görebilir.
  2. department — kompartmanlaştırma. Departman kısıtı YALNIZCA "gizli"
     dokümanlara uygulanır; "genel" dokümanlar departmandan bağımsız olarak
     herkese açıktır (bkz. config.PUBLIC_CLASSIFICATION, db._filter_clause).
     `allowed_departments=None` = tüm departmanların gizli dokümanları.

Bu ayrım kritiktir: bir kurumun izin politikası "genel" ama department="IK"
olabilir; departman kısıtı genel dokümanlara da uygulansaydı, bir BT çalışanı
şirketin herkese açık izin politikasını göremez ve düşük-ayrıcalıklı "calisan"
rolü, "bt_uzmani"den daha çok doküman görürdü (ayrıcalık ters dönmesi).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Role:
    name: str
    allowed_classifications: tuple[str, ...]
    allowed_departments: tuple[str, ...] | None
    """Rolün gizli dokümanlarını görebileceği departmanlar. None = tüm
    departmanların gizli dokümanları. Genel (herkese açık) dokümanlar bu
    kısıttan muaftır — departmandan bağımsız görünür (bkz. modül docstring'i,
    db._filter_clause)."""


ROLES: dict[str, Role] = {
    "calisan": Role("calisan", allowed_classifications=("genel",), allowed_departments=None),
    "ik_uzmani": Role(
        "ik_uzmani", allowed_classifications=("genel", "gizli"), allowed_departments=("IK", "Genel")
    ),
    "bt_uzmani": Role(
        "bt_uzmani", allowed_classifications=("genel", "gizli"), allowed_departments=("BT", "Genel")
    ),
    "yonetici": Role("yonetici", allowed_classifications=("genel", "gizli"), allowed_departments=None),
}

DEFAULT_ROLE = "calisan"


def get_role(name: str) -> Role:
    if name not in ROLES:
        raise ValueError(f"Bilinmeyen rol: '{name}'. Geçerli roller: {sorted(ROLES)}")
    return ROLES[name]
