"""connect()'in strict-offline davranışı: LOCAL_RAG_STRICT_OFFLINE=1 iken
yerelde indirilmemiş bir alias için 'foundry model download' (tek ağa çıkan
adım) ASLA çağrılmamalı, bunun yerine hemen FoundryModelNotFound fırlatılmalı.
Gerçek Foundry Local süreçlerine dokunmadan test edebilmek için manager/CLI/
katalog çağrıları monkeypatch'lenir."""

import pytest

import local_rag.foundry_client as fc


class _FakeManager:
    def __init__(self, *args, **kwargs):
        self.endpoint = "http://127.0.0.1:0"
        self.api_key = "fake"

    def start_service(self):
        pass


def _patch_common(monkeypatch, catalog, run_cli_calls):
    monkeypatch.setattr(fc.shutil, "which", lambda name: "C:/fake/foundry.exe")
    monkeypatch.setattr(fc, "FoundryLocalManager", _FakeManager)
    monkeypatch.setattr(fc, "_list_catalog", lambda: catalog)

    def fake_run_cli(*args):
        run_cli_calls.append(args)
        return ""

    monkeypatch.setattr(fc, "_run_cli", fake_run_cli)


def test_strict_offline_blocks_download_for_uncached_alias(monkeypatch):
    monkeypatch.setenv(fc.STRICT_OFFLINE_ENV_VAR, "1")
    catalog = [{"alias": "qwen2.5-7b", "id": "qwen2.5-7b:v1", "cached": False}]
    run_cli_calls: list[tuple] = []
    _patch_common(monkeypatch, catalog, run_cli_calls)

    with pytest.raises(fc.FoundryModelNotFound):
        fc.connect("qwen2.5-7b")

    assert not any(call[:2] == ("model", "download") for call in run_cli_calls)


def test_strict_offline_allows_load_for_already_cached_alias(monkeypatch):
    monkeypatch.setenv(fc.STRICT_OFFLINE_ENV_VAR, "1")
    catalog = [{"alias": "qwen2.5-7b", "id": "qwen2.5-7b:v1", "cached": True}]
    run_cli_calls: list[tuple] = []
    _patch_common(monkeypatch, catalog, run_cli_calls)

    conn = fc.connect("qwen2.5-7b")

    assert not any(call[:2] == ("model", "download") for call in run_cli_calls)
    assert any(call[:2] == ("model", "load") for call in run_cli_calls)
    assert conn.model_id == "qwen2.5-7b"


def test_non_strict_mode_downloads_as_before(monkeypatch):
    monkeypatch.delenv(fc.STRICT_OFFLINE_ENV_VAR, raising=False)
    catalog = [{"alias": "qwen2.5-7b", "id": "qwen2.5-7b:v1", "cached": False}]
    run_cli_calls: list[tuple] = []
    _patch_common(monkeypatch, catalog, run_cli_calls)

    conn = fc.connect("qwen2.5-7b")

    assert any(call[:2] == ("model", "download") for call in run_cli_calls)
    assert any(call[:2] == ("model", "load") for call in run_cli_calls)
    assert conn.model_id == "qwen2.5-7b"


def test_strict_offline_enabled_reads_env_var_truthy_values(monkeypatch):
    for value in ("1", "true", "True", "yes", "YES"):
        monkeypatch.setenv(fc.STRICT_OFFLINE_ENV_VAR, value)
        assert fc.strict_offline_enabled() is True

    for value in ("0", "false", "", "no"):
        monkeypatch.setenv(fc.STRICT_OFFLINE_ENV_VAR, value)
        assert fc.strict_offline_enabled() is False
