<#
.SYNOPSIS
    requirements.txt'teki tüm Python paketlerini air-gapped kurulum için
    yerel bir wheel klasörüne indirir.

.DESCRIPTION
    İNTERNETE BAĞLI bir makinede çalıştırılır. Çıktı klasörü (varsayılan:
    offline_wheels/) USB ile air-gapped makineye taşınır; orada aşağıdaki
    komutla internet gerekmeden kurulum yapılır:

        .venv/Scripts/python.exe -m pip install --no-index --find-links offline_wheels -r requirements.txt
#>

param(
    [string]$OutputDir = (Join-Path $PSScriptRoot "..\offline_wheels")
)

$RepoRoot = Split-Path $PSScriptRoot -Parent
$VenvPython = Join-Path $RepoRoot ".venv\Scripts\python.exe"
$RequirementsFile = Join-Path $RepoRoot "requirements.txt"

if (-not (Test-Path $VenvPython)) {
    Write-Error "Sanal ortam bulunamadı: $VenvPython (önce 'python -m venv .venv' çalıştırın)"
    exit 1
}

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

& $VenvPython -m pip download -r $RequirementsFile -d $OutputDir
if ($LASTEXITCODE -ne 0) {
    Write-Error "pip download başarısız oldu (exit code $LASTEXITCODE)"
    exit $LASTEXITCODE
}

Write-Host ""
Write-Host "[+] Wheel'ler indirildi: $OutputDir"
Write-Host "[+] Air-gapped makinede kurulum:"
Write-Host "    .venv/Scripts/python.exe -m pip install --no-index --find-links `"$OutputDir`" -r requirements.txt"
