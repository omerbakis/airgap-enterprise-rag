<#
.SYNOPSIS
    Air-gapped dağıtım için outbound ağ erişimini kısıtlar (OS-seviyesi aktif
    engelleme).

.DESCRIPTION
    Bu makinedeki Python sanal ortamının ve Foundry Local çalıştırılabilir
    dosyalarına, yalnızca localhost'a (127.0.0.1) giden bağlantılara izin
    verip geri kalan HER ŞEYİ engelleyen Windows Defender Firewall kuralları
    ekler. Bu, uygulama kodu güvenilir olsa da bağımlılıklardaki (3. parti
    pip paketleri) beklenmeyen bir dış çağrıya karşı ikinci bir savunma
    katmanıdır (defense-in-depth).

    Windows Firewall, aynı yöndeki (outbound) kurallar arasında EN SPESİFİK
    kapsam eşleşenini tercih eder — bu yüzden "yalnızca 127.0.0.1'e izin ver"
    kuralı, "her şeyi engelle" kuralından daha spesifik olduğu için önceliklidir;
    kural EKLEME SIRASI önemli değildir, KAPSAM (RemoteAddress) önemlidir.

.NOTES
    ÖNEMLİ: Bu script YALNIZCA gerçek air-gapped dağıtım makinesinde
    çalıştırılmalıdır. Geliştirme makinesinde çalıştırmak; pip güncellemelerini,
    Foundry Local model indirmelerini ve HuggingFace model indirmelerini
    engeller. Yönetici (Administrator) PowerShell oturumu gerektirir.

    Geri almak için: scripts/airgap_firewall_unblock.ps1
#>

#Requires -RunAsAdministrator

param(
    [string]$PythonExePath = (Join-Path (Split-Path $PSScriptRoot -Parent) ".venv\Scripts\python.exe")
)

$foundryCmd = Get-Command foundry -ErrorAction SilentlyContinue

if (-not (Test-Path $PythonExePath)) {
    Write-Error "Python çalıştırılabilir dosyası bulunamadı: $PythonExePath"
    exit 1
}

$targets = @{ "Local-RAG Python" = $PythonExePath }
if ($foundryCmd) {
    $targets["Foundry Local"] = $foundryCmd.Source
}

foreach ($name in $targets.Keys) {
    $exePath = $targets[$name]

    New-NetFirewallRule -DisplayName "$name - Allow Loopback" `
        -Direction Outbound -Program $exePath -Action Allow `
        -RemoteAddress "127.0.0.1" -Profile Any -ErrorAction Stop | Out-Null

    New-NetFirewallRule -DisplayName "$name - Block Outbound" `
        -Direction Outbound -Program $exePath -Action Block `
        -Profile Any -ErrorAction Stop | Out-Null

    Write-Host "[+] $name icin outbound kisitlama eklendi: $exePath"
}

Write-Host ""
Write-Host "Dogrulama onerileri:"
Write-Host "  - 'foundry server status' calismaya devam etmeli (loopback serbest)"
Write-Host "  - Streamlit arayuzu (http://127.0.0.1:8501) normal calismali"
Write-Host "  - Test-NetConnection <herhangi-bir-internet-adresi> -Port 443 basarisiz olmali"
Write-Host ""
Write-Host "Geri almak icin: scripts/airgap_firewall_unblock.ps1"
