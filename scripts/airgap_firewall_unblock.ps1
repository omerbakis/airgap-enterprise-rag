<#
.SYNOPSIS
    airgap_firewall_block.ps1 tarafından eklenen kuralları kaldırır.

.NOTES
    Yönetici (Administrator) PowerShell oturumu gerektirir.
#>

#Requires -RunAsAdministrator

$removed = 0
foreach ($pattern in @("Local-RAG Python - *", "Foundry Local - *")) {
    $rules = Get-NetFirewallRule -DisplayName $pattern -ErrorAction SilentlyContinue
    foreach ($rule in $rules) {
        Remove-NetFirewallRule -Name $rule.Name
        $removed++
    }
}

Write-Host "[+] $removed adet air-gap firewall kurali kaldirildi."
