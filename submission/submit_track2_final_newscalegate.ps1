<#
.SYNOPSIS
    Skrip PowerShell Pendaftaran & Pemaketan Submission Track 2 Agenthon 2026 (CodaBench) - FINAL newscalegate
.DESCRIPTION
    Skrip ini memaketkan descriptor submission_t2_final_newscalegate.json menjadi arsip terotentikasi
    resmi track2-final-newscalegate-20261022.zip menggunakan alat resmi qfbench2 di WSL.
    Diadaptasi dari submit_track2_final_newscale.ps1 (13 Okt); hanya folder, nama descriptor dan nama zip berubah.
    Team Key tidak pernah disentuh oleh skrip ini selain diteruskan ke qfbench2 (~/.team_key di WSL, bila ada).
#>

[CmdletBinding()]
param(
    [string]$TargetFolder = "C:\Users\Saifuddin\Documents\Agenthon-2026\submissions\2026-10-22",
    [string]$Descriptor = "submission_t2_final_newscalegate.json",
    [string]$ZipName = "track2-final-newscalegate-20261022.zip",
    [int]$TeamNumber = 210
)

$ErrorActionPreference = "Stop"

Write-Host "=================================================================" -ForegroundColor Cyan
Write-Host "   SKRIP PENDAFTARAN & PEMAKETAN SUBMISSION TRACK 2 (FINAL NEWSCALEGATE) " -ForegroundColor Cyan
Write-Host "=================================================================" -ForegroundColor Cyan

if (-not (Test-Path $TargetFolder)) {
    New-Item -ItemType Directory -Path $TargetFolder -Force | Out-Null
}

$drive = $TargetFolder.Substring(0, 1).ToLower()
$wslDir = "/mnt/$drive" + $TargetFolder.Substring(2).Replace("\", "/")
$descPath = Join-Path $TargetFolder $Descriptor
$outZipPath = Join-Path $TargetFolder $ZipName

if (-not (Test-Path $descPath)) {
    Write-Host "[ERROR] Descriptor tidak ditemukan: $descPath" -ForegroundColor Red
    exit 1
}

$descContent = Get-Content $descPath -Raw | ConvertFrom-Json
Write-Host "`n[1/3] Memeriksa Metadata Descriptor:" -ForegroundColor Yellow
Write-Host "  - Track          : $($descContent.track)"
Write-Host "  - Phase          : $($descContent.phase)"
Write-Host "  - Competition ID : $($descContent.competition_id)"
Write-Host "  - Image Registry : $($descContent.image.registry)"
Write-Host "  - Repository     : $($descContent.image.repository)"
Write-Host "  - Digest SHA256  : $($descContent.image.digest)"

Write-Host "`n[2/3] Memaketkan Signature Kriptografi Tim $TeamNumber via qfbench2..." -ForegroundColor Yellow

$wslCmd = "cd '$wslDir' && if [ -f ~/.team_key ]; then ~/venvs/agenthon/bin/qfbench2 submission pack --team-number $TeamNumber --team-key-file ~/.team_key --descriptor $Descriptor --out $ZipName --force; else ~/venvs/agenthon/bin/qfbench2 submission pack --team-number $TeamNumber --descriptor $Descriptor --out $ZipName --force; fi"

wsl -e bash -lc $wslCmd

if (Test-Path $outZipPath) {
    Write-Host "`n[3/3] Verifikasi Hasil Berkas ZIP:" -ForegroundColor Green
    $zipItem = Get-Item $outZipPath
    $hash = Get-FileHash -Path $outZipPath -Algorithm SHA256

    $downloadsPath = "C:\Users\Saifuddin\Downloads\$ZipName"
    Copy-Item $outZipPath -Destination $downloadsPath -Force

    Write-Host "  - Berkas Utama : $outZipPath ($($zipItem.Length) bytes)" -ForegroundColor Green
    Write-Host "  - Salinan Unduh: $downloadsPath" -ForegroundColor Green
    Write-Host "  - Checksum     : $($hash.Hash)" -ForegroundColor Green

    Write-Host "`n=================================================================" -ForegroundColor Cyan
    Write-Host "   STATUS: PAKET BERHASIL DIBUAT DAN SIAP DI-SUBMIT!            " -ForegroundColor Cyan
    Write-Host "=================================================================" -ForegroundColor Cyan
    Write-Host "`nInstruksi Submit ke CodaBench (upload KEDUA Final Track 2; menggantikan newscale):" -ForegroundColor Yellow
    Write-Host "1. Buka browser: https://www.codabench.org/competitions/ (cari kompetisi Agenthon 2026 Track 2 FINAL; Dev-nya 17766)"
    Write-Host "2. Masuk ke tab 'My Submissions'"
    Write-Host "3. Klik tombol upload dan pilih file:"
    Write-Host "   $downloadsPath" -ForegroundColor White
    Write-Host "4. Batas: 26 Okt 2026 11:59 UTC (18:59 WIB)."
} else {
    Write-Host "`n[ERROR] Gagal membuat file zip!" -ForegroundColor Red
    exit 1
}
