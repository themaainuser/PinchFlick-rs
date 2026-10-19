<#
.SYNOPSIS
    Fetches the three ONNX hand models into models\ and verifies SHA256.

.DESCRIPTION
    Source: takoyakisoft/vtuberkit-vision-landmarkers-onnx - ONNX conversions
    of the official Google MediaPipe hand models, Apache-2.0 (the same files
    the Rust port was built and gated against).

      GitHub original : https://github.com/takoyakisoft/vtuberkit-vision-landmarkers-onnx  (gone - 404)
      Hugging Face    : https://huggingface.co/takoyakisoft/vtuberkit-vision-landmarkers-onnx  (public, no login)

    The SHA256 values below are the Hugging Face LFS object hashes for those
    exact files; the copies this project was developed against match them
    byte-for-byte. A download that does not hash to the expected value is
    deleted, never used.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\download-models.ps1

.EXAMPLE
    # offline check of what is already in models\ (exit 1 = anything wrong)
    powershell -ExecutionPolicy Bypass -File scripts\download-models.ps1 -VerifyOnly
#>
param([switch]$VerifyOnly)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'

$repoRoot = Split-Path -Parent $PSScriptRoot
$modelsDir = Join-Path $repoRoot 'models'
$baseUrl = 'https://huggingface.co/takoyakisoft/vtuberkit-vision-landmarkers-onnx/resolve/main/onnx/hand'

$files = @(
    @{ Name = 'palm_detector.onnx'; Sha256 = '939e53b8c438bdbdcaaf2e5589c8204cb6c92d383a6fe87c939ca3bb52f605c7' }
    @{ Name = 'hand_landmark.onnx'; Sha256 = '716f2d75eb41cbb8f222cda9099eb617ec0a3248ecb96e77c0dc916759b5bb1c' }
    @{ Name = 'hand_recrop.onnx';   Sha256 = 'd71150bd45f3a0060805bda7147302d3d9215a26e7bc270ef840f9718f55d797' }
)

New-Item -ItemType Directory -Force -Path $modelsDir | Out-Null
$failed = $false

foreach ($f in $files) {
    $target = Join-Path $modelsDir $f.Name

    if (Test-Path -LiteralPath $target) {
        $hash = (Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash.ToLower()
        if ($hash -eq $f.Sha256) {
            Write-Host "OK       $($f.Name)  (present, SHA256 verified)"
            continue
        }
        if ($VerifyOnly) {
            Write-Host "BAD      $($f.Name)  (SHA256 mismatch: got $hash)"
            $failed = $true
            continue
        }
        Write-Host "REPLACE  $($f.Name)  (SHA256 mismatch - re-downloading)"
        Remove-Item -LiteralPath $target -Force
    }
    elseif ($VerifyOnly) {
        Write-Host "MISSING  $($f.Name)"
        $failed = $true
        continue
    }

    Write-Host "GET      $($f.Name)"
    $tmp = "$target.download"
    try {
        Invoke-WebRequest -Uri "$baseUrl/$($f.Name)" -OutFile $tmp -UseBasicParsing
        $hash = (Get-FileHash -LiteralPath $tmp -Algorithm SHA256).Hash.ToLower()
        if ($hash -ne $f.Sha256) {
            Write-Host "FAILED   $($f.Name)  SHA256 mismatch after download:"
            Write-Host "         got  $hash"
            Write-Host "         want $($f.Sha256)"
            exit 1
        }
        Move-Item -LiteralPath $tmp -Destination $target -Force
        Write-Host "OK       $($f.Name)  (downloaded, SHA256 verified)"
    }
    finally {
        # A network drop mid-download must not leave a partial file behind.
        # (After success Move-Item already consumed $tmp -> this is a no-op.)
        if (Test-Path -LiteralPath $tmp) { Remove-Item -LiteralPath $tmp -Force }
    }
}

if ($failed) {
    Write-Host 'model check FAILED'
    exit 1
}
Write-Host 'all models verified'
