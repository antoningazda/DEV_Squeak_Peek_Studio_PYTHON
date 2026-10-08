<#
.SYNOPSIS
    Authenticode-sign one or more Windows binaries, if a certificate is configured.

.DESCRIPTION
    Reads the certificate from environment variables so the release workflow
    can call this unconditionally:

      WINDOWS_CERTIFICATE_PFX       base64-encoded .pfx / .p12 code-signing cert
      WINDOWS_CERTIFICATE_PASSWORD  its password
      WINDOWS_TIMESTAMP_URL         optional; defaults to DigiCert's RFC3161 server

    With WINDOWS_CERTIFICATE_PFX unset or empty the script prints a notice and
    exits 0 — unsigned builds still produce a usable installer, they just trip
    SmartScreen (see packaging/SIGNING.md).

    Timestamping is not optional when a cert IS present: without it every
    signature stops validating the day the certificate expires, including on
    copies users already installed.

.EXAMPLE
    pwsh packaging/sign_windows.ps1 dist/SqueakPeekStudio/SqueakPeekStudio.exe
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true, ValueFromRemainingArguments = $true)]
    [string[]] $Path
)

$ErrorActionPreference = 'Stop'

$pfxBase64 = $env:WINDOWS_CERTIFICATE_PFX
if ([string]::IsNullOrWhiteSpace($pfxBase64)) {
    Write-Host '==> No WINDOWS_CERTIFICATE_PFX set; skipping Authenticode signing (unsigned release).'
    exit 0
}

if ([string]::IsNullOrWhiteSpace($env:WINDOWS_CERTIFICATE_PASSWORD)) {
    throw 'WINDOWS_CERTIFICATE_PASSWORD must be set alongside WINDOWS_CERTIFICATE_PFX.'
}

$timestampUrl = if ([string]::IsNullOrWhiteSpace($env:WINDOWS_TIMESTAMP_URL)) {
    'http://timestamp.digicert.com'
} else {
    $env:WINDOWS_TIMESTAMP_URL
}

# signtool ships with the Windows SDK; the GitHub runner has several SDK
# versions side by side, so take the newest x64 one rather than hardcoding
# a version that will rot.
$signtool = Get-ChildItem -Path 'C:\Program Files (x86)\Windows Kits\10\bin\*\x64\signtool.exe' `
    -ErrorAction SilentlyContinue |
    Sort-Object FullName |
    Select-Object -Last 1

if (-not $signtool) {
    throw 'signtool.exe not found. Install the Windows SDK signing tools.'
}
Write-Host "==> Using $($signtool.FullName)"

$pfxPath = Join-Path ([System.IO.Path]::GetTempPath()) "sps-codesign-$([guid]::NewGuid()).pfx"
try {
    [System.IO.File]::WriteAllBytes($pfxPath, [System.Convert]::FromBase64String($pfxBase64))

    foreach ($target in $Path) {
        if (-not (Test-Path -LiteralPath $target)) {
            throw "Nothing to sign at: $target"
        }
        Write-Host "==> Signing $target"
        & $signtool.FullName sign `
            /f $pfxPath `
            /p $env:WINDOWS_CERTIFICATE_PASSWORD `
            /fd SHA256 `
            /tr $timestampUrl `
            /td SHA256 `
            /d 'Squeak Peek Studio' `
            $target
        if ($LASTEXITCODE -ne 0) { throw "signtool failed on $target (exit $LASTEXITCODE)" }

        & $signtool.FullName verify /pa /v $target
        if ($LASTEXITCODE -ne 0) { throw "signature verification failed on $target" }
    }
}
finally {
    # The decoded private key must not outlive the signing step, even on failure.
    if (Test-Path -LiteralPath $pfxPath) {
        Remove-Item -LiteralPath $pfxPath -Force -ErrorAction SilentlyContinue
    }
}

Write-Host '==> Windows signing complete.'
