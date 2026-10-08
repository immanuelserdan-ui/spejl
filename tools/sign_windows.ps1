<#
.SYNOPSIS
    Authenticode-sign Windows files with a code-signing certificate (.pfx).

.DESCRIPTION
    Used by .github/workflows/release.yml when the SIGNING_CERT_PFX_BASE64 and
    SIGNING_CERT_PASSWORD secrets are set, and usable by hand on a PC:

        $env:SIGNING_CERT_PFX_BASE64 = [Convert]::ToBase64String([IO.File]::ReadAllBytes("cert.pfx"))
        $env:SIGNING_CERT_PASSWORD   = "..."
        ./tools/sign_windows.ps1 dist-installer\Spejl\Spejl.exe

    Works with a certificate from a public CA or from a company's own
    certificate authority (which only that company's PCs trust). Every file is
    timestamped, so signatures stay valid after the certificate expires.
#>
param(
    [Parameter(Mandatory = $true, ValueFromRemainingArguments = $true)]
    [string[]] $Files
)
$ErrorActionPreference = "Stop"

if (-not $env:SIGNING_CERT_PFX_BASE64) { throw "SIGNING_CERT_PFX_BASE64 is not set." }
$timestamp = if ($env:SIGNING_TIMESTAMP_URL) { $env:SIGNING_TIMESTAMP_URL } else { "http://timestamp.digicert.com" }

$signtool = Get-ChildItem "${env:ProgramFiles(x86)}\Windows Kits\10\bin\*\x64\signtool.exe" -ErrorAction SilentlyContinue |
    Sort-Object FullName | Select-Object -Last 1
if (-not $signtool) { throw "signtool.exe not found (install the Windows SDK)." }

$tempRoot = if ($env:RUNNER_TEMP) { $env:RUNNER_TEMP } else { [IO.Path]::GetTempPath() }
$pfx = Join-Path $tempRoot ("spejl-signing-" + [guid]::NewGuid() + ".pfx")
try {
    [IO.File]::WriteAllBytes($pfx, [Convert]::FromBase64String($env:SIGNING_CERT_PFX_BASE64))
    foreach ($file in $Files) {
        $path = (Resolve-Path $file).Path
        & $signtool.FullName sign /f $pfx /p $env:SIGNING_CERT_PASSWORD /fd sha256 /tr $timestamp /td sha256 `
            /d "Spejl" $path
        if ($LASTEXITCODE -ne 0) { throw "signtool failed for $path (exit $LASTEXITCODE)." }
        # A company-CA certificate is not trusted on the build machine, so check
        # that the signature is present rather than that Windows trusts it here.
        $signature = Get-AuthenticodeSignature $path
        if (-not $signature.SignerCertificate) { throw "No signature found on $path after signing." }
        Write-Host "Signed $path by $($signature.SignerCertificate.Subject) ($($signature.Status))"
    }
}
finally {
    if (Test-Path $pfx) { Remove-Item $pfx -Force }
}
