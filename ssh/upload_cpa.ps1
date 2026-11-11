[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$Server,
    [Parameter(Mandatory = $true)][string]$RemoteDir,
    [ValidateRange(1, 65535)][int]$Port = 22,
    [string]$IdentityFile
)

$ErrorActionPreference = 'Stop'
# Restrict remote arguments because ssh ultimately passes its command to a shell.
if ($Server -notmatch '^[A-Za-z0-9_][A-Za-z0-9_.@-]*$') {
    throw 'Server must be a host or user@host (use -Port for the port).'
}
if ($RemoteDir -notmatch '^/[A-Za-z0-9_./-]+$' -or $RemoteDir -match '(^|/)\.\.(/|$)') {
    throw 'RemoteDir must be an absolute Linux path without spaces or parent traversal.'
}
$RemoteDir = $RemoteDir.TrimEnd('/')
$ProjectRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$Files = @(
    'main.py',
    'byzantine/attacks.py',
    'byzantine/cpa_info/stl10-cov.pt',
    'tests/test_cpa.py',
    'ssh/run_cpa_spc_matrix.sh',
    'docs/cpa_experiment.md'
)
$SshArgs = @('-p', "$Port")
$ScpArgs = @('-P', "$Port")
if ($IdentityFile) {
    $SshArgs += @('-i', $IdentityFile)
    $ScpArgs += @('-i', $IdentityFile)
}
foreach ($File in $Files) {
    if (-not (Test-Path -LiteralPath (Join-Path $ProjectRoot $File) -PathType Leaf)) {
        throw "Missing local file: $File"
    }
}

& ssh @SshArgs $Server "mkdir -p '$RemoteDir/byzantine/cpa_info' '$RemoteDir/tests' '$RemoteDir/ssh' '$RemoteDir/docs'"
if ($LASTEXITCODE -ne 0) { throw 'Could not prepare remote directories.' }
foreach ($File in $Files) {
    $LocalFile = Join-Path $ProjectRoot $File
    & scp @ScpArgs $LocalFile "${Server}:${RemoteDir}/${File}"
    if ($LASTEXITCODE -ne 0) { throw "Upload failed: $File" }
    $ExpectedHash = (Get-FileHash -LiteralPath $LocalFile -Algorithm SHA256).Hash.ToLowerInvariant()
    $RemoteOutput = & ssh @SshArgs $Server "sha256sum '$RemoteDir/$File'"
    if ($LASTEXITCODE -ne 0 -or -not $RemoteOutput) { throw "Remote hash failed: $File" }
    $ActualHash = (($RemoteOutput -join "`n") -split '\s+')[0].ToLowerInvariant()
    if ($ActualHash -ne $ExpectedHash) { throw "Hash mismatch: $File" }
    Write-Host "Verified: $File"
}
Write-Host "Upload complete: ${Server}:${RemoteDir}. Training has not been started."
