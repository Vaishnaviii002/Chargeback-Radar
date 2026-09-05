[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
$repositoryRoot = (
    Resolve-Path (
        Join-Path $PSScriptRoot ".."
    )
).Path

$virtualEnvironmentPython = Join-Path (
    $repositoryRoot
) ".venv\Scripts\python.exe"

if (Test-Path -LiteralPath $virtualEnvironmentPython) {
    $python = $virtualEnvironmentPython
}
else {
    $python = (Get-Command python -ErrorAction Stop).Source
}

# Force every optional external integration into its deterministic, offline
# mode. Values are never printed by this script.
$env:EVIDENCE_AI_ENABLED = "false"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"
$env:MODEL_EXPLANATION_AI_ENABLED = "false"
$env:SUPPORT_SIGNAL_AI_ENABLED = "false"
$env:RAZORPAY_INTEGRATION_ENABLED = "false"
$env:RAZORPAY_TEST_MODE = "true"
$env:RAZORPAY_WEBHOOK_ENABLED = "false"

Push-Location $repositoryRoot

try {
    & $python "scripts/verify_release.py"

    if ($LASTEXITCODE -ne 0) {
        throw "Release verification failed."
    }
}
finally {
    Pop-Location
}
