<#
.SYNOPSIS
    Run a v5 script in the muscle-model environment, from anywhere.

.DESCRIPTION
    Resolves .venv-myo's interpreter by absolute path and clears the three
    environment variables that break it. A stale PYTHONHOME in particular
    makes the venv redirector report

        No Python at '"C:\Users\...\python.exe'

    even when that file is right there -- the redirector finds the target but
    Python then refuses to start against a conflicting home.

.EXAMPLE
    .\run.ps1 scripts\run_reward.py --stage W
    .\run.ps1 scripts\validate_env.py
    .\run.ps1 -m pytest tests -q
#>
param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Args
)

$ErrorActionPreference = "Stop"

$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path (Split-Path -Parent $here) ".venv-myo\Scripts\python.exe"

if (-not (Test-Path $python)) {
    Write-Error @"
No interpreter at $python

Create it with:
    pip install uv
    uv python install 3.12
    uv venv --python 3.12 .venv-myo
    uv pip install --python .venv-myo/Scripts/python.exe -r v5/requirements.txt
"@
    exit 1
}

# The usual culprits when a venv interpreter refuses to start.
foreach ($name in "PYTHONHOME", "PYTHONPATH", "VIRTUAL_ENV") {
    Remove-Item "Env:$name" -ErrorAction SilentlyContinue
}

if (-not $Args) { $Args = @("-c", "import sys; print(sys.version)") }

Push-Location $here
try {
    & $python @Args
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
