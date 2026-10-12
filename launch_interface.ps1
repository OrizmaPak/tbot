$ErrorActionPreference = "Stop"

Set-Location -LiteralPath $PSScriptRoot

if (-not (Test-Path -LiteralPath ".\.venv\Scripts\streamlit.exe")) {
    Write-Host "Streamlit is not installed in .venv. Installing project dependencies..."
    .\.venv\Scripts\python.exe -m pip install -e .
}

.\.venv\Scripts\streamlit.exe run src\tradingbot\ui.py
