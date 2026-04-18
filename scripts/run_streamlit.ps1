param(
    [int]$Port = 8501
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$python = ".\.venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) {
    throw "Missing .venv Python: $python"
}

if (-not $env:PYTHONPATH) {
    $env:PYTHONPATH = "src"
}
else {
    $env:PYTHONPATH = "src;$env:PYTHONPATH"
}

& $python -m streamlit run "src/expert_digest/app/streamlit_app.py" --server.port $Port

