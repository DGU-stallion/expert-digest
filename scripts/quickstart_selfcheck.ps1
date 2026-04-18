param(
    [string]$DbPath = "data/processed/quickstart.sqlite3",
    [string]$SampleJsonl = "data/sample/articles.jsonl",
    [string]$EmbeddingModel = "hash-bow-v1",
    [switch]$InstallDeps,
    [switch]$SkipQualityChecks
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$python = ".\.venv\Scripts\python.exe"
$cli = ".\.venv\Scripts\expert-digest.exe"

if (-not (Test-Path -LiteralPath $python)) {
    throw "Missing .venv Python: $python"
}
if (-not (Test-Path -LiteralPath $cli)) {
    throw "Missing expert-digest entrypoint: $cli"
}

function Run-Step {
    param(
        [string]$Name,
        [scriptblock]$Action
    )
    Write-Host "==> $Name"
    & $Action
}

if ($InstallDeps) {
    Run-Step -Name "Install dependencies (dev+app+mcp)" -Action {
        & $python -m pip install -e ".[dev,app,mcp]"
    }
}

Run-Step -Name "Import sample JSONL" -Action {
    & $cli import-jsonl $SampleJsonl --db $DbPath
}

Run-Step -Name "Rebuild chunks" -Action {
    & $cli rebuild-chunks --db $DbPath --max-chars 1200 --min-chars 80
}

Run-Step -Name "Rebuild embeddings" -Action {
    & $cli rebuild-embeddings --db $DbPath --model $EmbeddingModel
}

Run-Step -Name "Ask smoke query (JSON)" -Action {
    $json = & $cli ask "长期主义在项目管理里的核心是什么？" --db $DbPath --model $EmbeddingModel --format json
    New-Item -ItemType Directory -Force -Path "data/outputs" | Out-Null
    $json | Out-File -LiteralPath "data/outputs/quickstart_ask.json" -Encoding utf8
}

Run-Step -Name "Generate deterministic handbook" -Action {
    & $cli generate-handbook --db $DbPath --model $EmbeddingModel --synthesis-mode deterministic --output "data/outputs/quickstart_handbook.md"
}

Run-Step -Name "Build author profile (JSON)" -Action {
    & $cli build-author-profile --db $DbPath --format json --output "data/outputs/quickstart_profile.json"
}

Run-Step -Name "Generate skill draft" -Action {
    & $cli generate-skill-draft --db $DbPath --output "data/outputs/quickstart_skill.md"
}

if (-not $SkipQualityChecks) {
    Run-Step -Name "Ruff check" -Action {
        & $python -m ruff check src tests
    }
    Run-Step -Name "Pytest" -Action {
        & $python -m pytest -q
    }
}

Write-Host "`nSelf-check completed."
Write-Host "Artifacts:"
Write-Host "- data/outputs/quickstart_ask.json"
Write-Host "- data/outputs/quickstart_handbook.md"
Write-Host "- data/outputs/quickstart_profile.json"
Write-Host "- data/outputs/quickstart_skill.md"

