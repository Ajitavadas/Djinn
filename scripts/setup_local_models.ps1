<#
.SYNOPSIS
    Build Djinn's local models in Ollama from Qwen3.5-4B Q3_K_M.

.DESCRIPTION
    Creates two Ollama models from one set of weights:

      qwen3.5-4b-q3km-text   text only - Djinn's default local model
      qwen3.5-4b-q3km        same weights + vision projector, used for images

    They are separate because the vision projector takes ~0.7 GB of VRAM on
    every query. On a 4 GB card that pushes a quarter of the model onto the
    CPU, so Djinn only loads it when a message carries an image.

    Ollama's library has no Q3_K_M of this model, so the GGUFs come from
    unsloth on Hugging Face (2.3 GB + 0.7 GB). Ollama copies them into its own
    store (~/.ollama/models), after which the downloads are deleted.

.PARAMETER GgufDir
    Where to download to (or where the GGUFs already are). Defaults to a
    temporary folder.

.PARAMETER Keep
    Keep the downloaded GGUFs instead of deleting them afterwards.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\setup_local_models.ps1
#>
param(
    [string]$GgufDir = (Join-Path $env:TEMP "djinn-models"),
    [switch]$Keep
)

$ErrorActionPreference = "Stop"

$ollama = (Get-Command ollama -ErrorAction SilentlyContinue).Source
if (-not $ollama) {
    $ollama = Join-Path $env:LOCALAPPDATA "Programs\Ollama\ollama.exe"
}
if (-not (Test-Path $ollama)) {
    throw "Ollama not found. Install it from https://ollama.com/download first."
}

$repo = "https://huggingface.co/unsloth/Qwen3.5-4B-GGUF/resolve/main"
$files = @{
    weights   = @{ remote = "Qwen3.5-4B-Q3_K_M.gguf"; local = "Qwen3.5-4B-Q3_K_M.gguf" }
    projector = @{ remote = "mmproj-F16.gguf";        local = "Qwen3.5-4B-mmproj-F16.gguf" }
}

New-Item -ItemType Directory -Force $GgufDir | Out-Null

foreach ($f in $files.Values) {
    $path = Join-Path $GgufDir $f.local
    if (Test-Path $path) {
        Write-Host "Using existing $path"
        continue
    }
    Write-Host "Downloading $($f.remote) ..."
    # curl.exe ships with Windows 10+ and is far faster than Invoke-WebRequest.
    & curl.exe -L --fail --progress-bar -o "$path.part" "$repo/$($f.remote)"
    if ($LASTEXITCODE -ne 0) { throw "Download failed: $($f.remote)" }
    Move-Item -Force "$path.part" $path
}

$weights = Join-Path $GgufDir $files.weights.local
$projector = Join-Path $GgufDir $files.projector.local

# RENDERER / PARSER are Ollama's built-in Qwen3.5 handlers - the same ones its
# own qwen3.5 models use. They are what make tool calling and thinking work.
# Sampling follows Qwen's model card for non-thinking mode.
$common = @"
TEMPLATE {{ .Prompt }}
RENDERER qwen3.5
PARSER qwen3.5
PARAMETER temperature 0.7
PARAMETER top_k 20
PARAMETER top_p 0.8
"@

$models = [ordered]@{
    "qwen3.5-4b-q3km-text" = "FROM $weights`n$common"
    "qwen3.5-4b-q3km"      = "FROM $weights`nFROM $projector`n$common"
}

foreach ($name in $models.Keys) {
    $modelfile = Join-Path $GgufDir "Modelfile.$name"
    Set-Content -Path $modelfile -Value $models[$name] -Encoding ascii
    Write-Host "Creating $name ..."
    & $ollama create $name -f $modelfile
    if ($LASTEXITCODE -ne 0) { throw "ollama create $name failed" }
    Remove-Item $modelfile
}

if (-not $Keep) {
    Remove-Item (Join-Path $GgufDir $files.weights.local), (Join-Path $GgufDir $files.projector.local)
    Write-Host "Removed the downloads - Ollama keeps its own copy."
}

Write-Host "`nDone. Installed:"
& $ollama list | Select-String "qwen3.5-4b-q3km"
