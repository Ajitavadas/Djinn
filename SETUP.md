# Setting up Djinn on a new machine

Every step to get Djinn running on a fresh Windows PC after cloning, in order.
The commands are for **PowerShell**.

## 0. What you need

| | |
|---|---|
| OS | Windows 10 or 11 |
| GPU | NVIDIA with an up-to-date driver (recommended; CPU works, slower) |
| RAM | 8 GB minimum |
| Disk | ~10 GB free, on an SSD if you have one |
| Internet | For setup: about 6 GB of downloads in total |

What gets downloaded:

| What | Size | Where |
|---|---|---|
| Python packages | ~1.5 GB | `.venv\` in the repo |
| Whisper speech model (first run) | 1.5 GB | `%USERPROFILE%\.cache\huggingface` |
| Kokoro voice files | 115 MB | `djinn\data\models\` |
| Local LLM (optional) | 3 GB | `%USERPROFILE%\.ollama\models` |

## 1. Install the tools

```powershell
winget install --id Git.Git -e
winget install --id astral-sh.uv -e
winget install --id Ollama.Ollama -e        # only for the local model (step 6)
winget install --id Google.CloudSDK -e      # only for Gemini (step 5)
```

**Close and reopen the terminal** afterwards, so it picks up the new `PATH`.
You don't need to install Python. `uv` downloads the right version itself.

## 2. Clone the repo

```powershell
git clone https://github.com/Ajitavadas/Djinn.git
cd Djinn
```

If you copied the folder from another PC instead of cloning, git may refuse
to run with "detected dubious ownership". Fix it with:

```powershell
git config --global --add safe.directory ((Get-Location).Path.Replace('\','/'))
```

## 3. Install the Python dependencies

```powershell
uv sync
```

This creates `.venv\` with Python 3.11 and every package, including the CUDA
libraries Whisper needs.

## 4. Download the voice files

Kokoro, the offline text-to-speech engine, needs two files. They aren't in git
because of their size:

```powershell
New-Item -ItemType Directory -Force djinn\data\models | Out-Null
$base = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0"
curl.exe -L -o djinn\data\models\kokoro-v1.0.int8.onnx "$base/kokoro-v1.0.int8.onnx"
curl.exe -L -o djinn\data\models\voices-v1.0.bin       "$base/voices-v1.0.bin"
```

## 5. Gemini (the cloud model)

Skip this step if you only want the local model. Without Gemini, every query
goes to the local model, and `web_search` isn't available.

Djinn uses Vertex AI, so Google Cloud credits pay for it. You need a Google
Cloud project with billing enabled.

```powershell
gcloud auth application-default login
gcloud auth application-default set-quota-project <your-project-id>
gcloud services enable aiplatform.googleapis.com --project <your-project-id>
```

Then create your `.env`:

```powershell
Copy-Item .env.example .env
notepad .env        # set GOOGLE_CLOUD_PROJECT=<your-project-id>
```

`.env` is gitignored, so never commit it. To use an AI Studio API key
instead, see "Vertex AI vs AI Studio" in the README.

## 6. The local model (offline, free)

Make sure Ollama is running: it starts with Windows after installing. Check
with `ollama list`, and if that fails, run `ollama serve` in another
terminal. Then:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup_local_models.ps1
```

This downloads Qwen3.5-4B (Q3_K_M, 3 GB) and creates two Ollama models:
`qwen3.5-4b-q3km-text`, used for normal queries, and `qwen3.5-4b-q3km`, the
same model with vision, used for images. Check with:

```powershell
ollama list
```

## 7. Personalise the config

Open `djinn\config.yaml` and change these to match your setup:

| Setting | What to put |
|---|---|
| `djinn.name` | Your name. It's how Djinn addresses you. |
| `vad.input_device` | Part of your microphone's name, e.g. `"Headset"` or `"Realtek"`. The default Windows mic is often a silent dummy device. Leave it blank to use the default. |
| `tools.workspace` | The folder Djinn may read and write files in (default `~/Djinn`) |
| `ollama.num_gpu` | Leave at `99`. If you have no NVIDIA GPU, set `0`. |

To list your microphones:

```powershell
uv run python -c "import sounddevice as sd; [print(i, d['name']) for i, d in enumerate(sd.query_devices()) if d['max_input_channels'] > 0]"
```

## 8. Run it

Start with text mode, which needs no microphone:

```powershell
uv run djinn --text-only               # chat window
uv run djinn --text-only --mode local  # chat window, local model only
```

Then voice:

```powershell
uv run djinn                 # hold Ctrl+Alt+D and speak
uv run djinn --mode local    # voice, local model only
uv run djinn --no-gpu        # if you have no NVIDIA GPU
```

The first voice run downloads the Whisper model (1.5 GB), so it takes a few
minutes. After that, startup takes about 15 seconds.

In the chat window, `/auto` `/fast` `/pro` `/local` switch modes,
`/think <question>` makes the model reason first, and `/clear` resets the
conversation.

### What a healthy start looks like

```
Silero VAD loaded (ONNX, threshold=0.50, chunk=512 samples)
Whisper loaded in 5.3s
CUDA warmed up in 5.5s
Microphone: <your mic>
Gemini ready via Vertex AI (...)                      <- if you did step 5
Local model ready: qwen3.5-4b-q3km-text via Ollama    <- if you did step 6
Djinn is ready!
```

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `detected dubious ownership in repository` | The folder came from another user account. See step 2. |
| `uv` / `ollama` not recognised | The terminal is older than the install. Open a new one. |
| `An Application Control policy has blocked this file ... torch.dll` | Something reinstalled PyTorch, which Windows Smart App Control blocks. Djinn doesn't need it. Find what pulled it in with `uv tree --invert --package torch` and remove that dependency. |
| `Library cublas64_12.dll is not found` | The CUDA wheels are missing. Run `uv sync` again, or use `--no-gpu`. |
| Voice mode hears nothing | Wrong microphone. Set `vad.input_device` (step 7). |
| `Ollama not reachable` | Ollama isn't running. Start it from the Start menu, or run `ollama serve`. |
| `... is not installed. Run scripts/setup_local_models.ps1` | Step 6 wasn't run, or it failed. |
| `Gemini unavailable — every query goes to the local model` | Step 5 is missing or incomplete. Check `.env` and `gcloud auth application-default login`. |
| Local model slow, `ollama ps` shows less than 100% GPU | Another program is using GPU memory, such as an external monitor on the NVIDIA card or a game. Close it, or set `ollama.num_gpu: null`. |
| `Vertex AI ... 403 / permission denied` | The API isn't enabled or there's no quota project. Re-run the `gcloud` commands in step 5. |
