# Djinn

A local-first personal voice assistant for Windows. Hold a hotkey, speak, and it
answers out loud — and it can actually *do* things: search the web, open apps,
read and write files, control the system.

Speech recognition runs on your GPU, text-to-speech runs locally on your CPU, and
only the reasoning goes to the cloud.

```
Hotkey → Microphone → Silero VAD → faster-whisper → Router
                                                       ↓
Speakers ← Kokoro TTS ← Gemini (fast / deep tier) ← Tools
                                                       ↓
                                              Working memory
```

## How it works

A **router** classifies each query with regex, so no latency is wasted deciding:

| Route | Handled by | Typical latency |
|---|---|---|
| `local` | Answered in-process — time, date. No model call. | ~0ms |
| `fast` | Gemini 2.5 Flash, thinking disabled | ~1s |
| `deep` | Gemini 2.5 Pro, thinking enabled | ~3s |

You can override the router at any time — see **Modes** below.

## Requirements

- Windows, Python 3.11+
- An NVIDIA GPU is recommended for Whisper (CPU works via `--no-gpu`)
- A Google Cloud project with billing enabled

## Setup

```bash
git clone https://github.com/Ajitavadas/djinn.git
cd djinn
uv sync            # or: pip install -r requirements.txt
```

**Authenticate with Vertex AI.** Djinn uses Application Default Credentials, not
an API key — so your Google Cloud credits pay for it:

```bash
gcloud auth application-default login
gcloud auth application-default set-quota-project <your-project-id>
gcloud services enable aiplatform.googleapis.com
```

Then copy `.env.example` to `.env` and set `GOOGLE_CLOUD_PROJECT`.

> **Vertex AI vs AI Studio.** These are different billing systems for the same
> Gemini models. Google Cloud credits work on **Vertex only** — they do *not*
> apply to an AI Studio API key, which needs its own prepaid balance. Djinn
> defaults to Vertex for this reason. To use AI Studio instead, set
> `gemini.backend: aistudio` and put `GEMINI_API_KEY` in `.env`.

**Download the TTS voices** into `djinn/data/models/` (they are gitignored — 115 MB):
`kokoro-v1.0.int8.onnx` and `voices-v1.0.bin` from
[kokoro-onnx releases](https://github.com/thewh1teagle/kokoro-onnx/releases).

## Run

```bash
uv run djinn                 # voice mode — hold Ctrl+Alt+D and speak
uv run djinn --text-only     # chat window, no microphone
uv run djinn --mode pro      # pin the deep tier
uv run djinn --no-gpu        # CPU-only Whisper
```

## Modes

Djinn picks a model per query, but you can pin one:

| Mode | Behaviour |
|---|---|
| `auto` | The router decides (default) |
| `fast` | Always the fast tier |
| `pro` | Always the deep tier |

Switch at runtime: click the badge in the chat window, press `Ctrl+M`, type
`/auto` `/fast` `/pro`, or press `Ctrl+Alt+M` in voice mode. Local shortcuts
(time, date) stay instant in every mode.

## Tools

| Tool | Notes |
|---|---|
| `web_search` | Google Search grounding, returns sources |
| `open_app`, `close_app` | Launch/close applications |
| `set_volume`, `lock_screen` | System control |
| `get_clipboard`, `set_clipboard` | Clipboard access |
| `read_file`, `write_file`, `list_dir` | Confined to the workspace folder |
| `run_python` | **Disabled by default** — see Security |

### Security

Web search results are fed back into the model, which means a hostile page can
attempt a prompt injection. Two boundaries exist because of that:

- **File access is confined to a workspace** (`tools.workspace`, default
  `~/Djinn`). Paths that escape it via `..`, an absolute path, or a symlink are
  resolved and refused.
- **`run_python` is off by default.** It is timed out and runs in a scratch
  directory, but it is **not sandboxed** — a snippet can reach your disk and
  network with your privileges. Enable `tools.allow_code_execution` only if you
  accept that.

Never commit `.env`. It is gitignored.

## Configuration

Everything lives in `djinn/config.yaml` — models, hotkeys, VAD sensitivity,
voice, memory window, tool settings.

To change models, edit two lines:

```yaml
flash_model: "gemini-2.5-flash"   # fast tier
pro_model: "gemini-2.5-pro"       # deep tier
```

Note that Gemini charges *thinking* tokens against `max_output_tokens`. If
thinking is left uncapped on a small budget it will consume the entire allowance
and return an empty reply, so the deep tier budgets the two separately.

## Project layout

```
djinn/
  core/       orchestrator, router, voice input/output
  brain/      Gemini client (Vertex + AI Studio)
  tools/      web, apps, files, code + the registry
  memory/     working memory (long-term is a stub)
  ui/         tkinter chat window
```

## Speech and latency

Djinn speaks each sentence as it is generated, rather than waiting for the whole
reply, and synthesizes the *next* sentence while the current one plays. Typical
time to first spoken word is **~1.5–2s**.

Two TTS engines ship, and which one is primary matters more than it looks:

| Engine | Speed | Notes |
|---|---|---|
| **Edge** (default) | ~0.3x real time | Cloud, free, no API key |
| Kokoro | ~1.6x real time on CPU | Fully local, works offline |

The ratio is the point. Edge synthesizes *faster* than it plays, so synthesis
disappears behind playback entirely. Kokoro on a CPU is **slower than real time**,
so it can never be hidden and speech drags no matter how it is pipelined. Edge is
primary for that reason; Kokoro stays loaded as the offline fallback. Set
`tts.primary: kokoro` if you would rather never touch the network.

## Status

Working: voice pipeline, both model tiers, tools, streaming speech, working
memory, mode switching.

Not yet built: long-term memory (`memory/longterm.py`), offline Ollama fallback
(`brain/local_llm.py`), system tray and gaming overlay (`ui/`).
