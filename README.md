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
| `ollama` | Qwen3.5-4B on your own machine, via Ollama | ~2-10s on a 4 GB GPU |

You can override the router at any time — see **Modes** below. The local
model is never picked automatically while Gemini works. It is used when you pin
`local` mode, or as a fallback when Gemini is unconfigured or unreachable.

## Requirements

- Windows, Python 3.11+
- An NVIDIA GPU is recommended for Whisper (CPU works via `--no-gpu`)
- A Google Cloud project with billing enabled

Djinn doesn't need PyTorch. The voice detector is Silero v6 running on
onnxruntime, using the copy of the model that faster-whisper ships. This
matters on Windows 11 with **Smart App Control** on, because it blocks
`torch.dll`. Don't add a dependency that pulls PyTorch back in.

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

**Optional: the local model.** Install [Ollama](https://ollama.com/download),
then build the models (downloads 3 GB from Hugging Face):

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup_local_models.ps1
```

This creates two Ollama models from the same Qwen3.5-4B Q3_K_M weights:

| Model | Used for | Loaded size |
|---|---|---|
| `qwen3.5-4b-q3km-text` | Every normal query | 2.5 GB |
| `qwen3.5-4b-q3km` | Messages that carry an image | 3.0 GB + buffers |

Vision is a separate model because its projector takes ~0.7 GB of VRAM on
every query. On a 4 GB card that pushes a quarter of the language model onto
the CPU, so Djinn only loads it when there is an image to look at.

Why this model? On a GTX 1650 Ti, with 12 coding problems checked by unit
tests (`scripts/code_eval.py`):

| Model | Passed | Per problem | Speed | On GPU |
|---|---|---|---|---|
| Qwen3.5-2B Q4_K_M | 2/12 | ~7s | 74 tok/s | 100% |
| Qwen3.5-4B Q4_K_M (`qwen3.5:4b`) | 8/12 | ~62s | 7.5 tok/s | 48% |
| **Qwen3.5-4B Q3_K_M, text only** | 8–10/12 | ~10s | 25–38 tok/s | 74–100% |
| Qwen3.5-9B IQ2_XXS | 1/12 | ~111s | 5 tok/s | 43% |

Q3 scores as well as Q4 (the gap is within noise), and fits on the GPU. The
2-bit 9B writes plausible-looking code that does not work. Note that Windows
reserves ~0.8 GB of any GPU for itself, so a 4 GB card has ~2.8 GB left for
models. Djinn forces every layer onto the GPU, because Ollama's own estimate is
cautious. The 4B and Whisper fit together (3.56 of 4 GB), running at 38
tokens/s, against 26 when Ollama decides the split.

## Run

```bash
uv run djinn                 # voice mode — hold Ctrl+Alt+D and speak
uv run djinn --text-only     # chat window, no microphone
uv run djinn --mode pro      # pin the deep tier
uv run djinn --mode local    # pin the local Ollama model
uv run djinn --no-gpu        # CPU-only Whisper
```

## Modes

Djinn picks a model per query, but you can pin one:

| Mode | Behaviour |
|---|---|
| `auto` | The router decides (default) |
| `fast` | Always the fast tier |
| `pro` | Always the deep tier |
| `local` | Always the local model. Offline and free. |

Switch at runtime: click the badge in the chat window, press `Ctrl+M`, type
`/auto` `/fast` `/pro` `/local`, or press `Ctrl+Alt+M` in voice mode. Local
shortcuts (time, date) stay instant in every mode.

The local model gets the same tools as Gemini, except `web_search`, which runs
through Gemini and so needs a connection.

**Thinking.** In the chat window, `/think <question>` makes the model reason
before answering, for that one question. On the local model this is slow (1–3
minutes on a 4 GB GPU) but catches mistakes it would otherwise blurt out. With
Gemini it uses the deep tier.

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
  brain/      Gemini client (Vertex + AI Studio), local Ollama client
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

Working: voice pipeline, both Gemini tiers, the local Ollama model (pinned or
as the offline fallback), tools, streaming speech, working memory, mode
switching.

Not yet built: long-term memory (`memory/longterm.py`), system tray and gaming
overlay (`ui/`).
