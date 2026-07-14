"""Audition TTS voices and pick Djinn's voice.

Plays every matching voice speaking a sample line, one at a time, and lets
you set the one you like straight into config.yaml.

Usage:
    uv run python scripts/audition_voices.py                    # all English voices
    uv run python scripts/audition_voices.py --gender male
    uv run python scripts/audition_voices.py --engine edge --locale en-IN
    uv run python scripts/audition_voices.py --engine kokoro

At each voice:
    Enter  next voice
    r      replay
    s      set as Djinn's voice (writes config.yaml)
    b      back to the previous voice
    q      quit
"""
import argparse
import asyncio
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import sounddevice as sd

from djinn.core.voice_output import EdgeTTS, KokoroTTS

CONFIG = ROOT / "djinn" / "config.yaml"
SAMPLE = "Hello! I am Djinn, your personal assistant. This is how I would sound. Shall we get to work?"


def edge_voices(locale: str, gender: str) -> list[tuple[str, str]]:
    import edge_tts

    voices = asyncio.run(edge_tts.list_voices())
    out = []
    for v in voices:
        if not v["Locale"].lower().startswith(locale.lower()):
            continue
        if gender != "all" and v["Gender"].lower() != gender:
            continue
        out.append((v["ShortName"], v["Gender"]))
    return sorted(out)


def kokoro_voices(kokoro: KokoroTTS, gender: str) -> list[tuple[str, str]]:
    # Names encode language and gender: a=American b=British, then f/m.
    names = sorted(kokoro._kokoro.get_voices())
    out = []
    for n in names:
        if n[0] not in "ab" or len(n) < 2:
            continue
        g = {"f": "Female", "m": "Male"}.get(n[1])
        if g is None:
            continue
        if gender != "all" and g.lower() != gender:
            continue
        out.append((n, g))
    return out


def set_config_voice(engine: str, name: str) -> None:
    key = "edge_voice" if engine == "edge" else "kokoro_voice"
    text = CONFIG.read_text(encoding="utf-8")
    new, n = re.subn(rf'{key}: ".*"', f'{key}: "{name}"', text, count=1)
    if not n:
        print(f"   !! could not find {key} in config.yaml — set it by hand")
        return
    CONFIG.write_text(new, encoding="utf-8")
    print(f"   ✓ {key}: \"{name}\" written to config.yaml")
    if engine == "kokoro":
        print('     (Kokoro is the fallback engine — set tts.primary: "kokoro" to make it the main voice)')


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--engine", choices=["edge", "kokoro", "all"], default="all")
    ap.add_argument("--locale", default="en",
                    help='Edge locale prefix, e.g. "en" (all English), "en-IN", "en-GB"')
    ap.add_argument("--gender", choices=["male", "female", "all"], default="all")
    args = ap.parse_args()

    entries: list[tuple[str, str, str]] = []
    kokoro = None

    if args.engine in ("edge", "all"):
        print("Fetching the Edge voice list...")
        entries += [("edge", n, g) for n, g in edge_voices(args.locale, args.gender)]
    if args.engine in ("kokoro", "all"):
        kokoro = KokoroTTS()
        kokoro.load()
        if kokoro.available:
            entries += [("kokoro", n, g) for n, g in kokoro_voices(kokoro, args.gender)]

    if not entries:
        print("No voices matched those filters.")
        return

    print(f"\n{len(entries)} voices.  Enter=next  r=replay  s=set as Djinn's voice  b=back  q=quit\n")

    edge = EdgeTTS()
    i = 0
    while i < len(entries):
        engine, name, gender = entries[i]
        print(f"[{i + 1}/{len(entries)}] {engine:6s} {name}  ({gender})")

        if engine == "edge":
            edge.voice = name
            result = asyncio.run(edge.synthesize(SAMPLE))
        else:
            kokoro.voice = name
            result = kokoro.synthesize(SAMPLE)

        if result is None:
            print("   synthesis failed, skipping")
            i += 1
            continue

        samples, rate = result
        sd.play(samples, samplerate=rate)
        sd.wait()

        cmd = input("   > ").strip().lower()
        if cmd == "q":
            break
        if cmd == "r":
            continue
        if cmd == "b":
            i = max(0, i - 1)
            continue
        if cmd == "s":
            set_config_voice(engine, name)
        i += 1


if __name__ == "__main__":
    main()
