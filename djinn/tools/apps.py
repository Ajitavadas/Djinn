"""App and system control — launch apps, volume, clipboard, lock.

Windows-only. Everything here has a real side effect on the machine, so the
surface is deliberately narrow: launching is done via the shell's own app
resolution rather than arbitrary command execution, and there is no "run this
command" escape hatch (that lives in code.py, behind its own switch).
"""
import logging
import os
import subprocess
import shutil

log = logging.getLogger("djinn.tools.apps")

# Friendly name -> what to actually launch. Anything not listed here is passed
# to `start`, which resolves registered apps (and is why "spotify" works even
# though it is not in this table).
KNOWN_APPS = {
    "chrome": "chrome",
    "google chrome": "chrome",
    "firefox": "firefox",
    "edge": "msedge",
    "notepad": "notepad",
    "calculator": "calc",
    "calc": "calc",
    "explorer": "explorer",
    "file explorer": "explorer",
    "files": "explorer",
    "terminal": "wt",
    "cmd": "cmd",
    "powershell": "powershell",
    "task manager": "taskmgr",
    "settings": "ms-settings:",
    "vscode": "code",
    "vs code": "code",
    "code": "code",
    "spotify": "spotify",
    "discord": "discord",
    "steam": "steam",
}


def open_app(name: str) -> str:
    """Launch an application by name.

    Args:
        name: App name, e.g. "chrome", "notepad", "spotify".

    Returns:
        What happened, phrased for speaking aloud.
    """
    key = name.strip().lower()
    target = KNOWN_APPS.get(key, key)

    try:
        # `start` goes through the shell's app resolution, so registered apps
        # and URI handlers (ms-settings:) work without hardcoding paths.
        subprocess.Popen(
            ["cmd", "/c", "start", "", target],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        log.info("open_app(%r) -> launched %r", name, target)
        return f"Opening {name}."
    except Exception as e:
        log.error("open_app(%r) failed: %s", name, e)
        return f"I couldn't open {name}."


def close_app(name: str) -> str:
    """Close an application by name.

    Args:
        name: App name, e.g. "notepad".

    Returns:
        What happened.
    """
    key = name.strip().lower()
    target = KNOWN_APPS.get(key, key)
    image = target if target.lower().endswith(".exe") else f"{target}.exe"

    try:
        result = subprocess.run(
            ["taskkill", "/IM", image, "/F"],
            capture_output=True,
            text=True,
            timeout=10,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        if result.returncode == 0:
            log.info("close_app(%r) -> killed %s", name, image)
            return f"Closed {name}."
        return f"{name} doesn't seem to be running."
    except Exception as e:
        log.error("close_app(%r) failed: %s", name, e)
        return f"I couldn't close {name}."


def set_volume(action: str) -> str:
    """Change the system volume.

    Args:
        action: "up", "down", or "mute".

    Returns:
        What happened.
    """
    keys = {
        "up": 0xAF,      # VK_VOLUME_UP
        "down": 0xAE,    # VK_VOLUME_DOWN
        "mute": 0xAD,    # VK_VOLUME_MUTE
    }
    act = action.strip().lower()
    if act not in keys:
        return "I can only turn the volume up, down, or mute it."

    try:
        import ctypes

        vk = keys[act]
        presses = 1 if act == "mute" else 5  # each step is ~2%, so 5 is audible
        for _ in range(presses):
            ctypes.windll.user32.keybd_event(vk, 0, 0, 0)
            ctypes.windll.user32.keybd_event(vk, 0, 2, 0)

        log.info("set_volume(%r)", act)
        return {"up": "Volume up.", "down": "Volume down.", "mute": "Muted."}[act]
    except Exception as e:
        log.error("set_volume(%r) failed: %s", act, e)
        return "I couldn't change the volume."


def get_clipboard() -> str:
    """Read the current clipboard text.

    Returns:
        The clipboard contents, or a note that it is empty.
    """
    try:
        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command", "Get-Clipboard"],
            capture_output=True,
            text=True,
            timeout=10,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        text = (result.stdout or "").strip()
        return text if text else "The clipboard is empty."
    except Exception as e:
        log.error("get_clipboard failed: %s", e)
        return "I couldn't read the clipboard."


def set_clipboard(text: str) -> str:
    """Put text on the clipboard.

    Args:
        text: What to copy.

    Returns:
        Confirmation.
    """
    try:
        subprocess.run(
            ["clip"],
            input=text,
            text=True,
            timeout=10,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        return "Copied to clipboard."
    except Exception as e:
        log.error("set_clipboard failed: %s", e)
        return "I couldn't copy that."


def lock_screen() -> str:
    """Lock the workstation.

    Returns:
        Confirmation.
    """
    try:
        import ctypes

        ctypes.windll.user32.LockWorkStation()
        return "Locking."
    except Exception as e:
        log.error("lock_screen failed: %s", e)
        return "I couldn't lock the screen."
