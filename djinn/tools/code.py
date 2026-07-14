"""Code execution — run a short Python snippet and report what it printed.

SECURITY: this is the sharpest tool in the box, so it is OFF by default
(tools.allow_code_execution in config.yaml).

The snippet runs in a separate interpreter process, in a scratch directory,
under a wall-clock timeout, and is killed if it overruns. That bounds runaway
loops and keeps the assistant responsive. It does NOT sandbox the code: a
snippet can still reach the filesystem and the network with the privileges of
whoever is running Djinn. Enable it only if you accept that — and remember
web_search output flows back into the model, so a hostile page could try to
talk Djinn into running something. Off is the safe default.
"""
import logging
import subprocess
import sys
import tempfile
from pathlib import Path

log = logging.getLogger("djinn.tools.code")

TIMEOUT_SECONDS = 15
MAX_OUTPUT_CHARS = 4000

_enabled = False


def configure(enabled: bool) -> None:
    """Enable or disable code execution."""
    global _enabled
    _enabled = enabled
    log.info("Code execution %s", "ENABLED" if enabled else "disabled")


def run_python(code: str) -> str:
    """Run a Python snippet and return its output.

    Args:
        code: Python source. Print the result you care about.

    Returns:
        Whatever it printed, or the error it raised.
    """
    if not _enabled:
        return (
            "Code execution is turned off. Enable it with "
            "tools.allow_code_execution in config.yaml."
        )

    try:
        with tempfile.TemporaryDirectory(prefix="djinn-code-") as workdir:
            script = Path(workdir) / "snippet.py"
            script.write_text(code, encoding="utf-8")

            result = subprocess.run(
                [sys.executable, str(script)],
                capture_output=True,
                text=True,
                timeout=TIMEOUT_SECONDS,
                cwd=workdir,  # scratch dir: relative writes land nowhere important
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )

        stdout = (result.stdout or "").strip()
        stderr = (result.stderr or "").strip()

        if result.returncode != 0:
            log.info("run_python -> exit %d", result.returncode)
            return f"The code failed:\n{stderr[-MAX_OUTPUT_CHARS:] or '(no error message)'}"

        if not stdout:
            return "The code ran without errors, but printed nothing."

        log.info("run_python -> %d chars of output", len(stdout))
        return stdout[:MAX_OUTPUT_CHARS]

    except subprocess.TimeoutExpired:
        log.warning("run_python timed out after %ds", TIMEOUT_SECONDS)
        return f"The code was still running after {TIMEOUT_SECONDS} seconds, so I stopped it."
    except Exception as e:
        log.error("run_python failed: %s", e)
        return f"I couldn't run that: {e}"
