"""Tool registry: declares tools to Gemini and dispatches its calls.

One place that knows every tool. The brain asks for declarations(), Gemini
decides what to call, and dispatch() runs it.

Note there is no google_search tool here. Vertex rejects a request that mixes
a search tool with custom functions ("Multiple tools are supported only when
they are all search tools"), so web_search is an ordinary function that runs
its own grounded Gemini call internally — see tools/web.py.
"""
import asyncio
import inspect
import logging
from typing import Any, Callable

from djinn.tools import apps, code, files, web

log = logging.getLogger("djinn.tools")

# name -> (callable, description, {param: (type, description, required)})
_SPECS: dict[str, tuple[Callable, str, dict]] = {
    "web_search": (
        web.web_search,
        "Search the web for current or factual information. Use this for "
        "anything recent, anything you are unsure about, and anything that "
        "happened after your training data — news, prices, releases, people, "
        "events. Prefer searching over guessing.",
        {"query": ("string", "What to search for.", True)},
    ),
    "open_app": (
        apps.open_app,
        "Launch an application on the user's PC.",
        {"name": ("string", "App name, e.g. chrome, notepad, spotify.", True)},
    ),
    "close_app": (
        apps.close_app,
        "Close a running application.",
        {"name": ("string", "App name, e.g. notepad.", True)},
    ),
    "set_volume": (
        apps.set_volume,
        "Change the system volume.",
        {"action": ("string", "One of: up, down, mute.", True)},
    ),
    "get_clipboard": (
        apps.get_clipboard,
        "Read whatever text is currently on the clipboard.",
        {},
    ),
    "set_clipboard": (
        apps.set_clipboard,
        "Copy text to the clipboard.",
        {"text": ("string", "The text to copy.", True)},
    ),
    "lock_screen": (
        apps.lock_screen,
        "Lock the workstation.",
        {},
    ),
    "read_file": (
        files.read_file,
        "Read a text file from the user's workspace folder.",
        {"path": ("string", "Path relative to the workspace.", True)},
    ),
    "write_file": (
        files.write_file,
        "Write a text file into the user's workspace folder.",
        {
            "path": ("string", "Path relative to the workspace.", True),
            "content": ("string", "What to write.", True),
        },
    ),
    "list_dir": (
        files.list_dir,
        "List the files in a workspace folder.",
        {"path": ("string", "Path relative to the workspace. Defaults to the root.", False)},
    ),
    "run_python": (
        code.run_python,
        "Run a short Python snippet and return whatever it prints. Use this "
        "for calculations, data wrangling, or anything easier to compute than "
        "to reason about. Always print the result.",
        {"code": ("string", "Python source. Must print its result.", True)},
    ),
}

# Tools that are only offered when explicitly enabled in config.
_GATED = {"run_python": "allow_code_execution"}


_enabled: set[str] = set()


def configure(
    client,
    search_model: str,
    workspace: str,
    allow_code_execution: bool = False,
) -> None:
    """Wire up every tool's dependencies. Call once at startup."""
    web.configure(client, search_model)
    files.configure(workspace)
    code.configure(allow_code_execution)

    flags = {"allow_code_execution": allow_code_execution}

    _enabled.clear()
    for name in _SPECS:
        gate = _GATED.get(name)
        if gate is None or flags.get(gate, False):
            _enabled.add(name)

    withheld = sorted(set(_SPECS) - _enabled)
    log.info("Tools ready: %s", ", ".join(sorted(_enabled)))
    if withheld:
        log.info("Tools disabled: %s", ", ".join(withheld))


def declarations() -> list:
    """Build the Gemini FunctionDeclaration list for the enabled tools."""
    from google.genai import types

    type_map = {
        "string": types.Type.STRING,
        "integer": types.Type.INTEGER,
        "boolean": types.Type.BOOLEAN,
    }

    decls = []
    for name in sorted(_enabled):
        fn, description, params = _SPECS[name]

        properties = {
            param: types.Schema(type=type_map[ptype], description=pdesc)
            for param, (ptype, pdesc, _req) in params.items()
        }
        required = [p for p, (_t, _d, req) in params.items() if req]

        decls.append(
            types.FunctionDeclaration(
                name=name,
                description=description,
                parameters=(
                    types.Schema(
                        type=types.Type.OBJECT,
                        properties=properties,
                        required=required,
                    )
                    if properties
                    else None
                ),
            )
        )
    return decls


def tools() -> list:
    """The Gemini `tools` argument — a single Tool holding every declaration."""
    from google.genai import types

    decls = declarations()
    return [types.Tool(function_declarations=decls)] if decls else []


async def dispatch(name: str, args: dict[str, Any]) -> str:
    """Execute one tool call and return its result as text."""
    if name not in _SPECS:
        return f"Unknown tool: {name}"
    if name not in _enabled:
        return f"The {name} tool is disabled."

    fn = _SPECS[name][0]
    log.info("-> %s(%s)", name, ", ".join(f"{k}={v!r}"[:60] for k, v in args.items()))

    try:
        if inspect.iscoroutinefunction(fn):
            result = await fn(**args)
        else:
            # Tools here block (subprocess, file I/O), so keep them off the loop.
            result = await asyncio.to_thread(lambda: fn(**args))
        return str(result)
    except TypeError as e:
        # The model passed the wrong arguments. Tell it, so it can retry.
        return f"Bad arguments for {name}: {e}"
    except Exception as e:
        log.error("%s failed: %s", name, e)
        return f"{name} failed: {e}"
