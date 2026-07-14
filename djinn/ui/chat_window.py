"""Lightweight tkinter chat window for Djinn text-only mode.

Provides a minimal dark-themed GUI with an input field and scrollable
chat history so the user doesn't have to rely on terminal stdin.
"""
import asyncio
import logging
import queue
import threading
import tkinter as tk
from tkinter import scrolledtext

log = logging.getLogger("djinn.ui.chat_window")


class ChatWindow:
    """A minimal dark-themed chat window using tkinter.

    Runs tkinter's mainloop on a dedicated thread so it doesn't block
    the asyncio event loop. User input is placed into a thread-safe
    queue that the orchestrator can await.
    """

    # Cycle order for the mode toggle.
    MODES = ("auto", "fast", "pro")
    MODE_STYLE = {
        "auto": ("AUTO", "#53d8fb", "router decides"),
        "fast": ("FAST", "#3ddc84", "always fast tier"),
        "pro": ("PRO", "#e94560", "always deep tier"),
    }

    def __init__(
        self,
        on_close: callable = None,
        on_mode_change: callable = None,
        mode: str = "auto",
    ):
        self._input_queue: queue.Queue[str] = queue.Queue()
        self._on_close = on_close
        self._on_mode_change = on_mode_change
        self._mode = mode if mode in self.MODES else "auto"
        self._root: tk.Tk | None = None
        self._chat_display: scrolledtext.ScrolledText | None = None
        self._input_field: tk.Entry | None = None
        self._mode_btn: tk.Button | None = None
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()

    # ------------------------------------------------------------------
    # Public API (called from asyncio thread)
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Launch the chat window on a background thread."""
        self._thread = threading.Thread(target=self._run_tk, daemon=True)
        self._thread.start()
        self._ready.wait(timeout=5.0)
        log.info("Chat window opened")

    @property
    def mode(self) -> str:
        return self._mode

    def stop(self) -> None:
        """Close the window from any thread."""
        if self._root:
            try:
                self._root.after(0, self._root.destroy)
            except tk.TclError:
                pass

    async def get_input(self) -> str | None:
        """Await the next user input (non-blocking for asyncio)."""
        loop = asyncio.get_event_loop()
        try:
            return await loop.run_in_executor(None, self._input_queue.get)
        except Exception:
            return None

    def append_message(self, role: str, text: str) -> None:
        """Append a message to the chat display (thread-safe)."""
        if self._root and self._chat_display:
            self._root.after(0, self._do_append, role, text)

    # ------------------------------------------------------------------
    # Internal — tkinter thread
    # ------------------------------------------------------------------

    def _run_tk(self) -> None:
        """Build and run the tkinter window (runs on dedicated thread)."""
        root = tk.Tk()
        self._root = root
        root.title("Djinn — Text Mode")
        root.geometry("620x500")
        root.minsize(400, 300)
        root.configure(bg="#1a1a2e")
        root.protocol("WM_DELETE_WINDOW", self._handle_close)

        # --- Colour palette ---
        bg = "#1a1a2e"
        panel_bg = "#16213e"
        fg = "#e0e0e0"
        accent = "#0f3460"
        highlight = "#e94560"
        input_bg = "#0f3460"
        input_fg = "#ffffff"
        user_colour = "#53d8fb"
        djinn_colour = "#e94560"
        border_colour = "#0f3460"

        # --- Title bar ---
        title_frame = tk.Frame(root, bg=accent, height=38)
        title_frame.pack(fill=tk.X)
        title_frame.pack_propagate(False)
        tk.Label(
            title_frame,
            text="🔮 Djinn",
            font=("Segoe UI", 12, "bold"),
            bg=accent,
            fg="#ffffff",
        ).pack(side=tk.LEFT, padx=12)
        tk.Label(
            title_frame,
            text="text mode",
            font=("Segoe UI", 9),
            bg=accent,
            fg="#8888aa",
        ).pack(side=tk.LEFT, padx=(0, 8))

        # Mode toggle: click to cycle AUTO -> FAST -> PRO
        self._mode_btn = tk.Button(
            title_frame,
            font=("Segoe UI", 9, "bold"),
            bg=accent,
            activebackground=accent,
            relief=tk.FLAT,
            cursor="hand2",
            borderwidth=0,
            padx=10,
            command=self._cycle_mode,
        )
        self._mode_btn.pack(side=tk.RIGHT, padx=10)
        self._paint_mode_btn()

        # --- Chat display ---
        # height=1 is deliberate. A Text widget defaults to 24 rows, which is
        # TALLER THAN THE WINDOW; packed with expand=True it then eats the
        # entire frame and the input box below gets zero pixels — unmapped,
        # unfocusable, untypeable. Ask for almost nothing and let expand=True
        # grow it into whatever is left after the input row is reserved.
        self._chat_display = scrolledtext.ScrolledText(
            root,
            wrap=tk.WORD,
            font=("Consolas", 11),
            height=1,
            bg=panel_bg,
            fg=fg,
            insertbackground=fg,
            relief=tk.FLAT,
            padx=12,
            pady=10,
            state=tk.DISABLED,
            cursor="arrow",
            borderwidth=0,
            highlightthickness=1,
            highlightbackground=border_colour,
            highlightcolor=border_colour,
        )
        # NOTE: packed further down, *after* the input row, so the input row
        # claims its height first. Pack order is what protects it.

        # Tag styles for user / djinn messages
        self._chat_display.tag_configure("user_tag", foreground=user_colour, font=("Consolas", 11, "bold"))
        self._chat_display.tag_configure("djinn_tag", foreground=djinn_colour, font=("Consolas", 11, "bold"))
        self._chat_display.tag_configure("text_tag", foreground=fg, font=("Consolas", 11))
        self._chat_display.tag_configure("system_tag", foreground="#888888", font=("Consolas", 10, "italic"))

        # --- Input area ---
        # Packed to the BOTTOM *before* the chat display, so it reserves its
        # height up front. Pack it after and an expanding chat display starves
        # it to 1px. This ordering is load-bearing, not cosmetic.
        input_frame = tk.Frame(root, bg=bg)
        input_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=8, pady=(2, 8))

        self._input_field = tk.Entry(
            input_frame,
            font=("Consolas", 12),
            bg=input_bg,
            fg=input_fg,
            insertbackground=input_fg,
            relief=tk.FLAT,
            borderwidth=0,
            highlightthickness=1,
            highlightbackground=border_colour,
            highlightcolor=highlight,
        )
        self._input_field.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=8, padx=(0, 6))
        self._input_field.bind("<Return>", self._on_enter)
        # Ctrl+M also cycles the mode, so you never have to reach for the mouse.
        root.bind("<Control-m>", lambda _e: self._cycle_mode())

        send_btn = tk.Button(
            input_frame,
            text="Send",
            font=("Segoe UI", 10, "bold"),
            bg=highlight,
            fg="#ffffff",
            activebackground="#c73650",
            activeforeground="#ffffff",
            relief=tk.FLAT,
            cursor="hand2",
            command=lambda: self._on_enter(None),
            padx=16,
            pady=4,
        )
        send_btn.pack(side=tk.RIGHT, ipady=4)

        # Now that the input row owns its height, let the chat display expand
        # into everything that remains.
        self._chat_display.pack(
            fill=tk.BOTH, expand=True, padx=8, pady=(6, 2), before=input_frame
        )

        # Welcome message
        self._do_append(
            "system",
            "Djinn is ready. Type and press Enter.  "
            "Ctrl+M or the badge above cycles AUTO / FAST / PRO "
            "(or type /auto, /fast, /pro).",
        )

        # --- Keyboard focus ---------------------------------------------
        # Getting the caret into the Entry is fiddlier than it looks:
        #
        #  * focus_set() is silently dropped while the window itself is
        #    unfocused, which it is at startup, so the focus must be *forced*.
        #  * Toggling -topmost off resets focus to the toplevel, blowing away
        #    any focus we just set. So we don't touch -topmost at all.
        #  * Anything focusable (the buttons, the scrollback) can hold focus
        #    instead of the Entry, so they opt out via takefocus=False.
        #
        # Net effect without this: the window takes keystrokes (root bindings
        # like Ctrl+M fire) but the Entry never sees them — a dead text box.
        self._chat_display.configure(takefocus=False)
        send_btn.configure(takefocus=False)
        self._mode_btn.configure(takefocus=False)

        def _focus_entry(_event=None) -> None:
            try:
                self._input_field.focus_force()
                self._input_field.icursor(tk.END)
            except tk.TclError:
                pass

        def _startup_focus() -> None:
            try:
                root.deiconify()
                root.lift()
                root.focus_force()
                _focus_entry()
            except tk.TclError:
                pass

        root.after(100, _startup_focus)
        # Whenever the window is activated, put the caret back in the Entry.
        root.bind("<FocusIn>", _focus_entry, add="+")
        # Clicking anywhere in the window returns the caret to the input box.
        root.bind("<Button-1>", _focus_entry, add="+")
        # Typing anywhere in the window types into the input box.
        root.bind("<Key>", lambda _e: _focus_entry(), add="+")

        self._ready.set()
        root.mainloop()

    # ------------------------------------------------------------------
    # Mode toggle
    # ------------------------------------------------------------------

    def _paint_mode_btn(self) -> None:
        """Redraw the mode badge to match the current mode."""
        if not self._mode_btn:
            return
        label, colour, _desc = self.MODE_STYLE[self._mode]
        self._mode_btn.configure(text=f"● {label}", fg=colour, activeforeground=colour)

    def _cycle_mode(self) -> None:
        """Advance AUTO -> FAST -> PRO -> AUTO (tk thread)."""
        nxt = self.MODES[(self.MODES.index(self._mode) + 1) % len(self.MODES)]
        self.set_mode(nxt, announce=True)

    def set_mode(self, mode: str, announce: bool = False) -> None:
        """Set the mode. Safe to call from any thread."""
        if mode not in self.MODES or mode == self._mode:
            return
        self._mode = mode
        _label, _colour, desc = self.MODE_STYLE[mode]

        def _apply() -> None:
            self._paint_mode_btn()
            if announce:
                self._do_append("system", f"Mode: {mode.upper()} — {desc}")
            if self._input_field:
                self._input_field.focus_set()

        if self._root:
            self._root.after(0, _apply)
        if self._on_mode_change:
            self._on_mode_change(mode)

    def _on_enter(self, event) -> None:
        """Handle Enter key or Send button click."""
        text = self._input_field.get().strip()
        if not text:
            return

        self._input_field.delete(0, tk.END)
        self._do_append("You", text)
        self._input_queue.put(text)

    def _do_append(self, role: str, text: str) -> None:
        """Append a message to the chat area (must be called on tk thread)."""
        display = self._chat_display
        if not display:
            return

        display.configure(state=tk.NORMAL)

        if role == "system":
            display.insert(tk.END, f"  {text}\n\n", "system_tag")
        elif role == "You":
            display.insert(tk.END, "You: ", "user_tag")
            display.insert(tk.END, f"{text}\n\n", "text_tag")
        else:
            display.insert(tk.END, "Djinn: ", "djinn_tag")
            display.insert(tk.END, f"{text}\n\n", "text_tag")

        display.configure(state=tk.DISABLED)
        display.see(tk.END)

    def _handle_close(self) -> None:
        """Handle window close — push a sentinel so get_input unblocks."""
        self._input_queue.put(None)  # Sentinel
        if self._on_close:
            self._on_close()
        try:
            self._root.destroy()
        except tk.TclError:
            pass
