"""Djinn — Personal AI Assistant entry point."""
import asyncio
import sys
import os
import signal
import argparse
import logging

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv() # Load environmental variables from .env

from djinn.core.orchestrator import Orchestrator


def setup_logging(level: str = "INFO") -> None:
    """Configure logging with clean format."""
    # Djinn logs emoji route labels and box-drawing characters. Windows
    # consoles default to cp1252, which cannot encode them, and logging then
    # dies with UnicodeEncodeError. Force UTF-8 on the streams first.
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (OSError, ValueError):
                pass

    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s │ %(name)-20s │ %(levelname)-7s │ %(message)s",
        datefmt="%H:%M:%S",
    )
    # Quiet noisy libraries. Each of these emits a line per request or per
    # utterance and drowns out Djinn's own logs.
    for noisy in (
        "faster_whisper",
        "httpx",
        "urllib3",
        "google_genai.models",      # "AFC is enabled with max remote calls: 10"
        "google_genai._api_client",
        "google.auth",              # "No project ID could be determined"
        "google.auth._default",
        "phonemizer",               # "words count mismatch" on every sentence
    ):
        logging.getLogger(noisy).setLevel(logging.ERROR)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Djinn — Personal AI Assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml"),
        help="Path to config file (default: djinn/config.yaml)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )
    parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="Force CPU-only mode (no GPU for Whisper)",
    )
    parser.add_argument(
        "--text-only",
        action="store_true",
        help="Text input mode — skip voice pipeline, type queries instead",
    )
    parser.add_argument(
        "--mode",
        default=None,
        choices=["auto", "fast", "pro"],
        help=(
            "Which model tier to use. auto = router decides per query "
            "(default), fast = always the fast tier, pro = always the deep "
            "tier. Switch at runtime with Ctrl+Alt+M, or /fast /pro /auto."
        ),
    )
    return parser.parse_args()


def main() -> None:
    """Main entry point for Djinn."""
    args = parse_args()
    setup_logging(args.log_level)

    log = logging.getLogger("djinn")
    log.info("━" * 50)
    log.info("  DJINN — Personal AI Assistant")
    log.info("━" * 50)

    orchestrator = Orchestrator(
        config_path=args.config,
        force_cpu=args.no_gpu,
        text_only=args.text_only,
        mode=args.mode,
    )

    # Handle graceful shutdown
    def shutdown_handler(sig, frame):
        log.info("Shutdown signal received...")
        orchestrator.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)

    try:
        asyncio.run(orchestrator.run())
    except KeyboardInterrupt:
        log.info("Djinn shutting down.")
        orchestrator.shutdown()
        sys.exit(0)


if __name__ == "__main__":
    main()
