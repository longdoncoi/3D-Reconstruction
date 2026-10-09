"""Logging setup without import-time side effects."""
from __future__ import annotations

import logging
import logging.handlers
import os
import sys
import time
from datetime import datetime
from pathlib import Path


def force_utf8_stream(stream):
    """Wrap a binary stream buffer with UTF-8 TextIOWrapper (Qt Creator compatibility)."""
    try:
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
            return stream
        if hasattr(stream, "buffer"):
            import io
            return io.TextIOWrapper(stream.buffer, encoding="utf-8", errors="replace", line_buffering=True)
    except Exception:
        pass
    return stream


def setup_logging(logs_dir: Path | str) -> tuple[logging.Logger, Path]:
    """Configure application logging and return the main logger and current log file path."""
    # Configure global stream encodings for Qt Creator compatibility
    os.environ["PYTHONUTF8"] = "1"
    sys.stdout = force_utf8_stream(sys.stdout)
    sys.stderr = force_utf8_stream(sys.stderr)
    
    logs_path = Path(logs_dir)
    logs_path.mkdir(parents=True, exist_ok=True)
    
    cleanup_old_logs(logs_path)
    
    log_filename = logs_path / f"server_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    fmt_console = logging.Formatter("[%(asctime)s] %(levelname)-8s %(message)s", "%Y-%m-%d %H:%M:%S")
    fmt_file = logging.Formatter("[%(asctime)s] [%(levelname)s] %(name)s: %(message)s", "%Y-%m-%d %H:%M:%S")
    
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt_console)
    ch.setLevel(logging.INFO)
    if hasattr(ch, "stream") and hasattr(ch.stream, "reconfigure"):
        try:
            ch.stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    root.addHandler(ch)

    fh = logging.handlers.RotatingFileHandler(
        str(log_filename), maxBytes=10*1024*1024, backupCount=5, encoding="utf-8"
    )
    fh.setFormatter(fmt_file)
    fh.setLevel(logging.DEBUG)
    root.addHandler(fh)

    # Silence noisy third-party loggers
    for n in ("httpx", "httpcore", "urllib3", "sentence_transformers", 
              "huggingface_hub", "faiss", "uvicorn.access"):
        logging.getLogger(n).setLevel(logging.WARNING)

    return logging.getLogger("chatbot_server"), log_filename


def cleanup_old_logs(logs_dir: Path | str, max_days: int = 7, max_size_mb: int = 100) -> None:
    """Clean up old log files."""
    logs_path = Path(logs_dir)
    if not logs_path.is_dir():
        return

    now = time.time()
    max_age_sec = max_days * 86_400

    log_files = sorted(
        (f for f in logs_path.iterdir() if f.is_file() and (".log" in f.name)),
        key=lambda p: p.stat().st_mtime
    )

    # Step 1: Remove by age
    remaining_files = []
    for fpath in log_files:
        try:
            if now - fpath.stat().st_mtime > max_age_sec:
                fpath.unlink()
            else:
                remaining_files.append(fpath)
        except OSError:
            remaining_files.append(fpath)

    # Step 2: Remove by total size
    def _dir_size_mb() -> float:
        return sum(f.stat().st_size for f in remaining_files) / (1024 * 1024)

    while remaining_files and _dir_size_mb() > max_size_mb:
        try:
            remaining_files[0].unlink()
        except OSError:
            pass
        remaining_files.pop(0)


def get_agent_logger(agent_name: str) -> logging.Logger:
    """Return a logger writing this agent's events to its own file."""

    from ai_assistant.config.paths import get_paths

    safe_name = "".join(char if char.isalnum() or char in "-_" else "_" for char in agent_name.lower())
    agent_logger = logging.getLogger(f"agent.{safe_name}")
    agent_logger.setLevel(logging.DEBUG)
    agent_logger.propagate = True
    if not agent_logger.handlers:
        logs_dir = get_paths().logs_dir
        logs_dir.mkdir(parents=True, exist_ok=True)
        handler = logging.handlers.RotatingFileHandler(
            logs_dir / f"agent_{safe_name}.log",
            maxBytes=5 * 1024 * 1024,
            backupCount=3,
            encoding="utf-8",
        )
        handler.setFormatter(logging.Formatter(
            "[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
            "%Y-%m-%d %H:%M:%S",
        ))
        handler.setLevel(logging.DEBUG)
        agent_logger.addHandler(handler)
    return agent_logger


__all__ = [
    "cleanup_old_logs",
    "force_utf8_stream",
    "get_agent_logger",
    "setup_logging",
]
