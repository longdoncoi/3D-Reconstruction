"""Hardware and memory management utilities."""
from __future__ import annotations

import atexit
import gc
import sys
from typing import Any

_llama_cpp_cleanup_registered = False
_llama_cpp_null_streams: list[Any] = []


def _remember_llama_cpp_null_streams() -> None:
    """Retain backend streams when a test reloads the optional module."""
    utils = sys.modules.get("llama_cpp._utils")
    if utils is None:
        return
    for name in ("outnull_file", "errnull_file"):
        stream = getattr(utils, name, None)
        if stream is not None and all(known is not stream for known in _llama_cpp_null_streams):
            _llama_cpp_null_streams.append(stream)


def close_llama_cpp_null_streams() -> None:
    """Close llama-cpp-python's Windows ``nul`` streams when the process exits.

    ``llama_cpp._utils`` opens these two module-level streams to silence the
    native library, but the installed package does not close them.  Looking up
    an already-imported module avoids importing an optional backend merely to
    clean it up.
    """
    _remember_llama_cpp_null_streams()
    for stream in _llama_cpp_null_streams:
        if stream is not None and not stream.closed:
            try:
                stream.close()
            except OSError:
                pass


def register_llama_cpp_cleanup() -> None:
    """Register the one process-exit cleanup for llama-cpp-python streams."""
    global _llama_cpp_cleanup_registered
    _remember_llama_cpp_null_streams()
    if not _llama_cpp_cleanup_registered:
        atexit.register(close_llama_cpp_null_streams)
        _llama_cpp_cleanup_registered = True


def release_ml_memory() -> None:
    """Force garbage collection and clear GPU caches."""
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
