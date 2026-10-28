"""Windows proactor pipe-reset shim, kept out of the composition root.

The server runs on the asyncio proactor loop on Windows. When a client drops
the connection abruptly, the loop's pipe-shutdown callback raises
``ConnectionResetError`` *after* the peer is gone; the traceback is noise and
used to be logged as a server error. The shim wraps
``_ProactorBasePipeTransport._call_connection_lost`` and swallows exactly that
exception, leaving every other failure visible.

The module is standard-library only: the entry point imports it at module load,
so it must not pull asyncio internals (the proactor module exists on POSIX too,
but the shim is only installed on Windows).
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any


def silence_connection_reset(transport_cls: type) -> None:
    """Wrap ``_call_connection_lost`` so an abrupt client reset is not an error.

    Windows' proactor event loop raises ``ConnectionResetError`` from the pipe
    shutdown callback after a client disconnects abruptly; the traceback is
    noise, so the wrapper swallows exactly that exception.
    """
    original = transport_cls._call_connection_lost  # type: ignore[attr-defined]

    def _silenced(self: Any, exc: Exception) -> None:
        try:
            original(self, exc)
        except ConnectionResetError:
            pass

    transport_cls._call_connection_lost = _silenced  # type: ignore[attr-defined]


def patch_proactor_transport(*, platform: str, silence: Callable[[type], None]) -> None:
    """Install the Windows proactor pipe-reset shim (no-op on POSIX).

    ``platform`` and ``silence`` are injected instead of read/imported here so
    the composition root keeps the single decision point: the tests patch
    ``server.sys.platform`` and ``server._silence_connection_reset`` and drive
    the real branch through ``server._patch_proactor_transport``.
    """
    if platform != "win32":
        return
    from asyncio.proactor_events import _ProactorBasePipeTransport

    silence(_ProactorBasePipeTransport)


__all__ = ["patch_proactor_transport", "silence_connection_reset"]
