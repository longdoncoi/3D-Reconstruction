"""Authenticated encryption for state persisted on local disk.

Task messages, metadata, results, event logs and pending approvals contain
user prompts, tool arguments and model output. They used to reach SQLite and
the pending-action JSON file as plaintext; this module wraps each sensitive
text field in AES-256-GCM before the adapter writes it.

Key management is explicit and fail-closed:

* ``AI_STORAGE_KEY`` (``base64:``/``hex:``/bare URL-safe base64, exactly 32
  bytes) is the production/KMS path. The raw environment value is parsed in
  :mod:`ai_assistant.settings`; this adapter only validates the decoded key.
* ``AI_STORAGE_KEY_FILE`` points at a mounted secret. A missing or invalid
  file fails startup; it never silently generates a replacement.
* Otherwise one random key is generated at
  :func:`ai_assistant.settings.storage_key_location` (outside
  ``APP_DATA_DIR``) with owner-only permissions on POSIX and the account ACL
  on Windows, then reused.

Format: ``enc:v1:<url-safe base64(nonce || ciphertext)>``. The 96-bit nonce is
random per field and the field name is included as associated data, so a
ciphertext moved to a different column fails authentication. Legacy plaintext
values are returned unchanged by :meth:`AesGcmFieldCipher.open`; the stores
re-seal them during their schema migration.

Whole binary artifacts (the RAG index, the chunk corpus, the BM25 model) use
the same AEAD through :meth:`AesGcmFieldCipher.seal_bytes`, which keeps the
marker but stores ``nonce || ciphertext`` raw instead of base64 — base64
would inflate a multi-megabyte artifact by a third for nothing. That form has
no legacy escape hatch: an unsealed file is a miss, never a value to load.

Changing the key is a *re-key*, not a fresh start: :func:`reseal_text` opens
every value with the old cipher and seals it again with the new one, bound to
the same context. The stores call it inside a transaction, so a value the old
key cannot open aborts the whole rotation and leaves the state untouched
(ADR 0030).
"""
from __future__ import annotations

import base64
import os
import secrets
from collections.abc import Mapping
from pathlib import Path

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from ai_assistant.ports import (
    FieldCipher,
    StorageCryptoError,
    StorageDecryptionError,
    StorageKeyError,
)
from ai_assistant.settings import storage_key, storage_key_location

#: The exception types and the protocol live in ``ports`` (the port layer owns
#: the contract, this adapter implements it); they are re-exported here so
#: the persistence package has a single import site for them.
__all__ = [
    "SEALED_BINARY_PREFIX",
    "SEALED_PREFIX",
    "AesGcmFieldCipher",
    "FieldCipher",
    "StorageCryptoError",
    "StorageDecryptionError",
    "StorageKeyError",
    "cipher_from_key_file",
    "cipher_from_key_text",
    "is_sealed",
    "is_sealed_bytes",
    "load_storage_cipher",
    "reseal_text",
]

SEALED_PREFIX = "enc:v1:"
#: The binary envelope keeps the same marker, but stores ``nonce || ciphertext``
#: raw instead of base64: the RAG cache artifacts are megabytes of binary, and
#: base64 would inflate them by a third for no gain.
SEALED_BINARY_PREFIX = b"enc:v1:"
KEY_BYTES = 32
_NONCE_BYTES = 12
_AAD_PREFIX = b"ai_assistant.local_state.v1:"


def is_sealed(value: str) -> bool:
    """Return whether ``value`` starts with the versioned sealed envelope."""
    return value.startswith(SEALED_PREFIX)


def is_sealed_bytes(value: bytes) -> bool:
    """Return whether ``value`` starts with the versioned binary envelope."""
    return value.startswith(SEALED_BINARY_PREFIX)


class AesGcmFieldCipher:
    """AES-256-GCM field cipher with a context-bound associated data value."""

    def __init__(self, key: bytes) -> None:
        if len(key) != KEY_BYTES:
            message = f"at-rest key must decode to exactly {KEY_BYTES} bytes"
            raise StorageKeyError(message)
        self._aead = AESGCM(key)

    def seal(self, value: str, *, context: str) -> str:
        nonce = os.urandom(_NONCE_BYTES)
        ciphertext = self._aead.encrypt(nonce, value.encode("utf-8"), _associated_data(context))
        return SEALED_PREFIX + base64.urlsafe_b64encode(nonce + ciphertext).decode("ascii")

    def open(self, value: str, *, context: str) -> str:
        if not is_sealed(value):
            return value
        try:
            raw = base64.b64decode(
                value[len(SEALED_PREFIX):].encode("ascii"),
                altchars=b"-_",
                validate=True,
            )
        except ValueError as error:
            message = f"{context} has an invalid encrypted envelope"
            raise StorageDecryptionError(message) from error
        if len(raw) <= _NONCE_BYTES:
            message = f"{context} has an invalid encrypted envelope"
            raise StorageDecryptionError(message)
        try:
            plaintext = self._aead.decrypt(
                raw[:_NONCE_BYTES], raw[_NONCE_BYTES:], _associated_data(context),
            )
        except (InvalidTag, ValueError) as error:
            # Do not echo the ciphertext or key material into the exception:
            # the message is logged by callers and may reach an operator UI.
            message = f"{context} is encrypted with a different key or was tampered with"
            raise StorageDecryptionError(message) from error
        try:
            return plaintext.decode("utf-8")
        except UnicodeDecodeError as error:
            message = f"{context} does not decode as UTF-8"
            raise StorageDecryptionError(message) from error

    def is_sealed(self, value: str) -> bool:
        return is_sealed(value)

    def seal_bytes(self, value: bytes, *, context: str) -> bytes:
        """Return the binary envelope: marker, nonce and ciphertext, raw.

        The text envelope is base64 because it has to survive a JSON or INI
        round trip; the artifacts this seals (RAG index, chunk corpus, BM25
        model) are binary files, so base64 would only inflate them.
        """
        nonce = os.urandom(_NONCE_BYTES)
        ciphertext = self._aead.encrypt(nonce, value, _associated_data(context))
        return SEALED_BINARY_PREFIX + nonce + ciphertext

    def open_bytes(self, value: bytes, *, context: str) -> bytes:
        """Return the plaintext bytes; refuse anything that is not sealed.

        Unlike the text form there is no "legacy plaintext" escape hatch: the
        binary artifacts are whole files, so an unsealed file is either a
        cache from before encryption was enabled or a planted file. Both are
        a miss, never a value to unpickle.
        """
        if not is_sealed_bytes(value):
            message = f"{context} is not a sealed envelope"
            raise StorageDecryptionError(message)
        raw = value[len(SEALED_BINARY_PREFIX):]
        if len(raw) <= _NONCE_BYTES:
            message = f"{context} has an invalid encrypted envelope"
            raise StorageDecryptionError(message)
        try:
            return self._aead.decrypt(
                raw[:_NONCE_BYTES], raw[_NONCE_BYTES:], _associated_data(context),
            )
        except (InvalidTag, ValueError) as error:
            message = f"{context} is encrypted with a different key or was tampered with"
            raise StorageDecryptionError(message) from error


def load_storage_cipher(env: Mapping[str, str] | None = None) -> AesGcmFieldCipher:
    """Load the configured key, generating the local key exactly once.

    The function is intentionally fail-closed: an explicitly configured file
    that does not exist, a malformed environment key or a key of the wrong
    length aborts composition instead of starting with unreadable state or a
    fresh key that would orphan every existing row.
    """
    configured = storage_key(env)
    if configured:
        return AesGcmFieldCipher(_decode_key(configured, source="AI_STORAGE_KEY"))

    path, explicit = storage_key_location(env)
    if path.exists():
        return AesGcmFieldCipher(_read_key_file(path))
    if explicit:
        message = f"AI_STORAGE_KEY_FILE does not exist: {path}"
        raise StorageKeyError(message)

    key = secrets.token_bytes(KEY_BYTES)
    if _create_key_file(path, key):
        return AesGcmFieldCipher(key)
    # Another process won the O_EXCL race; its key is the one on disk.
    return AesGcmFieldCipher(_read_key_file(path))


def cipher_from_key_text(value: str, *, source: str) -> AesGcmFieldCipher:
    """Return the cipher for a key written in one of the accepted formats.

    ``base64:<url-safe>``, ``hex:<digits>`` or a bare URL-safe base64 value
    that decodes to exactly 32 bytes — the formats ``AI_STORAGE_KEY``
    accepts. Operator tooling (the key-rotation script) parses keys through
    this function so the accepted formats can never drift apart from the ones
    the application reads.
    """
    return AesGcmFieldCipher(_decode_key(value, source=source))


def cipher_from_key_file(path: Path) -> AesGcmFieldCipher:
    """Return the cipher for the key stored in ``path``.

    The file holds the key in one of the formats
    :func:`cipher_from_key_text` accepts; an unreadable file or a malformed
    key aborts instead of falling back to a generated key.
    """
    return AesGcmFieldCipher(_read_key_file(path))


def reseal_text(
    old: FieldCipher,
    new: FieldCipher,
    value: str,
    *,
    context: str,
) -> tuple[str, bool]:
    """Return ``(value, changed)`` — the same content sealed under ``new``.

    The value is opened with ``old`` and sealed again with ``new``, bound to
    the same ``context`` (the field name is the AEAD associated data, so the
    context must not change). A value the old key cannot open is **not**
    rewritten: it is either already bound to the new key — a repeated
    re-key, so the value is left untouched instead of being sealed twice —
    or it is bound to a third key, and then ``new.open`` raises and the
    caller aborts (fail closed, ADR 0030).

    Legacy plaintext is not bound to any key: it is opened unchanged by
    ``old.open`` and sealed under ``new``, so a re-key leaves no readable
    value behind.
    """
    try:
        plaintext = old.open(value, context=context)
    except StorageDecryptionError:
        new.open(value, context=context)
        return value, False
    return new.seal(plaintext, context=context), True


def _associated_data(context: str) -> bytes:
    return _AAD_PREFIX + context.encode("utf-8")


def _decode_key(value: str, *, source: str) -> bytes:
    text = value.strip()
    if text.startswith("hex:"):
        try:
            key = bytes.fromhex(text[4:])
        except ValueError as error:
            message = f"{source} is not valid hexadecimal"
            raise StorageKeyError(message) from error
    else:
        if text.startswith("base64:"):
            text = text[7:]
        try:
            key = base64.b64decode(
                text.encode("ascii") + b"=" * (-len(text) % 4),
                altchars=b"-_",
                validate=True,
            )
        except ValueError as error:
            message = f"{source} is not valid base64"
            raise StorageKeyError(message) from error
    if len(key) != KEY_BYTES:
        message = f"{source} must decode to exactly {KEY_BYTES} bytes"
        raise StorageKeyError(message)
    return key


def _read_key_file(path: Path) -> bytes:
    try:
        text = path.read_text(encoding="ascii")
    except (OSError, ValueError) as error:
        message = f"cannot read at-rest key file {path}: {error}"
        raise StorageKeyError(message) from error
    return _decode_key(text, source=str(path))


def _create_key_file(path: Path, key: bytes) -> bool:
    """Create ``path`` with owner-only permissions; return False on a race."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        return False
    payload = f"base64:{base64.urlsafe_b64encode(key).decode('ascii')}\n".encode("ascii")
    try:
        os.write(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    _fsync_directory(path.parent)
    return True


def _fsync_directory(directory: Path) -> None:
    """Persist the directory entry so a crash cannot drop the key file.

    ``fsync`` on the file only guarantees that the *data* reached the disk. The
    directory entry that makes the key file visible is separate metadata, and
    without flushing it a power loss can leave the file missing even though
    its contents were written. The next start would then generate a *fresh*
    key and orphan every row the file was meant to protect — the exact
    failure the fail-closed key policy exists to prevent.

    Windows has no directory fsync (opening a directory as a file handle is
    refused), and NTFS orders its own metadata, so the step is a no-op there.
    """
    try:
        descriptor = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        # The entry is already written; a filesystem that refuses to flush the
        # directory must not stop the key from being created.
        pass
    finally:
        os.close(descriptor)
