"""Cross-language contract for the at-rest envelope (ADR 0032).

The Qt client seals a chat attachment and the AI service opens it. The
envelope layout, the associated data and the nonce/tag sizes therefore have
to agree across two languages, in two repositories-worth of code, with no
shared header.

The existing interop test pins a *captured* vector on each side. A captured
vector is only as good as the paste: regenerating it on one side and
forgetting the other leaves two different vectors, each of which still opens
correctly on its own side — and the break surfaces as "the assistant cannot
see the image" in production.

This file closes that gap from the Python side, where the CI gate runs:

* the **format constants** are read out of the C++ implementation
  (``src/core/AtRestCipher.cpp``) and compared with the Python ones, so a
  change on either side fails the side that changed;
* the **reference envelope** is read out of the C++ test
  (``tests/TestAtRestCipher.cpp``) and compared with the Python test's copy,
  so a vector that was re-pinned on one side only fails here.

Both checks run in the Python CI, which triggers on the C++ files as well —
so no cross-language trigger is needed.
"""
from __future__ import annotations

import re
import unittest
from pathlib import Path

from ai_assistant.adapters.persistence.storage_crypto import (
    SEALED_BINARY_PREFIX,
    AesGcmFieldCipher,
)

ROOT = Path(__file__).resolve().parents[3]
CPP_SOURCE = ROOT / "src" / "core" / "AtRestCipher.cpp"
CPP_TEST = ROOT / "tests" / "TestAtRestCipher.cpp"

#: The tag length of AES-GCM. The C++ side names it ``kTagBytes``; the Python
#: side gets it from the AEAD implementation, so the constant is spelled out
#: here to make the comparison explicit.
GCM_TAG_BYTES = 16

#: The context both sides use for a chat attachment.
ATTACHMENT_CONTEXT = "chat.attachment"

#: The key both sides use for the interop vector: 32 x 0x5A.
INTEROP_KEY = bytes([0x5A]) * 32

#: The plaintext both sides seal in the interop vector.
INTEROP_PLAINTEXT = b"ai-assistant at-rest interop vector"


def _cpp_constant(source: str, name: str) -> str | None:
    """Read a ``const char kX[] = "..."`` literal out of the C++ source."""
    match = re.search(rf'const char {re.escape(name)}\[\]\s*=\s*"([^"]*)"', source)
    return match.group(1) if match else None


def _cpp_int(source: str, name: str) -> int | None:
    """Read a ``constexpr int kX = N;`` literal out of the C++ source."""
    match = re.search(rf"constexpr int {re.escape(name)}\s*=\s*(\d+)", source)
    return int(match.group(1)) if match else None


def _cpp_hex_vector(source: str) -> str | None:
    """Read the hex envelope literal out of the C++ interop test.

    The literal is written as two adjacent C string literals; they are
    concatenated in the order they appear. The *last* occurrence of the test
    name is used: the first one is the declaration in the class header, not
    the definition that carries the vector.
    """
    anchor = source.rfind("thePythonServiceEnvelopeIsReadable")
    if anchor < 0:
        return None
    body = source[anchor:anchor + 2000]
    literals = re.findall(r'"([0-9a-fA-F]{16,})"', body)
    if not literals:
        return None
    return "".join(literals).lower()


class AtRestFormatContractTests(unittest.TestCase):
    """The two implementations must describe the *same* envelope."""

    @classmethod
    def setUpClass(cls) -> None:
        if not CPP_SOURCE.exists():
            # A contract that silently skips is worse than no contract: the
            # gate would report green while checking nothing. The C++ side is
            # part of this repository, so its absence is a failure.
            cls.fail(f"AtRestCipher.cpp not found at {CPP_SOURCE}")
        cls.source = CPP_SOURCE.read_text(encoding="utf-8")

    def test_the_marker_matches(self) -> None:
        expected = _cpp_constant(self.source, "kMarker")
        if expected is None:
            self.fail("kMarker not found in AtRestCipher.cpp")
        self.assertEqual(expected.encode("ascii"), SEALED_BINARY_PREFIX)

    def test_the_associated_data_prefix_matches(self) -> None:
        expected = _cpp_constant(self.source, "kAadPrefix")
        if expected is None:
            self.fail("kAadPrefix not found in AtRestCipher.cpp")
        # The Python side keeps the prefix private; the contract is that both
        # derive the same associated data from the same context.
        from ai_assistant.adapters.persistence import storage_crypto

        self.assertEqual(expected.encode("ascii"), storage_crypto._AAD_PREFIX)

    def test_the_nonce_size_matches(self) -> None:
        expected = _cpp_int(self.source, "kNonceBytes")
        if expected is None:
            self.fail("kNonceBytes not found in AtRestCipher.cpp")
        from ai_assistant.adapters.persistence import storage_crypto

        self.assertEqual(expected, storage_crypto._NONCE_BYTES)

    def test_the_tag_size_matches(self) -> None:
        expected = _cpp_int(self.source, "kTagBytes")
        if expected is None:
            self.fail("kTagBytes not found in AtRestCipher.cpp")
        self.assertEqual(expected, GCM_TAG_BYTES)

    def test_a_sealed_envelope_has_the_documented_length(self) -> None:
        """The Python writer must produce exactly the layout the C++ reader
        expects: marker, nonce, ciphertext, tag."""
        envelope = AesGcmFieldCipher(INTEROP_KEY).seal_bytes(
            INTEROP_PLAINTEXT, context=ATTACHMENT_CONTEXT,
        )
        self.assertTrue(envelope.startswith(SEALED_BINARY_PREFIX))
        self.assertEqual(
            len(envelope),
            len(SEALED_BINARY_PREFIX) + 12 + len(INTEROP_PLAINTEXT) + GCM_TAG_BYTES,
        )


class AtRestVectorContractTests(unittest.TestCase):
    """The captured vector must be the *same* on both sides."""

    @classmethod
    def setUpClass(cls) -> None:
        if not CPP_TEST.exists():
            cls.fail(f"TestAtRestCipher.cpp not found at {CPP_TEST}")
        cls.source = CPP_TEST.read_text(encoding="utf-8")

    def test_the_cpp_vector_is_the_one_the_python_test_pins(self) -> None:
        from tests.unit.test_sealed_attachment_read import _PYTHON_ENVELOPE_HEX

        cpp_vector = _cpp_hex_vector(self.source)
        if cpp_vector is None:
            self.fail("the interop vector literal not found in TestAtRestCipher.cpp")
        self.assertEqual(
            cpp_vector,
            _PYTHON_ENVELOPE_HEX.lower(),
            "the at-rest interop vector was re-pinned on one side only — "
            "regenerate it with TestAtRestCipher::dumpInteropEnvelope and "
            "update both sides in the same commit",
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
