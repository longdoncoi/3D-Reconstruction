"""Cross-language contract for the chat-attachment directory (ADR 0035).

The desktop client seals the attachment it stores and writes it into the
platform's upload directory (``AppConfig::uploadDir``). The AI service has
to open the very same file, and it may not open an arbitrary client-named
path — so it has to know that directory. The two sides live in two
languages, in two repositories-worth of code, with no shared header, and a
mismatch would surface as "the assistant cannot see the image".

The literals are therefore read out of the C++ implementation and compared
with the Python ones, so a change on either side fails the side that
changed. This runs in the Python CI, which triggers on the C++ files as well
— so no cross-language trigger is needed (the pattern ADR 0032 established
for the at-rest envelope).
"""
from __future__ import annotations

import re
import unittest
from pathlib import Path

from ai_assistant.config.paths import PRODUCT_DATA_DIR_NAME, UPLOAD_DIR_NAME

ROOT = Path(__file__).resolve().parents[3]
CPP_SOURCE = ROOT / "src" / "core" / "AppConfig.cpp"


class UploadDirContractTests(unittest.TestCase):
    """The two implementations must name the same directory."""

    @classmethod
    def setUpClass(cls) -> None:
        if not CPP_SOURCE.exists():
            # A contract that silently skips is worse than no contract: the
            # gate would report green while checking nothing. The C++ side is
            # part of this repository, so its absence is a failure.
            cls.fail(f"AppConfig.cpp not found at {CPP_SOURCE}")
        cls.source = CPP_SOURCE.read_text(encoding="utf-8")

    def test_the_upload_directory_name_matches(self) -> None:
        """``AppConfig::uploadDir`` is ``<data root>/Upload``."""
        if '"/Upload"' not in self.source:
            self.fail('the "/Upload" literal not found in AppConfig.cpp')
        self.assertEqual("Upload", UPLOAD_DIR_NAME)

    def test_the_product_data_directory_name_matches(self) -> None:
        """The installed deployment puts the data root under the product
        folder in the per-user application data directory."""
        match = re.search(r'writableLocation\([^)]*\)\s*\+\s*"([^"]+)"', self.source)
        if match is None:
            self.fail("the data-root literal not found in AppConfig.cpp")
        self.assertEqual(f"/{PRODUCT_DATA_DIR_NAME}", match.group(1))
