"""Import-order regression: the supervisor must not silently disable A2A.

``orchestration/supervisor`` imports ``adapters.a2a_protocol`` inside a
module-scope ``try/except`` (graceful no-op when optional deps are missing).
The adapter must therefore never import ``orchestration.supervisor`` at module
level: doing so creates an import cycle that makes the ``try/except`` swallow
an ``ImportError`` and install the A2A no-op stubs even though the adapter is
fully available. This was observed on the real ``StartChatbotServer.py`` import
order, after which ``supervisor.delegate()`` never consults the remote registry.

The probe runs in a subprocess so each import order starts from a clean,
deterministic interpreter state (no ``sys.modules`` pollution across tests).
"""
from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

_SRC = Path(__file__).resolve().parents[2] / "src"


class A2AImportOrderTest(unittest.TestCase):
    def test_a2a_stays_enabled_in_both_import_orders(self) -> None:
        script = f"""\
import sys
sys.path.insert(0, {str(_SRC)!r})

# Order A: StartChatbotServer order -- a2a_protocol before supervisor.
import ai_assistant.adapters.a2a_protocol  # noqa: F401
import ai_assistant.orchestration.supervisor as supervisor_a
print("ORDER_A", supervisor_a._A2A_IMPORT_OK)

# Order B: anything that imports supervisor first (other entry points, tests).
for mod in list(sys.modules):
    if mod.startswith("ai_assistant"):
        del sys.modules[mod]
import ai_assistant.orchestration.supervisor as supervisor_b  # noqa: F401
print("ORDER_B", supervisor_b._A2A_IMPORT_OK)
"""
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(
            result.returncode,
            0,
            msg=f"probe failed:\nstdout={result.stdout}\nstderr={result.stderr}",
        )
        flags = dict(
            line.split()
            for line in result.stdout.splitlines()
            if line.startswith("ORDER_")
        )
        self.assertEqual(flags.get("ORDER_A"), "True", msg=result.stdout)
        self.assertEqual(flags.get("ORDER_B"), "True", msg=result.stdout)


if __name__ == "__main__":
    unittest.main()
