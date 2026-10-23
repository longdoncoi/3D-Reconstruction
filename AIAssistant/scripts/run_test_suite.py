"""Run the Python suite and turn shutdown-time resource warnings into failures."""
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

_RESOURCE_WARNING = re.compile(r"(?:ResourceWarning:|unclosed file)")


def main() -> int:
    """Run coverage under a child interpreter so its shutdown diagnostics are observable."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError):
            pass
    project_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONWARNINGS"] = "error::ResourceWarning"
    source_dir = str(project_root / "src")
    env["PYTHONPATH"] = source_dir + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(
        [
            sys.executable,
            "-W",
            "error::ResourceWarning",
            "-m",
            "coverage",
            "run",
            "-m",
            "unittest",
            "discover",
            "-s",
            "tests",
            "-p",
            "test_*.py",
        ],
        cwd=project_root,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    sys.stdout.write(result.stdout)
    sys.stderr.write(result.stderr)
    if result.returncode:
        return result.returncode
    if _RESOURCE_WARNING.search(result.stdout) or _RESOURCE_WARNING.search(result.stderr):
        print("STRICT WARNING GATE FAIL: ResourceWarning was emitted by the test process.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
