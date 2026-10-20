"""The locks must still satisfy the requirement files they were compiled from.

CI installs ``requirements-test.lock`` with ``--require-hashes`` and
production installs ``requirements.lock``, so a requirement file edited
without regenerating its lock would silently ship the wrong dependency set
(e.g. a bumped CVE floor that never reaches the install). These cases parse
the direct requirements with ``packaging`` and require each one to appear in
the matching lock with a version inside the declared range (ADR 0023).
"""
from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from packaging.version import Version

_ROOT = Path(__file__).resolve().parents[2]
_SBOM_SCRIPT = _ROOT / "scripts" / "generate_sbom.py"


def _load_sbom_script():
    spec = importlib.util.spec_from_file_location("generate_sbom_sync", _SBOM_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _direct_requirements(path: Path) -> list[Requirement]:
    """Requirement objects from a file, skipping comments and options."""
    requirements: list[Requirement] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or line.startswith("-"):
            continue
        requirements.append(Requirement(line))
    return requirements


def _violations(requirements_path: Path, lock_path: Path, parse_lock) -> list[str]:
    """Every direct requirement must be satisfied by a pin in its own lock."""
    pins: dict[str, list[str]] = {}
    for component in parse_lock(lock_path):
        pins.setdefault(canonicalize_name(component["name"]), []).append(
            component["version"]
        )
    messages: list[str] = []
    for requirement in _direct_requirements(requirements_path):
        versions = pins.get(canonicalize_name(requirement.name), [])
        if not versions:
            messages.append(f"{requirement.name} is not pinned in {lock_path.name}")
            continue
        if not any(
            requirement.specifier.contains(Version(version), prereleases=True)
            for version in versions
        ):
            messages.append(
                f"{requirement.name}{requirement.specifier} matches no pin in "
                f"{lock_path.name} ({', '.join(versions)}) — regenerate the lock"
            )
    return messages


class CheckerTests(unittest.TestCase):
    """The checker itself must fail on drift, not only pass on clean input."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.script = _load_sbom_script()

    def _pair(self, requirements: str, lock: str) -> list[str]:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            req_path = base / "requirements.txt"
            lock_path = base / "requirements.lock"
            req_path.write_text(requirements, encoding="utf-8")
            lock_path.write_text(lock, encoding="utf-8")
            return _violations(req_path, lock_path, self.script.parse_lock)

    def test_a_satisfied_pair_reports_nothing(self) -> None:
        messages = self._pair("fastapi>=0.111.0\n", "fastapi==0.136.1\n")
        self.assertEqual(messages, [])

    def test_a_range_the_lock_no_longer_satisfies_is_flagged(self) -> None:
        messages = self._pair("fastapi>=99.0.0\n", "fastapi==0.136.1\n")
        self.assertEqual(len(messages), 1)
        self.assertIn("fastapi>=99.0.0", messages[0])
        self.assertIn("regenerate the lock", messages[0])

    def test_a_requirement_missing_from_the_lock_is_flagged(self) -> None:
        messages = self._pair("fastapi>=0.111.0\n", "pydantic==2.14.1\n")
        self.assertEqual(messages, ["fastapi is not pinned in requirements.lock"])


class TrackedFileSyncTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.script = _load_sbom_script()

    def test_runtime_requirements_are_satisfied_by_the_runtime_lock(self) -> None:
        messages = _violations(
            _ROOT / "requirements.txt", _ROOT / "requirements.lock",
            self.script.parse_lock,
        )
        self.assertEqual(messages, [])

    def test_test_requirements_are_satisfied_by_the_test_lock(self) -> None:
        messages = _violations(
            _ROOT / "requirements-test.txt",
            _ROOT / "requirements-test.lock",
            self.script.parse_lock,
        )
        self.assertEqual(messages, [])

    def test_docker_requirements_are_satisfied_by_the_docker_lock(self) -> None:
        """The container installs its own lock (ADR 0032). It has to satisfy
        its own requirement file, exactly like the other two pairs."""
        messages = _violations(
            _ROOT / "requirements-docker.txt",
            _ROOT / "requirements-docker.lock",
            self.script.parse_lock,
        )
        self.assertEqual(messages, [])

    def test_the_docker_requirements_mirror_the_runtime_ones(self) -> None:
        """The container's dependency set is the runtime's, minus the GPU
        branch. A package that appears in only one of the two lists is drift:
        the container would ship a dependency the desktop app does not have,
        or miss one it does."""
        runtime = {
            canonicalize_name(requirement.name)
            for requirement in _direct_requirements(_ROOT / "requirements.txt")
        }
        docker = {
            canonicalize_name(requirement.name)
            for requirement in _direct_requirements(_ROOT / "requirements-docker.txt")
        }
        self.assertEqual(
            docker - runtime,
            set(),
            "the container requires a package the runtime does not",
        )
        # The runtime file declares faiss twice, once per platform marker;
        # the container declares the CPU one. Every other runtime package has
        # to be in the container as well.
        self.assertEqual(
            runtime - docker - {"faiss-gpu", "faiss-cpu"},
            set(),
            "the container is missing a runtime dependency",
        )
        self.assertIn("faiss-cpu", docker)

    def test_the_docker_lock_does_not_pull_a_cuda_wheel(self) -> None:
        """The slim CPU image must not download faiss-gpu and its CUDA
        runtime — hundreds of megabytes the container can never use."""
        components = self.script.parse_lock(_ROOT / "requirements-docker.lock")
        names = {canonicalize_name(component["name"]) for component in components}
        self.assertNotIn("faiss-gpu", names)
        self.assertNotIn("nvidia-cublas-cu12", names)
        self.assertIn("faiss-cpu", names)

    def test_both_files_still_declare_direct_requirements(self) -> None:
        # A parser regression that reads nothing would make the two cases
        # above vacuous, so the input files are pinned to be non-trivial too.
        for name in ("requirements.txt", "requirements-test.txt",
                     "requirements-docker.txt"):
            with self.subTest(file=name):
                self.assertGreater(len(_direct_requirements(_ROOT / name)), 10)


if __name__ == "__main__":
    unittest.main()
