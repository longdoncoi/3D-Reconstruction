"""The gates run where the product runs, and the real-dependency check is not
a nightly-only event.

Two structural facts about this package used to be true only in prose:

* the product ships on Windows and the package touches Windows-specific
  filesystem semantics — the ``O_EXCL`` advisory lock, the atomic publish, the
  sharing violations of a file that is being replaced — while CI ran on Linux
  only, so a Windows-only regression could reach main unnoticed;
* ``--resume`` is the one behaviour a fake backend cannot vouch for: the
  framework has to have written a checkpoint that carries its epoch counter
  and optimizer state, and has to accept ``resume=True``. The offline suite
  drives the pipeline with a stub, so the real check lived in a nightly job
  that a pull request never triggered.

Both are now contracts, read from the workflow files (ADR 0012):

* the gate workflow runs on ``windows-latest`` as well as on Linux, and the
  platform-sensitive decisions are mutation-checked on Windows too;
* the real-dependency workflow still interrupts a run and resumes it, and it
  runs for pull requests that touch the package — so the check cannot be
  silently deleted, and cannot silently stop covering the package.
"""

from __future__ import annotations

import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = ROOT / ".github" / "workflows"
GATE_WORKFLOW = WORKFLOWS / "Build-cvtrain-ci.yml"
NIGHTLY_WORKFLOW = WORKFLOWS / "cvtrain-nightly.yml"

#: The platforms the gates have to run on. The product ships on Windows; the
#: Linux runner is kept because it is the cheaper of the two and because the
#: mutation sweep runs there.
REQUIRED_RUNNERS = ("ubuntu-latest", "windows-latest")

#: The assertions the real-dependency job has to keep making about the resume
#: seam. Each one is a behaviour only the *real* framework can show, and the
#: effective-fact assertions are what separate "it resumed" from "it resumed
#: the run we interrupted with the budget that run had" (ADR 0013).
RESUME_ASSERTIONS = (
    "--resume",
    "Resuming yolo11n.onnx",
    "-name 'train-*'",
    'test "$BEFORE" -eq "$AFTER"',
    'RESUME_RUNS="$GITHUB_WORKSPACE/ResumeRuns"',
    "c['epoch'] >= 0",
    "c['train_args']['epochs'] == 5",
    "e['epochs'] == 5",
    "e['epochs_completed'] == 5",
    "e['resume_from'].endswith('last.pt')",
)


def _job_block(text: str, job: str) -> str:
    """The body of one job of a workflow file.

    The workflows are read as text, not as YAML: the package must stay
    importable without PyYAML (the *package* is import-light by design), and
    the structure a guard needs — "which runners does this job's matrix name"
    — is a handful of lines, not a document model.
    """

    lines = text.splitlines()
    start = None
    for index, line in enumerate(lines):
        if line == f"  {job}:":
            start = index
            break
    if start is None:
        return ""
    body: list[str] = []
    for line in lines[start + 1 :]:
        if line.strip() and not line.startswith("    "):
            break
        body.append(line)
    return "\n".join(body)


def _matrix_runners(block: str) -> set[str]:
    """The runners a job's matrix names (``os: [a, b]`` or ``- os: a``)."""

    runners: set[str] = set()
    for line in block.splitlines():
        stripped = line.strip()
        if stripped.startswith("os:"):
            value = stripped.split(":", 1)[1].strip().strip("[]")
            runners.update(part.strip().strip("'\"") for part in value.split(",") if part.strip())
        elif stripped.startswith("- os:"):
            runners.add(stripped.split(":", 1)[1].strip().strip("'\""))
    return runners


@unittest.skipUnless(GATE_WORKFLOW.exists(), f"{GATE_WORKFLOW} not found")
class PlatformCoverageTests(unittest.TestCase):
    """The behavioural gates run on the platform the product ships on."""

    def test_the_gates_run_on_every_required_runner(self) -> None:
        # The *gates* job is checked, not the whole file: the mutation job
        # names the same runners, so a file-wide search would stay green while
        # the behavioural gates stopped running on Windows.
        block = _job_block(GATE_WORKFLOW.read_text(encoding="utf-8"), "gates")
        self.assertTrue(block, "the gate workflow has no 'gates' job")
        runners = _matrix_runners(block)
        for runner in REQUIRED_RUNNERS:
            self.assertIn(runner, runners, f"the gates do not run on {runner}")

    def test_the_platform_jobs_run_the_suite_and_the_coverage_gate(self) -> None:
        block = _job_block(GATE_WORKFLOW.read_text(encoding="utf-8"), "gates")
        self.assertIn("coverage run -m unittest discover", block, "the unit suite is not part of the platform jobs")
        self.assertIn("coverage report", block, "the coverage gate is not part of the platform jobs")

    def test_the_platform_sensitive_decisions_are_mutation_checked_on_windows(self) -> None:
        # The advisory lock, the atomic publish and the retention rules are
        # the decisions whose behaviour depends on the platform; the rest of
        # the sweep inverts platform-independent logic and runs once.
        block = _job_block(GATE_WORKFLOW.read_text(encoding="utf-8"), "mutation")
        self.assertIn("mutation_check.py --files cvtrain/fsio.py", block, "Windows does not re-check the filesystem decisions")
        self.assertIn("windows-latest", _matrix_runners(block), "the mutation sweep does not run on Windows")


@unittest.skipUnless(NIGHTLY_WORKFLOW.exists(), f"{NIGHTLY_WORKFLOW} not found")
class RealDependencyCheckTests(unittest.TestCase):
    """The real-framework check stays, and it covers the pull request."""

    def test_the_resume_seam_is_still_exercised_against_the_real_framework(self) -> None:
        text = NIGHTLY_WORKFLOW.read_text(encoding="utf-8")
        for assertion in RESUME_ASSERTIONS:
            self.assertIn(assertion, text, f"the real-dependency job no longer asserts {assertion!r}")

    def test_the_resume_check_uses_its_own_runs_directory(self) -> None:
        # The earlier one-epoch runs leave *completed* ``last.pt`` files under
        # the default ``runs/`` tree. A find over that tree would pick one of
        # them, kill the new run before it wrote anything, and then "resume"
        # an already-completed run — the fail-open case ADR 0013 closes.
        text = NIGHTLY_WORKFLOW.read_text(encoding="utf-8")
        self.assertIn('RESUME_RUNS="$GITHUB_WORKSPACE/ResumeRuns"', text)
        self.assertIn('--runs-dir "$RESUME_RUNS"', text, "the resume run is not isolated from the earlier completed runs")

    def test_the_resume_check_asserts_the_effective_facts(self) -> None:
        # A resume that continues *some* run proves less than a resume that
        # continues the interrupted run with its own budget. The manifest is
        # the durable record: the budget, the completed count and the source
        # checkpoint must all describe the continued experiment.
        text = NIGHTLY_WORKFLOW.read_text(encoding="utf-8")
        self.assertIn("e['epochs'] == 5", text, "the resumed run's effective budget is not asserted")
        self.assertIn("e['epochs_completed'] == 5", text, "the resumed run's completed count is not asserted")
        self.assertIn("e['resume_from'].endswith('last.pt')", text, "the resumed run's source checkpoint is not asserted")

    def test_the_run_is_interrupted_before_it_is_resumed(self) -> None:
        # A resume check that never interrupts anything proves nothing: the
        # checkpoint has to come from a run that was actually killed.
        text = NIGHTLY_WORKFLOW.read_text(encoding="utf-8")
        self.assertIn("kill -9", text, "the real-dependency job no longer interrupts a run")
        self.assertIn("last.pt", text, "the real-dependency job no longer looks for the checkpoint the framework wrote")

    def test_the_real_dependency_check_runs_on_the_product_platform(self) -> None:
        # The export stack is native code (torch / ONNX Runtime / onnxslim's
        # onnx stack), and the product ships on Windows. A real-dependency
        # check that only runs on Linux never installs, loads or exports with
        # the wheels the users get, so the Windows job is part of the contract
        # (ADR 0014), not a convenience.
        block = _job_block(NIGHTLY_WORKFLOW.read_text(encoding="utf-8"), "windows-train-smoke")
        self.assertTrue(block, "the real-dependency workflow has no Windows job")
        self.assertIn("runs-on: windows-latest", block)
        self.assertIn("--det", block, "the Windows job does not train a model")
        self.assertIn("--verify-manifest", block, "the Windows job does not verify the manifest it wrote")

    def test_the_check_runs_for_pull_requests_touching_the_package(self) -> None:
        text = NIGHTLY_WORKFLOW.read_text(encoding="utf-8")
        self.assertIn("pull_request:", text, "the real-dependency check only runs on a schedule")
        self.assertIn("AIComputerVision/**", text, "the real-dependency check is not triggered by the package")


class WorkflowPresenceTests(unittest.TestCase):
    """A moved workflow must not switch the guard off silently."""

    def test_the_workflows_are_present_when_the_package_is(self) -> None:
        package = ROOT / "AIComputerVision"
        if not package.exists():
            self.skipTest("Python-only checkout: there is no workflow to check")
        if not WORKFLOWS.exists():
            self.skipTest("Python-only checkout: there are no workflows")
        self.assertTrue(GATE_WORKFLOW.exists(), f"{GATE_WORKFLOW} moved; the platform gates are gone")
        self.assertTrue(NIGHTLY_WORKFLOW.exists(), f"{NIGHTLY_WORKFLOW} moved; the real-dependency check is gone")


if __name__ == "__main__":
    unittest.main()
