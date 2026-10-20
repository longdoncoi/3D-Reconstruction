"""Line-coverage gate for the C++ guard and service layer (OpenCppCoverage).

The C++ side of the repository had no coverage measurement at all: the
architecture guard, the mutation check and the CTest suite proved behaviour,
but nothing prevented an untested branch from being added to the files that
every plugin depends on. This script runs the owning tests under
OpenCppCoverage, merges the per-test cobertura reports and enforces the
per-file floors below.

Scope: every translation unit of ``src/`` that a test can reach.

* The kernel/guard layer (``src/core``) and the composition root
  (``src/app``) — the files every plugin transitively depends on.
* The service layer (``src/services``) and the shared utility layer
  (``src/utils``) — the behaviour the application is made of.
* The plugin layer (``src/modules``) is *visible* (the plugin DLLs are
  instrumented and loaded by the smoke test) even though the shipped plugins
  are GUI shells: what is not executed is printed as "coverage debt" on every
  run instead of disappearing from the report.

Two gates beyond the per-file floors:

1. **Completeness** — a translation unit that is missing from the report
   fails the run.  A file can only go missing when it is not compiled into
   an instrumented module (or when the unity build inlined it away), and
   both are architectural drift, not a coverage detail.
2. **Debt visibility** — the files with 0 % are listed on every run with
   their line count, so the remaining work is named, not implied.

Header only templates (``ServiceRegistry.h``, ``MailThreadGuard.h``) compile inline
into their callers under ``/O2 /Ob2`` and cannot be mapped line-by-line by the
tool, so they are pinned by ``tests/mutation_check.py`` instead (null
registration, GUI-thread warning) — see tests/README.md for the split.

The floors are a ratchet, not an aspiration: they equal the measured value of
the current suite (rounded down), so any new uncovered line in these files
fails the gate. Raise a floor when its file reaches the next integer.

Usage (from the repository root, PATH prepared like the CTest step):

    python tests/coverage_gate.py            # enforce the floors
    python tests/coverage_gate.py --measure  # just print the numbers

CI runs the enforcing form right after the CTest step in Build-C++-ci.yml
(OpenCppCoverage is installed by the workflow there; locally the tool is
discovered through OPENCPPCOVERAGE_EXE, PATH or the known install paths).
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent

# One row per test executable that owns part of the gate scope.
# (ctest name, exe file name, extra environment)
#
# TestServiceRegistry is deliberately absent: it only exercises the
# header-only ServiceRegistry.h (no scope .cpp) and can finish without loading
# AppCore.dll, so OpenCppCoverage has nothing to export. Its null-registration
# decision is pinned by mutation_check.py instead.
TESTS: tuple[tuple[str, str, dict[str, str]], ...] = (
    ("TestPluginDependencyResolver", "TestPluginDependencyResolver.exe", {}),
    ("TestPluginHost", "TestPluginHost.exe", {}),
    # Loads the real plugin DLLs through the default QPluginLoader path —
    # the only test covering PluginHost::loadWithQPluginLoader.
    ("TestPluginSmoke", "TestPluginSmoke.exe", {}),
    ("TestAppShell", "TestAppShell.exe", {"QT_QPA_PLATFORM": "offscreen"}),
    ("TestUserManagerCrypto", "TestUserManagerCrypto.exe", {}),
    ("TestMailCredentialLifecycle", "TestMailCredentialLifecycle.exe", {}),
    ("TestChatSessionManager", "TestChatSessionManager.exe", {}),
    # The at-rest cipher (AES-256-GCM envelope shared with the Python service).
    ("TestAtRestCipher", "TestAtRestCipher.exe", {}),
    ("TestSettingsService", "TestSettingsService.exe", {}),
    ("TestShellService", "TestShellService.exe", {}),
    ("TestLanguageManager", "TestLanguageManager.exe", {}),
    ("TestAppConfig", "TestAppConfig.exe", {}),
    ("TestAIRequestQueue", "TestAIRequestQueue.exe", {}),
    # AppCore-owned worker lifetime/cancellation (owns a private thread pool).
    ("TestAIModelLoadService", "TestAIModelLoadService.exe", {}),
    # ── Service layer ────────────────────────────────────────────────────────
    # The rows above only exercise the kernel/guard layer.  The rows below own
    # the *service* layer: without them the service sources were reported with
    # 0 % (their lines are only reachable through a test that links the
    # concrete service translation units).  Each of them is a cheap test that
    # already exists in tests/CMakeLists.txt — the gate simply has to run it
    # under the coverage exporter.
    ("TestSignalBus", "TestSignalBus.exe", {"QT_QPA_PLATFORM": "offscreen"}),
    ("TestAgentActions", "TestAgentActions.exe", {"QT_QPA_PLATFORM": "offscreen"}),
    ("TestAIAssistant", "TestAIAssistant.exe", {"QT_QPA_PLATFORM": "offscreen"}),
    ("TestAIYoloPostProcessor", "TestAIYoloPostProcessor.exe", {}),
    # Mail: MIME parsing (the largest service TU) and the SMTP outbox adapter.
    ("TestMailMimeParser", "TestMailMimeParser.exe", {}),
    ("TestSmtpOutbox", "TestSmtpOutbox.exe", {"QT_QPA_PLATFORM": "offscreen"}),
    # Reconstruction: the pipeline family (16 translation units compiled into
    # the test executable) plus the two parsers/filters with their own tests.
    ("TestReconstructionPipeline", "TestReconstructionPipeline.exe",
     {"QT_QPA_PLATFORM": "offscreen"}),
    ("TestCameraParamsParser", "TestCameraParamsParser.exe", {}),
    ("TestPointCloudFilter", "TestPointCloudFilter.exe",
     {"QT_QPA_PLATFORM": "offscreen"}),
    # Scene: the DICOM crosshair math (pure header + TU, no VTK needed) and
    # the crosshair overlay manager (VTK, no GL context required).
    ("TestCrosshairGeometry", "TestCrosshairGeometry.exe", {}),
    ("TestCrosshairManager", "TestCrosshairManager.exe", {}),
    # Viewer: the file → VTK actor loaders for 2-D images and 3-D models.
    ("TestViewerLoaders", "TestViewerLoaders.exe", {}),
    # Plugin helpers that carry real logic (mail preview, inbox row, attachment
    # preview, prediction logger).
    ("TestPluginHelpers", "TestPluginHelpers.exe", {"QT_QPA_PLATFORM": "offscreen"}),
    # Shared utility layer (src/utils/**) — the plugin-facing helpers.
    ("TestUtilsLayer", "TestUtilsLayer.exe", {"QT_QPA_PLATFORM": "offscreen"}),
)

# Measured floors: repository-relative path -> minimum line coverage in %.
# A file listed here must appear in the merged report; the measurement mode
# prints candidates for files that are not listed yet. The number next to each
# entry is the floor; the comment records the measurement it was ratcheted
# from (2026-10, MSVC 2022 / VS 17.12, Release). Raise a floor when its file
# reaches the next integer — never lower one to make a change pass.
FLOORS: dict[str, float] = {
    # ── Kernel + guard layer: the files every plugin transitively depends on.
    "src/core/pluginhost.cpp": 92,               # measured 92.0
    "src/core/plugindependencyresolver.cpp": 93,  # measured 93.3
    "src/core/secretvault.cpp": 92,               # measured 92.9
    "src/core/cryptoutils.cpp": 100,              # measured 100
    "src/core/atrestcipher.cpp": 100,             # measured 100
    "src/core/licensekeyledger.cpp": 100,         # measured 100
    "src/core/otpstore.cpp": 100,                 # measured 100
    "src/core/useraccountstore.cpp": 100,         # measured 100
    "src/core/usersettingsstore.cpp": 97,         # measured 97.3
    "src/core/usermanager.cpp": 100,              # measured 100
    "src/core/languagemanager.cpp": 96,           # measured 96.5
    "src/core/appconfig.cpp": 98,                 # measured 98.8 (79/80)
    "src/core/signalbus.cpp": 100,                # measured 100
    # Composition root + the service files the wiring test owns end-to-end.
    "src/app/appshell.cpp": 100,                  # measured 100
    "src/services/ai/chatsessionmanager.cpp": 100,  # measured 100
    "src/services/ai/chatsessionstore.cpp": 100,    # measured 100
    "src/services/ai/airequestqueue.cpp": 100,      # measured 100
    "src/services/ai/aimodelloadservice.cpp": 100,  # measured 100
    "src/services/ai/aiyolopostprocessor.cpp": 100,  # measured 100
    "src/services/ai/agentactionmanifest.cpp": 45,  # measured 45.7
    "src/services/ai/aiassistant.cpp": 40,          # measured 40.4
    "src/services/ai/aiserverprocesscontroller.cpp": 34,  # measured 34.6
    "src/services/ai/aiservice.cpp": 10,           # measured 10.3
    "src/services/ai/aiprocessor.cpp": 9,          # measured 9.5
    "src/services/mail/mailmimeparser.cpp": 66,     # measured 67.0
    "src/services/mail/mailservice.cpp": 21,       # measured 22.0
    "src/services/mail/smtpoutbox.cpp": 87,        # measured 87.5
    "src/services/reconstruction/cameraparamsparser.cpp": 97,  # measured 97.6
    "src/services/reconstruction/featureextractor.cpp": 100,   # measured 100
    "src/services/reconstruction/featurematcher.cpp": 97,      # measured 97.7
    "src/services/reconstruction/incrementalsfm.cpp": 79,      # measured 79.0
    "src/services/reconstruction/multiviewtriangulator.cpp": 100,  # measured 100
    "src/services/reconstruction/pointcloudfilter.cpp": 66,   # measured 66.4
    "src/services/reconstruction/poseestimator.cpp": 61,      # measured 61.9
    "src/services/reconstruction/reconstructionpipeline.cpp": 44,  # measured 44.2
    "src/services/reconstruction/reconstructionpostprocessor.cpp": 12,  # measured 12.6
    "src/services/reconstruction/reconstructionservice.cpp": 53,  # measured 53.3
    "src/services/reconstruction/reconstructionworker.cpp": 100,  # measured 100
    "src/services/reconstruction/sfmstage.cpp": 100,  # measured 100
    "src/services/reconstruction/tracksfm.cpp": 93,   # measured 93.2
    "src/services/reconstruction/triangulator.cpp": 100,  # measured 100
    "src/services/scene/crosshairgeometry.cpp": 88,  # measured 88.2
    "src/services/scene/crosshairmanager.cpp": 56,   # measured 56.1 (120/214)
    "src/services/scene/sceneservice.cpp": 4,        # measured 4.9
    "src/services/settings/settingsservice.cpp": 100,  # measured 100
    "src/services/shell/shellservice.cpp": 98,    # measured 98.2 (one closing brace)
    "src/services/viewer/dicomloader.cpp": 1,      # measured 1.5
    "src/services/viewer/image2dloader.cpp": 95,    # measured 95.8 (68/71)
    "src/services/viewer/model3dloader.cpp": 46,    # measured 46.9 (30/64)
    "src/services/viewer/viewerservice.cpp": 13,   # measured 13.6
    # Shared utility layer — the helpers every plugin links against.
    "src/utils/abstractribbonpanel.cpp": 100,     # measured 100
    "src/utils/customprogressdialog.cpp": 95,      # measured 95.2
    "src/utils/fileutilities.cpp": 100,           # measured 100 (54/54)
    "src/utils/htmlutilities.cpp": 95,            # measured 95.3
    "src/utils/iconfactory.cpp": 100,             # measured 100
    "src/utils/logger.cpp": 69,                   # measured 69.5
    "src/utils/moderndialog.cpp": 71,             # measured 71.4
    "src/utils/modernmessagebox.cpp": 46,         # measured 46.8
    "src/utils/stylemanager.cpp": 90,             # measured 90.2
    # Plugin layer — the shipped plugins are GUI shells; the floor is what
    # their initialize()/cleanup() cycle covers.  The rest of the plugin code
    # (dialogs, dock widgets) is listed as coverage debt.
    "src/modules/aiassistantplugin/chatbotdockwidget.cpp": 96,      # measured 96.4
    "src/modules/viewerplugin/viewernavigatorui.cpp": 94,           # measured 94.7
    "src/modules/reconstructionplugin/reconstructionribbonui.cpp": 57,  # measured 57.1
    "src/modules/userauthplugin/userauthplugin.cpp": 46,            # measured 46.2
    "src/modules/sendmailplugin/maildockwidget.cpp": 45,            # measured 45.9
    "src/modules/viewerplugin/viewerplugin.cpp": 41,                # measured 41.5
    "src/modules/sendmailplugin/sendmailplugin.cpp": 36,            # measured 36.4
    "src/modules/aiprocessorplugin/aitraindockwidget.cpp": 32,       # measured 32.0
    "src/modules/aiprocessorplugin/aiprocessorplugin.cpp": 24,       # measured 24.6
    "src/modules/reconstructionplugin/reconstructionplugin.cpp": 18, # measured 18.9
    "src/modules/aiassistantplugin/aiassistantplugin.cpp": 16,       # measured 16.9
    "src/modules/viewerplugin/viewerlistui.cpp": 7,                 # measured 7.0
    "src/modules/aiassistantplugin/chatmessagerenderer.cpp": 5,      # measured 5.7
    "src/modules/userauthplugin/themeselectiondialog.cpp": 3,        # measured 3.0
    "src/modules/aiassistantplugin/aiattachmentpreviewfactory.cpp": 100,  # measured 100
    "src/modules/aiprocessorplugin/aipredictionlogger.cpp": 100,           # measured 100
    "src/modules/sendmailplugin/mailinboxitemfactory.cpp": 100,            # measured 100
    "src/modules/sendmailplugin/mailmessageformatter.cpp": 84,              # measured 84.7
}

_TOOL_CANDIDATES = (
    Path(os.environ.get("LOCALAPPDATA", "")) / "Tools/OpenCppCoverage/app/OpenCppCoverage.exe",
    Path("C:/Program Files/OpenCppCoverage/OpenCppCoverage.exe"),
    Path("C:/Program Files (x86)/OpenCppCoverage/OpenCppCoverage.exe"),
    BASE / "tools/OpenCppCoverage/OpenCppCoverage.exe",
)


def find_tool() -> Path:
    override = os.environ.get("OPENCPPCOVERAGE_EXE")
    if override:
        path = Path(override)
        if not path.is_file():
            sys.exit(f"OPENCPPCOVERAGE_EXE is set but does not exist: {path}")
        return path
    on_path = shutil.which("OpenCppCoverage")
    if on_path:
        return Path(on_path)
    for candidate in _TOOL_CANDIDATES:
        if candidate.is_file():
            return candidate
    sys.exit(
        "OpenCppCoverage not found. Install it (choco install opencppcoverage) "
        "or set OPENCPPCOVERAGE_EXE to OpenCppCoverage.exe."
    )


@dataclass
class FileCoverage:
    lines: dict[int, int] = field(default_factory=dict)

    @property
    def valid(self) -> int:
        return len(self.lines)

    @property
    def covered(self) -> int:
        return sum(self.lines.values())

    @property
    def percent(self) -> float:
        return 100.0 * self.covered / self.valid if self.valid else 0.0


# Translation units that cannot be measured — each entry carries its reason.
# The list is deliberately explicit: a new translation unit that nobody can
# measure has to be listed here (with a reason) instead of silently
# disappearing from the report.
_NOT_MEASURABLE = {
    # PCH carriers: the dummy .cpp exists only to hold the precompiled header.
    "src/pch_dummy.cpp",
    "src/pch_light_dummy.cpp",
    # The application entry point and the main window are compiled into the
    # application executable, which no test executes — they are exercised by
    # running the application itself.
    "src/app/main.cpp",
    "src/app/mainwindow.cpp",
    # VTK boilerplate only (vtkStandardNewMacro), no logic of its own.
    "src/services/scene/panstyle.cpp",
}


def _source_translation_units() -> set[str]:
    """Every real .cpp under ``src/``, repository-relative and lower-case."""
    units = {
        path.relative_to(BASE).as_posix().lower()
        for path in (BASE / "src").rglob("*.cpp")
    }
    return units - _NOT_MEASURABLE


def _windows_path(path: Path) -> str:
    return str(path).replace("/", "\\")


def _module_patterns(build_dir: Path, exe_path: Path, include_plugins: bool) -> list[str]:
    """Module patterns the exporter has to instrument for one test run.

    Only the modules that can contribute coverage are instrumented:

    * the test executable itself (some tests compile the translation unit
      under test directly into their own binary),
    * ``AppCore.dll`` — the kernel/guard layer every test imports,
    * the application executable — it carries the service layer, so the
      service sources are part of the report even when no test runs them
      (they show up with 0 hits, which is what makes the debt visible),
    * the plugin DLLs — only for the test that loads them, so the whole
      plugin layer (``src/modules/**``) stops being invisible.

    Instrumenting *everything* in the output directory (the previous
    behaviour) made the first run of a clean build tree pay for ~35 modules
    — tens of minutes before the first report appeared.
    """
    patterns: list[str] = [_windows_path(exe_path)]
    for name in ("AppCore.dll", "3D-Reconstruction.exe"):
        module = build_dir / "Release" / name
        if module.is_file():
            patterns.append(_windows_path(module))
    if include_plugins:
        plugins = build_dir / "plugins" / "Release"
        if plugins.is_dir():
            patterns.append(_windows_path(plugins) + "\\*")
    return patterns


def run_tests(tool: Path, build_dir: Path, out_dir: Path) -> None:
    exe_dir = build_dir / "Release"
    for name, exe, extra_env in TESTS:
        exe_path = exe_dir / exe
        if not exe_path.is_file():
            sys.exit(f"missing test executable: {exe_path} (build the tests first)")
        env = os.environ.copy()
        env.update(extra_env)
        report = out_dir / f"{name}.xml"
        # Never accept a stale report from a previous run if this run fails to
        # produce one.
        report.unlink(missing_ok=True)
        argv = [
            str(tool), "--quiet", "--optimized_build",
            "--sources", _windows_path(BASE / "src"),
            # The plugin layer is exercised in child processes (one per
            # plugin, so a crash cannot take the suite down): without this
            # flag the exporter would only see the parent and the plugin
            # code would measure 0 %.
            "--cover_children",
        ]
        for pattern in _module_patterns(build_dir, exe_path, name == "TestPluginSmoke"):
            argv += ["--modules", pattern]
        argv += [
            "--export_type", f"cobertura:{_windows_path(report)}",
            "--", exe,
        ]
        print(f"  coverage: {name}")
        result = subprocess.run(argv, cwd=exe_dir, env=env, capture_output=True,
                                text=True, errors="replace", timeout=1800,
                                check=False)
        if result.returncode != 0:
            sys.exit(
                f"{name} failed under OpenCppCoverage (exit {result.returncode}):\n"
                f"{result.stdout[-2000:]}\n{result.stderr[-2000:]}"
            )
        # OpenCppCoverage can report success before Windows/AV makes the
        # cobertura file visible (observed intermittently on Win11). Poll for
        # a non-empty, stable file instead of failing the gate spuriously.
        deadline = time.monotonic() + 30.0
        last_size = -1
        while time.monotonic() < deadline:
            if report.is_file():
                size = report.stat().st_size
                if size > 0 and size == last_size:
                    break
                last_size = size
            time.sleep(0.1)
        if not report.is_file():
            sys.exit(f"OpenCppCoverage produced no report for {name}")


def parse_reports(out_dir: Path) -> dict[str, FileCoverage]:
    """Merge every cobertura report; keep files under this repository's ``src/``.

    A line counts as covered when *any* owning test executes it (union of
    hits), never as a sum: the same AppCore lines appear in every report of a
    test that loads the DLL, so adding report denominators would dilute the
    rate of files that only one test touches.
    """
    merged: dict[str, FileCoverage] = {}
    for report in sorted(out_dir.glob("*.xml")):
        root = ET.parse(report).getroot()
        for cls in root.iter("class"):
            key = _repo_relative(cls.get("filename") or "")
            if key is None:
                continue
            entry = merged.setdefault(key, FileCoverage())
            for line in cls.iter("line"):
                number = int(line.get("number", "0"))
                if number not in entry.lines:
                    entry.lines[number] = 0
                if int(line.get("hits", "0")) > 0:
                    entry.lines[number] = 1
    return merged


def _repo_relative(filename: str) -> str | None:
    """Map any path shape the exporter emits to ``src/<tail>``.

    Observed shapes: ``F:\\repo\\src\\a.cpp``, ``PROJECTS\\...\\3D-Reconstruction
    \\src\\a.cpp`` (drive letter dropped) and relative ``..\\..\\src\\a.cpp``.
    Everything else — including ONNX Runtime's build-machine paths that also
    contain ``\\src\\`` — is rejected by matching the repository directory name
    (plus the ``..\\..\\src`` shape for sources compiled from the build tree).
    """
    lowered = filename.replace("/", "\\").lower()
    markers = (
        f"{BASE.name.lower()}\\src\\",
        "..\\..\\src\\",  # relative to build\\Release of a top-level target
        "..\\src\\",      # relative to build\\tests of a test target
    )
    for marker in markers:
        if marker in lowered:
            tail = lowered[lowered.rindex(marker) + len(marker):]
            if tail:
                return "src/" + tail.replace("\\", "/")
    return None


def print_table(merged: dict[str, FileCoverage]) -> None:
    print()
    print(f"{'file':<58} {'covered':>7} {'valid':>6} {'rate':>7}")
    print("-" * 82)
    for name in sorted(merged):
        entry = merged[name]
        print(f"{name:<58} {entry.covered:>7} {entry.valid:>6} "
              f"{entry.percent:>6.1f}%")


def print_debt(merged: dict[str, FileCoverage]) -> None:
    """Print the translation units that are visible but never executed.

    These files carry no floor (a floor of 0 would pin nothing), so they would
    otherwise be invisible in the report.  Listing them on every run keeps the
    debt explicit: it is measured, named and reviewable, not forgotten.
    """
    dark = sorted(
        (name, entry) for name, entry in merged.items()
        if name.endswith(".cpp") and entry.covered == 0 and entry.valid > 0
    )
    if not dark:
        return
    total = sum(entry.valid for _, entry in dark)
    print()
    print(f"Coverage debt: {len(dark)} translation unit(s), {total} line(s) "
          f"visible but never executed")
    for name, entry in dark:
        print(f"  {name:<62} {entry.valid:>5} lines")


def enforce(merged: dict[str, FileCoverage]) -> int:
    failures: list[str] = []
    # Completeness first: a translation unit that is missing from the report
    # is either not linked into any instrumented module (a plugin DLL outside
    # the module pattern) or was added/renamed without the gate noticing.
    # Either way the file is invisible to the measurement, so the gate must
    # not pass while that is true.
    missing_units = sorted(_source_translation_units() - set(merged))
    if missing_units:
        shown = ", ".join(missing_units[:10])
        if len(missing_units) > 10:
            shown += f", … (+{len(missing_units) - 10} more)"
        failures.append(
            f"{len(missing_units)} translation unit(s) escape the measurement: {shown}"
        )
    for path, floor in sorted(FLOORS.items()):
        entry = merged.get(path)
        if entry is None or entry.valid == 0:
            failures.append(f"{path}: no coverage data (file missing from the report)")
            continue
        if entry.percent + 1e-9 < floor:
            failures.append(
                f"{path}: {entry.percent:.1f}% < floor {floor:.1f}% "
                f"({entry.covered}/{entry.valid} lines)")
    print()
    if failures:
        print("Coverage gate FAILED:")
        for failure in failures:
            print(f"  {failure}")
        return 1
    print(f"Coverage gate OK: {len(FLOORS)} file(s) at or above their floors.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", default=str(BASE / "build"))
    parser.add_argument("--measure", action="store_true",
                        help="print measured rates without enforcing the floors")
    args = parser.parse_args()

    tool = find_tool()
    # Resolve before run_tests(): the test executable runs with cwd=
    # <build>/Release, so a relative report path would be resolved there
    # instead of the repository root (and silently fail to export).
    build_dir = Path(args.build_dir).resolve()
    out_dir = build_dir / "coverage"
    out_dir.mkdir(parents=True, exist_ok=True)
    # The merged report is the union of *this* run's per-test reports. A
    # leftover report from an earlier run (a test that was removed from TESTS,
    # or a run that crashed) would silently inflate the numbers, so the
    # directory is emptied before the measurement starts.
    for stale in out_dir.glob("*.xml"):
        stale.unlink()

    print(f"OpenCppCoverage: {tool}")
    run_tests(tool, build_dir, out_dir)
    merged = parse_reports(out_dir)
    if not merged:
        sys.exit("no source files were found in the coverage reports")
    print_table(merged)
    print_debt(merged)

    if args.measure:
        missing = sorted(set(FLOORS) - set(merged))
        if missing:
            print("\nFloors without data: " + ", ".join(missing))
        return 0
    return enforce(merged)


if __name__ == "__main__":
    sys.exit(main())
