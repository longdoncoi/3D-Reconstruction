"""C++ mutation check for the load-bearing architecture guards.

Statement coverage says the guards are *executed*; it does not say they are
*pinned*. This script inverts one decision at a time in a guard, rebuilds only
the target that owns it, and re-runs only the owning CTest test:

    KILLED  — the test fails: the guard's behaviour is pinned by the suite.
    SURVIVED— the test stays green: the suite would not notice the guard
              being disabled (the mutation check fails the run).
    STALE   — the anchor no longer matches the source: the mutant cannot be
              applied, and the run fails instead of silently shrinking.
    INVALID — the mutant does not compile: it is a broken mutant, not a kill.

Discipline (same as AIAssistant/scripts/mutation_check.py):

* Every owning test must pass on the unmutated tree first (pre-flight). A
  missing Qt/vcpkg DLL on PATH would otherwise masquerade as killed mutants.
* The mutated file is restored byte-identical in a ``finally``; a lock file in
  the build directory refuses overlapping runs and, left behind by a crash,
  blocks the next run until the source is restored by hand.
* After the run every mutated target is rebuilt once, so the build tree ends
  up with the original binaries (the final verification re-runs the suite).
* The architecture guard (tests/CheckArchitectureBoundaries.cmake) is itself
  mutated with "planted violation" mutants: a forbidden include / bare
  ``waitForFinished()`` / heavy SDK include in src/utils is appended to a real
  source file and the guard test must fail — a guard nobody can trip is a
  guard nobody knows is alive.

Usage (from the repository root, PATH prepared like the CTest step):

    python tests/mutation_check.py [--only SUBSTR] [--config Release]

Exit code 0 = every mutant killed, no stale/invalid, suite restored green.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent

# Architecture-guard mutants run no build (the guard is a CMake script).
_PLANT: None = None

#: A build that fails because the *output* is still open by a just-finished
#: test process is an environmental race, not a broken mutant. The linker
#: says so explicitly.
_BUILD_RETRIES = 6
_BUILD_RETRY_DELAY = 3.0


def _is_locked_output(output: str) -> bool:
    """Whether the build failed because a file was locked by a live process."""
    return "LNK1104" in output or "LNK1181" in output or "cannot open file" in output


@dataclass(frozen=True)
class Mutant:
    """One deliberate logic inversion, with the target + test that own it."""

    file: str
    original: str
    mutated: str
    target: str | None
    test: str
    label: str
    timeout_seconds: int = 300

    @property
    def is_plant(self) -> bool:
        return self.target is None


MUTANTS: tuple[Mutant, ...] = (
    # ── ServiceRegistry: null registration refusal (silent nullptr services
    #    would crash at the first call site instead of failing at wiring time)
    Mutant(
        "src/core/ServiceRegistry.h",
        "if (!impl) {\n            qWarning() << \"ServiceRegistry: refusing to register null service\"",
        "if (false) {\n            qWarning() << \"ServiceRegistry: refusing to register null service\"",
        "TestServiceRegistry",
        "TestServiceRegistry",
        "null service registration accepted (wiring bug deferred to a crash)",
    ),
    # ── PluginHost: API-version gate (stale-ABI plugin must be rejected)
    Mutant(
        "src/core/PluginHost.cpp",
        "if (plugin->apiVersion() != IPlugin::kApiVersion) {",
        "if (false) {",
        "TestPluginHost",
        "TestPluginHost",
        "stale-ABI Plugin loaded (apiVersion gate removed)",
    ),
    # ── PluginHost: toolchain/ABI fingerprint gate (same virtual surface but a
    #    different Qt/MSVC/runtime must be rejected instead of corrupted)
    Mutant(
        "src/core/PluginHost.cpp",
        "if (plugin->abiFingerprint() != IPlugin::kAbiFingerprint) {",
        "if (false) {",
        "TestPluginHost",
        "TestPluginHost",
        "ABI fingerprint gate removed (mismatched toolchain loaded)",
    ),
    # ── PluginHost: discovery must not probe non-libraries / non-plugin files
    Mutant(
        "src/core/PluginHost.cpp",
        "if (!QLibrary::isLibrary(fileName)) continue;",
        "if (false) continue;",
        "TestPluginHost",
        "TestPluginHost",
        "non-library files probed by QPluginLoader (discovery filter removed)",
    ),
    # ── PluginHost: failed initialize() must roll back already-initialized
    #    plugins (leaked half-initialized plugins on abort)
    Mutant(
        "src/core/PluginHost.cpp",
        "for (auto it = m_initialized.rbegin(); it != m_initialized.rend(); ++it) {\n                (*it)->cleanup();\n            }",
        "for (auto it = m_initialized.rbegin(); it != m_initialized.rend(); ++it) {\n                (void)it;\n            }",
        "TestPluginHost",
        "TestPluginHost",
        "initialize() rollback removed (earlier plugins stay initialized)",
    ),
    # ── PluginHost: cleanup must run in reverse init order (dependents die
    #    before their dependencies)
    Mutant(
        "src/core/PluginHost.cpp",
        "void PluginHost::shutdown() {\n    for (auto it = m_initialized.rbegin(); it != m_initialized.rend(); ++it) {",
        "void PluginHost::shutdown() {\n    for (auto it = m_initialized.begin(); it != m_initialized.end(); ++it) {",
        "TestPluginHost",
        "TestPluginHost",
        "shutdown cleans in init order (dependencies torn down before dependents)",
    ),
    # ── PluginDependencyResolver: unknown phase is a wiring bug, not a hint
    Mutant(
        "src/core/PluginDependencyResolver.cpp",
        "if (phaseRank(plugins[i].phase) < 0) {",
        "if (false) {",
        "TestPluginDependencyResolver",
        "TestPluginDependencyResolver",
        "unknown loadPhase silently accepted",
    ),
    # ── PluginDependencyResolver: missing dependency must fail fast
    Mutant(
        "src/core/PluginDependencyResolver.cpp",
        "result.issues.append({Severity::Error, plugins[i].name,\n                                      QStringLiteral(\"depends on \\\"%1\\\" but that plugin is not loaded\")",
        "result.issues.append({Severity::Warning, plugins[i].name,\n                                      QStringLiteral(\"depends on \\\"%1\\\" but that plugin is not loaded\")",
        "TestPluginDependencyResolver",
        "TestPluginDependencyResolver",
        "missing dependency downgraded to a warning (partial order accepted)",
    ),
    # ── PluginDependencyResolver: a cycle must be reported, not ignored
    Mutant(
        "src/core/PluginDependencyResolver.cpp",
        "if (order.size() != count) {",
        "if (false) {",
        "TestPluginDependencyResolver",
        "TestPluginDependencyResolver",
        "dependency cycle not reported",
    ),
    # ── SecretVault: unprotected legacy values must pass through and be
    #    migrated on save (treating them as DPAPI blobs loses the secret)
    Mutant(
        "src/core/SecretVault.cpp",
        "if (!storedValue.startsWith(QString::fromLatin1(kEncodedPrefix))) {",
        "if (false) {",
        "TestUserManagerCrypto",
        "TestUserManagerCrypto",
        "legacy plaintext secret treated as an encrypted blob (value lost)",
    ),
    # ── SecretVault: empty input stays empty (never store a blob for "")
    Mutant(
        "src/core/SecretVault.cpp",
        "if (plainValue.isEmpty()) return {};",
        "if (false) return {};",
        "TestUserManagerCrypto",
        "TestUserManagerCrypto",
        "empty secret encrypted into a non-empty blob",
    ),
    # ── UserManager: PBKDF2 iteration count is part of the stored credential
    #    contract; weakening it must be visible.
    Mutant(
        "src/core/CryptoUtils.cpp",
        "constexpr int kPbkdf2Iterations = 100000;",
        "constexpr int kPbkdf2Iterations = 1;",
        "TestUserManagerCrypto",
        "TestUserManagerCrypto",
        "PBKDF2 iterations weakened 100000 -> 1",
    ),
    # ── UserManager: a consumed license key must stay consumed
    Mutant(
        "src/core/UserManager.cpp",
        "if (m_licenseLedger.isUsed(key)) {",
        "if (false) {",
        "TestUserManagerCrypto",
        "TestUserManagerCrypto",
        "already-used license key accepted (key reuse)",
    ),
    # ── OTP: successful verification must consume the code
    Mutant(
        "src/core/OtpStore.cpp",
        "    m_entries.erase(it);\n    return true;",
        "    return true;",
        "TestUserManagerCrypto",
        "TestUserManagerCrypto",
        "OTP remains valid after successful use (replay)",
    ),
    # ── AIRequestQueue: never start a second request while a reply is active
    Mutant(
        "src/services/ai/AIRequestQueue.cpp",
        "if (!m_pending.isEmpty() || m_queued.isEmpty()) return false;",
        "if (m_queued.isEmpty()) return false;",
        "TestAIRequestQueue",
        "TestAIRequestQueue",
        "queued request starts while another reply is still pending",
    ),
    # ── ChatSessionStore: the chat history must be sealed with the OS vault
    #    before it reaches the disk (plaintext JSON leaks the conversation)
    Mutant(
        "src/services/ai/ChatSessionStore.cpp",
        "    const QString sealed = SecretVault::encrypt(QString::fromUtf8(payload));",
        "    const QString sealed = QString::fromUtf8(payload);",
        "TestChatSessionManager",
        "TestChatSessionManager",
        "ChatSessionStore: chat history written unencrypted (vault bypassed)",
    ),
    # ── ChatSessionStore: a sealed history must be unsealed before it is
    #    parsed (skipping the step turns the file into unreadable garbage)
    Mutant(
        "src/services/ai/ChatSessionStore.cpp",
        "    const QString plain = SecretVault::decrypt(QString::fromUtf8(raw));",
        "    const QString plain = QString::fromUtf8(raw);",
        "TestChatSessionManager",
        "TestChatSessionManager",
        "ChatSessionStore: chat history read without unsealing (sealed file parsed as JSON)",
    ),
    # ── AtRestCipher: the attachment must be sealed before it reaches the disk
    #    (the copied original and its thumbnail are the user's files).
    Mutant(
        "src/core/AtRestCipher.cpp",
        "    return kMarker + nonce + cipher + tag;",
        "    return plain;",
        "TestAtRestCipher",
        "TestAtRestCipher",
        "AtRestCipher: seal() stores the payload in the clear (envelope removed)",
    ),
    # ── AtRestCipher: a sealed payload must be unsealed before it is used
    #    (returning the envelope would hand a ciphertext to the caller).
    Mutant(
        "src/core/AtRestCipher.cpp",
        "    if (!gcmDecrypt(keyMaterial, nonce, associatedData(context), cipher, tag, &plain)) { qWarning() << \"AtRestCipher: the payload could not be unsealed (wrong key, wrong context or tampered bytes).\"; return {}; }\n"
        "    return plain;",
        "    if (!gcmDecrypt(keyMaterial, nonce, associatedData(context), cipher, tag, &plain)) { qWarning() << \"AtRestCipher: the payload could not be unsealed (wrong key, wrong context or tampered bytes).\"; return {}; }\n"
        "    return sealed;",
        "TestAtRestCipher",
        "TestAtRestCipher",
        "AtRestCipher: open() hands back the sealed blob (unseal skipped)",
    ),
    # ── FileUtilities: the copied attachment must be sealed in place — a copy
    #    of the data folder must not hand the user's file over in the clear.
    Mutant(
        "src/utils/FileUtilities.cpp",
        "    const QByteArray sealedOriginal =\n"
        "        AtRestCipher::seal(originalBytes, QString::fromLatin1(AttachmentContext));",
        "    const QByteArray sealedOriginal =\n"
        "        originalBytes;",
        "TestUtilsLayer",
        "TestUtilsLayer",
        "FileUtilities: the copied attachment is stored unencrypted (seal removed)",
    ),
    # ── Mail thread guard: blocking mail I/O on the GUI thread must warn
    Mutant(
        "src/services/mail/MailThreadGuard.h",
        "if (app != nullptr && QThread::currentThread() == app->thread()) {",
        # /WX: the mutant must stay warning-clean — keep `app` referenced with
        # a runtime condition that can never be true (thread() is never null).
        "if (app != nullptr && app->thread() == nullptr) {",
        "TestMailCredentialLifecycle",
        "TestMailCredentialLifecycle",
        "GUI-thread mail I/O no longer warns (contract guard disabled)",
    ),
    # ── AppShell: deferred phase must be idempotent (live instances held by
    #    dock widgets and plugins are never replaced)
    Mutant(
        "src/app/AppShell.cpp",
        "if (m_reconService) return;",
        "if (false) return;",
        "TestAppShell",
        "TestAppShell",
        "deferred services re-created on every call (held pointers dangle)",
    ),
    # ── AppConfig: runtime AI settings must be validated before use
    Mutant(
        "src/core/AppConfig.cpp",
        "return (value > 0 && value <= 65535) ? value : 8080;",
        "return value;",
        "TestAppConfig",
        "TestAppConfig",
        "out-of-range AI server port accepted",
    ),
    Mutant(
        "src/core/AppConfig.cpp",
        'return value.isEmpty() ? QStringLiteral("python") : value;',
        "return value;",
        "TestAppConfig",
        "TestAppConfig",
        "blank Python interpreter accepted (empty launch command)",
    ),
    # ── Architecture guard itself: a forbidden plugin include must fail
    Mutant(
        "src/modules/SendMailPlugin/SendMailPlugin.cpp",
        "",
        "\n#include \"SceneService.h\" // planted by mutation_check — guard must reject\n",
        _PLANT,
        "TestArchitectureBoundaries",
        "planted: plugin includes a concrete service header",
    ),
    # ── Architecture guard itself: bare waitForFinished() must fail
    Mutant(
        "src/services/mail/SmtpMailer.cpp",
        "",
        "\nvoid mutation_check_probe(QProcess &p) { p.waitForFinished(); } // planted by mutation_check\n",
        _PLANT,
        "TestArchitectureBoundaries",
        "planted: untimed waitForFinished() in src/",
    ),
    # ── Architecture guard itself: heavy SDK in src/utils must fail
    Mutant(
        "src/utils/Logger.h",
        "",
        "\n#include <vtkActor.h> // planted by mutation_check — utils must stay light\n",
        _PLANT,
        "TestArchitectureBoundaries",
        "planted: heavy SDK include in src/utils (light-plugin ABI surface)",
    ),
)


def _anchor_variants(anchor: str) -> list[str]:
    """The file on disk may use CRLF while the anchors are written with LF."""
    if "\n" not in anchor or "\r\n" in anchor:
        return [anchor]
    return [anchor, anchor.replace("\n", "\r\n")]


class MutationCheck:
    def __init__(self, build_dir: Path, config: str, only: str | None) -> None:
        self.build_dir = build_dir
        self.config = config
        self.only = only
        self.lock_file = build_dir / ".mutation_check.lock"
        self.killed: list[str] = []
        self.survived: list[str] = []
        self.stale: list[str] = []
        self.invalid: list[str] = []
        self.touched_targets: set[str] = set()
        self._restored = False

    # ── infrastructure ───────────────────────────────────────────────────────

    def _run(self, argv: list[str], timeout: int) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            argv, cwd=BASE, capture_output=True, text=True, errors="replace",
            timeout=timeout, check=False,
        )

    def build(self, target: str, timeout: int = 2400) -> bool:
        """Build one target; a locked output DLL is retried, not a verdict.

        The mutant that just ran leaves its test executable — and therefore
        ``AppCore.dll`` — loaded for a moment after the process exits, and
        the next relink then fails with ``LNK1104``. That is an
        environmental race, not a mutant that does not compile, so it is
        retried instead of being reported as INVALID.
        """
        for attempt in range(_BUILD_RETRIES):
            result = self._run(
                ["cmake", "--build", str(self.build_dir), "--config", self.config,
                 "--target", target],
                timeout,
            )
            if result.returncode == 0:
                return True
            output = f"{result.stdout}\n{result.stderr}"
            if not _is_locked_output(output) or attempt == _BUILD_RETRIES - 1:
                print(f"  build output (tail):\n{result.stdout[-3000:]}\n{result.stderr[-3000:]}")
                return False
            print(f"  the build output is locked by a running process — retrying "
                  f"({attempt + 1}/{_BUILD_RETRIES})")
            time.sleep(_BUILD_RETRY_DELAY)
        return False

    def run_test(self, test: str, timeout: int) -> bool:
        result = self._run(
            ["ctest", "--test-dir", str(self.build_dir), "-C", self.config,
             "-R", f"^{test}$", "--output-on-failure", "--timeout", str(timeout)],
            timeout + 120,
        )
        return result.returncode == 0

    def acquire_lock(self) -> None:
        if self.lock_file.exists():
            sys.exit(
                f"refusing to start: {self.lock_file} exists — a previous run "
                "crashed with a mutation still applied. Restore the sources "
                "(git diff) and delete the lock file."
            )
        self.lock_file.write_text("mutation_check running\n", encoding="utf-8")

    def release_lock(self) -> None:
        self.lock_file.unlink(missing_ok=True)

    # ── mutant application ───────────────────────────────────────────────────

    def apply(self, mutant: Mutant) -> bytes | None:
        """Write the mutant, return the original bytes (None = STALE)."""
        path = BASE / mutant.file
        original_bytes = path.read_bytes()
        text = original_bytes.decode("utf-8")

        if mutant.is_plant:
            if mutant.mutated in text:
                return None
            path.write_text(text + mutant.mutated, encoding="utf-8", newline="")
            return original_bytes

        for candidate in _anchor_variants(mutant.original):
            if text.count(candidate) != 1:
                continue
            replacement = mutant.mutated
            if candidate != mutant.original:  # matched the CRLF variant
                replacement = replacement.replace("\n", "\r\n")
            path.write_text(text.replace(candidate, replacement, 1),
                            encoding="utf-8", newline="")
            return original_bytes
        return None

    @staticmethod
    def restore(mutant: Mutant, original_bytes: bytes) -> None:
        (BASE / mutant.file).write_bytes(original_bytes)

    # ── run ──────────────────────────────────────────────────────────────────

    def preflight(self, mutants: tuple[Mutant, ...]) -> None:
        tests = sorted({m.test for m in mutants})
        print(f"Pre-flight on unmutated tree: {len(tests)} owning test(s) must pass")
        for test in tests:
            if not self.run_test(test, 300):
                sys.exit(
                    f"pre-flight failed: '{test}' does not pass on the unmutated "
                    "tree — fix the environment (Qt/vcpkg DLLs on PATH) before "
                    "trusting any mutation result."
                )
        print("Pre-flight OK\n")

    def run(self) -> int:
        selected = tuple(
            m for m in MUTANTS if self.only is None or self.only in m.label
        )
        if not selected:
            sys.exit(f"no mutant matches --only {self.only!r}")

        self.acquire_lock()
        try:
            self.preflight(selected)
            for index, mutant in enumerate(selected, start=1):
                self.run_one(index, len(selected), mutant)
            if self.stale:
                self.report_stale()
            if self.survived or self.stale or self.invalid:
                self.summarize()
                return 1
            self.verify_restored(selected)
            self.summarize()
            return 0
        finally:
            if self.touched_targets and not self._restored:
                print("Aborted with mutants applied to the build tree; rebuilding "
                      f"{len(self.touched_targets)} target(s) before exit ...")
                for target in sorted(self.touched_targets):
                    if not self.build(target):
                        print(f"WARNING: could not rebuild {target}; its binaries "
                              "may still contain a mutant")
            self.release_lock()

    def run_one(self, index: int, total: int, mutant: Mutant) -> None:
        print(f"[{index}/{total}] {mutant.label}")
        original_bytes = self.apply(mutant)
        if original_bytes is None:
            self.stale.append(f"{mutant.file}: anchor for {mutant.label!r} not found")
            print("  STALE (anchor missing or not unique)\n")
            return
        try:
            if not mutant.is_plant:
                self.touched_targets.add(mutant.target)
                if not self.build(mutant.target):
                    self.invalid.append(
                        f"{mutant.file}: mutant for {mutant.label!r} does not compile"
                    )
                    print("  INVALID (build failed)\n")
                    return
            passed = self.run_test(mutant.test, mutant.timeout_seconds)
            if passed:
                self.survived.append(
                    f"{mutant.file}: {mutant.label!r} — {mutant.test} stayed green"
                )
                print(f"  SURVIVED ({mutant.test} did not notice)\n")
            else:
                self.killed.append(mutant.label)
                print(f"  KILLED ({mutant.test} failed as expected)\n")
        finally:
            self.restore(mutant, original_bytes)

    def report_stale(self) -> None:
        for entry in self.stale:
            print(f"STALE: {entry}")

    def verify_restored(self, mutants: tuple[Mutant, ...]) -> None:
        print(f"Restoring build tree ({len(self.touched_targets)} target(s)) ...")
        for target in sorted(self.touched_targets):
            if not self.build(target):
                sys.exit(f"restore build of {target} failed")
        tests = sorted({m.test for m in mutants})
        for test in tests:
            if not self.run_test(test, 300):
                sys.exit(f"post-restore verification failed: '{test}' is not green")
        self._restored = True

    def summarize(self) -> None:
        total = len(self.killed) + len(self.survived) + len(self.stale) + len(self.invalid)
        print("-" * 72)
        print(
            f"Mutation check: {len(self.killed)}/{total} killed, "
            f"{len(self.survived)} survived, {len(self.stale)} stale, "
            f"{len(self.invalid)} invalid"
        )
        for label in self.survived:
            print(f"  SURVIVED: {label}")
        for label in self.stale:
            print(f"  STALE:    {label}")
        for label in self.invalid:
            print(f"  INVALID:  {label}")


def main() -> int:
    # The console of a Windows shell is usually cp1252/cp437: printing a
    # build log that contains a character outside that code page used to kill
    # the whole run with a UnicodeEncodeError — the gate has to report the
    # mutant, not crash on the way to the report.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", default=str(BASE / "build"))
    parser.add_argument("--config", default="Release")
    parser.add_argument("--only", default=None,
                        help="run only mutants whose label contains this text")
    args = parser.parse_args()
    return MutationCheck(Path(args.build_dir), args.config, args.only).run()


if __name__ == "__main__":
    sys.exit(main())
