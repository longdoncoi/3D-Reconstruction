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
#: A source write can hit a transient Windows sharing lock (indexer, scanner,
#: IDE, build). A failed restore leaves a mutant behind, so retry briefly and
#: then fail loudly instead of letting the next step read a mutated tree.
_RESTORE_RETRIES = 8
_RESTORE_RETRY_DELAY = 2.0


def _is_locked_output(output: str) -> bool:
    """Whether the build failed because a file was locked by a live process."""
    # MSB6003 is the compiler task failing to delete/replace an object file
    # that another build still holds (`... cannot access the file ... because
    # it is being used by another process`). An IDE building the same tree is
    # the usual source; it is a race, not a broken mutant.
    return (
        "LNK1104" in output
        or "LNK1181" in output
        or "cannot open file" in output
        or "MSB6003" in output
        or "being used by another process" in output
        # C1041: the compiler cannot open the program database because a
        # second build holds it (`cannot open program database ... vc143.pdb`).
        # Same race, different tool.
        or "C1041" in output
        or "cannot open program database" in output
    )


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
        "src/include/ServiceRegistry.h",
        "if (!impl) {\n            qWarning() << \"ServiceRegistry: refusing to register null service\"",
        "if (false) {\n            qWarning() << \"ServiceRegistry: refusing to register null service\"",
        "TestServiceRegistry",
        "TestServiceRegistry",
        "null service registration accepted (wiring bug deferred to a crash)",
    ),
    # ── ServiceRegistry: a duplicate registration must be REFUSED. Silently
    #    letting the second one win lets a plugin take over a service the
    #    composition root installed.
    Mutant(
        "src/include/ServiceRegistry.h",
        "            qWarning() << \"ServiceRegistry: duplicate registration of\"\n"
        "                       << key << \"- keeping the existing entry.\";\n"
        "            return;",
        "            qWarning() << \"ServiceRegistry: duplicate registration of\"\n"
        "                       << key << \"- keeping the existing entry.\";",
        "TestServiceRegistry",
        "TestServiceRegistry",
        "duplicate service registration replaces the existing service",
    ),
    # ── ReconstructionPostProcessor: the Poisson trim replaces the vertex
    #    list, so the faces have to be remapped WITH it. Keeping the old
    #    indices leaves the faces pointing at vertices that no longer exist —
    #    a corrupt mesh.
    Mutant(
        "src/services/reconstruction/ReconstructionPostProcessor.cpp",
        "                face.vertices.push_back(remap[static_cast<size_t>(vertexIndex)]);",
        "                face.vertices.push_back(vertexIndex);",
        "TestReconstructionPipeline",
        "TestReconstructionPipeline",
        "the Poisson trim keeps the old face indices (mesh points at removed vertices)",
    ),
    # ── ReconstructionPostProcessor: a face that lost a vertex must be
    #    dropped, not kept with a dangling index.
    Mutant(
        "src/services/reconstruction/ReconstructionPostProcessor.cpp",
        "            if (!complete) continue;",
        "            if (false) continue;",
        "TestReconstructionPipeline",
        "TestReconstructionPipeline",
        "a Poisson face that lost a vertex is kept (dangling vertex index)",
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
    # ── IPlugin: the ABI fingerprint must carry the content hash of the
    #    plugin-facing interface surface. Without it, changing an I*.h without
    #    bumping kApiVersion would silently load an incompatible plugin.
    Mutant(
        "src/include/IPlugin.h",
        "    static constexpr quint64 kAbiFingerprint =\n"
        "        kToolchainFingerprint ^ (kInterfaceHash * 0x9E3779B97F4A7C15ULL);",
        "    static constexpr quint64 kAbiFingerprint =\n"
        "        kToolchainFingerprint;",
        "TestPluginHost",
        "TestPluginHost",
        "the ABI fingerprint ignores the plugin-facing interface hash",
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
        "for (auto it = m_initialized.rbegin(); it != m_initialized.rend(); ++it) {\n                cleanupPlugin(*it);\n            }",
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
        "    if (plainValue.isEmpty()) return {};\n\n#ifdef Q_OS_WIN",
        "    if (false) return {};\n\n#ifdef Q_OS_WIN",
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
    # ── CryptoUtils: the work factor comes from the credential file, so it
    #    needs a ceiling — a tampered file must not turn a login into a hang.
    Mutant(
        "src/core/CryptoUtils.cpp",
        "constexpr int kMaxPbkdf2Iterations = 1000000;",
        "constexpr int kMaxPbkdf2Iterations = 0;",
        "TestUserManagerCrypto",
        "TestUserManagerCrypto",
        "the ceiling on the stored PBKDF2 work factor is removed",
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
    # ── AIAssistant: an auto-generated title must clear the placeholder flag.
    #    Setting the title alone leaves the sidebar rendering "New Session
    #    <date>", so the prompt never appears in the session list.
    Mutant(
        "src/services/ai/AIAssistant.cpp",
        "    session.titleIsDefault = false;\n    return true;",
        "    session.titleIsDefault = true; // mutant: the placeholder survives\n    return true;",
        "TestAIAssistant",
        "TestAIAssistant",
        "AIAssistant: the prompt title never replaces the 'New Session' placeholder",
    ),
    # ── ChatSessionManager: the load-time repair must clear the placeholder
    #    flag, otherwise the sidebar keeps rendering "New Session <date>" for
    #    histories written before auto-title cleared it (and every start
    #    rewrites the sealed history).
    Mutant(
        "src/services/ai/ChatSessionManager.cpp",
        "        session.titleIsDefault = false;\n        changed = true;",
        "        session.titleIsDefault = true; // mutant: the repair does not stick\n        changed = true;",
        "TestChatSessionManager",
        "TestChatSessionManager",
        "ChatSessionManager: the legacy prompt title never replaces the placeholder",
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
    # ── SecretVault: the protection failure must FAIL CLOSED on a platform
    #    that has an OS backend. A transient DPAPI failure must cost the user
    #    a re-entry, not the confidentiality of the secret.
    Mutant(
        "src/core/SecretVault.cpp",
        "    // Fail closed: this platform HAS an OS backend, so a protection failure\n"
        "    // must not turn into a plaintext write. An empty result tells the caller\n"
        "    // \"do not write anything — keep the value that is already stored\".\n"
        "    return {};",
        "    // Fail closed: this platform HAS an OS backend, so a protection failure\n"
        "    // must not turn into a plaintext write. An empty result tells the caller\n"
        "    // \"do not write anything — keep the value that is already stored\".\n"
        "    return QString::fromLatin1(kPlainPrefix) + plainValue;",
        "TestSecretProtection",
        "TestSecretProtection",
        "SecretVault: a protection failure downgrades the secret to cleartext (fail open)",
    ),
    # ── UserSettingsStore: a secret key must never reach the plaintext
    #    preference namespace, whichever setter the caller uses.
    Mutant(
        "src/core/UserSettingsStore.cpp",
        "    if (isSecretKey(key)) {\n        storeSecret(username, key, value);\n        return;\n    }",
        "    if (false) {\n        storeSecret(username, key, value);\n        return;\n    }",
        "TestSecretProtection",
        "TestSecretProtection",
        "a secret key is written in the clear (the preference API bypasses the vault)",
    ),
    # ── UserSettingsStore: the migration must retire the plaintext copy, not
    #    only seal a second copy of it.
    Mutant(
        "src/core/UserSettingsStore.cpp",
        "        m_settings->setValue(\"User_\" + username + \"/secret_\" + key, sealed);\n"
        "        m_settings->remove(fullKey);",
        "        m_settings->setValue(\"User_\" + username + \"/secret_\" + key, sealed);",
        "TestSecretProtection",
        "TestSecretProtection",
        "the legacy plaintext secret is sealed but the cleartext copy stays in the INI",
    ),
    # ── UserManager: a password reset needs a proof of ownership — the OTP
    #    that verifyOtp() granted. Without it the service would be a way to
    #    change anybody's password.
    Mutant(
        "src/core/UserManager.cpp",
        "    if (proof == m_authorizedResets.constEnd()\n"
        "        || QDateTime::currentDateTime() > proof.value()) {",
        "    if (proof != m_authorizedResets.constEnd()\n"
        "        && QDateTime::currentDateTime() > proof.value()) {",
        "TestUserManagerCrypto",
        "TestUserManagerCrypto",
        "a password reset no longer needs the proof of ownership",
    ),
    # ── UserManager: the proof is single-use — one OTP changes one password.
    Mutant(
        "src/core/UserManager.cpp",
        "    // Single use: the proof is spent even if the rest of the reset fails.\n"
        "    m_authorizedResets.remove(proofKey);",
        "    // Single use: the proof is spent even if the rest of the reset fails.\n"
        "    // (mutated: the proof is not consumed)",
        "TestUserManagerCrypto",
        "TestUserManagerCrypto",
        "the proof of ownership is not consumed (the same OTP resets twice)",
    ),
    # ── UserManager: a reset is subject to the same password policy as a
    #    change — otherwise the reset path is the weak one.
    Mutant(
        "src/core/UserManager.cpp",
        "    m_authorizedResets.remove(proofKey);\n\n    if (newPass.length() < 6) {",
        "    m_authorizedResets.remove(proofKey);\n\n    if (newPass.length() < 0) {",
        "TestUserManagerCrypto",
        "TestUserManagerCrypto",
        "the password policy is not applied to a reset",
    ),
    # ── UserAuthPlugin: cleanup() must release the widgets the plugin
    #    installed into the shell — the shell outlives the plugin.
    Mutant(
        "src/modules/UserAuthPlugin/UserAuthPlugin.cpp",
        "    releaseShellWidgets();\n    if (s_users) s_users->saveConfig();",
        "    if (s_users) s_users->saveConfig();",
        "TestPluginSmoke",
        "TestPluginSmoke",
        "UserAuthPlugin: cleanup() leaves the shell widgets behind (dead UI)",
    ),
    # ── PluginHost: the shutdown must release the plugin objects. The root
    #    component has no parent — only the loader's unload() destroys it.
    Mutant(
        "src/core/PluginHost.cpp",
        "        if (!loader->unload()) {",
        "        if (false) {",
        "TestPluginSmoke",
        "TestPluginSmoke",
        "PluginHost: the plugin objects are not released (the whole QObject tree stays)",
    ),
    # ── Architecture guard itself: a plugin that installs an event filter on
    #    a shell widget must be able to remove it — the filter's vtable lives
    #    in the plugin DLL.
    Mutant(
        "src/modules/ViewerPlugin/ViewerPlugin.cpp",
        "",
        "\nvoid mutation_probe(QObject *o) { o->installEventFilter(new QObject(o)); }"
        " // planted by mutation_check — the filter can never be removed\n",
        _PLANT,
        "TestArchitectureBoundaries",
        "planted: an event filter is installed on a shell widget and never removed",
    ),
    # ── Architecture guard itself: the shell window comes from IShellService,
    #    not from a scan of the top-level widgets.
    Mutant(
        "src/modules/SendMailPlugin/SendMailPlugin.cpp",
        "",
        "\nvoid mutation_probe() { QApplication::topLevelWidgets(); }"
        " // planted by mutation_check — the shell must come from IShellService\n",
        _PLANT,
        "TestArchitectureBoundaries",
        "planted: the shell window is found by scanning the top-level widgets",
    ),
    # ── Architecture guard itself: blocking I/O must not go through the
    #    global pool — it may run the task inline on the GUI thread.
    Mutant(
        "src/modules/AIProcessorPlugin/AIProcessorPlugin.cpp",
        "",
        "\nvoid mutation_probe() { QtConcurrent::run([]() {}); }"
        " // planted by mutation_check — blocking I/O needs blocking::run\n",
        _PLANT,
        "TestArchitectureBoundaries",
        "planted: blocking I/O scheduled on the global thread pool",
    ),
    # ── AIProcessorPlugin: the inference entry points must leave the GUI thread.
    #    The ONNX/OpenCV call used to run inside the slot, freezing the window
    #    for the whole inference; blocking::run owns a dedicated QThread.
    Mutant(
        "src/modules/AIProcessorPlugin/AIProcessorPlugin.cpp",
        "    return blocking::run(std::move(task), &m_inferenceTasks);",
        "    QPromise<cv::Mat> promise;\n"
        "    promise.start();\n"
        "    promise.addResult(task());\n"
        "    promise.finish();\n"
        "    return promise.future();",
        "TestPluginUi",
        "TestPluginUi",
        "AIProcessorPlugin: the inference runs synchronously on the GUI thread (window freeze)",
    ),
    # ── AIProcessorPlugin: only ONE inference may be in flight at a time — the
    #    ONNX session is shared, and two results race for the same viewer.
    Mutant(
        "src/modules/AIProcessorPlugin/AIProcessorPlugin.cpp",
        "    if (m_inferenceRunning) return;",
        "    if (false) return;",
        "TestPluginUi",
        "TestPluginUi",
        "AIProcessorPlugin: a second inference races the first (a stale result can win)",
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
    # ── Architecture guard itself: a plugin's include path must not contain
    #    src/core — the public surface is src/include, so adding src/core
    #    would hand the plugin the concrete kernel headers.
    Mutant(
        "src/modules/ViewerPlugin/CMakeLists.txt",
        "",
        "\ntarget_include_directories(ViewerPlugin PRIVATE ${PROJECT_SOURCE_DIR}/src/core)"
        " // planted by mutation_check — the plugin must not see the kernel\n",
        _PLANT,
        "TestArchitectureBoundaries",
        "planted: a plugin's include path hands it the concrete kernel headers",
    ),
    # ── AppShell: the composition root must register the SEGREGATED user
    #    contracts under their own keys — otherwise a consumer that depends
    #    on the narrow interface (IAuthService, IUserPreferences, …) resolves
    #    nothing and silently loses the capability.
    Mutant(
        "src/app/AppShell.cpp",
        "    m_registry->registerService<IAuthService>(UserManager::instance());",
        "    // m_registry->registerService<IAuthService>(UserManager::instance());",
        "TestAppShell",
        "TestAppShell",
        "the composition root does not register the segregated IAuthService",
    ),
    # ── AIProcessor: a FAILED model load must not report the model as ready
    #    (the readiness flag is what the plugin asks before it uses a model).
    Mutant(
        "src/services/ai/AIProcessor.cpp",
        "  } catch (const Ort::Exception &e) {\n"
        "    qWarning() << \"ORT Exception during loading detection model:\" << e.what();\n"
        "    isDetModelLoaded = false;\n"
        "    return false;",
        "  } catch (const Ort::Exception &e) {\n"
        "    qWarning() << \"ORT Exception during loading detection model:\" << e.what();\n"
        "    isDetModelLoaded = true;\n"
        "    return true;",
        "TestAIService",
        "TestAIService",
        "a failed model load reports the model as ready",
    ),
    # ── ChatMessage: the UI-only agent trace must never be accepted as a
    #    model role — sending it to FastAPI was the source of HTTP 422.
    Mutant(
        "src/include/ChatMessage.h",
        "        return role == ChatRole::User || role == ChatRole::Assistant\n"
        "               || role == ChatRole::System;",
        "        return true; // mutant: the UI-only agent trace would reach the model",
        "TestChatSessionManager",
        "TestChatSessionManager",
        "the UI-only agent trace is accepted as a model role",
    ),
    # ── ReconstructionService: the published image list must be read through
    #    its own lock, not the pipeline lock the worker holds for the whole
    #    run — taking the pipeline lock freezes the UI (ReconstructionPlugin
    #    calls getImageList() right after starting a run).
    Mutant(
        "src/services/reconstruction/ReconstructionService.cpp",
        "std::lock_guard pub(m_publishedMutex);\n    return m_imageList;",
        "std::shared_lock lock(m_mutex);\n    return m_imageList;",
        "TestReconstructionService",
        "TestReconstructionService",
        "ReconstructionService: getImageList() waits on the pipeline lock (UI freeze)",
        600,
    ),
    # ── AIModelLoadService: an exception from a loader is a failure. Reporting
    #    it as success would hide the broken model and (before the try/catch)
    #    left m_running stuck / no terminal signal.
    Mutant(
        "src/services/ai/AIModelLoadService.cpp",
        "} catch (const std::exception& e) {\n                ok = false;",
        "} catch (const std::exception& e) {\n                ok = true;",
        "TestAIModelLoadService",
        "TestAIModelLoadService",
        "AIModelLoadService: a throwing loader is reported as success",
    ),
    # ── AIServerProcessController: a start() that arrives while the wrapper is
    #    still exiting must be deferred and launched when it dies, not dropped
    #    (the whole point of making stop() non-blocking).
    Mutant(
        "src/services/ai/AIServerProcessController.cpp",
        "m_pendingModelIndex = modelIndex;\n        requestStop();",
        "m_pendingModelIndex = -1;\n        requestStop();",
        "TestAIServerProcessController",
        "TestAIServerProcessController",
        "AIServerProcessController: a deferred start() is dropped",
    ),
    # ── AIServerProcessController: a stop()/restart() we asked for exits
    #    non-zero and must NOT be reported as a crash (the owner would run its
    #    recovery restart and spin).
    Mutant(
        "src/services/ai/AIServerProcessController.cpp",
        "if (!wasStopping)\n        emit processFinished(exitCode, status == QProcess::CrashExit);",
        "if (!wasStopping || true)\n        emit processFinished(exitCode, status == QProcess::CrashExit);",
        "TestAIServerProcessController",
        "TestAIServerProcessController",
        "AIServerProcessController: a requested stop is reported as a crash",
    ),
    # ── AIServerProcessController: the kill fallback must be armed, or a
    #    wrapper that ignores terminate() (console app) never dies.
    Mutant(
        "src/services/ai/AIServerProcessController.cpp",
        "m_killTimer.start(AppConstants::AIServer::STOP_SERVER_TIMEOUT_MS);",
        "(void)0; // kill fallback disarmed",
        "TestAIServerProcessController",
        "TestAIServerProcessController",
        "AIServerProcessController: the kill fallback is not armed",
    ),
    # ── BlockingCall: the promise must be finished EXPLICITLY. A promise that
    #    is destroyed unfinished is CANCELLED by QPromise's destructor, so the
    #    result reaches the caller in the cancelled state — and the natural
    #    `if (watcher.isCanceled()) return;` guard drops it silently.
    Mutant(
        "src/utils/BlockingCall.h",
        "            promise->finish();\n            // Always released",
        "            // The promise is never finished: the result arrives cancelled.\n"
        "            // Always released",
        "TestUtilsLayer",
        "TestUtilsLayer",
        "BlockingCall: the promise is never finished (the result arrives cancelled)",
    ),
    # ── BlockingCall: an exception must not escape the worker thread. It
    #    reaches std::terminate() and ABORTS THE WHOLE PROCESS (0xC0000409).
    Mutant(
        "src/utils/BlockingCall.h",
        "            } catch (const std::exception &e) {\n"
        "                qWarning() << \"blocking::run: the task failed:\" << e.what();\n"
        "                if constexpr (!std::is_void_v<Result>) {\n"
        "                    promise->addResult(Result{});\n"
        "                }\n"
        "            } catch (...) {",
        "            } catch (const std::exception &e) {\n"
        "                throw;\n"
        "            } catch (...) {",
        "TestUtilsLayer",
        "TestUtilsLayer",
        "BlockingCall: an exception in the callable escapes the worker thread (the app aborts)",
    ),
    # ── AIInferenceGuard: an inference exception must be CONTAINED. ONNX
    #    Runtime and OpenCV throw; the exception reaches std::terminate() when
    #    it leaves a Qt slot or a QThread::run() and kills the application.
    #    (The message is still built before the rethrow, so the mutated code
    #    compiles under /W4 /WX — an unused catch parameter would fail the
    #    build and the mutant would be reported as INVALID, not KILLED.)
    Mutant(
        "src/services/ai/AIInferenceGuard.cpp",
        "        outcome.image = call(input);\n"
        "        outcome.ok = true;\n"
        "        return outcome;\n"
        "    } catch (const cv::Exception& e) {\n"
        "        outcome.error = describeException(stage, e);\n"
        "    } catch (const std::exception& e) {\n"
        "        outcome.error = describeException(stage, e);",
        "        outcome.image = call(input);\n"
        "        outcome.ok = true;\n"
        "        return outcome;\n"
        "    } catch (const cv::Exception& e) {\n"
        "        outcome.error = describeException(stage, e);\n"
        "        throw;\n"
        "    } catch (const std::exception& e) {\n"
        "        outcome.error = describeException(stage, e);\n"
        "        throw;",
        "TestAIService",
        "TestAIService",
        "AIInferenceGuard: an inference exception is not contained (the app aborts)",
    ),
    # ── VideoTrackerThread::run() is a THREAD ENTRY POINT: an exception that
    #    leaves it reaches std::terminate() and aborts the application.
    Mutant(
        "src/modules/AIProcessorPlugin/VideoTrackerThread.cpp",
        "    } catch (const std::exception &e) {\n"
        "        // Contained: the application survives and the failure is reported.\n"
        "        emit errorOccurred(QStringLiteral(\"Tracking failed: %1\")\n"
        "                               .arg(QString::fromUtf8(e.what())));\n"
        "        emit finishedTracking();\n"
        "    } catch (...) {",
        "    } catch (const std::exception &e) {\n"
        "        throw;\n"
        "    } catch (...) {",
        "TestPluginWidgets",
        "TestPluginWidgets",
        "VideoTrackerThread: an exception in the tracking loop escapes the thread (the app aborts)",
    ),
    # ── ReconstructionService: the lock split is a CLASS of decisions, not one
    #    method. Every UI-facing query must read the PUBLISHED state, never
    #    the pipeline (the worker holds the pipeline lock for the whole run).
    Mutant(
        "src/services/reconstruction/ReconstructionService.cpp",
        "    std::lock_guard pub(m_publishedMutex);\n    return m_snapshot;",
        "    std::shared_lock lock(m_mutex);\n"
        "    std::lock_guard pub(m_publishedMutex);\n    return m_snapshot;",
        "TestReconstructionService",
        "TestReconstructionService",
        "ReconstructionService: pointCloudSnapshot() waits on the pipeline lock (UI freeze)",
        600,
    ),
    Mutant(
        "src/services/reconstruction/ReconstructionService.cpp",
        "    std::lock_guard pub(m_publishedMutex);\n    return m_publishedPointCount;",
        "    std::shared_lock lock(m_mutex);\n"
        "    std::lock_guard pub(m_publishedMutex);\n    return m_publishedPointCount;",
        "TestReconstructionService",
        "TestReconstructionService",
        "ReconstructionService: pointCount() waits on the pipeline lock (UI freeze)",
        600,
    ),
    Mutant(
        "src/services/reconstruction/ReconstructionService.cpp",
        "    std::lock_guard pub(m_publishedMutex);\n    return m_snapshot != nullptr;",
        "    std::shared_lock lock(m_mutex);\n"
        "    std::lock_guard pub(m_publishedMutex);\n    return m_snapshot != nullptr;",
        "TestReconstructionService",
        "TestReconstructionService",
        "ReconstructionService: hasResult() waits on the pipeline lock (UI freeze)",
        600,
    ),
    # ── ReconstructionService: a mutator must REFUSE the change while a run
    #    is in flight, not wait for the run to release the pipeline lock.
    Mutant(
        "src/services/reconstruction/ReconstructionService.cpp",
        "    std::unique_lock<std::shared_mutex> lock(m_mutex, std::try_to_lock);\n"
        "    if (!lock.owns_lock()) {\n"
        "        qWarning() << \"ReconstructionService: the pipeline is busy — the image list was not changed.\";",
        "    std::unique_lock<std::shared_mutex> lock(m_mutex);\n"
        "    if (!lock.owns_lock()) {\n"
        "        qWarning() << \"ReconstructionService: the pipeline is busy — the image list was not changed.\";",
        "TestReconstructionService",
        "TestReconstructionService",
        "ReconstructionService: setImages() waits for the running pipeline (UI freeze)",
        600,
    ),
    # ── Architecture guard itself: an inference entry point that bypasses the
    #    exception guard must be rejected — the guard exists so an ONNX/OpenCV
    #    exception can never leave a Qt slot or a thread entry point.
    Mutant(
        "src/services/ai/AIService.cpp",
        "",
        "\nvoid mutation_check_probe(AIService* s, const cv::Mat& m)"
        " { (void)s->m_processor->runTracking(m); }"
        " // planted by mutation_check — the inference must go through the guard\n",
        _PLANT,
        "TestArchitectureBoundaries",
        "planted: an AI inference call bypasses the exception guard",
    ),
    # ── AIInferenceGuard: the guard hands the INPUT back on failure so a naive
    #    renderer cannot paint a blank frame. The SERVICE boundary must consume
    #    that flag, because the plugin reports a failure only for an EMPTY
    #    image — without this a failed inference looked exactly like a
    #    successful one that changed nothing.
    Mutant(
        "src/services/ai/AIInferenceGuard.cpp",
        "    if (!outcome.ok)\n        return {};",
        "    if (false) // mutant: a failed inference is rendered as the input\n"
        "        return {};",
        "TestAIService",
        "TestAIService",
        "AIService: a failed inference is rendered as the input frame",
    ),
    # ── AIService: a model that was never loaded cannot infer. AIProcessor
    #    returns the INPUT unchanged, which reads as "detected nothing"; the
    #    service must refuse instead (IAIService.h documents "empty on error").
    Mutant(
        "src/services/ai/AIService.cpp",
        "    if (!m_processor->isDetectionModelLoaded()) {",
        "    if (false) { // mutant: detection runs without a loaded model",
        "TestAIService",
        "TestAIService",
        "AIService: detection without a model is reported as success",
    ),
    # ── PluginHost: a plugin is third-party code, so initialize() may throw.
    #    Without the barrier the exception aborts the process before the
    #    rollback (the whole point of a failed init) can run.
    Mutant(
        "src/core/PluginHost.cpp",
        "        qWarning() << \"Plugin\" << plugin->pluginName()\n"
        "                   << \"threw an unknown exception during initialize()\";\n"
        "    }\n"
        "    return false;\n"
        "}",
        "        qWarning() << \"Plugin\" << plugin->pluginName()\n"
        "                   << \"threw an unknown exception during initialize()\";\n"
        "    }\n"
        "    return true; // mutant: a throwing plugin is reported as initialized\n"
        "}",
        "TestPluginHost",
        "TestPluginHost",
        "PluginHost: a plugin that throws in initialize() is reported as started",
    ),
    # ── AIPredictionLogService: a QRunnable runs inside QThread::run(), so an
    #    exception that leaves it reaches std::terminate(). cv::imwrite throws
    #    on a failed encode, and a diagnostic write must never kill the process
    #    it is describing.
    Mutant(
        "src/services/ai/AIPredictionLogService.cpp",
        "        } catch (const std::exception &e) {\n"
        "            qWarning() << \"[AIPredictionLog] writing\" << logPath << \"failed:\" << e.what();",
        "        } catch (const std::exception &e) {\n"
        "            (void)e;\n"
        "            throw; // mutant: the encoder exception escapes the pool thread",
        "TestPluginHelpers",
        "TestPluginHelpers",
        "AIPredictionLogService: an encoder exception escapes the pool thread",
    ),
    # ── VideoTrackerThread: the service reports a failed inference as an EMPTY
    #    image (nothing is thrown). Forwarding that frame keeps the view frozen
    #    on the last picture and the UI in Tracking mode: a silent no-op that
    #    ends with the success message.
    Mutant(
        "src/modules/AIProcessorPlugin/VideoTrackerThread.cpp",
        "            if (trackedFrame.empty()) {",
        "            if (false) { // mutant: an empty result is forwarded as a frame",
        "TestPluginWidgets",
        "TestPluginWidgets",
        "VideoTrackerThread: a failed inference is forwarded as a tracked frame",
    ),
    # ── AIProcessorPlugin: errorOccurred() is followed by finishedTracking(),
    #    so a failed run would end with the "tracking finished" box on top of
    #    the error — the failure is reported as a success too.
    Mutant(
        "src/modules/AIProcessorPlugin/AIProcessorPlugin.cpp",
        "    if (!m_trackingFailed) {",
        "    if (true) { // mutant: a failed run ends with the success box",
        "TestPluginUi",
        "TestPluginUi",
        "AIProcessorPlugin: a failed tracking run also reports success",
    ),
)

# Deleting a mutant is a decision, not an accident: the floor turns a shrinking
# gate into a failing one. It is not lowered to make a run pass — a mutant that
# no longer applies is replaced, not removed.
#
# 71 is the number of load-bearing decisions the guard layer pins today
# (PluginHost, ServiceRegistry, SecretVault, CryptoUtils, UserManager,
# LicenseKeyLedger, OtpStore, UserSettingsStore, ReconstructionService,
# ReconstructionPostProcessor, AIModelLoadService, AIServerProcessController,
# IPlugin ABI fingerprint, AppShell service registration, ChatMessage roles,
# AIProcessor model loading, AIProcessor inference dispatch,
# BlockingCall, AIInferenceGuard, VideoTrackerThread, architecture guard,
# AIService failure semantics, AIPredictionLogService pool barrier,
# VideoTrackerThread failed-inference stop, AIProcessorPlugin false success).
MIN_MUTANTS = 71


#: Header-only files on the plugin-facing surface carry load-bearing decisions
#: that the coverage gate CANNOT measure: it enumerates `.cpp` translation
#: units only, so a header is invisible to it. Without this list such a
#: decision would be checked by NOTHING — the exact blind spot that let
#: `BlockingCall.h` ship an uncontained exception and an unfinished promise.
#:
#: Every file listed here must be pinned by at least one mutant; the gate
#: fails when one of them has none.
PINNED_HEADERS: tuple[str, ...] = (
    "src/include/IPlugin.h",
    "src/include/ServiceRegistry.h",
    "src/include/ChatMessage.h",
    "src/services/mail/MailThreadGuard.h",
    "src/utils/BlockingCall.h",
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

    #: A test whose completed-run case is slow declares a larger `TIMEOUT` in
    #: tests/CMakeLists.txt; mirror it here. The pre-flight and the post-restore
    #: verification run the *complete* test, while a mutant only has to make the
    #: test fail — so the mutant's own budget stays the fail-fast one, but the
    #: green runs need the slow case to finish. Keep this map in step with the
    #: CMake property.
    _TEST_TIMEOUTS = {"TestReconstructionService": 900}

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
            if "LNK1103" in output:
                # The link read debug records that no longer match the objects
                # left by an earlier parallel compile. The linker names the
                # remedy: recompile the module. Waiting does not heal it.
                print(f"  corrupt debug info in the build tree for {target} — "
                      "forcing a clean recompile")
                self.force_recompile(target)
                continue
            if not _is_locked_output(output) or attempt == _BUILD_RETRIES - 1:
                print(f"  build output (tail):\n{result.stdout[-3000:]}\n{result.stderr[-3000:]}")
                return False
            print(f"  the build output is locked by a running process — retrying "
                  f"({attempt + 1}/{_BUILD_RETRIES})")
            time.sleep(_BUILD_RETRY_DELAY)
        return False

    def force_recompile(self, target: str) -> None:
        """Drop one target's objects and PDB so the next build recompiles it.

        ``LNK1103`` means the object files carry debug records for a PDB that
        was replaced (MSVC's ``/MP`` compiles share one PDB). Deleting the
        whole ``<target>.dir`` is not an option: CMake generates support
        scripts there (``copy_idb_pdb_*.cmake``) and only a reconfigure
        restores them. The missing ``.obj`` files are enough to make MSBuild
        recompile the target, and the PDB is regenerated.
        """
        for target_dir in self.build_dir.rglob(f"{target}.dir"):
            for obj in target_dir.rglob("*.obj"):
                obj.unlink(missing_ok=True)
        for pdb in self.build_dir.rglob(f"{target}.pdb"):
            pdb.unlink(missing_ok=True)

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
        """Write the original bytes back, retrying a transient file lock.

        A scanner, an indexer or a build can hold the source for a moment. A
        failed restore leaves the mutant in the tree, so the write is retried
        and then fails loudly: the gate must never continue as if the tree
        were clean (running gate three died exactly this way, with mutant 59
        still applied).
        """
        path = BASE / mutant.file
        for attempt in range(_RESTORE_RETRIES):
            try:
                path.write_bytes(original_bytes)
                return
            except PermissionError:
                if attempt == _RESTORE_RETRIES - 1:
                    print(f"  FATAL: {mutant.file} is STILL MUTATED — the "
                          "restore could not write it; fix the file before "
                          "running the gate again")
                    raise
                time.sleep(_RESTORE_RETRY_DELAY)

    # ── run ──────────────────────────────────────────────────────────────────

    def preflight(self, mutants: tuple[Mutant, ...]) -> None:
        self.check_sources(mutants)
        self.check_pinned_headers()
        tests = sorted({m.test for m in mutants})
        print(f"Pre-flight on unmutated tree: {len(tests)} owning test(s) must pass")
        for test in tests:
            budget = self._TEST_TIMEOUTS.get(test, 300)
            if self.run_test(test, budget):
                continue
            # A single failure right after a rebuild is usually the
            # environment, not the tree: the freshly written DLL can still be
            # locked, and a GPU-backed test can lose its device for a moment.
            # The pre-flight is a "is the environment sane?" check — so it gets
            # one retry before it is treated as a real failure. A test that
            # fails twice is a real failure.
            print(f"  pre-flight: '{test}' failed once — retrying")
            if self.run_test(test, budget):
                print(f"  pre-flight: '{test}' passed on the retry")
                continue
            sys.exit(
                f"pre-flight failed: '{test}' does not pass on the unmutated "
                "tree — fix the environment (Qt/vcpkg DLLs on PATH) before "
                "trusting any mutation result.\n"
                "A previous run killed mid-way restores the SOURCES but not "
                "the BINARIES: build/ can still hold a target compiled with a "
                "mutant, which looks exactly like a broken environment. "
                "Rebuild first: cmake --build build --config Release."
            )
        print("Pre-flight OK\n")

    def check_sources(self, mutants: tuple[Mutant, ...]) -> None:
        """Every mutant must point at a file that EXISTS and at an anchor that
        is present EXACTLY ONCE.

        A stale path used to abort the whole run with a FileNotFoundError —
        after the pre-flight had already passed, so the gate looked like it
        had run and reported nothing at all. Three commits carried a mutant
        that pointed at `src/core/ServiceRegistry.h` after the file had moved
        to `src/include/`, and the "52/52 mutant" claim was never actually
        verified. A gate that cannot find its own target must fail LOUDLY,
        BEFORE any rebuild.
        """
        problems: list[str] = []
        for mutant in mutants:
            path = BASE / mutant.file
            if not path.is_file():
                problems.append(
                    f"{mutant.file}: the file does not exist (mutant: {mutant.label!r})"
                )
                continue
            if mutant.is_plant:
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            if not any(text.count(variant) == 1
                       for variant in _anchor_variants(mutant.original)):
                problems.append(
                    f"{mutant.file}: the anchor is missing or not unique "
                    f"(mutant: {mutant.label!r})"
                )
        if problems:
            sys.exit(
                "the mutation gate is STALE — it does not match the tree:\n  - "
                + "\n  - ".join(problems)
                + "\n\nFix the mutant (new path / new anchor) in the SAME commit "
                "that changed the file. A gate that cannot find its own target "
                "must fail before it reports a result, not after."
            )
        print(f"Source check OK: {len(mutants)} mutant(s) point at existing, "
              "unambiguous anchors")

    def check_pinned_headers(self) -> None:
        """Every header-only file that carries a load-bearing decision must be
        pinned by at least one mutant.

        The coverage gate enumerates `.cpp` translation units, so a header is
        invisible to it. Without this list, a decision in a header would be
        checked by nothing at all.
        """
        referenced = {m.file for m in MUTANTS}
        missing = [header for header in PINNED_HEADERS if header not in referenced]
        if missing:
            sys.exit(
                "the mutation gate does not pin every header-only decision:\n  - "
                + "\n  - ".join(missing)
                + "\n\nEach of these files carries a load-bearing decision that the "
                "coverage gate cannot see (it enumerates .cpp translation units "
                "only). Add a mutant that inverts the decision, in the same "
                "commit that changed the header."
            )
        print(f"Header pinning OK: {len(PINNED_HEADERS)} header-only decision(s) "
              "are pinned")

    def run(self) -> int:
        selected = tuple(
            m for m in MUTANTS if self.only is None or self.only in m.label
        )
        if not selected:
            sys.exit(f"no mutant matches --only {self.only!r}")

        # The floor applies to a FULL run. --only is a developer tool for one
        # mutant, so it is exempt; deleting a mutant from the tuple is not.
        if self.only is None and len(selected) < MIN_MUTANTS:
            sys.exit(
                f"mutation gate shrank: {len(selected)} mutant(s) declared, "
                f"MIN_MUTANTS is {MIN_MUTANTS}. A new guard means a new mutant "
                f"in the same commit; an obsolete one is replaced, and the "
                f"floor is never lowered to make the run pass."
            )

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
            passed = self.run_test(
                mutant.test,
                max(mutant.timeout_seconds, self._TEST_TIMEOUTS.get(mutant.test, 0)),
            )
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
            if not self.run_test(test, self._TEST_TIMEOUTS.get(test, 300)):
                sys.exit(f"post-restore verification failed: '{test}' is not green")
        self._restored = True

    def summarize(self) -> None:
        total = len(self.killed) + len(self.survived) + len(self.stale) + len(self.invalid)
        print("-" * 72)
        print(
            f"Mutation check: {len(self.killed)}/{total} killed, "
            f"{len(self.survived)} survived, {len(self.stale)} stale, "
            f"{len(self.invalid)} invalid "
            f"(floor {MIN_MUTANTS} mutant(s))"
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
