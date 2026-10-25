# Repository guide for AI agents

## Scope first

- C++/Qt application code is in `src/`; CMake tests are in `tests/`.
- The Python service is isolated in `AIAssistant/`.
- The YOLO training/export pipeline lives in `AIComputerVision/` (`cvtrain/`
  package). Exported `Models/*.onnx` files are Git LFS artifacts and
  `Models/manifest.json` records their provenance; keep the file names in sync
  with `src/include/AIConstants.h` (the constants live there — `AppConstants.h`
  only includes it). `AIComputerVision/Docs/SCORECARD.md` holds the measured
  gates and the claims this package has pinned.
- Treat `build*/`, `.vs/`, `AIAssistant/Cache/`, `AIAssistant/logs/`, and
  `AIComputerVision/runs/` as generated output. Never inspect or edit them
  unless a task explicitly concerns a generated artifact.
- Read the smallest relevant file set. Use `rg` before opening broad folders.

## Qt kit vs vcpkg Qt (load-failure footgun)

The runtime PATH loads the local Qt kit (`C:/Qt/6.9.3/msvc2022_64/bin`)
while plain `cmake -S . -B build` resolves `find_package(Qt6 ...)` to the
vcpkg tree, which carries a *different Qt version*. Binaries built against
vcpkg Qt then fail at load time with `0xc0000139`
(STATUS_ENTRYPOINT_NOT_FOUND) or exit code 127 (ERROR_PROC_NOT_FOUND).
Always configure from Qt Creator (which sets `QT_QMAKE_EXECUTABLE`) or pass
`-DQT_QMAKE_EXECUTABLE=C:/Qt/6.9.3/msvc2022_64/bin/qmake.exe` explicitly so
every target — exe, `AppCore.dll`, plugins, tests — uses the kit Qt.
CI is unaffected: no Qt Creator there, so the vcpkg Qt stays correct.

## Offline builds (no network at configure time)

Once dependencies are provisioned (Qt kit, the vcpkg `installed` tree and a
pre-extracted ONNX Runtime) the configure/build/test loop needs **no network** —
verified end-to-end behind a dead proxy. Two knobs make that work:

- `-DVCPKG_MANIFEST_INSTALL=OFF` reuses the existing `VCPKG_INSTALLED_DIR` tree
  instead of running vcpkg's manifest install plan (which needs the internet for
  `libaec`; CI handles that with a GitHub-mirror overlay port).
- `-DFETCHCONTENT_SOURCE_DIR_ONNXRUNTIME=<dir>` removes the **only**
  configure-time download in `CMakeLists.txt` (`FetchContent`, ~323 MB GPU zip).
  The directory must be an unzipped release with `include/` and `lib/` at its
  root: the versioned `onnxruntime-win-x64-gpu-<ver>/` wrapper from the zip has
  to be stripped (that is what `FetchContent` would have produced).

```powershell
cmake -B build -S . -G "Visual Studio 17 2022" `
  -DCMAKE_TOOLCHAIN_FILE=<vcpkg>/scripts/buildsystems/vcpkg.cmake `
  -DVCPKG_INSTALLED_DIR=<vcpkg>/installed -DVCPKG_MANIFEST_INSTALL=OFF `
  -DQT_QMAKE_EXECUTABLE=C:/Qt/6.9.3/msvc2022_64/bin/qmake.exe `
  -DBUILD_TESTING=ON `
  -DFETCHCONTENT_SOURCE_DIR_ONNXRUNTIME=<extracted onnxruntime-win-x64-gpu-<ver>>
```

Needs the MSVC developer environment (`vcvars64.bat`) or Qt Creator. CI
provisions the same directory once (`ONNXRUNTIME_DIR`), caches it, and reads the
version from `CMakeLists.txt`, so Configure never downloads it.

## Build layout (AppCore split)

`src/core`, `src/utils` and `src/services` compile into **`AppCore.dll`**
(the plugin-facing ABI: interfaces + kernel + services). The executable only
contains the `src/app` shell and imports from AppCore; plugins and tests link
`AppCore` instead of the exe (no circular link, no `ENABLE_EXPORTS`). Qt
plugins and translations are deployed next to the exe by POST_BUILD steps;
`AppCore.dll` is copied there too (the CI package step picks up
`build/Release/*.dll` wholesale).

**The plugin-facing surface is `src/include/` + `src/utils/`.** The `I*.h`
interfaces, the allow-listed core glue (`Global.h`, `AppConstants.h`,
`AIMode.h`, `SignalBus.h`, `ServiceRegistry.h`) and the DTOs they expose live
in `src/include`; `src/core` holds only the **concrete** kernel
(`UserManager.h`, `SecretVault.h`, `AppConfig.h`, `PluginHost.h`, …) and is
**PRIVATE** on AppCore. A plugin that includes a concrete header therefore does
not compile — the boundary is enforced by the build graph, not only by the
regex guard. Rule 11 of `tests/CheckArchitectureBoundaries.cmake` keeps a
plugin's *include path* free of `src/core` too, so one
`target_include_directories` line cannot undo the whole thing.

Heavy third-party SDKs (VTK/OpenCV/PCL/Eigen/ONNX Runtime) are **PRIVATE** to
AppCore — they never propagate to plugin link lines. Two PCH targets encode
the heavy/light split: `SharedPCH` (`src/pch.h`: Qt + heavy SDK) for the exe,
AppCore and heavy plugins; `SharedPCHLight` (`src/pch_light.h`: Qt/STL only)
for light plugins. A plugin whose sources include VTK/OpenCV/PCL must declare
those includes/libs itself and reuse `SharedPCH`; every other plugin must
reuse `SharedPCHLight`. `tests/CheckArchitectureBoundaries.cmake` derives the
classification from the plugin sources, fails on drift, and also forbids
heavy SDK includes in `src/utils/**` (shared ABI surface — heavy code lives
in `src/services/*`, e.g. `scene/PanStyle`).

**AppCore stays a single DLL by design.** Splitting it physically into
AppContracts / AppPlatform / AppRuntime was evaluated and deliberately
deferred: the measurable defect (light plugins inheriting VTK/OpenCV/PCL/ONNX
headers and libs) is fixed by the PRIVATE heavy links + PCH split above,
without paying for extra DLL boundaries, export macros and PCH parity
problems. Revisit only if plugins from outside this repo must be supported.

## Plugin-facing service interfaces

Plugins talk to app services only through `src/core/I*.h` interfaces resolved
from the `ServiceRegistry` (`ctx->services()->get<T>()`), never through
concrete singletons or exe classes. Conventions:

- Plugin init order is resolved before any `initialize()` call by
  `PluginDependencyResolver` (compiled into AppCore): `dependencies()` are
  hard edges, `loadPhase()` is a hint over the canonical phases
  (`kernel < platform < default < ui < late`), then `loadOrder()` and
  directory-discovery order break ties. A missing dependency, duplicate
  name, unknown phase or dependency cycle aborts plugin loading with a
  concrete error. Plugin classes must not re-implement ordering logic.
- The load → resolve → initialize → cleanup lifecycle itself lives in
  `PluginHost` (AppCore), not in the exe shell: discovery filters by file
  name, a resolution failure drops every plugin, `initialize()` failure rolls
  back the already initialized plugins (the failing one included), and
  `shutdown()` cleans up in reverse init order. `IPlugin::apiVersion()` is the
  ABI gate — a plugin built against a different contract version is rejected
  instead of crashing.
- The ABI gate has a **second, automatic** dimension: `IPlugin::kAbiFingerprint`
  mixes the toolchain fingerprint with the **content hash of the plugin-facing
  interface surface** (the transitive quoted-include closure of every header in
  `src/include/` — the `I*.h` interfaces, the DTOs they expose, the
  allow-listed core glue — and `src/utils/`). CMake generates that hash into
  `InterfaceAbiHash.h` (`cmake/InterfaceAbi.cmake`, `cmake/PluginInterface.cmake`)
  whenever one of those files changes, so **changing a plugin-facing header
  without bumping `kApiVersion` can no longer silently load an incompatible
  plugin** — the host rejects it. `TestInterfaceAbi` (CTest) guards the hash
  itself: it must be fresh and must react to a change of any interface header.
  `tests/TestPluginHost.cpp` covers the contract with an injected loader;
  `tests/TestPluginSmoke.cpp` loads the real DLLs and needs the plugin
  dependencies on `PATH` (CI provides it).
- **Widget ownership is a contract, not a convention.** The shell outlives the
  plugins: a widget a plugin installed in the shell stays there as dead UI
  after `cleanup()`. Every plugin must therefore delete (synchronously — a
  `deleteLater()` may not run in time) everything it installed: docks through
  `IShellService::deleteDockWidget()`, ribbon groups by deleting the
  `AbstractRibbonPanel` (its destructor removes the group from the tab panel),
  menu-bar widgets and event filters explicitly. `TestPluginSmoke` measures
  this: after `cleanup()` the shell's widget tree must hold no object the
  plugin installed, and the check is not vacuous (the plugin must have
  installed something in the first place).
- The plugin's *library* stays mapped for the whole session: Qt's
  `QPluginLoader` pins it with `QLibrary::PreventUnloadHint` and the load
  hints can only be OR-ed, never cleared. `PluginHost::shutdown()` still
  calls `unload()`, which destroys the plugin's root component (and its whole
  QObject tree) — `PluginHost::releasedPluginCount()` makes that observable.
  So a surviving object is dead UI, not an immediate crash; the contract
  above is still what has to hold.
- `IAppContext` no longer exposes the concrete `QMainWindow` (the deprecated
  `mainWindow()` was removed): shell interaction goes through
  `IShellService` (`shell()`), resolved from the context or registry. A
  plugin must not hunt for the shell among `QApplication::topLevelWidgets()`
  either — the architecture guard rejects that.

- QObject-based interfaces (`ILanguageService`, `IReconstructionService`)
  derive **only** from `QObject` via the interface — concrete classes inherit
  the interface alone (no duplicate `QObject` base, no diamond). Signals live
  on the interface, not the impl. Export with `APP_EXPORT`.
- Plain interfaces (`IViewerService`, `ISettingsService`) need no signals.
- `ISceneService` is the documented exception that exposes VTK types — keep it
  to actor/renderer plumbing.
- The shell-owned progress dialog is `IProgressDialog`; `CustomProgressDialog`
  remains an AppCore implementation detail. Plugins must not include the
  concrete widget header (the old contract leaked it into every plugin ABI).
- AI model loading is owned by AppCore `AIModelLoadService`
  (`IAIModelLoadService`), not by `AIProcessorPlugin`: an AppCore worker can
  outlive or be cancelled during plugin unload without executing plugin code.
- Translation goes through `ILanguageService` / `IAppContext::translate()`;
  the `LM`/`LM_TR` macros are gone. `UserAuthPlugin::translate()` is a
  module-scoped facade over the registry-bound service.
- `UserManager::instance()` appears in exactly one production site: the
  AppShell registration (composition root). Everything else — plugins,
  dialogs, AppCore services — resolves `IUserManager` from the registry
  (ctor-injected where the component has no context).
- **`IUserManager` is the AGGREGATE, not the contract to depend on.** The
  user service is split into `IAuthService` (sign in/out, the current
  identity, the password/OTP surface), `IUserProfileService` (the profile of
  the current account), `IUserAdminService` (administering the accounts),
  `ISmtpCredentialService` (the outgoing-mail credentials),
  `IUserPreferences` (per-user preferences and secrets) and
  `ILicenseService` (licence queries and activation). Each has its own
  `ServiceRegistry` key; the composition root registers the SAME
  `UserManager` instance under all of them. A consumer that needs one
  capability must depend on that one interface — the plugins that only want
  "who am I" + "my preferences" resolve `IAuthService` +
  `IUserPreferences`, never the aggregate. `UserTypes.h` holds the value
  types (`UserInfo`, `AuthError`, `UserRole`) so the narrow interfaces can
  share them without a circular include.
- `LanguageManager` is owned by AppShell (no singleton). `AppConfig::bootstrap()`
  is the bootstrap/composition-root accessor (Main, AppShell, Logger only);
  every other component resolves `IAppPaths` from the registry. The
  architecture guard enforces both confinement rules.
- Tab ids and build order live in `AppConstants::TabIds` (`UIConstants.h`);
  plugins look panels up by id only — there is no translated-label fallback,
  so renaming a language can never change panel routing.
- License keys are stored DPAPI-encrypted (`SecretVault`, `enc1:`) with
  SHA-256-hashed INI path buckets; legacy plaintext entries migrate on load.
- **The vault fails closed.** On a platform that HAS an OS backend, a failing
  protection call must never degrade into a plaintext write: `encrypt()` then
  returns an empty string and the caller keeps the value it already has
  (`UserSettingsStore::storeSecret`, `LicenseKeyLedger::save`). On a platform
  without a backend the value is stored marked as `plain:` — a documented
  platform limit, never a silent fallback. `SecretVault::protectionFailureFallback()`
  is the named decision (and `setProtectionFailureForTesting()` is the seam
  that lets the callers' fail-closed behaviour be tested).
- **Two namespaces, two confidentiality levels.** A *preference* is stored in
  the clear; a *secret* (mail password, AI provider API key) is stored through
  the OS vault. `UserSettingsStore::secretKeys()` declares which keys are
  secrets: whichever setter a caller uses, a secret key is encrypted and the
  plaintext copy is retired. The AI provider key used to be written in the
  clear through `setUserPref` — that hole is closed at the storage layer, so
  a call site cannot reopen it.
- **A password reset needs a proof of ownership.** `verifyOtp()` records the
  proof in the core; `resetPassword(email, newPass, …)` consumes it, so one
  OTP changes one password and no holder of `IUserManager` can reset a
  password without having proven ownership. The reset is subject to the same
  password policy as `changePassword`.
- Password hashes are PBKDF2-HMAC-SHA256 through OpenSSL (the mail stack
  already needs libcrypto at runtime, so this adds no DLL). The stored
  format is unchanged, so existing credentials keep working. The work factor
  comes from the credential file and is therefore capped
  (`kMaxPbkdf2Iterations`) — a tampered file must not turn a login into a
  hang.
- **A long-running worker must not hold the lock the UI reads.** The
  reconstruction worker holds the pipeline lock (`ReconstructionService::
  m_mutex`) **exclusively for the whole run**, so *every* UI-facing method of
  `IReconstructionService` must read the **published** state instead
  (`m_publishedMutex`) and every mutator must **refuse** the change
  (`try_to_lock`) rather than wait. A query that takes the pipeline lock
  freezes the GUI for the entire reconstruction; a mutator that waits does the
  same. The published snapshot and point count are written by the worker when
  the run **ends** (successful or cancelled — the UI must see the state of the
  run that just ended, never the state of an older one), so the UI never
  touches the pipeline — not even to count its points. Lock order is
  invariant: `m_mutex` → `m_publishedMutex`, never the other way round.

## Blocking I/O contracts

- `IMailService` (and the mail stack under it) is synchronous and blocking:
  callers must run it off the GUI thread via
  `blocking::run(...)` (pattern: MailDockWidget, MailSettingsDialog,
  ForgotPasswordDialog, SettingsDialog).  `blocking::run` lives in
  `src/utils/BlockingCall.h` and always uses a dedicated `QThread`, so the
  "never on the GUI thread" part of the contract is guaranteed.  Do **not**
  use `QtConcurrent::run` for blocking work: a pool may execute the task
  *inline* on the calling thread (QThreadPool::doStart), which freezes the
  GUI for the whole socket timeout.  `mail::enforceBackgroundThread()` warns
  in every build and asserts in debug builds when a blocking entry point is
  invoked on the GUI thread — it is the last line of defence, not the
  mechanism.
- Teardown paths bound every process wait with an explicit timeout
  (`waitForFinished(<ms>)`; the argument-less call waits Qt's 30 s default and
  freezes shutdown). The architecture guard rejects new un-timed calls.
- A background task must not outlive what it uses. `blocking::run(callable,
  &tracker)` registers the task in a `blocking::TaskTracker`; the owner waits
  for it (bounded) in its teardown path, *before* it releases the service the
  task calls. A task that outlived the service would call into a destroyed
  object. The plugin layer does this in the destructors of `MailDockWidget`,
  `MailSettingsDialog`, `SettingsDialog` and `ForgotPasswordDialog`.
- **AI inference from a GUI slot is blocking work too.** `AIProcessorPlugin`
  dispatches `IAIService::runDetection` / `runSegmentation` through
  `dispatchInference()` → `blocking::run` (a dedicated `QThread`), applies the
  result in a `QFutureWatcher<cv::Mat>`, keeps **one** inference in flight
  (the ONNX session is shared) and joins the task in `cleanup()` with a bounded
  wait before the widgets are deleted. A new GUI entry point that calls
  `IAIService::run*` directly re-opens the freeze that the architecture guard
  cannot see — its regex rejects `QtConcurrent::run`, not a synchronous call.
  `TestPluginUi` asserts the worker thread id, the one-at-a-time rule and the
  contained failure; two mutants pin them.
- `blocking::run` makes **two** guarantees, both pinned by mutants:
  1. the returned future is **completed**, never *cancelled*. `QPromise`'s
     destructor cancels a promise that was never finished, so a helper that
     only calls `addResult()` hands the caller a cancelled future — and the
     natural `if (watcher.isCanceled()) return;` guard then drops the result
     silently. `run()` calls `finish()` itself.
  2. an exception thrown by the callable is **contained**: it is logged and
     the `TaskTracker` semaphore is released regardless. An exception that
     leaves `QThread::run()` reaches `std::terminate()` and aborts the whole
     application. The exception is deliberately **not** stored on the
     promise: `QFuture::result()` re-throws a stored exception, and that
     re-throw happens in the *caller's* slot — back to `std::terminate()`,
     one stack frame further from the cause. A callable that must report a
     domain error catches it itself and returns it in the result (the mail
     stack does that: `errorMsg` is filled by the service).

## Exception barriers

An exception that leaves a **Qt slot** or a **`QThread::run()`** reaches
`std::terminate()` and aborts the whole application (0xC0000409). ONNX
Runtime, OpenCV and the allocator all throw, so every entry point that can
reach them needs a barrier.

- **AI inference** goes through one place: `ai::runInferenceGuarded()` /
  `ai::runDetectionGuarded()` (`src/services/ai/AIInferenceGuard.h`). The
  service boundary `AIService` routes all four inference entry points
  through it; architecture-guard rule 12 requires *one guard per raw
  `m_processor->…` call* in `AIService.cpp`, so a new entry point cannot
  forget the guard.
- **A thread entry point** wraps its body and reports the failure through a
  signal (`VideoTrackerThread::run()` emits `errorOccurred` +
  `finishedTracking`). An exception that escapes is a process abort.
- **A blocking call on a worker thread** is contained by `blocking::run`
  (see above).
- The model-**load** path already had its barrier (`AIProcessor::
  loadDetectionModel` and friends). Keep the two paths in step: a new
  inference entry point is a new mutant in the same commit.

## Desktop agent actions and chat sessions

- Demo fixtures for desktop-agent actions come from
  `Config/agent_action_manifest.json` → top-level `sample_data`
  (`AgentActionManifest::sampleData` / `applySampleDefaults`). The AI service
  fills `path` / `folder` parameters before emitting
  `SignalBus::agentUiActionRequested`; plugins must consume those parameters
  and never hard-code demo paths. `AIAssistant/tests/**` only requires the
  `actions` array, extra top-level keys are safe.
- Chat sessions are owned by `ChatSessionManager` (session list, current
  pointer, attachment/thumbnail cleanup on `deleteSession`); persistence stays
  in `ChatSessionStore` (username + config path injected — no singleton).
  A message is the **typed** `ChatMessage` (`src/include/ChatMessage.h`): the
  role is a closed enum (`ChatRole`), the attachments are a path list, and
  `isModelRole()` is the single decision about what the model may see — the
  UI-only agent trace can no longer leak into the request (it used to be a
  free-form string and once came back as HTTP 422). The wire and disk format
  is unchanged (`toJson()`/`fromJson()`), so a sealed history still loads.
  `AIAssistant` orchestrates sessions, payload building and network replies;
  the QProcess/readiness handshake lives in `AIServerProcessController`, and
  pending/queued request bookkeeping lives in `AIRequestQueue`. `AIAssistant`
  takes an optional injected `QNetworkAccessManager` and exposes the protected
  `markServerReady()` transition so `TestAIAssistant` can drive the request
  state machine without sockets or a Python child process.
- Password hashes never leave `UserAccountStore` (owned behind `UserManager`);
  `UserInfo` is profile-only, so `updateUser()` cannot clobber credentials.
  Tests observe the hash through the INI (`User_<name>/password_hash`).
  `UserManager` is a facade over `CryptoUtils` (PBKDF2/constant-time helpers),
  `LicenseKeyLedger`, `OtpStore` and `UserSettingsStore`; keep new auth data
  formats inside those collaborators, not back in the facade.
- `src/core/AppManifest.json` is an authored, tracked contract file (read by
  `AIComputerVision/tests/test_contract.py`); it is not generated by CMake.

## Required checks

- C++ changes: configure/build the affected target; run the relevant CTest
  tests when dependencies are available. `tests/CMakeLists.txt` forces
  `QT_ASSUME_STDERR_HAS_CONSOLE=1` for every test so QtTest output (which goes
  to stderr) survives `ctest --output-on-failure`; without it Windows drops
  the report and failures look empty. First-party targets compile with
  `/W4 /WX` (`/external:anglebrackets /external:W0` keeps SDK headers out of
  the stream): a new warning fails the build instead of accumulating.
- Guard-layer C++ changes (`src/core/Plugin*`, `SecretVault`, `UserManager`,
  `AppShell`, chat-session/settings/shell services): run
  `python tests/coverage_gate.py` (measured floors; Release must ship PDBs —
  CMakeLists.txt section 5b) and `python tests/mutation_check.py` (inverts one
  decision per guard; the owning test must fail). Both need Qt + vcpkg `bin` on
  `PATH` and are enforced in the C++ workflow after the test step. See
  `tests/README.md` for the floor table and the coverage/mutation split for
  header-only guards.
- The same rule now covers the **service layer** guards that carry a load-bearing
  decision: `ReconstructionService` (published state vs. the pipeline lock the
  worker holds), `AIModelLoadService` (a throwing loader is a failure) and
  `AIServerProcessController` (non-blocking stop, deferred start, kill
  fallback). A change to any of them needs a new mutant in the same commit and
  a test that fails when the decision is inverted.
- **A guard must pin the CLASS of decisions, not one representative.** A guard
  that was added to a single method of a class left its siblings unprotected,
  and the live defects were exactly there. Two classes were closed this way
  (`Docs/ARCHITECTURE_SRC.md` §4.8, §4.10): the pipeline lock (8 methods of
  `IReconstructionService`, 4 mutants) and the exception barrier
  (`AIInferenceGuard`, `VideoTrackerThread::run`, `blocking::run`, 3 mutants).
  When you add a guard, ask which siblings share the decision and pin them
  too.
- **Header-only code is invisible to the coverage gate** (it enumerates `.cpp`
  translation units). `tests/mutation_check.py` therefore carries a
  `PINNED_HEADERS` list: every header-only file that carries a load-bearing
  decision must have at least one mutant, or the gate fails. Add your header
  to that list in the same commit that adds its decision.
- The coverage gate covers the **whole** `src/` tree, not only the guard
  layer: every translation unit must appear in the merged report
  (completeness check) and every measured file carries a floor. A new `.cpp`
  that no test can reach fails the gate until it is reachable or explicitly
  exempted with a reason. Do **not** enable the unity build for a test whose
  subject is a source file (`tests/CMakeLists.txt` sets `UNITY_BUILD OFF`
  there): an inlined-away file emits no code, the PDB forgets it and it
  measures 0 % silently. The same applies to the application target
  (`CMakeLists.txt`).
- **The coverage floor is a one-way street.** `tests/coverage_floors.json` is
  the committed baseline; the gate fails when a floor is *lowered* instead of
  the code being fixed, and when a floor has no baseline entry. Raising a
  floor and the baseline happen in the same commit
  (`python tests/coverage_gate.py --update-baseline`). A coverage regression
  is fixed in the code, never in the number.
- The C++ side of the "10/10" contract (the five measurable criteria and the
  remaining debt) is tracked in `Docs/ARCHITECTURE_SRC.md`.
- **Test targets compile with `/utf-8`** (`tests/CMakeLists.txt`): the test
  sources contain Vietnamese literals, and without the flag MSVC decodes them
  with the system code page — the expected strings end up mangled and the
  assertions compare garbage. The application targets already had the flag.
- The **plugin layer** (`src/modules/**`) is measured through
  `TestPluginSmoke`: every shipped plugin is initialized and cleaned up
  against a stub context (`tests/core/mock/PluginContextMock.h`) — one plugin
  per **child process** (the test executable re-run with `--init-plugin
  <name>`), so widget ownership cannot leak from one plugin into the next and
  a crash becomes a reported failure instead of a dead suite. The coverage
  exporter needs `--cover_children` for this. Real exit-crash bugs were found
  this way (`AIProcessorPlugin` deleted a `QWidget` through
  `deleteDockWidget(QDockWidget*)`; `ViewerPlugin::cleanup()` was empty) — a
  plugin must delete every widget it installs into the shell **before** the
  DLL is unloaded. The same test now asserts the ownership contract
  directly: after `cleanup()` the shell's widget tree must hold no object the
  plugin installed, and the plugin must have installed something (otherwise
  the check would be vacuous).
- Python changes: from `AIAssistant/`, run the whole gate — `ruff check .`,
  `python -m compileall -q evals src`, `python -m mypy src/ai_assistant`,
  `PYTHONPATH=src lint-imports --no-logo`, then
  `PYTHONPATH=src python -m coverage run -m unittest discover -s tests -p "test_*.py"`
  and `python -m coverage report` (the coverage gate is 100% statements **and**
  branches), then `PYTHONPATH=src python scripts/mutation_check.py`.
  `StartChatbotServer.py` was removed with the V6 refactor; the syntax check
  referenced a file that no longer exists. The mutation gate pins the package's
  load-bearing decisions (150 mutants, ADR 0032–0041) and declares a **floor**
  (`MIN_MUTANTS`): deleting a mutant is a decision, and the gate turns it into
  a failing one instead of a silent one. A new guard means a new mutant in the
  same commit. Measured gates, the debt table and the claim-to-test map live in
  `AIAssistant/Docs/SCORECARD.md` — the same "claim must be pinned or removed"
  rule as ADR 0015. The scorecard is **tracked**: a number nobody else can read
  pins nothing.
- AIComputerVision changes: from `AIComputerVision/`, install the gates from the
  hash-pinned lock first (`pip install --require-hashes -r
  requirements-test.lock` — ruff/mypy/coverage *decide* green or red, so they
  are pinned like the AIAssistant side, ADR 0012), then run `ruff check .`,
  `python -m mypy`, `coverage run -m unittest discover -s tests` and
  `coverage report` (the coverage gate is 100% statements + branches), then
  `python scripts/mutation_check.py`. The mutation gate pins the package's
  load-bearing decisions (101 mutants, ADR 0007/0008/0009/0011/0012/0013) and
  declares the same **floor** (`MIN_MUTANTS = 101`); it runs in a sandbox copy of
  the tree, so it never touches `Models/`, `runs/` or the local dataset, and
  `--files` runs a subset (the Windows job re-sweeps `cvtrain/fsio.py`, the
  platform-sensitive decisions) without ever lowering the floor. A new guard
  means a new mutant in the same commit. The gates live in
  `.github/workflows/Build-cvtrain-ci.yml` and run on `ubuntu-latest` **and**
  `windows-latest` — the product ships on Windows. Measured gates, the debt
  table and the claim-to-test map live in `AIComputerVision/Docs/SCORECARD.md` —
  the same "claim must be pinned or removed" rule as `AIAssistant` ADR 0015.
- Workflow changes: keep `workflow_dispatch`, path filters, and a finite job
  timeout. Do not cancel long dependency-bootstrap jobs. Keep the ONNX Runtime
  provision/cache steps (`ONNXRUNTIME_DIR`) and their
  `-DFETCHCONTENT_SOURCE_DIR_ONNXRUNTIME` configure flag in both C++ workflows —
  they are what keeps Configure from downloading the 323 MB zip on every run.

## Change discipline

- Keep C++ and Python changes independent unless the interface requires both.
- Do not commit model weights outside Git LFS, caches, logs, build trees, IDE
  state, or secrets.
- Prefer one focused change plus a verification result over broad rewrites.
