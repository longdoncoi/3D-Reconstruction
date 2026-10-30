# C++ verification stack

Four gates prove the C++ side of the repository. All of them run locally with
the commands below and are enforced in `.github/workflows/Build-C++-ci.yml`
after the build step. The reference run recorded here is
2026-10, Windows 11, MSVC 2022 (VS 17.12), Qt 6.9.3, Release.

| # | Gate | Command | What it proves |
|---|------|---------|----------------|
| 1 | Architecture boundaries | `cmake -DSOURCE_DIR=<repo> -P tests/CheckArchitectureBoundaries.cmake` | Layering invariants (heavy/light PCH split, plugin includes, plugin **include path**, blocking-I/O waits, singleton confinement, event-filter ownership, shell-window lookup …). Runs in CTest as `TestArchitectureBoundaries`. |
| 2 | Unit + integration suite | `ctest --test-dir build -C Release --output-on-failure` | 51 CTest entries (44 test executables + 7 CMake-script tests): kernel units, plugin host/resolver with an injected loader, real-plugin smoke, `TestAppShell` composition-root wiring, `TestMainWindow` (the tab-panel contract a plugin resolves by tab id), `TestAIAssistant` request state machine (injected transport), AppCore-owned model-load lifetime (`TestAIModelLoadService`), the non-blocking server-process teardown (`TestAIServerProcessController`, which re-runs the test executable as a stand-in for the Python wrapper), the interface ABI guard (`TestInterfaceAbi`), the service-key uniqueness guard (`TestServiceKeys`), the AI service against the REAL ONNX models (`TestAIService` — needs `git lfs pull`), the at-rest cipher shared with the Python service (`TestAtRestCipher`), the at-rest confidentiality of the user INI (`TestSecretProtection`), the shared utility layer (`TestUtilsLayer`), the scene/viewer layers (`TestViewerService` covers 2-D paging and auto-navigation with a fake scene; `TestSceneService` drives the MPR layout through a headless render window), the reconstruction service boundary (`TestReconstructionService` — the UI-facing queries and mutators must not wait for the worker, and the published result must be complete), plus `TestReconstructionFinalize` and `TestReconstructionService` (both labelled `slow` — they need a completed reconstruction). |
| 3 | Coverage gate | `python tests/coverage_gate.py` | Measured line coverage of the kernel/guard, composition, service and utility files, with per-file floors (the minimum of the local and CI reference environments), a completeness check (no translation unit may escape the measurement), a **one-way ratchet** (a floor may never be lowered below the committed baseline), and a printed debt list; runs OpenCppCoverage over the owning tests. |
| 4 | Mutation check | `python tests/mutation_check.py` | Every load-bearing guard decision is *pinned*: one inverted decision per mutant, rebuilt and re-run; the owning test must fail. |

Run them from the repository root with Qt and the vcpkg `bin` directory on
`PATH` (the same environment the CTest step uses), e.g.:

```powershell
$env:PATH = "C:/Qt/6.9.3/msvc2022_64/bin;$env:VCPKG_INSTALLED_DIR/x64-windows/bin;$env:PATH"
ctest --test-dir build -C Release -LE slow --output-on-failure   # fast round
python tests/coverage_gate.py
python tests/mutation_check.py
```

## Gate 1b — the plugin-facing interface ABI

`IPlugin::kApiVersion` is a number somebody has to remember to bump when a
plugin-facing header changes. `IPlugin::kAbiFingerprint` closes that hole
automatically: it mixes the toolchain fingerprint, the **SDK-version
fingerprint** (VTK / OpenCV / ONNX Runtime — the libraries whose types cross
the boundary) and the **content hash of the plugin-facing interface surface**,
so a plugin built against different headers or a different SDK is rejected by
`PluginHost` instead of being loaded against a stale ABI.

| Piece | Where | What it does |
|-------|-------|--------------|
| the surface (one definition) | `cmake/PluginInterface.cmake` | every `src/include/*.h` (the `I*.h` interfaces, the allow-listed core glue and the DTOs they expose) + `src/utils/*.h`; `src/core` and `src/services/**` are PRIVATE on AppCore |
| the hash | `cmake/InterfaceAbi.cmake` | stable over file **name + content** (not the absolute path, so the value is the same on every machine); follows both `"..."` and `<...>` includes when the target is inside the surface |
| the SDK fingerprint | `cmake/InterfaceAbi.cmake` (`compute_sdk_fingerprint`) | SHA-256 of `VTK_VERSION\|OpenCV_VERSION\|ONNXRUNTIME_VERSION`; a stamp file in the custom command's `DEPENDS` regenerates the header when a dependency is bumped |
| the generator | `cmake/GenerateInterfaceAbi.cmake` | writes `InterfaceAbiHash.h`; the custom command depends on **every** file of the surface, so the hash can never go stale |
| the guard | `tests/CheckInterfaceAbi.cmake` (`TestInterfaceAbi`) | **freshness** (the generated hash equals a fresh computation) + **sensitivity** (changing any interface header changes the hash) + the same two properties for the SDK half |
| the consumer | `src/include/IPlugin.h` | `kInterfaceHash` + `kSdkFingerprint` → `kAbiFingerprint`; `TestPluginHost` pins that the fingerprint really carries both |

The surface is the **transitive** `#include` closure, not just the `I*.h`
files: a plugin sees `IReconstructionService.h` *and* the DTO it includes
(`PointCloudData.h`), so hashing only the interface files would miss a change
to that DTO. Angle-bracket spellings are part of the closure too — `src/include`
is on the include path, so `<PointCloudData.h>` compiles and a one-character
edit must not drop the DTO out of the hash.

`TestPluginSmoke` is the end-to-end proof: it loads the **real** plugin DLLs
through the ABI gate, so a hash that differs between `AppCore` and a plugin
turns the test red.

## Gate 3 — coverage floors

`tests/coverage_gate.py` runs the owning tests under OpenCppCoverage, merges
the per-test cobertura reports (a line counts as covered when *any* test
executes it) and enforces four things:

1. **Per-file floors** — the ratchet in `FLOORS`.
2. **Completeness** — every translation unit of `src/` that a test can reach
   must appear in the merged report. A file that goes missing fails the run:
   that happens when it is not compiled into an instrumented module, or when
   the unity build inlined it away (see below). Four units are exempt, each
   with a recorded reason (`_NOT_MEASURABLE` in the script): the two PCH
   carriers, `src/app/main.cpp` (the application entry point, exercised by
   running the application itself) and
   `src/services/scene/panstyle.cpp` (a `vtkStandardNewMacro` boilerplate).
3. **The ratchet is a one-way street** — `tests/coverage_floors.json` is the
   committed baseline. The gate fails when a floor is *lowered* (a coverage
   regression has to be fixed in the code, not in the number) and when a floor
   has no baseline entry. Raising a floor and the baseline happen in the same
   commit: `python tests/coverage_gate.py --update-baseline`.
4. **Debt visibility** — the files with 0 % are printed on every run with
   their line count, so the remaining work is named instead of implied.

A floor is the **conservative minimum of the supported reference
environments** — local VS 17.12 / Qt 6.9.3 and CI `windows-latest` (VS 18 /
Qt 6.5.3) — not the value of one machine. The compiler and PDB decide how many
executable lines a source file has (VS 18 attributes 177 lines to
`reconstructionservice.cpp`, VS 17.12 attributes 174 to the same code), so a
floor set to the local percentage fails a supported build without a coverage
regression. The table below records the local measurement and the CI value in
the note; the floor is the lower whole percent. Floors are raised deliberately
when the code genuinely gains coverage, never automatically from a single
measurement.

Scope: the kernel/guard layer, the composition root, the **service layer**,
the **shared utility layer** and the **plugin layer**.
The plugin DLLs are instrumented for `TestPluginSmoke`, so `src/modules/**`
no longer disappears from the report.

**Plugin measurement runs in child processes.** `TestPluginSmoke` initializes
and cleans up every shipped plugin against a stub context
(`tests/core/mock/PluginContextMock.h`) — but one plugin per **child
process**: the plugins install their widgets into the shell's widget tree,
and sharing one process leaks widget ownership from one plugin into the next
(plug A's widget is still parented to a shell widget when plug B tears its
own tree down).  The child is the test executable itself, re-run with
`--init-plugin <name>`; a crash in one plugin becomes a reported failure
instead of a dead suite.  The coverage exporter needs `--cover_children` to
see the children's code — that flag is set in `coverage_gate.py`.

This test found two real defects (both fixed):

* `AIProcessorPlugin::cleanup()` handed its `AITrainDockWidget` (a plain
  `QWidget`) to `IShellService::deleteDockWidget(QDockWidget*)` — the shell
  deleted it through the wrong vtable, **corrupting the heap** and killing
  the process on exit.
* `ViewerPlugin::cleanup()` was empty — the image list and the navigator it
  installs into the main window outlived the plugin DLL, so Qt deleted
  widgets whose vtable had already been unloaded (crash on exit).
* `MailMessageFormatter::decodeQuotedPrintable()` round-tripped the whole
  body through Latin-1 bytes and decoded the result as UTF-8 — every
  non-Latin-1 character (a plain-text Vietnamese mail body) was replaced by
  U+FFFD.  Only the `=XX` escapes are byte sequences; the rest of the text
  must stay untouched.

**The plugin library stays mapped.** Qt's `QPluginLoader` pins the library
with `QLibrary::PreventUnloadHint`, and `QLibraryPrivate::mergeLoadHints()`
can only OR the hints — the pin can never be removed. So the "the host
unloads the plugin DLL" wording of the older contract was never true and
cannot be made true with `QPluginLoader`. What the host *does* do is call
`unload()`, which destroys the plugin's root component and with it the whole
QObject tree; `PluginHost::releasedPluginCount()` makes that observable and
the smoke test asserts it. A plugin object that outlives `cleanup()` is
therefore dead UI rather than an immediate crash — the contract that has to
hold is the widget ownership one above.

Two more defects were found by the plugin-helper tests:

* `SmtpMailer::sendMail()` validated `to.isEmpty()` on the *list* of
  recipients, but the single-recipient overload wraps its argument into a
  list — an empty recipient therefore slipped through and the message was
  handed to the server without any `RCPT TO`.  The check now looks for at
  least one usable address.
* `tests/mutation_check.py` could die with a `UnicodeEncodeError` while
  printing a build log on a cp1252 console; the gate now reconfigures its
  streams with `errors="replace"` so it reports the mutant instead of
  crashing on the way to the report.

### Unity build vs. measurability

The exporter maps executed lines through the PDB. When a source file shares a
unity translation unit with its neighbours, MSVC can inline all of it away —
no code is emitted, the PDB does not mention the file and the exporter
reports an empty class. The file then measures 0 % *silently* (this is what
happened to `CrosshairGeometry.cpp`). The fix is applied in two places:

* `tests/CMakeLists.txt`: `UNITY_BUILD OFF` for every test whose *subject* is
  a source file (`TestCrosshairGeometry`, `TestMailMimeParser`,
  `TestCameraParamsParser`, `TestPointCloudFilter`, `TestAIYoloPostProcessor`).
* `CMakeLists.txt`: `UNITY_BUILD OFF` for the application target, so the app
  shell's own translation units are visible as well. The PCH still absorbs the
  heavy VTK/PCL headers (rebuild of the app target: ~57 s).

`AppCore` and the plugins keep the unity build: all of their translation
units are visible, and the completeness check catches a new one that is not.

The same effect hides *single functions*: a small helper that the compiler
inlines into its only caller is reported as never executed, even though the
code ran. `src/core/AtRestCipher.cpp` hits that case with its key-file reader
(`readKeyFile` — the compiler folded it into `loadKey`, and the exporter
attributed the hits to the call site). The helper is therefore marked
`ATREST_NOINLINE`, so the measurement stays honest instead of silently
reporting a gap.

### Measured reference (2026-10, MSVC 2022 / VS 17.12, Release)

**composition root**

| file | measured | floor |
|------|----------|-------|
| `src/app/appshell.cpp` | 100.0 % (72/72) | 100 |
| `src/app/mainwindow.cpp` | 70.7 % (123/174) | 70 |

**kernel + guard**

| file | measured | floor |
|------|----------|-------|
| `src/core/atrestcipher.cpp` | 100.0 % (142/142) | 100 |
| `src/core/cryptoutils.cpp` | 100.0 % (43/43) | 100 |
| `src/core/licensekeyledger.cpp` | 100.0 % (91/91) | 100 |
| `src/core/otpstore.cpp` | 100.0 % (22/22) | 100 |
| `src/core/signalbus.cpp` | 100.0 % (2/2) | 100 |
| `src/core/useraccountstore.cpp` | 100.0 % (107/107) | 100 |
| `src/core/usermanager.cpp` | 100.0 % (351/351) | 100 |
| `src/core/usersettingsstore.cpp` | 100.0 % (106/106) | 100 |
| `src/core/appconfig.cpp` | 98.8 % (79/80) | 98 |
| `src/core/pluginhost.cpp` | 98.1 % (152/155) | 97 |
| `src/core/languagemanager.cpp` | 96.5 % (55/57) | 96 |
| `src/core/secretvault.cpp` | 94.6 % (53/56) | 94 |
| `src/core/plugindependencyresolver.cpp` | 93.3 % (84/90) | 93 |

**service: ai**

| file | measured | floor |
|------|----------|-------|
| `src/services/ai/aimodelloadservice.cpp` | 100.0 % (47/47) | 100 |
| `src/services/ai/aipredictionlogservice.cpp` | 100.0 % (35/35) | 100 |
| `src/services/ai/airequestqueue.cpp` | 100.0 % (62/62) | 100 |
| `src/services/ai/aiyolopostprocessor.cpp` | 100.0 % (51/51) | 100 |
| `src/services/ai/chatsessionmanager.cpp` | 100.0 % (78/78) | 100 |
| `src/services/ai/chatsessionstore.cpp` | 100.0 % (62/62) | 100 |
| `src/services/ai/chatmessage.cpp` | 88.6 % (39/44) | 88 |
| `src/services/ai/agentactionmanifest.cpp` | 100.0 % (70/70) | 100 |
| `src/services/ai/aiassistant.cpp` | 55.7 % (306/549) | 55 |
| `src/services/ai/aiserverprocesscontroller.cpp` | 94.7 % (107/113) | 94 |
| `src/services/ai/aiinferenceguard.cpp` | 75.0 % (30/40) | 75 |
| `src/services/ai/aiservice.cpp` | 86.5 % (45/52) | 86 |
| `src/services/ai/aiprocessor.cpp` | 17.7 % (53/299) | 17 |

`aiserverprocesscontroller.cpp` was measured at 34.6 % before the non-blocking
stop rework; `TestAIServerProcessController` now drives the whole process
lifecycle (launch, deferred start, kill fallback, unexpected exit, destructor
reap) and the file measures **94.7 %** — the floor was raised to **94** in the
same commit as the measurement (`python tests/coverage_gate.py --update-baseline`).

`reconstructionpostprocessor.cpp` went the same way: the densifier, the Poisson
mesher and the texturing stage had no test at all (20.0 %). They are pure
CPU-side geometry, so `TestReconstructionPipeline` now covers them
(96.3 %, floor **96**) — including the invariant that a Poisson face never
references a vertex the trim removed.

**service: mail**

| file | measured | floor |
|------|----------|-------|
| `src/services/mail/smtpoutbox.cpp` | 87.5 % (14/16) | 87 |
| `src/services/mail/mailmimeparser.cpp` | 67.1 % (155/231) | 67 |
| `src/services/mail/mailservice.cpp` | 32.2 % (78/242) | 32 |
| `src/services/mail/smtpmailer.cpp` | 21.6 % (37/171) | 21 |

**service: reconstruction**

| file | measured | floor |
|------|----------|-------|
| `src/services/reconstruction/featureextractor.cpp` | 100.0 % (8/8) | 100 |
| `src/services/reconstruction/multiviewtriangulator.cpp` | 100.0 % (76/76) | 100 |
| `src/services/reconstruction/reconstructionworker.cpp` | 100.0 % (4/4) | 100 |
| `src/services/reconstruction/sfmstage.cpp` | 100.0 % (9/9) | 100 |
| `src/services/reconstruction/triangulator.cpp` | 100.0 % (10/10) | 100 |
| `src/services/reconstruction/featurematcher.cpp` | 97.7 % (42/43) | 97 |
| `src/services/reconstruction/cameraparamsparser.cpp` | 97.6 % (41/42) | 97 |
| `src/services/reconstruction/pointcloudfilter.cpp` | 97.2 % (104/107) | 97 |
| `src/services/reconstruction/tracksfm.cpp` | 96.8 % (61/63) | 96 |
| `src/services/reconstruction/incrementalsfm.cpp` | 80.2 % (198/247) | 80 |
| `src/services/reconstruction/poseestimator.cpp` | 100.0 % (21/21) | 100 |
| `src/services/reconstruction/reconstructionservice.cpp` | 89.1 % (155/174) | 87 |
| `src/services/reconstruction/reconstructionpipeline.cpp` | 59.5 % (119/200) | 59 |
| `src/services/reconstruction/reconstructionpostprocessor.cpp` | 96.3 % (183/190) | 96 |

**service: scene**

| file | measured | floor |
|------|----------|-------|
| `src/services/scene/crosshairgeometry.cpp` | 95.5 % (42/44) | 93 |
| `src/services/scene/crosshairmanager.cpp` | 99.5 % (213/214) | 97 |
| `src/services/scene/sceneservice.cpp` | 97.9 % (274/280) | 95 |

**service: viewer**

| file | measured | floor |
|------|----------|-------|
| `src/services/viewer/image2dloader.cpp` | 98.7 % (75/76) | 98 |
| `src/services/viewer/model3dloader.cpp` | 100.0 % (64/64) | 100 |
| `src/services/viewer/viewerservice.cpp` | 100.0 % (59/59) | 100 |
| `src/services/viewer/dicomloader.cpp` | 95.6 % (129/135) | 90 |

**service: settings**

| file | measured | floor |
|------|----------|-------|
| `src/services/settings/settingsservice.cpp` | 100.0 % (18/18) | 100 |

**service: shell**

| file | measured | floor |
|------|----------|-------|
| `src/services/shell/shellservice.cpp` | 98.2 % (56/57) | 98 |

**utility layer**

| file | measured | floor |
|------|----------|-------|
| `src/utils/abstractribbonpanel.cpp` | 100.0 % (42/42) | 100 |
| `src/utils/fileutilities.cpp` | 100.0 % (70/70) | 100 |
| `src/utils/iconfactory.cpp` | 100.0 % (19/19) | 100 |
| `src/utils/htmlutilities.cpp` | 95.3 % (41/43) | 95 |
| `src/utils/customprogressdialog.cpp` | 95.2 % (59/62) | 95 |
| `src/utils/stylemanager.cpp` | 90.2 % (46/51) | 90 |
| `src/utils/moderndialog.cpp` | 78.6 % (55/70) | 78 |
| `src/utils/logger.cpp` | 88.5 % (54/61) | 87 |
| `src/utils/modernmessagebox.cpp` | 46.8 % (22/47) | 46 |

**plugin: AI assistant**

| file | measured | floor |
|------|----------|-------|
| `src/modules/aiassistantplugin/aiassistantribbonui.cpp` | 100.0 % (10/10) | 100 |
| `src/modules/aiassistantplugin/aiattachmentpreviewfactory.cpp` | 100.0 % (13/13) | 100 |
| `src/modules/aiassistantplugin/chatimageviewer.cpp` | 100.0 % (35/35) | 100 |
| `src/modules/aiassistantplugin/chatbotdockwidget.cpp` | 96.4 % (135/140) | 96 |
| `src/modules/aiassistantplugin/aiassistantplugin.cpp` | 18.2 % (73/400) | 18 |
| `src/modules/aiassistantplugin/chatmessagerenderer.cpp` | 100.0 % (217/217) | 100 |

**plugin: AI processor**

| file | measured | floor |
|------|----------|-------|
| `src/modules/aiprocessorplugin/aiprocessorribbonui.cpp` | 100.0 % (17/17) | 100 |
| `src/modules/aiprocessorplugin/videotrackerthread.cpp` | 82.9 % (58/70) | 82 |
| `src/modules/aiprocessorplugin/aitraindockwidget.cpp` | 39.0 % (60/154) | 38 |
| `src/modules/aiprocessorplugin/aiprocessorplugin.cpp` | 46.0 % (209/454) | 45 |

**plugin: reconstruction**

| file | measured | floor |
|------|----------|-------|
| `src/modules/reconstructionplugin/reconstructionribbonui.cpp` | 100.0 % (21/21) | 100 |
| `src/modules/reconstructionplugin/reconstructionplugin.cpp` | 21.3 % (44/207) | 21 |

**plugin: send mail**

| file | measured | floor |
|------|----------|-------|
| `src/modules/sendmailplugin/mailinboxitemfactory.cpp` | 100.0 % (30/30) | 100 |
| `src/modules/sendmailplugin/sendmailribbonui.cpp` | 100.0 % (7/7) | 100 |
| `src/modules/sendmailplugin/mailmessageformatter.cpp` | 84.7 % (83/98) | 84 |
| `src/modules/sendmailplugin/mailfilterdialog.cpp` | 81.8 % (36/44) | 81 |
| `src/modules/sendmailplugin/mailsettingsdialog.cpp` | 65.6 % (42/64) | 65 |
| `src/modules/sendmailplugin/sendmailplugin.cpp` | 47.7 % (21/44) | 47 |
| `src/modules/sendmailplugin/maildockwidget.cpp` | 61.0 % (208/341) | 60 |

**plugin: user auth**

| file | measured | floor |
|------|----------|-------|
| `src/modules/userauthplugin/overlayresizer.cpp` | 100.0 % (42/42) | 100 |
| `src/modules/userauthplugin/settingsdialog.cpp` | 91.0 % (294/323) | 90 |
| `src/modules/userauthplugin/themeselectiondialog.cpp` | 90.6 % (144/159) | 90 |
| `src/modules/userauthplugin/aboutdialog.cpp` | 76.2 % (64/84) | 76 |
| `src/modules/userauthplugin/logindialog.cpp` | 68.9 % (71/103) | 68 |
| `src/modules/userauthplugin/licenseactivationdialog.cpp` | 68.8 % (64/93) | 68 |
| `src/modules/userauthplugin/forgotpassworddialog.cpp` | 65.6 % (86/131) | 65 |
| `src/modules/userauthplugin/userauthplugin.cpp` | 74.3 % (197/265) | 74 |
| `src/modules/userauthplugin/changepassworddialog.cpp` | 55.4 % (51/92) | 55 |
| `src/modules/userauthplugin/avatarcropperdialog.cpp` | 42.2 % (95/225) | 42 |
| `src/modules/userauthplugin/adminusermanagerdialog.cpp` | 40.6 % (101/249) | 40 |

**plugin: viewer**

| file | measured | floor |
|------|----------|-------|
| `src/modules/viewerplugin/viewerribbonui.cpp` | 100.0 % (8/8) | 100 |
| `src/modules/viewerplugin/viewernavigatorui.cpp` | 94.7 % (36/38) | 94 |
| `src/modules/viewerplugin/viewerplugin.cpp` | 51.4 % (74/144) | 51 |
| `src/modules/viewerplugin/viewerviewmodel.cpp` | 98.5 % (66/67) | 98 |
| `src/modules/viewerplugin/viewerlistui.cpp` | 100.0 % (59/59) | 100 |

Rules for the table:

* Floors are a one-way ratchet — the gate fails when one is lowered. A floor is
  the lower of the two reference measurements (local VS 17.12 / Qt 6.9.3 and CI
  VS 18 / Qt 6.5.3), so the same code cannot fail a supported compiler because
  its PDB attributes a few more lines. Raise a floor deliberately when the code
  gains coverage; there is no automatic raise from one environment.
* The remaining uncovered lines fall into documented classes:
  * **lines without executable code** — closing braces that MSVC attributes as
    lines (`ShellService` 56/57), `continue` statements
    (`PluginDependencyResolver` 105/111/136) and functions the optimizer
    inlined (`PluginHost` 59/184/187, the resolver's recursive lambda tail
    52-54). They can never be reported as executed;
  * **branches that need a broken environment** — the DPAPI failure path
    (`SecretVault` 40-41) and its non-Windows counterpart (67), the
    "translation file cannot be opened" warnings (`LanguageManager` 75/86,
    only in an installation without the translation files) and the
    `mkpath()` failure of `AppConfig` (38, only with a read-only profile);
  * **branches that need a fixture the headless CI cannot provide** — the
    ONNX-backed `aiprocessor` (the exporter multiplies every allocation and
    session creation dies with `std::bad_alloc`, so the gate sets
    `AI_SERVICE_NO_MODELS`) and the TLS SMTP dialogue (`smtpmailer`, the
    server must present a trusted certificate).  These files are *measured*
    (they have floors), so a regression in their reachable code still fails
    the gate; only the fixture-bound branches stay uncovered.  The DICOM
    pipeline, the ONNX service wrapper, the reconstruction post-processing
    and the VTK scene are covered by fixture-based tests; their numbers are
    in the table above and are deliberately not repeated here — a second copy
    is a number that can drift.
  The guard layer's remaining lines all fall into the classes above — no
  branch was left unnamed; the floors reflect that (90-100 %).
* The former
  `UserManager` gaps are gone: the auth/licence guard is now split into
  `CryptoUtils`,
  `UserAccountStore`, `LicenseKeyLedger`, `OtpStore` and `UserSettingsStore`,
  each exercised by the user-management flow tests (the store's credential map
  has its own direct contract test, so inlining under unity build cannot hide
  it).
* `--measure` prints every file seen in the reports without enforcing, so the
  next candidate for a floor is easy to pick.
* **A service file is measured from two line maps.** `src/services/**` is
  compiled into `AppCore.dll` *and* — for the targets that need it standalone
  — into a test executable, and the two compilations can attribute one
  statement to different lines. Only the test copy executes, so a line that
  exists in the AppCore map alone reads 0 %: `Triangulator.cpp` fell from 100 %
  to 86.7 % because adding an `else` made MSVC map the loop's closing brace to
  its own line in the AppCore build. The fix is structural, not a floor
  change: keep each statement on one line, keep the loop shape that was
  measured before, and pre-fill the result (`outPts.assign(...)`) instead of
  adding a second branch inside the loop. A single
  `OpenCppCoverage --modules <test>.exe --modules AppCore.dll` run prints both
  maps in seconds — do that before paying for a full gate run.

### Coverage debt (visible, not yet executed)

These translation units are part of the report but no line of them is
executed, so they carry no floor (a floor of 0 would pin nothing). The gate
prints this list on every run — the debt is measured and named, not hidden.

**There is no such translation unit left: every `.cpp` in `src/` executes at
least one line.**  What remains uncovered are lines *inside* measured files —
the documented classes above (no-code lines, branches that need a broken
environment) and the error branches of the service layer.

The plugin layer dominates it: the shipped plugins are GUI shells (dialogs,
ribbon panels, dock widgets) and their interactive paths need a human. The
plugin-contract test exists now (`TestPluginSmoke` drives
`initialize()`/`cleanup()` against a mocked `IAppContext`/`IShellService`, and
`TestPluginUi` / `TestUserAuthDialogs` cover the plugin UI), which lifted the
layer from 18.6 % to 54.6 % at the time (56.0 % after the §4.13 round).

**The widget-ownership contract is measured, not assumed.** After `cleanup()`
the shell's widget tree must hold no object the plugin installed, and the
plugin must have installed something in the first place — otherwise the check
would be vacuous. The stub context returns a *real* tab panel (with the
QHBoxLayout the shell uses) and adds/removes the docks on a real
`QMainWindow`, so a plugin that forgets to release its ribbon group, its menu
buttons or its event filter is reported by name. The failures found this way
were real:

* `UserAuthPlugin::cleanup()` deleted none of the four widgets it installed
  (the help group, the language and avatar buttons, the menu-bar corner) —
  dead UI in the shell after the plugin was gone;
* `UserAuthPlugin` installed an event filter (`OverlayResizer`, a class
  defined *inside* the plugin DLL) on the shell window and never removed it —
  the shell would call into the plugin's code after the plugin was gone. The
  class is now named (`ThemeSelectionDialog.h`), the plugin keeps the handle
  and removes it in `cleanup()`;
* `ReconstructionPlugin` and `ViewerPlugin` never deleted their ribbon panel,
  so their group stayed in the tab bar;
* `AIProcessorPlugin` deleted its dock without detaching it from the dock
  area, and added it to the shell twice.

| file | lines |
| _(khong con TU nao chua duoc thuc thi)_ | 0 |

**Header-only guards are not gated by coverage.** `ServiceRegistry.h`,
`MailThreadGuard.h` and `BlockingCall.h` compile inline into their callers
under `/O2 /Ob2`, so the tool cannot map their lines reliably.
`TestServiceRegistry` is therefore not part of the coverage run (it may not
load `AppCore.dll`, leaving OpenCppCoverage nothing to export). Their
decisions are pinned by gate 4 instead (null registration refused, GUI-thread
warning fires, the promise is finished and the exception is contained), and
`tests/mutation_check.py` carries a `PINNED_HEADERS` list that fails the gate
when one of them has no mutant.

**A case the exporter cannot host is skipped, not hidden.**
`TestReconstructionService` needs a *completed* reconstruction for the
published-result contract, and the heavy post-processing (MLS densify to 1M
points + Poisson meshing) exceeds the exporter's per-test budget. The
coverage gate sets `RECONSTRUCTION_NO_HEAVY_RUN=1` for that executable and
the case skips with a message; the same contract is measured by the cheap
cases of the same executable and by `TestReconstructionPipeline` (which owns
the snapshot conversion), and the completed run is proven by the normal CTest
run, where no exporter is involved. The same pattern is used for the ONNX
session (`AI_SERVICE_NO_MODELS`).

Release builds must ship PDBs for gate 3 to map anything: `CMakeLists.txt`
section 5b adds `/Zi` to the Release compile flags and `/DEBUG /OPT:REF
/OPT:ICF` to the link flags (symbols without changing Release codegen). The
tool is discovered through `OPENCPPCOVERAGE_EXE`, `PATH`, or the known install
locations; CI installs the pinned `0.9.9.0` upstream release into its temporary
directory and exports `OPENCPPCOVERAGE_EXE` for the gate.

## Gate 4 — mutation check

`tests/mutation_check.py` owns the "is the guard pinned?" question for the
guards that coverage can only say are *executed*:

* `ServiceRegistry` refuses null registration (mutation: accept it),
* `PluginHost` API-version gate, toolchain/ABI fingerprint, discovery library
  filter, initialize() rollback, reverse-order shutdown, **release of the
  plugin objects** (mutation: the unload is skipped — the plugin's whole
  QObject tree would stay alive),
* `PluginDependencyResolver` unknown phase, missing dependency, dependency
  cycle,
* `SecretVault` legacy-plaintext passthrough + empty-stays-empty + **fail-closed
  on a protection failure** (mutation: downgrade the secret to cleartext),
* `UserSettingsStore` **a secret key never reaches the plaintext namespace**
  (mutation: the preference API bypasses the vault) and **the migration
  retires the plaintext copy** (mutation: seal a second copy and leave the
  cleartext one),
* `CryptoUtils` PBKDF2 iteration count + **the ceiling on the stored work
  factor** (mutation: a tampered credential file may ask for an unbounded
  number of iterations),
* `UserManager` used-license-key rejection, **a password reset needs the
  proof of ownership** (mutation: the proof is not required), **the proof is
  single-use** (mutation: the same OTP resets twice) and **the password
  policy applies to a reset**,
* `UserAuthPlugin` **cleanup() releases the widgets it installed in the
  shell** (mutation: the shell keeps dead UI),
* `OtpStore` one-shot verification (no replay),
* `AIRequestQueue` single in-flight reply,
* `ChatSessionStore` seals the chat history before it reaches the disk and
  unseals it again on load (mutation: write it unencrypted / read it without
  unsealing),
* `AtRestCipher` seals a payload into the `enc:v1:` envelope and unseals it
  again (mutation: return the plaintext / hand back the sealed blob), and
  `FileUtilities` seals the copied attachment in place (mutation: store it
  unencrypted),
* **the cross-language envelope contract**: the Qt client seals the chat
  attachment and the Python service opens it, so the byte layout, the
  associated data and the key resolution have to match across the two
  languages. Each side pins the *other* side's output as a fixed vector
  (`TestAtRestCipher::thePythonServiceEnvelopeIsReadable` and
  `AIAssistant/tests/unit/test_sealed_attachment_read.py`) — a change that
  breaks the other language fails a test instead of surfacing as "the
  assistant cannot see the image",
* `AppConfig` AI port/interpreter validation,
* `MailThreadGuard` GUI-thread warning,
* `AppShell` deferred-phase idempotence,
* `ServiceRegistry` **a duplicate registration is refused** (mutation: the
  second one silently takes the service over),
* `IPlugin` **the ABI fingerprint carries the content hash of the
  plugin-facing interface surface AND the VTK/OpenCV/ONNX versions** (mutants:
  the fingerprint ignores the interfaces, or ignores the SDK, so a plugin
  built against different headers/SDK would load),
* `ReconstructionService` **the published image list is read through its own
  lock** (mutation: it waits on the pipeline lock the worker holds for the
  whole run — the UI freezes),
* `AIModelLoadService` **a throwing loader is a failure** (mutation: it is
  reported as success) and **the load leaves the caller's thread** (mutation:
  it runs inline, freezing the GUI for the whole load),
* `AIServerProcessController` **a deferred start is not dropped**, **a
  requested stop is not reported as a crash** and **the kill fallback is
  armed**,
* `ChatMessage` **the UI-only agent trace is not a model role** (mutation: it
  would reach FastAPI — the HTTP 422 of the past),
* `AppShell` **the composition root registers the segregated user contracts**
  (mutation: a consumer of `IAuthService` would resolve nothing),
* `AIProcessor` **a failed model load does not report the model as ready**
  (mutation: the readiness flag lies to the plugin),
* `BlockingCall` **the promise is finished explicitly** (mutation: the result
  is handed over *cancelled*, and the natural `isCanceled()` guard drops it)
  and **an exception in the callable is contained** (mutation: it escapes
  `QThread::run()`, reaches `std::terminate()` and aborts the application),
* `AIInferenceGuard` **an inference exception is contained** (mutation: an
  ONNX/OpenCV exception leaves the Qt slot or the worker thread and kills the
  application),
* `AIService` **a detection call cannot run without a loaded model, on either
  API** (mutants: `runDetection()` renders the input frame as the result, and
  `detectObjects()` reports `ok == true` with zero objects — "detected
  nothing" instead of "never ran"),
* `VideoTrackerThread` **the tracking loop contains its exceptions** (mutation:
  the thread entry point lets them escape),
* `AIProcessorPlugin` **the inference leaves the GUI thread** (mutation:
  `dispatchInference()` runs the task synchronously — the window freezes for
  the whole ONNX call) and **only one inference is in flight** (mutation: a
  second click races the first and a stale result can win the viewer),
* `ReconstructionService` **the whole published/pipeline split, as a class**:
  `pointCloudSnapshot()`, `pointCount()` and `hasResult()` read the published
  state (mutation: they wait on the pipeline lock the worker holds for the
  whole run — the UI freezes) and `setImages()` refuses the change instead of
  waiting for the running pipeline (mutation: it blocks the GUI thread),
* and seven **planted violations** that the architecture guard itself must
  reject: a concrete service include in a plugin, a bare
  `waitForFinished()` in `src/`, a heavy SDK include in `src/utils/`, an
  event filter that is installed on a shell widget and never removed, a
  shell window found by scanning the top-level widgets, blocking I/O
  scheduled on the global thread pool, and an AI inference call that bypasses
  the exception guard.

### The gate checks its own targets (added 2026-10-22)

The gate used to die **silently**: a mutant that pointed at a file which no
longer existed aborted the script with a `FileNotFoundError` *after* the
pre-flight had already gone green — so the run looked like it had happened
and reported nothing. Three commits carried such a mutant (the file moved from
`src/core/` to `src/include/` in the same commit that added the mutant), and
the "52/52 mutant" claim had never actually been verified.

The gate now verifies its own targets **before any rebuild**:

* every mutant must point at a file that **exists**;
* every anchor must occur **exactly once** in that file;
* every header-only file that carries a load-bearing decision must be
  **pinned** by at least one mutant (`PINNED_HEADERS`).

A violation is a **red gate with the list of broken mutants**, not a silent
pass. The header list exists because the coverage gate enumerates `.cpp`
translation units only — a header is invisible to it, so without this list a
decision in a header would be checked by nothing at all.

**The repaired gate immediately found a survivor** — the first hard evidence
that the old "52/52 mutant ✅" claim had never been verified:

> `AppShell: deferred services re-created on every call (held pointers dangle)`
> → SURVIVED (`TestAppShell` did not notice).

The old test compared the **registry pointers** before and after the second
call. But the registry **refuses** a duplicate registration (first wins), so
it *cannot* change — the comparison was vacuous. The real defect of a
non-idempotent deferred phase is that the shell **replaces** its instances
(the old ones are destroyed) while the registry keeps pointing at them: a
dangling pointer. The test now checks the **liveness** of the instances
(`QPointer` on the QObject-derived services) — and the mutant is killed.

A mutant that does not compile is INVALID, not a kill — so a mutation that
leaves a catch parameter unused would fail the `/W4 /WX` build and be
reported as INVALID. The mutants are written so that they compile.

Discipline (mirrors `AIAssistant/scripts/mutation_check.py`):

* every owning test must pass on the unmutated tree first — a missing DLL on
  `PATH` must never masquerade as a killed mutant. A pre-flight test that
  fails **once** is retried: right after a rebuild the freshly written DLL can
  still be locked, and a GPU-backed test can lose its device for a moment. A
  test that fails twice is a real failure;
* an anchor that no longer matches is STALE and fails the run;
* a mutant that does not compile is INVALID, not a kill — **except** when the
  build failed because the *output* was still open (`LNK1104`) or the compiler
  could not replace an object another build held (`MSB6003`): the test that
  just ran keeps `AppCore.dll` loaded for a moment after it exits, and that
  race is retried instead of being reported as a broken mutant. `LNK1103`
  ("debugging information corrupt; recompile module") is the incremental-link
  analogue: the target's objects and PDB are dropped and the target is
  recompiled, because a corrupt PDB is not a mutant verdict either;
* the source is restored byte-identical in a `finally`, a lock file in the
  build directory refuses overlapping runs, and the touched targets are
  rebuilt once so the build tree ends up with the original binaries. A
  transient sharing lock on the source (scanner, indexer, IDE) is retried; a
  restore that still cannot write fails loudly with the file name, because
  continuing would leave a mutant behind. The pre-flight, the mutant run and
  the post-restore verification all use the slow test's declared budget (`TestReconstructionService` declares
  `TIMEOUT 900` because its completed-run case reconstructs the scene); the
  mutant's own budget only bounds how long it may take to fail, not how long
  the green case may take to finish;
* **a run killed mid-way restores the sources, not the binaries.** The
  `finally` that rebuilds the touched targets does not execute when the
  process itself is terminated, so `build/` can still hold a target compiled
  with a mutant. The next run then fails its pre-flight and reports an
  environment problem, which is the wrong diagnosis: rebuild
  (`cmake --build build --config Release`) before retrying. This is exactly
  what happened here — `TestAppShell.exe` still contained the mutant that
  comments out `registerService<IAuthService>`, and the pre-flight blamed
  `PATH`;
* the same build directory must not be shared with an IDE build. A
  concurrent build can overwrite a mutated binary — making the verdict
  meaningless — or lock an object file during the restore. Close Qt Creator's
  build (or run the gate in CI) while it runs;
* `--only SUBSTR` runs a subset by label.

Reference result: **90/90 killed, 0 survived, 0 stale, 0 invalid**.
