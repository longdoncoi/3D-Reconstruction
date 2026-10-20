# C++ verification stack

Four gates prove the C++ side of the repository. All of them run locally with
the commands below and are enforced in `.github/workflows/Build-C++-ci.yml`
after the build step. The reference run recorded here is
2026-10, Windows 11, MSVC 2022 (VS 17.12), Qt 6.9.3, Release.

| # | Gate | Command | What it proves |
|---|------|---------|----------------|
| 1 | Architecture boundaries | `cmake -DSOURCE_DIR=<repo> -P tests/CheckArchitectureBoundaries.cmake` | Layering invariants (heavy/light PCH split, plugin includes, blocking-I/O waits, singleton confinement …). Runs in CTest as `TestArchitectureBoundaries`. |
| 2 | Unit + integration suite | `ctest --test-dir build -C Release --output-on-failure` | 31 tests: kernel units, plugin host/resolver with an injected loader, real-plugin smoke, `TestAppShell` composition-root wiring, `TestAIAssistant` request state machine (injected transport), AppCore-owned model-load lifetime (`TestAIModelLoadService`), the at-rest cipher shared with the Python service (`TestAtRestCipher`), the shared utility layer (`TestUtilsLayer`), the scene/viewer layers, plus `TestReconstructionFinalize` (labelled `slow`). |
| 3 | Coverage gate | `python tests/coverage_gate.py` | Measured line coverage of the kernel/guard, composition, service and utility files, with per-file floors, a completeness check (no translation unit may escape the measurement) and a printed debt list; runs OpenCppCoverage over the owning tests. |
| 4 | Mutation check | `python tests/mutation_check.py` | Every load-bearing guard decision is *pinned*: one inverted decision per mutant, rebuilt and re-run; the owning test must fail. |

Run them from the repository root with Qt and the vcpkg `bin` directory on
`PATH` (the same environment the CTest step uses), e.g.:

```powershell
$env:PATH = "C:/Qt/6.9.3/msvc2022_64/bin;$env:VCPKG_INSTALLED_DIR/x64-windows/bin;$env:PATH"
ctest --test-dir build -C Release -LE slow --output-on-failure   # fast round
python tests/coverage_gate.py
python tests/mutation_check.py
```

## Gate 3 — coverage floors

`tests/coverage_gate.py` runs the owning tests under OpenCppCoverage, merges
the per-test cobertura reports (a line counts as covered when *any* test
executes it) and enforces three things:

1. **Per-file floors** — the ratchet in `FLOORS`.
2. **Completeness** — every translation unit of `src/` that a test can reach
   must appear in the merged report. A file that goes missing fails the run:
   that happens when it is not compiled into an instrumented module, or when
   the unity build inlined it away (see below). Five units are exempt, each
   with a recorded reason (`_NOT_MEASURABLE` in the script): the two PCH
   carriers, `src/app/main.cpp` + `src/app/mainwindow.cpp` (they live in the
   application executable, which no test executes) and
   `src/services/scene/panstyle.cpp` (a `vtkStandardNewMacro` boilerplate).
3. **Debt visibility** — the files with 0 % are printed on every run with
   their line count, so the remaining work is named instead of implied.

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
| `src/app/appshell.cpp` | 100.0 % (63/63) | 100 |

**kernel + guard**

| file | measured | floor |
|------|----------|-------|
| `src/core/atrestcipher.cpp` | 100.0 % (139/139) | 100 |
| `src/core/cryptoutils.cpp` | 100.0 % (63/63) | 100 |
| `src/core/licensekeyledger.cpp` | 100.0 % (75/75) | 100 |
| `src/core/otpstore.cpp` | 100.0 % (22/22) | 100 |
| `src/core/signalbus.cpp` | 100.0 % (2/2) | 100 |
| `src/core/useraccountstore.cpp` | 100.0 % (107/107) | 100 |
| `src/core/usermanager.cpp` | 100.0 % (263/263) | 100 |
| `src/core/appconfig.cpp` | 98.8 % (79/80) | 98 |
| `src/core/usersettingsstore.cpp` | 97.3 % (36/37) | 97 |
| `src/core/languagemanager.cpp` | 96.5 % (55/57) | 96 |
| `src/core/pluginhost.cpp` | 95.0 % (95/100) | 95 |
| `src/core/plugindependencyresolver.cpp` | 93.3 % (84/90) | 93 |
| `src/core/secretvault.cpp` | 92.9 % (26/28) | 92 |

**service: ai**

| file | measured | floor |
|------|----------|-------|
| `src/services/ai/aimodelloadservice.cpp` | 100.0 % (31/31) | 100 |
| `src/services/ai/airequestqueue.cpp` | 100.0 % (62/62) | 100 |
| `src/services/ai/aiyolopostprocessor.cpp` | 100.0 % (51/51) | 100 |
| `src/services/ai/chatsessionmanager.cpp` | 100.0 % (56/56) | 100 |
| `src/services/ai/chatsessionstore.cpp` | 100.0 % (57/57) | 100 |
| `src/services/ai/agentactionmanifest.cpp` | 45.7 % (32/70) | 45 |
| `src/services/ai/aiassistant.cpp` | 40.4 % (224/554) | 40 |
| `src/services/ai/aiserverprocesscontroller.cpp` | 34.6 % (27/78) | 34 |
| `src/services/ai/aiservice.cpp` | 10.3 % (4/39) | 10 |
| `src/services/ai/aiprocessor.cpp` | 9.5 % (26/275) | 9 |

**service: mail**

| file | measured | floor |
|------|----------|-------|
| `src/services/mail/smtpoutbox.cpp` | 87.5 % (14/16) | 87 |
| `src/services/mail/mailmimeparser.cpp` | 67.0 % (152/227) | 66 |
| `src/services/mail/mailservice.cpp` | 22.0 % (49/223) | 21 |
| `src/services/mail/smtpmailer.cpp` | 19.7 % (28/142) | 19 |

**service: reconstruction**

| file | measured | floor |
|------|----------|-------|
| `src/services/reconstruction/featureextractor.cpp` | 100.0 % (8/8) | 100 |
| `src/services/reconstruction/multiviewtriangulator.cpp` | 100.0 % (76/76) | 100 |
| `src/services/reconstruction/reconstructionworker.cpp` | 100.0 % (4/4) | 100 |
| `src/services/reconstruction/triangulator.cpp` | 100.0 % (11/11) | 100 |
| `src/services/reconstruction/sfmstage.cpp` | 100.0 % (9/9) | 100 |
| `src/services/reconstruction/featurematcher.cpp` | 97.7 % (42/43) | 97 |
| `src/services/reconstruction/cameraparamsparser.cpp` | 97.6 % (41/42) | 97 |
| `src/services/reconstruction/tracksfm.cpp` | 93.2 % (55/59) | 93 |
| `src/services/reconstruction/incrementalsfm.cpp` | 79.0 % (196/248) | 79 |
| `src/services/reconstruction/pointcloudfilter.cpp` | 66.4 % (71/107) | 66 |
| `src/services/reconstruction/poseestimator.cpp` | 61.9 % (13/21) | 61 |
| `src/services/reconstruction/reconstructionservice.cpp` | 53.3 % (72/135) | 53 |
| `src/services/reconstruction/reconstructionpipeline.cpp` | 44.2 % (72/163) | 44 |
| `src/services/reconstruction/reconstructionpostprocessor.cpp` | 12.6 % (22/175) | 12 |

**service: scene**

| file | measured | floor |
|------|----------|-------|
| `src/services/scene/crosshairgeometry.cpp` | 92.9 % (39/42) | 88 |
| `src/services/scene/crosshairmanager.cpp` | 56.1 % (120/214) | 56 |
| `src/services/scene/sceneservice.cpp` | 4.9 % (13/264) | 4 |

**service: viewer**

| file | measured | floor |
|------|----------|-------|
| `src/services/viewer/image2dloader.cpp` | 95.8 % (68/71) | 95 |
| `src/services/viewer/model3dloader.cpp` | 46.9 % (30/64) | 46 |
| `src/services/viewer/viewerservice.cpp` | 13.6 % (8/59) | 13 |
| `src/services/viewer/dicomloader.cpp` | 1.5 % (2/135) | 1 |

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
| `src/utils/fileutilities.cpp` | 100.0 % (61/61) | 100 |
| `src/utils/iconfactory.cpp` | 100.0 % (19/19) | 100 |
| `src/utils/htmlutilities.cpp` | 95.3 % (41/43) | 95 |
| `src/utils/customprogressdialog.cpp` | 95.2 % (59/62) | 95 |
| `src/utils/stylemanager.cpp` | 90.2 % (46/51) | 90 |
| `src/utils/moderndialog.cpp` | 71.4 % (50/70) | 71 |
| `src/utils/logger.cpp` | 69.5 % (41/59) | 69 |
| `src/utils/modernmessagebox.cpp` | 46.8 % (22/47) | 46 |

**plugin: AI assistant**

| file | measured | floor |
|------|----------|-------|
| `src/modules/aiassistantplugin/aiattachmentpreviewfactory.cpp` | 100.0 % (13/13) | 100 |
| `src/modules/aiassistantplugin/aiassistantribbonui.cpp` | 100.0 % (10/10) | 100 |
| `src/modules/aiassistantplugin/chatimageviewer.cpp` | 100.0 % (35/35) | 100 |
| `src/modules/aiassistantplugin/chatbotdockwidget.cpp` | 96.4 % (134/139) | 96 |
| `src/modules/aiassistantplugin/aiassistantplugin.cpp` | 16.9 % (67/397) | 16 |
| `src/modules/aiassistantplugin/chatmessagerenderer.cpp` | 5.7 % (12/212) | 5 |

**plugin: AI processor**

| file | measured | floor |
|------|----------|-------|
| `src/modules/aiprocessorplugin/aipredictionlogger.cpp` | 100.0 % (9/9) | 100 |
| `src/modules/aiprocessorplugin/aiprocessorribbonui.cpp` | 100.0 % (17/17) | 100 |
| `src/modules/aiprocessorplugin/videotrackerthread.cpp` | 50.8 % (30/59) | 50 |
| `src/modules/aiprocessorplugin/aitraindockwidget.cpp` | 32.0 % (48/150) | 32 |
| `src/modules/aiprocessorplugin/aiprocessorplugin.cpp` | 24.6 % (99/402) | 24 |

**plugin: reconstruction**

| file | measured | floor |
|------|----------|-------|
| `src/modules/reconstructionplugin/reconstructionribbonui.cpp` | 57.1 % (12/21) | 57 |
| `src/modules/reconstructionplugin/reconstructionplugin.cpp` | 18.9 % (38/201) | 18 |

**plugin: send mail**

| file | measured | floor |
|------|----------|-------|
| `src/modules/sendmailplugin/mailinboxitemfactory.cpp` | 100.0 % (30/30) | 100 |
| `src/modules/sendmailplugin/sendmailribbonui.cpp` | 100.0 % (7/7) | 100 |
| `src/modules/sendmailplugin/mailmessageformatter.cpp` | 84.7 % (83/98) | 84 |
| `src/modules/sendmailplugin/mailfilterdialog.cpp` | 81.8 % (36/44) | 81 |
| `src/modules/sendmailplugin/mailsettingsdialog.cpp` | 63.9 % (39/61) | 63 |
| `src/modules/sendmailplugin/maildockwidget.cpp` | 45.9 % (147/320) | 45 |
| `src/modules/sendmailplugin/sendmailplugin.cpp` | 36.4 % (16/44) | 36 |

**plugin: user auth**

| file | measured | floor |
|------|----------|-------|
| `src/modules/userauthplugin/settingsdialog.cpp` | 86.1 % (272/316) | 86 |
| `src/modules/userauthplugin/aboutdialog.cpp` | 75.9 % (63/83) | 75 |
| `src/modules/userauthplugin/logindialog.cpp` | 68.6 % (70/102) | 68 |
| `src/modules/userauthplugin/licenseactivationdialog.cpp` | 68.5 % (63/92) | 68 |
| `src/modules/userauthplugin/themeselectiondialog.cpp` | 66.3 % (132/199) | 66 |
| `src/modules/userauthplugin/forgotpassworddialog.cpp` | 57.9 % (70/121) | 57 |
| `src/modules/userauthplugin/changepassworddialog.cpp` | 54.9 % (50/91) | 54 |
| `src/modules/userauthplugin/userauthplugin.cpp` | 48.5 % (127/262) | 48 |
| `src/modules/userauthplugin/avatarcropperdialog.cpp` | 42.8 % (95/222) | 42 |
| `src/modules/userauthplugin/adminusermanagerdialog.cpp` | 40.8 % (100/245) | 40 |

**plugin: viewer**

| file | measured | floor |
|------|----------|-------|
| `src/modules/viewerplugin/viewerribbonui.cpp` | 100.0 % (8/8) | 100 |
| `src/modules/viewerplugin/viewernavigatorui.cpp` | 94.7 % (36/38) | 94 |
| `src/modules/viewerplugin/viewerplugin.cpp` | 41.5 % (56/135) | 41 |
| `src/modules/viewerplugin/viewerviewmodel.cpp` | 9.0 % (6/67) | 8 |
| `src/modules/viewerplugin/viewerlistui.cpp` | 7.0 % (4/57) | 7 |

Rules for the table:

* Floors are a ratchet — raise one when its file reaches the next integer,
  never lower one to make a change pass.
* The remaining uncovered lines fall into documented classes:
  * **lines without executable code** — closing braces that MSVC attributes as
    lines (`ShellService` 56/57), `continue` statements
    (`PluginDependencyResolver` 105/111/136) and functions the optimizer
    inlined (`PluginHost::clearLoaders` 158-161, the resolver's recursive
    lambda tail 52-54). They can never be reported as executed;
  * **branches that need a broken environment** — the DPAPI failure path
    (`SecretVault` 38) and its non-Windows counterpart (43), the
    "translation file cannot be opened" warnings (`LanguageManager` 75/86,
    only in an installation without the translation files) and the
    `mkpath()` failure of `AppConfig` (38, only with a read-only profile);
  * **branches that need a fixture the headless CI cannot provide** — the
    DICOM pipeline (`dicomloader` 1.5 %), the ONNX-backed AI services
    (`aiservice` 10 %, `aiprocessor` 9.5 %), the TLS SMTP dialogue
    (`smtpmailer` 19.7 %, the server must present a trusted certificate),
    the heavy reconstruction post-processing (`reconstructionpostprocessor`
    12.6 %) and the VTK scene (`sceneservice` 4.9 % — the constructor needs a
    real `QVTKOpenGLNativeWidget`, i.e. an OpenGL context, which the
    offscreen platform does not provide).  These files are *measured* (they
    have floors), so a regression in their reachable code still fails the
    gate; only the fixture-bound branches stay uncovered.
  The guard layer therefore has **no uncovered executable line** left; the
  floors reflect that (95-100 %).
* The former
  `UserManager` gaps are gone: the auth/licence guard is now split into
  `CryptoUtils`,
  `UserAccountStore`, `LicenseKeyLedger`, `OtpStore` and `UserSettingsStore`,
  each exercised by the user-management flow tests (the store's credential map
  has its own direct contract test, so inlining under unity build cannot hide
  it).
* `--measure` prints every file seen in the reports without enforcing, so the
  next candidate for a floor is easy to pick.

### Coverage debt (visible, not yet executed)

These translation units are part of the report but no line of them is
executed, so they carry no floor (a floor of 0 would pin nothing). The gate
prints this list on every run — the debt is measured and named, not hidden.

**There is no such translation unit left: every `.cpp` in `src/` executes at
least one line.**  What remains uncovered are lines *inside* measured files —
the documented classes above (no-code lines, branches that need a broken
environment) and the error branches of the service layer.

The plugin layer dominates it: the shipped plugins are GUI shells (dialogs,
ribbon panels, dock widgets) and `TestPluginSmoke` only loads their metadata.
Closing it needs a plugin-contract test that calls `initialize()` with a
mocked `IAppContext`/`IShellService` — the next step on the roadmap.

| file | lines |
| _(khong con TU nao chua duoc thuc thi)_ | 0 |

**Header-only guards are not gated by coverage.** `ServiceRegistry.h` and
`MailThreadGuard.h` compile inline into their callers under `/O2 /Ob2`, so the
tool cannot map their lines reliably. `TestServiceRegistry` is therefore not
part of the coverage run (it may not load `AppCore.dll`, leaving OpenCppCoverage
nothing to export). Their decisions are pinned by gate 4 instead (null
registration refused, GUI-thread warning fires).

Release builds must ship PDBs for gate 3 to map anything: `CMakeLists.txt`
section 5b adds `/Zi` to the Release compile flags and `/DEBUG /OPT:REF
/OPT:ICF` to the link flags (symbols without changing Release codegen). The
tool is discovered through `OPENCPPCOVERAGE_EXE`, `PATH`, or the known install
locations; CI installs it with `choco install opencppcoverage --no-progress -y`.

## Gate 4 — mutation check

`tests/mutation_check.py` owns the "is the guard pinned?" question for the
guards that coverage can only say are *executed*:

* `ServiceRegistry` refuses null registration (mutation: accept it),
* `PluginHost` API-version gate, toolchain/ABI fingerprint, discovery library
  filter, initialize() rollback, reverse-order shutdown,
* `PluginDependencyResolver` unknown phase, missing dependency, dependency
  cycle,
* `SecretVault` legacy-plaintext passthrough + empty-stays-empty,
* `CryptoUtils` PBKDF2 iteration count,
* `UserManager` used-license-key rejection,
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
* and three **planted violations** that the architecture guard itself must
  reject: a concrete service include in a plugin, a bare
  `waitForFinished()` in `src/`, a heavy SDK include in `src/utils/`.

Discipline (mirrors `AIAssistant/scripts/mutation_check.py`):

* every owning test must pass on the unmutated tree first — a missing DLL on
  `PATH` must never masquerade as a killed mutant;
* an anchor that no longer matches is STALE and fails the run;
* a mutant that does not compile is INVALID, not a kill — **except** when the
  build failed because the *output* was still open (`LNK1104`): the test that
  just ran keeps `AppCore.dll` loaded for a moment after it exits, and that
  race is retried instead of being reported as a broken mutant;
* the source is restored byte-identical in a `finally`, a lock file in the
  build directory refuses overlapping runs, and the touched targets are
  rebuilt once so the build tree ends up with the original binaries;
* `--only SUBSTR` runs a subset by label.

Reference result: **27/27 killed, 0 survived, 0 stale, 0 invalid**.
