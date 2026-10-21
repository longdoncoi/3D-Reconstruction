# AIAssistant Scorecard

> Measured signals, each produced by a command in this repository — nothing
> here is an estimate. **Last measured: 2026-10-21**, Windows, Python 3.11,
> dependencies per `requirements-test.lock` (hash-pinned: `faiss-cpu`, `rank-bm25`, `mcp`).
> A stripped environment measures less: the optional-dependency branches flip
> with it (ADR 0014).
>
> This file is **tracked** (ADR 0032). It used to be listed in `.gitignore`,
> which meant the numbers below were readable only on the machine that
> produced them — a scorecard nobody else can see pins nothing.

## Quality gates

| Gate | Command | Measured 2026-10-21 |
|------|---------|---------------------|
| Lint | `python -m ruff check .` | 0 issues |
| Types | `python -m mypy src/ai_assistant` | 0 issues, 140 files, `strict = true` (3 SDK-boundary modules relax 2 flags, ADR 0022) |
| Layer contracts | `PYTHONPATH=src lint-imports --no-logo` | 9/9 contracts kept (ADR 0032: the application layer may no longer import the plugin package — the catalogue is a port) |
| Byte-compile | `python -m compileall -q evals src` | clean |
| Tests | `PYTHONPATH=src python -m unittest discover -s tests -p "test_*.py"` | 2 093 tests, 0 failures, 6 skipped |
| Coverage | `PYTHONPATH=src python -m coverage run … && python -m coverage report` | **100.00%** — 9 220 statements, 2 436 branches, 0 missed, 0 partial |
| Mutation | `PYTHONPATH=src python scripts/mutation_check.py` | **87/87 killed** (ADR 0032 added 7: the policy-tree write deny and its prefix match, the OpenAPI surface, the log redaction, the compare-and-set, the checkpoint seal, the adapter's CAS guard and the volatile-key rule; ADR 0033 added one: the write target is resolved, so a symlink into a policy tree is matched on the file it points at; ADR 0034 added five: the secret-file read guard, the PATH scan that never consults the current directory, the case-insensitive write deny, the sealed-regime read refusal and the value-shaped egress check; ADR 0035 added six: the path-shaped program name, the case-insensitive blocked-directory rule, the path confinement itself, the LSP server resolution, the fail-closed git resolution and the attachment gate, and one more: the registered-but-not-enabled declaration), 0 survived, 0 stale |
| Mutation floor | `scripts/mutation_check.py` | `MIN_MUTANTS = 87` — deleting a mutant now fails the gate (ADR 0032) |
| Policy evals | `python evals/run_tool_eval.py`, `python evals/run_agentic_eval.py` | exit 0; tool: 16/16 (adversarial included); agentic: 114 cases / 25 scenarios, 0 failed, **0 skipped** (ADR 0025) |
| Model pins | `python scripts/verify_models.py` | 5 artifact local khớp pin, 1 chưa tải, **0 mismatched** (ADR 0023) |
| Supply chain | `python scripts/generate_sbom.py` | CycloneDX 1.5: 171 distribution (kèm **3 941 SHA-256**) + 6 artifact GGUF kèm commit/SHA-256 (ADR 0023) |
| Lock verify | `pip install --require-hashes --dry-run -r requirements.lock` + `pip download --require-hashes -r requirements-test.lock` | runtime lock: exit 0, 0 ERROR; test lock: **83/83 artifact tải về, SHA-256 khớp** (CI job `test` cài đúng lock này bằng `--require-hashes`); runtime mẫu 6/6 (ADR 0023) |
| Lock sync | `PYTHONPATH=src python -m unittest tests.unit.test_lockfile_sync` | 9 case OK; mọi direct requirement của `requirements.txt`, `requirements-test.txt` **và `requirements-docker.txt`** được lock tương ứng thỏa mãn; checker tự bị test bằng ca drift giả (range vượt lock / thiếu pin) nên không thể là no-op |
| Container lock | `pip install --require-hashes --dry-run -r requirements-docker.lock` | exit 0; CPU-only (`faiss-cpu`, **không** `faiss-gpu`/CUDA) — ADR 0032 |
| Cross-language contract | `PYTHONPATH=src python -m unittest tests.unit.test_at_rest_cipher_contract` | 6 case OK: hằng số format đọc từ `src/core/AtRestCipher.cpp` và vector đọc từ `tests/TestAtRestCipher.cpp` phải khớp bên Python — ADR 0032 |


`python evals/run_model_in_loop_eval.py` chạy tay với backend thật (fallback 3B
đã pin, SHA-256 khớp): lần đo 2026-10-19 — **accuracy 6/7 (86%), coverage 7/7,
p95 3.94s**; case trượt `en_003` nằm nguyên trong report. Không có backend thì
script in `SKIP` và exit 0 **không kèm** số accuracy, nên một lần chạy xanh
không bao giờ là con số tổng hợp giả.

## Independent review follow-up (2026-10-21, ADR 0032)

A second architectural review ran against the source, not this page, and
re-ran every gate before reading the code. It found nine defects in the
*hardening* and in the *evidence* — not in the core architecture, which held
up. All nine are closed and pinned by regression tests:

- **The scorecard itself was not tracked.** `Docs/SCORECARD.md` and ADRs
  0026–0031 were listed in `.gitignore`, so the "10/10" evidence existed
  only on the machine that produced it. Both are tracked now.
- **The mutation count was not a gate.** Deleting a mutant was
  indistinguishable from passing it: the suite stayed green and the guard
  silently stopped being guarded. `scripts/mutation_check.py` now declares
  `MIN_MUTANTS` and fails below it.
- **`/docs`, `/redoc` and `/openapi.json` were unguarded.** They are
  *app-level* FastAPI routes, so the transport guard that protects every
  other surface never reached them — a peer that could reach the port
  received the full route schema. The OpenAPI surface is now opt-in
  (`AGENT_EXPOSE_API_DOCS` / `[api] expose_docs`), and a
  `TrustedHostMiddleware` is installed when the deployment declares a host
  allow-list.
- **The Agent Card advertised the request's `Host` header.** A peer sending
  `Host: evil.example` made this server publish a card pointing at the
  attacker. The card now advertises the *configured* base URL, injected by
  the composition root.
- **Tool results were logged verbatim.** The file handler runs at DEBUG, so
  the document a tool read landed in a plaintext log — the same content the
  at-rest cipher protects. A new domain helper `describe_tool_result`
  summarises a result by shape; the user prompt and the model output are
  scrubbed on *both* the local and the remote leg.
- **The RAG reset had two owners with two different field sets.** One
  cleared the pipeline but not the encoder, the other cleared the encoder
  but not the pipeline — so `/health` could report zero chunks while
  retrieval still served the previous corpus. `RagModule.reset_index_state()`
  is now the single owner and clears every handle under the lock.
- **Task state and its event log were two transactions.** A crash between
  them left a task `completed` with no terminal event, and an SSE subscriber
  waited for an event that would never arrive. `ports.TaskStore` now
  declares a single `commit(task, *, expected, events)`; the adapter writes
  both in one transaction, and the duck-typed read-then-write fallback — the
  very race the compare-and-set exists to close — is gone.
- **The LangGraph checkpoint store was plaintext.** It holds the system
  prompt, the user's messages and the tool results, and `.env.example`
  claimed `Cache/` was sealed. The SQLite checkpointer now writes through
  LangGraph's `EncryptedSerializer`, backed by the platform's at-rest key.
- **The write deny-list named two policy files.** The transport policy, the
  A2A scope map, the tool catalogue and the approval logic were all
  writable. The deny-list now covers the policy *trees*, matched on the
  resolved path so a symlink cannot walk around it.

Two further classes of gap were closed:

- **Six rules lived in a layer above the model they constrain**, so two
  layers each believed they owned them: the classification lattice, the
  capability→scope lattice, the terminal task states and the identity of a
  tool invocation. They now live in `domain`. The invocation fingerprint
  had *two* rules in *three* places, and the two disagreed about the
  volatile per-retry keys — so a retry that only changed its `request_id`
  slipped past the duplicate-call guard. One owner, two named rules.
- **CI did not measure what the scorecard claimed.** CI ran on Linux while
  the numbers were measured on Windows, the platform the product ships on,
  and two of the sandbox's security guards are `skipUnless(win32)` — they
  never executed in CI. Every action is now pinned to a commit SHA, `ruff`
  comes from the hash-pinned lock, and a **Windows job** runs the suite, the
  coverage gate, the import contracts and the type check. The container
  installs its own hash-pinned, CPU-only lock, its base image is pinned by
  digest, and its health probe checks liveness (it ships no weights, so the
  readiness probe could never pass).

One defect was found *by* the new gates rather than by the review: the RAG
cache tamper test flipped the last base64 character of the sealed envelope.
The last character of a base64 quantum contributes as few as two bits, so
the tamper could decode to the very same bytes — the seal still verified
and the test passed without having tampered anything, one run in four. The
tamper now lands in the middle of the envelope.

## Follow-up on the 2026-10-21 review (ADR 0035)

ADR 0034 closed eight defects and concluded that *every gate was green
because the gates could not see them*. To test that conclusion, the review
was re-run against the layers ADR 0034 never touched — RAG, the HTTP routes,
the plugin loader, the orchestration, the remaining tools — with the same
lens. It found **five more of the same class**, and one of them was a
*sibling* of a rule ADR 0034 had just fixed, twenty lines away, in the same
function:

| Defect | Why the gate was green |
|---|---|
| `.GIT/config` was readable (no approval) — it carries the remote URL and its credential helper | The *blocked-directory* rule compared case-**sensitively** while the *secret file name* rule right next to it compared case-**insensitively**: a visible asymmetry inside one function |
| `lsp_client` still resolved its program with `shutil.which` — a binary planted in the working directory ran | The test mocked that lookup and returned a **relative** name — exactly the shape a current-directory hit produces. It was the **second** place a program was resolved; ADR 0034 had fixed only the first |
| The LSP read gate compared the path *spelling* — a junction inside the project read out of it | Every other read gate resolves first and fails closed; this one did not |
| The git tools threw the verified absolute hit away with `or "git"` | The test *asserted* the unsafe behaviour was correct, with `subprocess.run` mocked |
| The chat route opened **any** client-named path, holding the at-rest key | `image_to_data_uri` was mocked; no case ever read a real file |

**The conclusion ADR 0034 drew is now twice measured: a green gate table is a
measurement of the *gate*, not of the *system*.** And the deeper point: if
thirteen defects share one class, then *counting the fixes* is a meaningless
number — the class survives, and the fourteenth instance is one more line of
code. Closing instances is the wrong move; closing the **class** is the right
one.

So ADR 0035 closes the class, not the instances:

- **Two kernel leaves own the rules.** `ai_assistant/path_policy.py`
  (`canonical_path`, `resolve_within`, `is_blocked_dir`, `is_secret_name`)
  and `ai_assistant/exec_policy.py` (`resolve_executable`,
  `executable_candidates`). Standard library only, so every layer may depend
  on them and they depend on none. Three places that each resolved a program
  their own way now use one rule; four places that each compared a path their
  own way now use one rule.
- **Every comparison is case-insensitive.** A Windows filesystem is
  case-insensitive, so `.GIT` *is* `.git`. The tests are **property-shaped**:
  they walk the rule sets themselves, so adding a name to a set automatically
  requires the case-insensitive behaviour — a drift-proof guard rather than a
  list of examples that can go stale.
- **The real target decides, not the spelling.** `resolve_within` checks the
  resolved target (links followed) and canonicalises both sides before
  comparing, so a junction out of the root is an escape and Windows' 8.3
  short form never mismatches the canonical long form. Three read gates that
  used three different methods now share one, and it is **fail-closed**.
- **`shutil.which` is banned — by a static guard.**
  `tests/integration/test_executable_resolution_guard.py` walks the AST of
  every module under `src/ai_assistant/**` and fails the build if any of
  them calls the platform lookup. The guard has to be static because the
  defect it prevents is invisible at runtime: the old mock returned a
  relative name, the very shape a current-directory hit produces, so the
  suite stayed green while the control did not work.
- **Fail closed instead of degrading.** The git tools no longer fall back to
  a bare name when `git` is not on `PATH`; the call is refused. The shipped
  container image has no `git` at all, so there the fallback was the
  *default* path.
- **The attachment gate.** The chat route opens a client-named path and holds
  the at-rest key, so the path is only opened when it resolves inside a
  directory the desktop client stores attachments in — derived exactly as
  `AppConfig::uploadDir` does and pinned by a new cross-language contract
  test — or when the file is a **sealed** envelope, which only the platform
  can produce.

`MIN_MUTANTS` 80 → **87**: seven new mutants, one per new guard.

### The class is closed at the structural level

| Class | Closed how | A new instance would |
|---|---|---|
| resolving a program | one kernel owner + a static AST guard | fail the build |
| comparing a path to a root | one kernel owner, fail-closed, canonical on both sides | use the owner and be correct |
| comparing a name to a rule set | one owner, case-insensitive, property-tested over the set | be correct by construction |
| naming a secret file | one owner, matched by name not extension | be correct by construction |
| opening a client-named file | one gate (upload roots or a sealed envelope) | be correct |

### Remaining tracked gaps (ADR 0035)

- **The two LSP tools are still refused by the supervisor.** They are
  advertised to the model and registered in the factory, but the
  supervisor's `ALLOWED_TOOLS` table has no row for them, so every call is
  refused. That refusal is what *masked* the `shutil.which` defect while it
  was live. The invariant is recorded: **do not enable the LSP tools before
  `lsp_client` uses the shared resolver** — it does now, so the road is safe,
  but enabling them is a separate decision.
- **ADR 0013 §6 describes the current state of those two tools incorrectly.**
- **The attachment gate accepts absolute paths that resolve into an upload
  root.** A deployment whose client stores attachments elsewhere has to
  declare another root.

## Follow-up on the 2026-10-21 review (ADR 0034)

A third architectural review ran against the source, not this page, and
re-ran every gate before reading the code. **Every gate was green** — lint
clean, mypy strict 0 issues, 9/9 import contracts, 2 034 tests, coverage
**100.00 %** (9 095 statements / 2 388 branches), mutation 74/74 killed — and
the core architecture held up. The review found **eight defects**, and they
share one property: *the gates were green because they could not see them.*

| Defect | Why the gate was green |
|---|---|
| `read_file` returned `AIAssistant/.env` — the at-rest key and **every** bearer token — **with no approval** | The block-list was an *extension* list, and a dotfile has no extension (`splitext(".env") == (".env", "")`), so the guard never fired |
| A binary planted in the working directory ran instead of the operator's own tool | The test that "pinned" the control **mocked `shutil.which` to return `None`** — a precondition that never holds on Windows |
| `PYPROJECT.TOML` overwrote the real `pyproject.toml` | The deny-list compared basenames **case-sensitively**; no test tried the variant |
| `AesGcmFieldCipher.open()` served unsealed plaintext — an **AEAD downgrade** | `open_bytes` (same class) refused it; the text branch did not, and no test reached it |
| A2A discovery **followed redirects** while dispatch refused them | Two legs of one transport used two different openers; the tests only covered the dispatch leg |
| A credential pasted into the user's own prompt left the trust boundary in clear | `has_sensitive_keys` did exactly what its name says — and the name was the bug |
| The composition root was **writable**: the OpenAPI gate, the dotenv path and the cipher's key | The deny-list covered the policy *files*, not the layer that assembles the platform |
| The symlink guard could not be measured on Windows — the mutation gate went **red on the shipping platform** | The test `skip`ped when the OS refused to create a symlink |

This is the same class ADR 0029 already fixed once for
`/v1/chat/completions`: *"every case mocked the collaborator — and a mock
answers any attribute"*. It came back in the sandbox layer: the control was
real, the test was real, and **the test could not reproduce the attack**.

All eight are closed and pinned by regression tests:

- **Secret files are blocked by name, at one chokepoint.** `is_secret_name`
  (an explicit set plus the whole `.env*` family) is checked inside
  `agent_safe_path` — the function every read tool resolves through — against
  both the lexical spelling and the real one a symlink points at.
  `search_text` walks the tree itself, so it uses the same rule: the walk
  reaches `secrets.json` (a `.json` file) long before any dotfile.
- **The PATH scan never consults the current directory.** `shutil.which` is
  gone: on Windows it prepends the CWD to the search path, so it returned a
  *relative* hit for a binary that was never on `PATH` — the allow-list was
  not enforced and the refusal never fired. `_resolve_executable` walks the
  `PATH` entries itself and returns an absolute path, so the OS cannot
  re-order the search when the child starts.
- **The write deny-list ignores the case of the file.** Windows filesystems
  are case-insensitive, so `PYPROJECT.TOML` **is** `pyproject.toml`; the
  substring fallback cannot cover it (`pyproject.toml` is the one entry that
  is not a path fragment).
- **A sealed column must be sealed.** `SqliteTaskStore` tracks a *sealed
  regime*: once the one-time migration has run, an unsealed value is refused
  with `StorageDecryptionError` instead of being served as authentic. The
  binary artifacts already refused an unsealed file for the same reason; the
  text columns now behave the same way. A legacy pending-action file is
  re-sealed and written back at the first read.
- **Discovery and dispatch share one opener.** The discovery leg used the
  *default* urllib opener while the dispatch leg deliberately refused 30x —
  the weaker leg was the one that runs at startup against an unvetted peer.
- **Egress asks about the value, not only the key.**
  `has_sensitive_content` joins `has_sensitive_keys` under one owner
  (ADR 0032: one owner, two named rules). The value shapes are deliberately
  unambiguous — a credential assignment with a long value, or a token format
  that carries its own marker — because a boundary that refused on prose
  would refuse ordinary requests. Parameters (a payload to be *executed*) are
  refused; the message (the user's own prompt) is masked.
- **The composition root is not writable.** `_DENIED_WRITE_TREES` now covers
  `bootstrap`, `llm`, `orchestration`, `plugins`, `rag`, `observability` and
  `tools` inside the `ai_assistant` package: the layer that assembles the
  platform *is* the policy.
- **The symlink guard is measurable on Windows.** ADR 0033's
  `_canonical_path` compares one spelling on both sides, so a case-variant
  write into a policy tree hits the policy branch — without needing the
  privilege to create a symlink. The gate is no longer red on the platform
  the product ships on.

### Controls that are only verified through a mock

Recorded so the next review does not have to guess which security controls
are proven by a test that cannot reproduce the attack. ADR 0035 moved most
of this list into the *structural* column — a control that is verified
through a mock but whose class is closed by a static guard is no longer a
silent gap.

| Control | The test proves | The test cannot prove | Closed at the class level? |
|---|---|---|---|
| `_resolve_executable` (PATH scan) | that a planted binary outside `PATH` is not resolved | — (exercised for real, against real files) | yes — static AST guard |
| `agent_safe_path` (secret names) | that `.env` and its family are refused | — (exercised for real, against real files) | yes — one owner, property-tested |
| the write deny-list (case) | that `PYPROJECT.TOML` is refused | — (exercised for real) | yes — one owner, property-tested |
| the sealed-regime read | that a substituted plaintext column is refused | — (exercised for real, against a real SQLite file) | partially — see the tracked gap on the schema version |
| the blocked-directory rule (case) | that `.GIT` is refused | — (exercised for real) | yes — one owner, property-tested |
| the LSP server resolution | that the shared resolver is used | — (the *planted binary* case is exercised for real) | yes — static AST guard |
| the git resolution | that an absent `git` is refused | — (exercised for real) | yes — one owner, fail-closed |
| the attachment gate | that a path outside the upload roots is refused | — (exercised for real, against a real file) | yes — one gate |
| the path confinement | that a link out of the root is refused | — (exercised for real where the platform allows a link; forced otherwise) | yes — one owner, fail-closed |

### The one control that is still mock-only

The **sealed-regime read** of the task store is exercised for real, but the
*signal* that says "this store has migrated" is `PRAGMA user_version` — which
lives **inside the file** an attacker with write access can also edit. A
forged version makes the store re-run its one-time migration, and the
migration seals the substituted rows. So the "leave it plaintext" downgrade
is closed (ADR 0034) while the "forge the version" one is not. Closing it
needs a trust anchor *outside* the file, which is the same question ADR
0034 recorded for the whole-file pending envelope. Tracked, not forgotten.


### Remaining tracked gaps (ADR 0034)

- **A whole-file pending-action envelope has no external anchor.** It cannot
  tell "plaintext written by an older version" from "plaintext substituted by
  an attacker" without something outside that file. The window is narrowed to
  one read; the rest is the operator's (ACLs, volume encryption), as before.
- **A short secret in prose has no shape to recognise.** The value rule needs
  a long enough value; a four-character password in a sentence is invisible
  to any pattern matcher. Written down rather than hidden.

## Follow-up on the 2026-10-21 review (ADR 0033)

One claim from the review above was **not true on the platform the product
ships on**. ADR 0032 closed the write deny-list over the policy *trees* and
matched them on the *resolved* path, so a symlink cannot walk around it — but
the comparison was between two *spellings* of one directory. Windows has two:
the canonical long form (`C:\Users\runneradmin`) and the 8.3 short form
(`C:\Users\RUNNER~1`), and the CI runner hands the sandbox the short one
(`TEMP=C:\Users\RUNNER~1\AppData\Local\Temp`). A symlink whose target does not
exist is resolved by *reading the link*, which keeps the short spelling, while
the project root is resolved by the OS into the long one — so `relative_to()`
failed and the write was reported as an **escape** instead of the policy denial
it is. The deny-list was never reached.

The write gate now compares one canonical spelling of the path on both sides
(`tools/sandbox._canonical_path`), so the ADR 0032 claim holds on Windows as
well. It is pinned twice: a Windows-only case builds the short-spelled link on
purpose (`GetShortPathNameW`) and goes red without the fix, and a new mutant
(the target is not resolved) is killed by the suite.

- **The write deny-list compared two spellings, not two paths.** Fixed, and
  pinned by
  `tests/unit/test_sandbox.py::test_a_link_spelled_with_the_short_path_is_denied`
  plus the new mutant (`scripts/mutation_check.py`, 75 mutants now).

## Independent review follow-up (2026-10-20)

The architectural review was re-run against the source, not this page. The
following defects it found are now closed and pinned by regression tests:

- **Remote A2A delegation was always refused.** The egress classifier treated
  `session_id`/`idempotency_key` as credentials. Redaction keys and egress
  secrets are now separate sets (`REDACTION_KEYS` vs `EGRESS_SECRET_KEYS`), and
  the real `graph_wiring` payload shape is tested end to end.
- **Approval grants were predictable.** `issue_grant_pair` now mints independent
  `secrets.token_urlsafe` capabilities instead of deriving them from the action
  id or invocation fingerprint.
- **Same-session queueing could consume the global run budget.** The global slot
  is now acquired after the session lock, so only a run that can execute holds
  a slot.
- **SSE subscriptions and agent runs had no deadline/disconnect path.** A2A
  subscriptions end at a bounded deadline; an agent SSE stream cancels its
  cooperative run when the client drops.
- **A crash left `submitted`/`working` tasks non-terminal.** `TaskService`
  reconciles them on construction; SQLite records a schema version so the
  migration runs once.
- **Remote inference buffered an unbounded response.** It now shares the A2A
  1 MiB cap.
- **The read sandbox followed links out of the project.** `agent_safe_path`
  now resolves symlinks/junctions and checks the real target, not just the
  lexical path.
- **A2A had no per-task ownership.** It now resolves a tenant from
  `AI_A2A_TENANTS` (bearer token → tenant) and every get/cancel/resume/list/SSE
  operation is owner-scoped; an empty mapping keeps single-tenant `"default"`
  behaviour. Unknown tokens fail closed with 401.
- **SSE subscribers were unbounded.** A2A admission now caps concurrent
  streams (`AI_A2A_MAX_SUBSCRIBERS`, default 32) and releases the slot on
  completion or client disconnect, so the 300 s deadline is not the only
  resource bound.
- **A failed `compose()` looked composed.** A build failure now rolls
  `_COMPOSED` back and re-raises, so the next access retries instead of
  exposing a half-built graph; startup logging is replaced, not duplicated.
- **Terminal tasks accumulated forever.** Startup now prunes completed /
  failed / canceled / rejected rows past `AI_TASK_RETENTION_DAYS` (default 30,
  `0` disables) together with their event log, while live tasks are untouched.
- **Importing the server loaded `.env`.** Environment merging moved into
  `compose()` before the modules that snapshot settings are imported, so the
  module import remains configuration-free.
- **A2A configuration was frozen at import.** `a2a_protocol` no longer reads
  `A2A_ENABLED`/`A2A_REMOTE_AGENTS`/card settings at module scope: the optional
  SDK probe and the discovery target list are resolved on first use, so
  importing the adapter has no environment-dependent side effects.
- **Remote A2A discovery was never invoked.** Startup now populates the remote
  registry from the merged deployment allowlist (`[a2a] trusted_endpoints` plus
  `A2A_REMOTE_AGENTS`) in a worker thread; an empty allowlist — the default —
  stays a network-free no-op, and a discovery failure is logged without
  blocking startup. Registry reads and writes are lock-guarded.
- **An admin model reload could swap the handle mid-run.** `LlmModule` now
  issues a per-run model lease; reload waits for every active lease and blocks
  new ones until the swap completes, so a run sees one model end to end.
- **The remote registry and supervisor lookup were process-wide state.** The
  registry is now a `RemoteAgentRegistry` instance owned by `compose()`, injected
  into both `A2ARouter` and a `functools.partial(delegate, remote_lookup=...)`
  handed to `AgentService`; `delegate()` takes the resolver as an argument and
  keeps the local default when none is supplied. A second composition (tests,
  embedded hosts) can no longer observe or overwrite another's A2A state.
- **Task and pending-action content was plaintext at rest.** It is now sealed
  per field with AES-256-GCM (`enc:v1:<base64(nonce||ciphertext)>`, 96-bit
  random nonce per field, the field name as associated data so a ciphertext
  moved to another column fails authentication). The key comes from
  `AI_STORAGE_KEY` → `AI_STORAGE_KEY_FILE` → one key generated outside
  `APP_DATA_DIR`; every source is fail-closed, and a missing or wrong key
  aborts the read instead of handing ciphertext to the caller. Legacy plaintext rows are
  sealed exactly once (`PRAGMA user_version = 3`). Pinned by
  `tests/unit/test_storage_crypto.py`, `test_sqlite_store_crypto.py`,
  `test_pending_store_crypto.py` and 10 new mutants (ADR 0027).
- **The RAG cache was a second plaintext copy of the user's document corpus.**
  `Cache/chunks.pkl` holds the full text of every indexed document plus the
  base64 of every image, `bm25.pkl` the tokenized corpus and
  `faiss_index.bin` the embeddings. The HMAC that used to guard them proved
  *integrity*, never *confidentiality* — a copy of the data volume carried a
  readable copy of the corpus. All three are now sealed with the **same key**
  as the task store (one key for everything the app writes; the separate
  `cache_hmac_key` file is gone), the FAISS index is serialized in memory so
  the plaintext never touches the disk, and a cache that cannot be
  authenticated (wrong key, tampered, unsealed, written by an older version)
  is a *miss* — discarded and rebuilt, never unpickled. Pinned by
  `tests/unit/test_rag_cache_seal.py`, `test_index.py` and 5 new mutants
  (ADR 0028).
- **`/v1/chat/completions` was dead in production, and every gate was green.**
  `ChatbotAgent` called the prompt-building methods on the LLM *runtime
  handle*; after that handle was split from the prompt builder, the calls
  raised `AttributeError` on every request. The suite could not see it
  because every case mocked the collaborator — and a mock answers any
  attribute. The prompt builder is now a real port (`PromptBuilder`), wired
  explicitly by the composition root, and
  `tests/integration/test_chatbot_agent.py::RealWiringTests` builds the
  *production* objects — the case is red against the old code (ADR 0029).
- **The at-rest key could not be rotated — changing it lost every row.**
  `scripts/rekey_storage.py` re-encrypts the task store in **one
  transaction**, re-seals the pending payload atomically, deletes the derived
  RAG cache, and refuses to write anything unless the old key actually opens
  the state. `--verify` answers "is this the right key?" read-only and
  `--dry-run` previews the rotation (ADR 0030).
- **The chat attachments and their thumbnails were plaintext.** They are now
  sealed with the **same platform at-rest key** the AI service uses — one
  key for everything, so the AI service can still open the file it has to
  send to the model, with the model's input bytes unchanged. The envelope
  layout, the associated data and the key-resolution order are a contract
  between the two processes and live in `ai_assistant/ports.py` (ADR 0031).
  The contract is pinned **in both directions**: each side holds the other
  side's output as a fixed vector
  (`tests/unit/test_sealed_attachment_read.py` and
  `tests/TestAtRestCipher.cpp`), so a change that breaks the other language
  fails a test instead of surfacing as "the assistant cannot see the image".
  ADR 0032 added the third pin that closed the gap this left: the format
  constants are read out of `src/core/AtRestCipher.cpp` and the reference
  vector out of `tests/TestAtRestCipher.cpp`, and compared with the Python
  ones — so a vector re-pinned on one side only fails the Python CI, which
  now triggers on the C++ files as well.
- **The HTTP surface was never exercised against the composed app.** Every
  route test injected a double, so the *binding* between the composition
  root and the router was never executed — the same class of gap as the
  `/v1/chat/completions` bug above.
  `tests/integration/test_real_http_surface.py` now composes the real
  object graph in a fresh interpreter, swaps only the model backend and
  drives a real request through the real router. The case is red against a
  broken binding.

Remaining tracked gaps (not measured-gate failures — each is a decision the
repository deliberately does not make, with the reason written down):

- **Unattended rotation and KMS/DPAPI integration.** The at-rest key can be
  rotated now — `scripts/rekey_storage.py` re-encrypts the task store in one
  transaction, re-seals the pending payload atomically, deletes the derived
  RAG cache, and refuses to write anything unless the old key actually opens
  the state (ADR 0030). What is missing is calling a KMS/DPAPI directly and
  rotating without an operator in the loop.
- **The source documents are still the user's own files.** ADR 0028 seals the
  *copy* the assistant creates; encrypting the original document tree is the
  user's and the operator's business (volume encryption).
- **Volume-level encryption stays an operator concern.** Field-level encryption
  covers the listed content columns, the cache artifacts and the chat
  attachments; BitLocker/LUKS, volume ACLs and backup encryption remain
  deployment responsibilities.
- **The key file relies on the account's filesystem ACL** (owner-only on POSIX,
  the account ACL on Windows). A deployment that needs stronger custody sets
  `AI_STORAGE_KEY_FILE` to a KMS-mounted secret instead.
- **The Postgres and Redis checkpoint backends** keep their published
  constructor signature (`from_conn_string` takes no `serde`), so their
  state is protected by the database's own encryption rather than by the
  platform key. The SQLite backend — the one that lands in the app's own
  data directory — is sealed (ADR 0032).
- **The two third-party C++ actions** (`jurplel/install-qt-action`,
  `lukka/run-vcpkg`) are still referenced by tag in the C++ workflow. They
  are outside this package's scope and were not pinned in ADR 0032; the
  Python workflows are fully SHA-pinned.

## Numbered signals

The `SCORECARD #N` labels below are the historical identifiers used in code
comments (`observability/metrics.py`, `application/coordination.py`,
`evals/run_model_in_loop_eval.py`) and in `ARCHITECTURE.md`; they resolve to
the rows here.

| Label | Signal | Where it is measured |
|-------|--------|----------------------|
| SCORECARD #1 | Share of the golden set actually graded by the loaded model (`model_coverage=`) | `evals/run_model_in_loop_eval.py` |
| SCORECARD #2 | Measured p95 latency per timed node (`plan`/`reason`/`tool`/`reflect`), in-process, no PromQL round-trip | `observability/metrics.py` — `p95_latency_by_node()` over the bounded span window |
| SCORECARD #4 | Cancel-to-stop recovery seconds, reported through the observer the composition root injects into `agent_recovery_seconds` | `application/coordination.py` — `TaskCoordinator.finish` |

`SCORECARD #3` has no consumer in this tree: no module, test or workflow
references it, so it is deliberately not defined here — a metric nobody pins is
removed rather than restated (ADR 0015).

## How this page stays honest

- Every row is a measurement with the command that reproduces it; CI re-runs
  the same gates (`Build-Python-ci.yml`, `agent-evals.yml`) — on **both**
  Linux and Windows, so the numbers describe the platform the product ships
  on (ADR 0032).
- Coverage and mutation are *outputs* of the suite, not targets set by hand:
  `pyproject.toml` owns the configuration (`branch = true`, `fail_under = 100`,
  `exclude_also`), so a local run and CI measure the same thing.
- The mutation **count** is a gate too (`MIN_MUTANTS`): deleting a mutant is
  a decision, and the gate turns it into a visible one (ADR 0032).
- This page is **tracked**, so the numbers are readable by anyone who clones
  the repository — a scorecard nobody else can see pins nothing (ADR 0032).
- The evaluation scripts exit non-zero on a failing **or not-run** case, so a
  green eval cannot come from an unchecked path.
