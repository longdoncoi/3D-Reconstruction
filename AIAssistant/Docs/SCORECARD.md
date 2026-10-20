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
| Types | `python -m mypy src/ai_assistant` | 0 issues, 138 files, `strict = true` (3 SDK-boundary modules relax 2 flags, ADR 0022) |
| Layer contracts | `PYTHONPATH=src lint-imports --no-logo` | 9/9 contracts kept (ADR 0032: the application layer may no longer import the plugin package — the catalogue is a port) |
| Byte-compile | `python -m compileall -q evals src` | clean |
| Tests | `PYTHONPATH=src python -m unittest discover -s tests -p "test_*.py"` | 2 034 tests, 0 failures, 3 skipped |
| Coverage | `PYTHONPATH=src python -m coverage run … && python -m coverage report` | **100.00%** — 9 095 statements, 2 388 branches, 0 missed, 0 partial |
| Mutation | `PYTHONPATH=src python scripts/mutation_check.py` | **74/74 killed** (ADR 0032 added 7: the policy-tree write deny and its prefix match, the OpenAPI surface, the log redaction, the compare-and-set, the checkpoint seal, the adapter's CAS guard and the volatile-key rule), 0 survived, 0 stale |
| Mutation floor | `scripts/mutation_check.py` | `MIN_MUTANTS = 74` — deleting a mutant now fails the gate (ADR 0032) |
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
