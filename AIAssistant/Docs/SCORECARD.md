# AIAssistant Scorecard

> Measured signals, each produced by a command in this repository — nothing
> here is an estimate. **Last measured: 2026-11-18**, Windows, Python 3.11,
> dependencies per `requirements-test.lock` (hash-pinned: `faiss-cpu`, `rank-bm25`, `mcp`).
> A stripped environment measures less: the optional-dependency branches flip
> with it (ADR 0014).
>
> This file is **tracked** (ADR 0032). It used to be listed in `.gitignore`,
> which meant the numbers below were readable only on the machine that
> produced them — a scorecard nobody else can see pins nothing.

## Quality gates

| Gate | Command | Measured 2026-11-18 |
|------|---------|---------------------|
| Lint | `python -m ruff check .` | 0 issues |
| Types | `python -m mypy src/ai_assistant` | 0 issues, 155 files, `strict = true` (3 SDK-boundary modules relax 2 flags, ADR 0022) |
| Layer contracts | `PYTHONPATH=src lint-imports --no-logo` | 10/10 contracts kept; no peer layer imports bootstrap and `internal` imports no `ai_assistant` layer |
| Byte-compile | `python -m compileall -q evals src` | clean |
| Tests | `PYTHONPATH=src python -m tests.run_suite` | **2 727 tests**, 0 failures, 8 skipped; executable `ResourceWarning`/unraisable leak gate (ADR 0056/0058/0060/0061/0062/0063/0064) |
| Coverage | `PYTHONPATH=src python -m coverage run -m tests.run_suite && python -m coverage report` | **100.00%** — 12 286 statements, 3 440 branches, 0 missed, 0 partial (ADR 0058/0059/0060/0061/0062/0063/0064 full baseline) |
| Mutation | `PYTHONPATH=src python scripts/mutation_check.py --jobs 4` | **363/363** killed, 0 survived, 0 stale; isolated workers with green owning-suite preflight (ADR 0056/0058/0059/0060/0061/0062/0063/0064; CI uses 2 workers) |
| Mutation floor | `scripts/mutation_check.py` | `MIN_MUTANTS = 363` — deleting a mutant now fails the gate (ADR 0032/0052/0053/0054/0055/0056/0057/0058/0059/0060/0061/0062/0063/0064); `PINNED_DECISION_FILES` adds a per-file floor |
| Policy evals | `python evals/run_tool_eval.py`, `python evals/run_agentic_eval.py` | exit 0; tool: 16/16 (adversarial included); agentic: 136 cases / 27 scenarios, 0 failed, **0 skipped** (ADR 0025/0052) |
| RAG framework eval | `python evals/run_rag_framework_eval.py` | **4/4 pass**; RAGAS 0.4.3 mean context precision **0.9375**, recall **1.0**; DeepEval 4.3.3 mean retrieval-F1 **0.9643**; offline/non-LLM (`requirements-eval.lock` / CI env) |
| Pinned model-in-loop | `python evals/run_model_in_loop_eval.py --model-path … --min-model-coverage 0.9 --min-accuracy 0.9 --min-slice-accuracy 0.75 --min-freshness-accuracy 0.9` | Verified pinned 3B (SHA-256 `9c9f56…3f94`), measured 2026-11-12: planner **32/32 (100%)**, model coverage **100%**, all 8 language × kind slices **4/4**, p95 **3.109 s**; production freshness **16/16 (100%)**, coverage **100%**, p95 **0.203 s**. |
| Model pins | `python scripts/verify_models.py --require-all` | **6/6 artifact local khớp pin**, 0 missing, **0 mismatched** (ADR 0023; measured 2026-11-12) |
| Supply chain | `python scripts/generate_sbom.py` | CycloneDX 1.5: 171 distribution (kèm **3 941 SHA-256**) + 6 artifact GGUF kèm commit/SHA-256 (ADR 0023) |
| Dependency audit | `python scripts/dependency_audit.py` | 5/5 hash-pinned lock sạch + direct-URL llama wheel được ánh xạ và audit theo version, 0 advisory chưa xử lý; 2 ngoại lệ không có bản vá được giới hạn phạm vi và ghi mã cụ thể (`PYSEC-2026-2447`, `PYSEC-2026-3046`) |
| Lock verify | `pip install --require-hashes --dry-run -r requirements.lock` + `pip download --require-hashes -r requirements-test.lock` | runtime lock: exit 0, 0 ERROR; test lock: **103/103 artifact tải về, SHA-256 khớp** (CI job `test` cài đúng lock này bằng `--require-hashes`); runtime mẫu 6/6 (ADR 0023) |
| Lock sync | `PYTHONPATH=src python -m unittest tests.unit.test_lockfile_sync` | 9 case OK; mọi direct requirement của `requirements.txt`, `requirements-test.txt` **và `requirements-docker.txt`** được lock tương ứng thỏa mãn; checker tự bị test bằng ca drift giả (range vượt lock / thiếu pin) nên không thể là no-op |
| Container lock | `pip install --require-hashes --dry-run -r requirements-docker.lock` | exit 0; CPU-only (`faiss-cpu`, **không** `faiss-gpu`/CUDA) — ADR 0032 |
| Cross-language contract | `PYTHONPATH=src python -m unittest tests.unit.test_at_rest_cipher_contract` | 6 case OK: hằng số format đọc từ `src/core/AtRestCipher.cpp` và vector đọc từ `tests/TestAtRestCipher.cpp` phải khớp bên Python — ADR 0032 |

## Verified RAG, bounded re-planning and selective token streaming (2026-11-17, ADR 0064)

RAG now returns typed hits with repository-relative provenance, rank, fusion,
rerank and confidence signals. The agent-facing path rejects missing or weak
evidence instead of presenting it as an answer, and retrieved text is treated
as untrusted data rather than executable instruction.

The workflow critic emits closed decisions (`continue`, `retry`, `replan`,
`ask_user`, `reconcile`, `stop`). Re-plans and tool calls have explicit
budgets. Only a response already classified as user-visible prose streams
tokens; planner, tool-selection and critic JSON stay atomic. SSE carries
ordered `status`, `delta`, `step` and terminal events through a bounded queue,
while Qt renders a draft and replaces it with the canonical final envelope.

Six decision mutants pin the new fail-closed boundaries and raise the floor
from 357 to **363/363 killed**.

## Final architecture enforcement closure (2026-11-17)

The two documentation-strength claims that were still weaker than their
enforcement are now executable contracts.  Every non-bootstrap package and
top-level module is forbidden from importing the composition root, while the
framework wrapper under `internal/` may import SDK/stdlib modules but no
`ai_assistant` layer.  The AST twins scan the complete source tree and include
synthetic self-tests, so a clean tree cannot make either policy vacuously
green.  Two decision mutants pin those self-tests and raise the floor to
**357/357 killed**.

The model-in-loop workflow no longer performs a version-only install from an
extra index.  `requirements-llama-cpu.lock` now contains the Linux/amd64 wheel
and its complete minimal dependency closure, all SHA-256 pinned, and the job
installs it with `--require-hashes`.  Ruff security exceptions are no longer
repository-wide: they are confined to named, independently tested boundary
modules, so a new subprocess, URL opener, silent cleanup, broad exception or
mutable class default elsewhere fails lint.

## Atomic composition follow-up (2026-11-17)

Composition now builds every public binding in local state and publishes the
complete graph through one `globals().update()` only after the last router is
mounted. A concurrent direct attribute reader can no longer observe a staged
`app` while `_COMPOSED` is false. The transaction also snapshots root logging
handlers: on failure it removes and closes only handlers owned by this service,
restores foreign handlers, and still contains a failing handler close.

The RAG framework import test is now enabled explicitly by eval CI instead of
by ambient package discovery. A developer environment containing packages
from both the unit and eval locks therefore cannot silently change the unit
suite's test selection. Two new decision mutants raise the floor from 353 to
**355/355 killed**.

## Transactional release boundary hardening (2026-11-17, ADR 0063)

The shipped image now contains the repository-level tool and desktop-action
contracts and CI executes both from inside that image. Composition publishes
the object graph only after a complete build; failure drains every allocated
worker/store and restores the preceding complete state. Authenticated A2A
egress is HTTPS-only unless development opts into plaintext explicitly.
Python locks and the built image now have blocking vulnerability audits.
The image refreshes its digest-pinned Debian base, upgrades the scanner-visible
APT/zlib packages to their fixed versions, and removes stale vulnerable
setuptools vendor metadata before overlaying the hash-pinned runtime prefix.

Nine decision mutants raise the floor from 344 to **353/353 killed**. The
changed A2A and composition owners were re-swept independently with 0 survived
and 0 stale; the full gate row above records the repository-wide sweep.

## Windows DPAPI storage-key custody (2026-11-16, ADR 0061)

The automatically generated Windows desktop key is no longer readable base64.
New keys are stored as user-scope `dpapi:v1` envelopes, and a legacy plaintext
key is atomically migrated before it can be used. DPAPI or migration failure
aborts startup instead of silently retaining plaintext. A real Windows native
round trip plus cross-platform failure tests pin the boundary. Five new
decision mutants raise the floor from 334 to **339/339 killed**.

## Verified write handles and container sandbox parity (2026-11-16, ADR 0062)

Write/create commits no longer reopen an authorised path string. Windows
holds a non-reparse directory-handle chain without delete sharing; POSIX uses
`dir_fd` plus `O_NOFOLLOW`, and both create and atomically replace relative to
the verified parent. The compose profile now enables command tools through
the already-running, read-only capability-dropped service container, and the
runtime accepts that boundary only after observing a Linux container marker.
The native mechanism is isolated in `tools/verified_write.py`, removing this
responsibility from the sandbox policy hotspot. Five new decision mutants
raise the measured floor from 339 to **344/344 killed**.

## Bounded LSP and durable worker handoff (2026-11-16, ADR 0060)

The LSP client now owns blocking pipe reads in one response-reader thread.
Callers wait on a bounded queue for the exact JSON-RPC id, so a silent live
server cannot defeat the deadline and a notification cannot replace the
requested response. Teardown independently closes both pipes and always
attempts terminate, bounded wait, kill and a final bounded reap.

The durable task lifecycle now holds its lifecycle lock across commit and
worker start. A failed initial start is compensated to `FAILED`; a failed
resume returns to `INPUT_REQUIRED` without the consumed resume payload. Real
subprocess and SQLite regressions pin both defects. Seven new decision mutants
raise the measured floor from 327 to **334/334 killed**.

## Verified read handles and executable resource ownership (2026-11-15, ADR 0058)

All package-owned readers of untrusted paths now consume data from a shared,
already-open handle that has passed `fstat`: it must be a regular file with one
link, and the path must still identify that handle. This closes the hard-link
alias bypass and the identity gap at the final open; it does not make earlier
path-based root/name policy checks atomic against a hostile concurrent local
writer. An AST boundary test rejects direct built-in `open()` in the guarded
reader set. The original contract was intentionally read-side only. ADR 0062
subsequently closed ADR 0033's write/create symlink/junction TOCTOU limitation
with a verified directory-handle chain; the wider OS sandbox/ACL boundary
remains explicit.

CI and local coverage now use `tests.run_suite`, which records ordinary and
unraisable `ResourceWarning` channels through a forced final collection. Two
new mutants pin the single-link decision and the leak-gate failure branch,
raising the measured floor from 313 to **315/315 killed**.

## Unified LangSmith/OpenTelemetry tracing (2026-11-15)

`agent.plan`, `agent.plan_reflect`, `agent.reason`, `agent.tool` and
`agent.reflect` now emit one unified OpenTelemetry span + LangSmith run per
node. LangSmith receives native, node-specific input/output objects; OTel
receives the same payload under `agent.node.input.*` and
`agent.node.output.*`, including searchable scalar fields and bounded JSON
previews. Tool traces preserve every result in a parallel batch, while reason
and reflect traces expose the decision, current plan step, observation and
verification that produced the next state. The shared governance redactor runs
before either exporter, and oversized strings/collections are bounded.

The same contract now covers `agent.session`, `agent.llm.completion`, local
`agent.tool.<name>`, `agent.a2a_delegate.<name>` and `agent.tool_batch` instead
of nesting a payload-poor OTel span around a separate LangSmith run. OTel also
receives the LangSmith metadata and the exact enriched output object, including
`outcome` and `duration_ms`; both exporters therefore observe the same logical
operation, inputs, metadata, outputs and failure lifecycle.

`test_graph_run_guards`, `test_observability`, `test_runner`,
`test_parallel_batch` and `test_checkpoint_langsmith_guards` pin all payload
shapes, unified backend export, nested LangSmith context and failure
containment. Five new
decision mutants pin redaction plus LangSmith/OTel input/output delivery,
raising the full measured mutation gate from 308 to **313/313 killed**.
Two additional parity mutants pin OTel metadata delivery and the enriched
output identity, raising the current floor from 315 to **317/317 killed**.

## Checkpoint retention runtime contract (2026-11-12, ADR 0057)

`cleanup_old_checkpoints(max_days=30, ...)` now enforces the advertised age
window even when the SQLite file is below its size cap. LangGraph's UUIDv6
checkpoint ids provide the timestamp without decrypting sealed payloads. The
owner deletes a whole inactive thread (checkpoints and writes together), keeps
the complete chain of an active thread, and preserves any thread containing a
legacy/custom id whose age cannot be proven during age-based pruning. Two
decision mutants raise the floor from 306 to 308; the targeted checkpoint
sweep is 9/9 killed.

## Isolated mutation workers and explicit teardown ownership (2026-11-12, ADR 0056)

The mutation gate no longer rewrites the checkout. It copies only the authored
test/source/contract inputs into one temporary sandbox per worker, verifies an
owning suite is green before mutation, keeps all mutants for that suite on one
worker, and automatically reclaims only a lock whose recorded process is no
longer alive. Two workers are the CI default; `--files` selects exact paths for
targeted diagnosis without weakening the global floor.

Repeated logging composition now closes only handlers marked as owned by this
service, while foreign handlers are detached without stealing their lifetime.
ML teardown clears CUDA only when Torch is already loaded, so teardown cannot
pull in a large optional stack merely to release it, and all CUDA probe failures
remain contained. Completion tests inject their llama grammar dependency rather
than importing the installed runtime accidentally; the full gate treats an
unclosed handle as a failure instead of leaving a `ResourceWarning` at process
exit. The two new lifecycle decisions are pinned by three mutants, raising the
floor from 303 to 306.

## Web-grounded external knowledge (2026-11-08, ADR 0052)

`web_search` is a separate, injected capability rather than a RAG fallback or
product-specific rule. The planner selects it for current, external or
existence-sensitive claims; the normal chat route asks the loaded model for
the same freshness decision. Endpoint, enablement, timeout, result count and
User-Agent are deployment settings. Credential-shaped queries are refused,
PII is scrubbed, redirects are not followed, responses are capped at 1 MB and
web text enters the prompt as untrusted evidence with source URLs.

## Owned streaming shutdown and honest freshness gate (2026-11-11, ADR 0053)

Agent SSE workers are now owned by the composition root: shutdown closes
admission, cancels active sessions and performs a bounded drain before any
checkpoint, memory or task store is released. A worker that ignores
cancellation keeps those dependencies open for process teardown rather than
racing a close underneath live code.

The release evaluator now covers the real web-routing surface. Its planner
matrix is balanced across Vietnamese/English and four kinds (32 cases), and a
separate 16-case matrix invokes the exact classifier used by normal chat. CI
gates freshness accuracy and model coverage independently. The expanded
pinned-model measurement is now closed by the 2026-11-12 verified 3B run in
the row above; the historical 23/24 matrix remains historical evidence only.

## Enforced tool deadlines and complete worker ownership (2026-11-11, ADR 0054)

`ToolSpec.timeout_seconds` is no longer decorative metadata. The single tool
gateway bounds the caller, reports an indeterminate late outcome honestly,
keeps the worker owned, closes admission during shutdown and drains it before
shared dependencies can close. Timeout values must be positive integers.

Shutdown now covers the whole class: SSE, A2A task and tool workers all report
their drain result. If any owner fails to drain, checkpoint and memory stores
remain open for process teardown. Six new decision mutants are 6/6 killed in
targeted runs, and the complete sweep at ADR 0054 was **286/286 killed**, 0 survived, 0
stale.

## Runtime path parity, web-grounding hardening and shutdown bounds (2026-11-11, ADR 0055)

Operational, web-grounding and worker-lifecycle cracks identified across the
whole package are now closed:

- **One root across Docker, Qt and CLI**: `config/paths.resolve_paths()` and
  `bootstrap/server.py` align `APP_DATA_DIR` / checkout `logs_dir` and resolve
  container stores (`tasks.sqlite` and `memory.sqlite`) from `PACKAGE_ROOT`,
  making `read_only: true` deployments consistent. `ChatbotAgentHolder` guards
  `get()`/`rebuild()` under a lock, and `server_manager.find_pid_on_port`
  matches exact port tokens after `:`.
- **Coherent, fail-closed web grounding**: `AI_WEB_SEARCH_ENABLED=0` omits the
  client and `web_search` tool catalog entry and skips the blocking freshness
  LLM call in `ChatbotAgent`; Rule 14 preserves `[UNTRUSTED WEB EVIDENCE]`
  public URL citations while keeping private RAG labels internal; character-like
  queries with web hits keep URL citations and `/v1/chat/completions` surfaces
  `web_sources` in `x_meta`; `tool_contract.py` aligns `web_search` timeout at
  30 s; `HtmlWebSearchClient` requires `http(s)://` endpoints with netloc and
  raises `WebSearchChallengeError` on anti-bot challenges.
- **Pre-commit shutdown gates, bounded workers and SSE safety**:
  `TaskService.submit()`/`resume()` refuse closing services before database
  writes and discard failed threads; `ToolExecutionService.execute()` checks
  `_is_closing()` before consuming single-use approval grants and caps lingering
  worker threads at `max_concurrent_workers=32` (`tool_workers_exhausted`);
  agent SSE stream workers catch `BaseException` so `_sse_events` never stalls;
  `uvicorn.Config` sets `timeout_graceful_shutdown=10`; `_PII_PATTERNS` matches
  `_EGRESS_ASSIGNMENT`.

Fourteen new decision mutants raise `MIN_MUTANTS` from **286 to 300** (adding
`tools/factory.py` and `tools/tool_contract.py` to `PINNED_DECISION_FILES`),
with **300/300 killed** across the full suite (**2 569 tests**, 0 failures, 6
skipped; **100.00%** coverage — 10 963 statements, 2 998 branches, 0 missed, 0
partial).

The 2026-11-11 (ADR 0055) pass re-ran lint, types, layer contracts,
byte-compile, tests (`test_lockfile_sync` and `test_at_rest_cipher_contract`
included), coverage, full mutation sweep (300/300), tool/agentic policy evals,
and `verify_models.py`. The RAG framework eval (`requirements-eval.lock`), SBOM,
pip lock dry-runs/downloads, and CI pinned-3B model-in-loop rows show their last
measurement from those dedicated environments; they were not re-run inside the
local dev environment in this pass.

## 10/10 re-validation (2026-11-06): authenticated egress and honest operations (ADR 0051)

An independent review invalidated the previous absolute closure claim in four
places: inbound A2A required a bearer while outbound discovery/dispatch sent
none; `/metrics` and `/v1/models` shared the public probe router without a
guard; the RUNBOOK queried graph-node outcome labels as if they were HTTP
codes; and the docs contract counted ADRs without checking their links.

ADR 0051 closes the class rather than one example:

- the composition root injects one explicit egress credential into both A2A
  discovery and dispatch, and a real client-to-router integration test proves
  the two halves interoperate;
- only `/health`, `/livez`, and `/readyz` remain public; metrics uses the admin
  guard and model inventory uses the agent/chat guard;
- `agent_http_requests_total` records actual status codes independently of
  graph-node `success`/`error` outcomes;
- Docker paths, PowerShell commands, auth examples, PromQL, the complete ADR
  index, and the Linux/Windows CI wording now match executable behavior;
- nine new mutants raise the floor from 260 to 269 and the docs test derives
  the current suite/mutation counts from the tree.

## Final 10/10 closure (2026-11-03): measured balanced model gate (ADR 0050)

The release blocker below is now closed against the exact pinned artifact
(`Qwen2.5-3B-Instruct-Q4_K_M.gguf`, SHA-256
`9c9f56a391a3abbd5b89d0245bf6106081bcc3173119d4229235dd9d23253f94`).
The seven-case smoke set was expanded to a balanced 24-case matrix over
Vietnamese/English and all three planner kinds. The measured result is
**23/24 correct (95.83%)**, **100% model coverage**, with every four-case
slice at or above 75%; the old `en_003` miss now passes. The source gates on
the same snapshot are 2 485 tests, 100% statement/branch coverage and
260/260 mutants killed. This is the evidence that closes the repository's
declared architecture score at **10/10**; deployment-owned controls remain
explicit boundaries of the threat model, not unmeasured application claims.

## 10/10 closure review (2026-11-03): authenticated loopback and a release-grade model gate (ADR 0048)

The last review found two claims weaker than the codebase's stated bar. A
loopback peer was treated as identity when no bearer was configured, and the
real-model CI accepted 50% accuracy although the pinned baseline was only 6/7.
Both are now fail-closed:

- the Qt owner generates a fresh 256-bit token for every Python child process,
  passes it only through the child environment, and signs every privileged
  request; admin, agent, chat, A2A and MCP refuse a missing configuration even
  on loopback, and A2A no longer invents an anonymous tenant;
- MCP is not mounted without its token, while health/readiness remain the only
  deliberately unauthenticated local probes;
- model-in-loop CI requires accuracy 0.9. With seven cases this means 7/7; the
  previous 6/7 measurement is a red release gate. The planner prompt now makes
  repository storage/configuration questions explicit RAG work, closing the
  known `en_003` classification gap. CI remains the authority for the pinned
  3B measurement because that artifact is not present in this checkout.

Measured locally after the change: **2 465 tests**, 0 failures, 6 skipped;
**100.00%** coverage (10 413 statements, 2 830 branches); Ruff/mypy/import
contracts clean. The new server_manager guard mutant brings the floor to
**258/258**. Qt contract tests additionally pin token generation, child
environment propagation and the Authorization header on all request classes.

Within the repository threat model this closed the source architecture score,
while the pinned-model release evidence was still pending at that point. The
KMS/DPAPI, whole-file rollback anchor, volume encryption and third-party
database encryption items below remain explicitly owned deployment controls.
ADR 0050 and the measured 23/24 balanced run above supersede that pending
release condition.

## Independent review follow-up (2026-11-02): the three boundaries that were still fail-open (ADR 0047)

A twelfth review read the source rather than the gate table and asked the
same question the eleventh did: *is the guard fail-closed, or does it only
look closed?* Three answers were no, and all three sat in layers earlier ADRs
had already claimed (0039/0042/0043/0047's siblings):

| Defect | Why the gate was green |
|---|---|
| `ToolSpec.side_effect` was typed `SideEffect` but never checked at runtime. A plugin could register `side_effect="delete"`: the registry's set-membership test missed it, so scope and approval were never demanded, and `tool_definition_from_plugin` mapped it to `read_only` — an unknown label **widened** authority | The enum is a type hint, and no test constructed a spec outside the vocabulary; the `.get(..., "read_only")` fallback was itself covered by the tests that asserted it returned read-only for a *known* value |
| `MemoryStore._shrink_to_bytes` and `BoundedShortTermMemory._evict_locked` were global oldest-first. One tenant writing a lot (or minting session ids) evicted another tenant's rows and context while the victim was far below any size | Every test used a single owner (`"u"`, `DEFAULT_MEMORY_OWNER`), so the global loop was indistinguishable from a per-owner one |
| `UIActionService.ui_action_result` accepted the 128-bit `request_id` alone, while `ApprovalService.approve` also required `session_id`. A queued or stale ack could resume a run in another window | Both paths were tested only for their happy case; the approval's session check had a test, the ack's non-existent check could not have one |

The fixes (ADR 0047):

- **The vocabulary closes at the record.** `ToolSpec.__post_init__` refuses
  any `side_effect` that is not a `SideEffect` member, so every construction
  path is covered, and the bridge uses a direct lookup — a `.get` fallback
  would reopen exactly what was just closed.
- **Retention is spent per owner.** `prune` keeps age global (a cutoff is a
  policy, not a contended resource) but splits the byte budget evenly across
  the owners present and shrinks each inside its own share; the short-term
  buffer tags each window with its owner and evicts LRU **within** that
  owner, with `AgentService` forwarding the run's owner.
- **The ack is bound to its session.** `AgentUiActionResultRequest` carries
  `session_id` (empty is a mismatch, never a pass), the service restores the
  parked action before refusing — refusing to answer may not destroy another
  session's capability — and the Qt client echoes it like `approve`.

Measured on 2026-11-03: **2 463 tests, 0 failures, 6 skipped**; coverage
**100.00 %** (10 408 statements, 2 828 branches, 0 missed, 0 partial); six
mutants added (`MIN_MUTANTS` **249 → 255**) and all of them killed; the
per-file floor gained `domain/tools.py` and `agents/short_term_memory.py`.

## Independent review follow-up (2026-10-31): the checkpoint backend's context manager

An eleventh review read the persistence adapters and the memory loop against
the installed LangGraph API instead of the test doubles. Three defects were
live while every gate above was green:

| Defect | Why the gate was green |
|---|---|
| `PostgresSaver.from_conn_string()` / `RedisSaver.from_conn_string()` return a **context manager**, not a saver. The adapter called `.setup()` on the manager and returned it as if it were the saver | The test fakes returned a bare saver, so they encoded the wrong API shape; the real call raised `AttributeError`, the catch-all logged it and silently fell back to `MemorySaver` — a deployment that configured a durable backend got a process-local one |
| The SQLite branch called `os.makedirs(os.path.dirname(path))` unconditionally. A bare relative `AGENT_CHECKPOINT_PATH` (`checkpoints.sqlite`) has an empty dirname, so `makedirs("")` raised and the same catch-all fell back to `MemorySaver` | Every test used a path under a temp directory, which has a dirname |
| `MemoryService.recall` ranked rehydrated rows outside the fail-closed guard. A malformed `created_at` (e.g. `None`) raised out of `recall_rank` and failed the run | The malformed-row test covered the store's own `_rehydrate`; the service-level ranker was only fed well-formed records |

The fixes:

- **`CheckpointerHandle` owns the resource.** `_open_context_saver()` enters
  the factory with an `ExitStack`, calls `setup()` on the *saver*, and hands
  the stack's `close` to the handle; a failing `setup()` closes the stack
  before re-raising. The SQLite branch closes its connection on a failing
  setup and on `close()`. `close()` is idempotent because shutdown paths may
  run twice.
- **The composition root owns the handle.** `bootstrap.server.checkpointer_handle`
  is the closeable resource; `AgentService` receives only `handle.saver`.
  `bootstrap.lifecycle` closes the handle before the memory and task stores,
  containing a failing close like the other two.
- **A bare relative path is valid.** `_ensure_parent_directory()` skips
  `makedirs` when the path has no directory component.
- **Recall ranking is fail-closed.** A ranker failure is logged and returns
  `[]`, exactly like a store failure: memory never fails a run.

Seven mutants pin the class (context manager entered, failing setup released,
bare path, idempotent close, malformed record, handle closed on shutdown, a
failing close contained), so `MIN_MUTANTS` is **242 → 249**. The mutation
gate's own anchor test (`tests/unit/test_mutation_gate.py`) caught the
`saver` → `handle` rename as a stale anchor — the gate doing its job.

Documentation is now a gate too: `tests/unit/test_docs_contract.py`
re-derives the ADR count/contiguity from `Docs/adr/`, the eval group and
case/scenario counts from `evals/run_agentic_eval.py`, and the mypy file
count from `src/ai_assistant/`. README and this page had drifted (40 ADRs
listed, 25 scenarios/124 cases, 140 mypy files); they now quote the measured
46 / 12 groups / 27 scenarios / 132 cases / 151 files. The coverage, test and
mutation figures in `README.md`/`pyproject.toml` and the RUNBOOK's eval counts
had drifted the same way; the contract test now cross-checks those claims
against this page too; the ADR count they quote moved to **47** when ADR 0047
landed (the test re-derives it from the directory).

Measured on 2026-10-31: **2 457 tests, 0 failures, 6 skipped**; coverage
**100.00 %** (10 380 statements, 2 816 branches, 0 missed, 0 partial);
mutation **249/249 killed** (`MIN_MUTANTS = 249`); lint, types, layer
contracts and byte-compile unchanged (0 issues, 151 files, 9/9 contracts);
tool eval 16/16, agentic eval 132/132 across 27 scenarios.

## Agentic capabilities (2026-10-26): memory and parallel tool batches (ADR 0043/0044)

The audit that opened this round asked seven questions of the service. Five
already had an answer with a gate behind it: multi-step plans run through
LangGraph, checkpoints are durable and resumable (ADR 0023/0027), HITL is a
single-use grant (ADR 0009), LangSmith + OpenTelemetry spans exist, and the
supervisor routes to specialists / A2A peers (ADR 0005/0039). Two did not:

| Gap | What was actually missing | Why no gate caught it |
|---|---|---|
| Memory | Nothing carried a session's context between calls beyond the request's own history, and nothing learned from a completed run. The service had retrieval *over the project corpus* (RAG), not memory *about the interaction* | Every existing test sent a self-contained request; "the second turn remembers the first" was not a question any suite asked |
| Parallel sub-tasks | Concurrency existed **between** tasks (`TaskService` threads, `agent_max_concurrent_runs`, session locks) but a workflow's own steps ran strictly one per reason→tool→reflect lap | No plan ever declared two independent steps, so the sequential path was always right |

ADR 0043 adds the memory subsystem (bounded short-term buffer per session,
SQLite long-term store with sealed payloads, keyword+recency recall, writes
only for a completed run, procedural plan seed). ADR 0044 adds the parallel
tool batch: a planner-declared group of `rag_search` steps fans out inside the
tool node (bounded by `AGENT_PARALLEL_MAX`, default 4, hard cap 8), merges as
the same ordered `tool_call`/`tool_result` pairs a sequential run produces,
and falls back to the sequential path for every group that is not provably
eligible (approval-gated, non-RAG member, restored from a checkpoint with a
stale spec).

Measured on 2026-10-26:

- **39 mutants for ADR 0043 and 22 for ADR 0044**, all killed:
  `MIN_MUTANTS` 160 → 199 → 221.
- **100.00 % coverage** over 10 217 statements / 2 792 branches, including
  every new refusal arm (a group that cannot be proven eligible never runs
  in parallel), the per-branch exception barrier, the mid-batch cancel and
  the delegation cache lock.
- New tests: `test_plan_parallel_groups.py`, `test_parallel_batch.py`,
  `test_graph_wiring_delegations.py`, `test_short_term_memory.py`,
  `test_memory_service.py`, `test_memory_store.py`, `test_memory_domain.py`,
  plus the procedural-seed and directive cases in the existing suites.
- `evals/run_agentic_eval.py` gained the `memory-policy` and `parallel-policy`
  scenarios (8 cases), measured **132/132 passed, 0 skipped**.


`python evals/run_model_in_loop_eval.py` chạy tay với backend thật (fallback 3B
đã pin, SHA-256 khớp): lần đo 2026-10-19 — **accuracy 6/7 (86%), coverage 7/7,
p95 3.94s**; case trượt `en_003` nằm nguyên trong report. Không có backend thì
script in `SKIP` và exit 0 **không kèm** số accuracy, nên một lần chạy xanh
không bao giờ là con số tổng hợp giả.

## Independent review follow-up (2026-10-28): the concurrency decision ADR 0044 exposed

The 2026-10-27 review asked whether the package had reached 10/10. It had
not: ADR 0044 added real concurrency **inside** one workflow and fixed the
check-then-set in `_delegation_cache` (§6), but the same shape remained in
`rag.embeddings.LazyEncoder` — on the exact path every `rag_search` branch
takes. A warm cache deliberately leaves the encoder unloaded, so two branches
could both call the factory and load two transformers (double RAM/VRAM, one
instance leaked when the second assignment overwrites it).

Closed in this round:

- **`LazyEncoder` is locked.** The load runs under a `threading.Lock` with a
  double-check inside it; the fast path after a completed load stays
  lock-free. `tests/unit/test_embedding_loader.py` starts eight threads on a
  barrier against a cold encoder and asserts the factory ran **once**; two
  mutants (lock removed, double-check removed) are killed by that test.
- **A tampered numeric column skips its own row.** `MemoryStore._rehydrate`
  coerces `created_at`/`last_used_at`/`salience` through `_as_float`; the old
  `float()` sat outside the per-row guard, so a non-numeric column raised out
  of `candidates` and the caller lost **every** healthy row instead of the one
  bad one. One mutant pins it.
- **Shutdown closes both stores independently.** A raising
  `memory_store.close()` used to skip `platform.tasks.close()` and leave the
  task store open; each close is now contained and logged. One mutant
  restores the un-contained close and is killed by
  `tests/integration/test_server_module.py` (the task store must still be
  closed after the memory store raises).
- **`Referer` is reduced to its origin.** A browser `Referer` is a full URL,
  so the verbatim comparison against an origin-only allow-list rejected every
  legitimate same-origin request (403). `urlsplit` normalises both headers;
  an opaque `null` and a non-HTTP `file:` referer stay fail-closed. One mutant
  compares the header verbatim again and is killed by the same-origin
  `Referer` test.
- **Long-term memory is scoped to the caller, not the process.** One
  `MemoryService` serves the whole process, so `MemoryService` now requires an
  owner and every read/write takes a per-call one; `AgentService.execute`
  requires `owner`; the A2A orchestrator forwards `task.owner` (the tenant),
  the loopback routes pass `DEFAULT_MEMORY_OWNER`; record ids carry
  `sha256(owner)[:16]` (the default keeps the historical shape), and the store
  refuses a cross-owner replacement. `tests/integration/test_memory_tenant_isolation.py`
  drives the real sealed store through the real orchestrator and proves two
  tenants — and the local default — cannot recall or overwrite each other;
  twelve mutants pin the class (id namespace, service, facade, store, adapter,
  orchestrator, both routes).
- **The mutation gate has a per-file floor.** `PINNED_DECISION_FILES` names
  the files whose decision is load-bearing and requires at least one mutant
  per file, so a global count cannot hide a file losing its last one.
- **`pending_lock` is a port.** `AgentService`, `AgentRunContext` and
  `RunDeps` type it as `LockLike` instead of `Any`.

Measured on 2026-10-28: **2 433 tests, 0 failures, 6 skipped**; coverage
**100.00 %** (10 321 statements, 2 808 branches, 0 missed, 0 partial);
mutation **238/238 killed** (`MIN_MUTANTS = 238`); lint, types, layer
contracts and byte-compile unchanged (0 issues, 151 files, 9/9 contracts).

## Consolidation-on-resume follow-up (2026-10-29): the write decision has one owner

The 2026-10-28 ownership round scoped every read and write to a tenant, but it
left the *completion* decision in one path. A fresh run consolidated inside
`AgentService._execute_locked`; an approval resume, a desktop-ack resume and a
desktop workflow that finalised without a graph resume all returned
`completed` without remembering anything. The owner also stopped at the
request boundary: the parked approval/UI-ack entry carried no tenant, so a
resume that did consolidate would have written into the default namespace.

Closed in this round:

- **The completion decision has one owner.** `AgentRunContext` holds the
  injected `MemoryService` and exposes `consolidate_completed(result, …)`;
  `AgentRunContext.run` requires `owner` and calls it after *every* graph
  completion. The fresh run no longer has a private consolidation block.
- **The non-graph completion paths call the same helper.** `ApprovalService.
  _approved` (direct execution when there is no graph to resume) and
  `UIActionService._finalize` (plain-text finalisation) both consolidate the
  completed envelope; a failed desktop ack and a parked/rejected/cancelled run
  still write nothing.
- **The owner travels with the parked action.** `run_langgraph_agent` requires
  `owner`, stores it in the pending payload, and a chained desktop action
  copies it onto the next queued entry. A legacy entry without the key falls
  back to `DEFAULT_MEMORY_OWNER` instead of crashing or writing to an empty
  namespace.
- **Six mutants pin the class** (parked owner dropped, parked/failed run
  consolidated, resume owner dropped, chained owner reset, either non-graph
  fallback skipped), and the resume tests assert the owner on both the runner
  call and the stored record. The four files that carry the decision
  (`agents/context.py`, `agents/response.py`, `agents/approval_service.py`,
  `agents/ui_action_service.py`) joined `PINNED_DECISION_FILES`, so a file
  cannot quietly lose its last mutant while the global count stays above the
  floor.

Measured on 2026-10-29: **2 443 tests, 0 failures, 6 skipped**; coverage
**100.00 %** (10 337 statements, 2 808 branches, 0 missed, 0 partial);
mutation **242/242 killed** (`MIN_MUTANTS = 242`); lint, types, layer
contracts and byte-compile unchanged (0 issues, 151 files, 9/9 contracts).

Design debt closed in the same round (ADR 0045/0046):

- **The agent prompt/grammar surface is a port.** `AgentToolSpec` and
  `AgentToolCatalog` describe the record `AgentService`, `AgentRunContext`,
  `RunDeps`, the prompt builder and the dispatch guard consume;
  `tools.registry.ToolRegistry` satisfies both ports structurally. The agent
  layer no longer imports the concrete `ToolRegistry`/factory at module load —
  `agents/catalog.py` builds the built-in default lazily, so a bare facade or
  context still works without a composition root.
- **The composition root is split by responsibility.** `bootstrap/lifecycle.py`
  owns the FastAPI lifespan, `bootstrap/cli.py` the uvicorn entry point and the
  graceful-stop watcher, and `bootstrap/platform_patch.py` the Windows proactor
  shim. `bootstrap/server.py` keeps `compose()` and the process-wide state, and
  re-exports the moved names, so the documented entry points
  (`server.main`, `python src/ai_assistant/bootstrap/server.py`) and the tests
  keep working. The two mutants that pin the shutdown decisions moved with
  their code, and `cli.py`/`lifecycle.py` joined `PINNED_DECISION_FILES`.

## Follow-up on the 2026-10-22 review (ADR 0039)

A sixth review asked three narrow questions the earlier rounds had not asked:
whether a plugin tool travels the *same* catalogue and policy as a builtin,
whether the policy/alias/MCP bridges fail **closed**, and whether per-session
state is bounded. All three had a reproducible defect while every gate above
was green:

| Defect | Why the gate was green |
|---|---|
| A plugin tool was **executable but invisible**: it registered in `PluginRegistry` and the gateway could run it, yet the agent catalogue came only from `Config/agent_tools.json` and `supervisor.authorise` only knew the builtin table | No test joined the two catalogues; "the plugin registered" and "the agent can route it" were separately true |
| Policy had **three sources**: `tool_contract.py`, `supervisor.ALLOWED_TOOLS` and the alias table inside `agents/completion.py`. An `app_action_*` alias carried `read_only` policy for a `desktop_ack`/`restricted` action; an unknown policy fell back to `read_only`; an unknown classification fell back to `RESTRICTED` instead of `REGULATED` | Each source was internally consistent, so each local test passed. MCP also advertised every plugin spec without asking the MCP principal |
| `AgentService` kept every session's `RLock` for the life of the process: 10,000 finished session ids left 10,000 locks | The map looked like a cache with a correctness reason not to evict; nothing measured its size after the runs finished |

ADR 0039 closes the class:

- `domain/specialists.py` owns the vocabulary (`Specialist.PLUGIN`, tool
  classes, `TOOL_SPECIALISTS`); `supervisor.ALLOWED_TOOLS` and
  `tool_contract_for()` are projections of that one table. LSP tools are now
  advertised read-only research tools instead of registered-but-refused.
- `tools/aliases.py` is the only alias table; the execution catalogue holds
  only the canonical `application_action`.
- A loaded plugin's tools are merged into the agent catalogue under
  `Specialist.PLUGIN`; the composition root binds the live row into
  `authorise` and passes the same policy to approval/desktop-ack resume.
  Builtins are always allowed, plugin registration is scoped to the loaded
  plugin, and mutating plugin tools require scope **and** approval.
- Unknown tool names, policy labels and classification labels fail closed
  (`REGULATED` for the last); alias names cannot become execution specs. MCP
  binds only what its principal can run: it drops approval-gated,
  over-scope, over-clearance and alias tools.
- `SessionLockPool` reference-counts holders and queued threads, so an entry is
  dropped when the last user leaves; `session_id` is capped at 128 characters.
  A2A excludes the local `plugin` role from its card.

`llm/` and `internal/` are no longer read-only: ADR 0039 adds 8 mutants for the
cloud policy/size bound and the model pin/reuse/mmproj rules, and 3 for the
LangGraph checkpointer/config/`get_state` forwarding. Mutation is **112 → 134**
(`MIN_MUTANTS = 134`), and the measured run is **134/134 killed, 0 survived, 0
stale**. The full suite is **2 200 tests, 0 failures, 6 skipped**; coverage is
**100.00 %** (9 591 statements, 2 588 branches).

## Follow-up on the 2026-10-22 review (ADR 0040)

A seventh review asked one narrow question the earlier rounds had not asked:
*does every entry point that uses the model hold the drain lease ADR 0032
introduced?* Two resumes and the chat route did not, while every gate above
was green:

| Path | Why the gate was green |
|---|---|
| `AgentService.approve` and `AgentService.ui_action_result` resume the graph through `AgentRunContext.run` | Each was tested only as "delegates to the sub-service"; nothing asserted that a resume participates in the drain |
| `POST /v1/chat/completions` snapshotted `llm_module.llm` under `llm_lock` and generated outside it | `llm_lock` protects the handle *read*, not the model's lifetime; the stream test never ran concurrently with a reload |

ADR 0040 closes the class:

- `execute`, `approve` and `ui_action_result` all open
  `AgentService._model_lease()`; a resume queues behind its session lock first,
  so it does not hold the model while waiting. `cancel` stays lock-free and
  lease-free.
- The chat route holds `llm_module.model_lease()` from prompt assembly (the
  vision flag decides the prompt format) through token estimation and the
  whole stream, and generates with the backend the lease returns. A reload
  that wins the race before the lease is granted yields `None` → 503; the
  early fast-path check is no longer the guarantee.
- `LLMModuleLike` declares the lease, so the adapter contract is typed.

Four mutants pin the class — one per entry point — and `MIN_MUTANTS` is
**134 → 138**. The measured run is **138/138 killed, 0 survived, 0 stale**;
the suite is **2 205 tests, 0 failures, 6 skipped**; coverage is
**100.00 %** (9 594 statements, 2 590 branches).

## Review follow-up (2026-10-23): four drifts around a green gate table

An eighth review re-ran every gate against the source and found four drifts
*around* the gate table — none in the core layering:

| Drift | Why no gate saw it |
|---|---|
| `README.md` still called this scorecard "local, not tracked" while the same README (line 592) says it is tracked and `git ls-files` confirms it | Documentation is not a gate |
| `AGENT_MAX_CONCURRENT_RUNS` was enforced by `AgentService.execute` only; `approve` and `ui_action_result` resume the graph without a slot | Each entry point was tested for its own delegation; nothing asserted the budget on a resume — the same shape ADR 0040 had just closed for the model lease |
| Two architecture tests owned the LangGraph isolation rule with opposite allow-lists (`orchestration/graph.py` allowed by one, forbidden by the other) | Both passed: `graph.py` reaches the engine through the port today, so the disagreement was latent |
| The AST twin of the application-layer contract allowed `plugins`, while the import-linter contract forbids it | No application module imports the plugin package, so the weaker twin never fired |

The fixes and their pins:

- The budget is a property of *running the graph*: `execute`, `approve` and
  `ui_action_result` all take one slot through `_acquire_run_slot()` /
  `_release_run_slot()`. A2A fresh runs already went through `execute`; A2A
  resumes now share the same cap. Three mutants pin the decision — one per
  resume site plus the shared release — so `MIN_MUTANTS` is **138 → 141**.
- `test_langgraph_framework_stays_isolated` is the single owner of the
  framework rule; the duplicate was deleted and the surviving docstring names
  the real adapter (`internal/langgraph_engine.py`) instead of a path that
  never existed.
- `APPLICATION_ALLOWED` mirrors the import-linter contract exactly
  (`application`, `domain`, `ports`), and its docstring says why.
- `README.md` and `RUNBOOK.md` describe the budget scope and the scorecard's
  tracked status truthfully.

Measured 2026-10-23: **141/141 killed, 0 survived, 0 stale**; suite
**2 206 tests, 0 failures, 6 skipped**; coverage **100.00 %** (9 605 statements,
2 592 branches). The suite count moved 2 205 → 2 206: the two resume-budget
tests were added and the duplicate architecture test was removed.

## Independent review follow-up (2026-10-23, ADR 0041)

A ninth review asked two questions the gate table could not answer: *when
`/admin/release-vram` returns `ok`, are the weights actually gone?* and *can a
burst create more worker threads than the process can afford?* Both had a
reproducible defect while every gate above was green:

| Defect | Why the gate was green |
|---|---|
| The backend had **two owners**: `LlmModule.llm` and a module-level `_global_backend` in `llm/model_loader.py`. `release-vram` dropped only the module reference, so the global kept the model object alive, `release_ml_memory()` reclaimed nothing, and `/health` reported `llm_loaded: false` while `/v1/models` still advertised the released model | The route test asserted the route's writes (`llm is None`, HTTP 200), not the property that matters: the object can be collected. Coverage measured lines, not ownership |
| `TaskService` spawned one thread per A2A task with no admission bound. `AGENT_MAX_CONCURRENT_RUNS` is enforced later in `AgentService` (503), after the thread and its native memory already exist; the rate limit defaults off | No test submitted more concurrent tasks than the executor limit, and there was no single "decision" to mutate — the bound was *missing*, not wrong |

ADR 0041 closes both classes:

- `model_loader` is a **pure factory** (`load_backend`); the process-wide
  backend, `get_backend`, `load_model` and `reload_model` are gone.
  `LlmModule` is the only owner and `release()` clears the reference and its
  metadata (`active_model_desc`, `is_vision_model`) in one critical section
  before asking the runtime to free memory. A `weakref` test proves the object
  is collectable after release; `/health` and `/v1/models` derive the
  advertised model from the backend actually held.
- `TaskService` takes `max_concurrent_tasks` (config
  `AGENT_MAX_CONCURRENT_TASKS`, `[runtime] max_concurrent_tasks`, default 8)
  and reserves a slot **before** the task row is committed and the thread is
  created. A refusal is a durable `REJECTED` task with its event on submit,
  `TaskCapacityError` (a `ServiceUnavailableError`) on resume, HTTP 429 with
  `Retry-After` on the A2A routes and JSON-RPC error `-32000`. The worker
  returns its slot in `finally`, including when `_start` refuses during
  shutdown.
- Nine mutants pin the class (two ownership/release, two health truths, three
  admission paths, two backpressure mappings), so `MIN_MUTANTS` is
  **141 → 150**. A review that trusts neither the scorecard nor the gates
  independently re-ran the suite and measured **150/150 killed, 0 survived,
  0 stale**; **2 209 tests, 0 failures, 6 skipped**; coverage **100.00 %**
  (9 642 statements, 2 600 branches). The count moved 2 210 → 2 209 because
  `LlmModule.reload_model()` and its test were removed: release +
  `load_model()` is now the single reload path, so there is no second
  implementation to test. README, this table and the `pyproject.toml`
  comment now quote the same run.

## Independent review follow-up (2026-10-24, ADR 0042)

A tenth review read the source instead of this page and asked three questions
a green gate table cannot answer: *who enforces the invariant*, *what happens
when configuration is missing*, and *does shutdown run cleanup*. All three had
a live defect while every gate above was green.

| Defect | Why the gate was green |
|---|---|
| `LlmModule.release()` / `load_model()` were still public. ADR 0041 made the drain contract correct, but any new caller could call the primitive directly and swap the model under a live lease | The route tests exercised `model_reload()`, and the only direct callers (startup, teardown) were not leases at that moment — so the bypass had no failing case |
| Three fail-open defaults: `scopes_for_capability(None)` returned `project.write`/`project.execute`; a tool without `policy` became read-only; a missing `Config/agent_tools.json` produced an empty registry | Each fallback was a *deliberate* line, covered by a test that asserted the fallback itself. Coverage measures execution, not whether the default is safe |
| `/admin/shutdown` called `os._exit(0)`, skipping the lifespan `finally` (task-store close, checkpoints, flush) | The test pinned the exit code with a mock, so the missing cleanup was invisible — there was no observable post-exit state |
| The HTTP adapters read `_reranker` / `_release_ml_memory`; `LLMModuleLike` re-declared the lease as `Any`; `ToolSpec.plugin_id` defaulted to a second `"builtin"` spelling | Names matched at runtime; only a type-level/one-source check would notice, and none was asserted |
| The composition-root docstring claimed "no module-level mutable state" while `_COMPOSED` is a process-wide singleton | Documentation is not a gate. The statement was simply false |

ADR 0042 closes the class:

- `_require_reload_window()` makes the owner, not the caller, enforce the
  drain: `release()` and `load_model()` raise outside `model_reload()`.
  `bootstrap_runtime()` takes the same window at startup, and the bootstrap
  test's fake loader calls the same precondition, so forgetting it fails.
- `scopes_for_capability(None)` is `READ_ONLY_SCOPES`; `tool_spec_from()`
  raises on a missing policy; `create_tool_registry()` raises
  `ConfigurationError` on a missing catalogue. A deployment either declares
  its authority or refuses to start.
- The RAG adapter surface is public (`reranker_active`, `release_ml_memory()`),
  and `/health` + chat status read it through the protocol.
- `/admin/shutdown` sets `RuntimeModules.shutdown_requested`; the server's
  `_stop_server_when_requested` watcher sets `uvicorn.Server.should_exit`, so
  lifespan cleanup runs. `os` is gone from the router.
- One lease type: `LLMRuntime.model_lease()` yields
  `AbstractContextManager[LLMBackend | None]`, and `LLMModuleLike` inherits it.
  `ToolSpec.plugin_id` defaults to `BUILTIN_PLUGIN_ID`.
- The composition root documents its real scope: one process-wide graph; two
  servers in one process are unsupported, not silently allowed.

Ten mutants pin the class (reload window + startup caller, scope map, missing
policy, missing catalogue, shutdown event/watcher/route, public reranker
property + health read), so `MIN_MUTANTS` is **150 → 160**. Measured again:
**160/160 killed, 0 survived, 0 stale**; **2 214 tests, 0 failures, 6
skipped**; coverage **100.00 %** (9 662 statements, 2 604 branches).

Tracked constraint: the broad `except Exception` handlers are unchanged in
kind, though their count moved from 72 to 77 (ADR 0043/0044 added three net,
and the 2026-10-28 shutdown fix added the two contained store closes). They
sit at boundaries (tool execution, HTTP translation, optional SDKs, background
workers) where a failure has to become a structured result or a log line
rather than a process abort. Narrowing them is behavioural work, not a
security gate — every security decision has its own guard above.

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

## Follow-up on the 2026-10-21 review (ADR 0038)

A fifth architectural review ran against the layers **nobody had examined
closely** — `rag/`, `llm/`, `internal/`, `config/`, `observability/` — with
the same lens. It found three defects of the class ADR 0037 had just named,
in the one place a second reader of the filesystem hides: the **RAG corpus
builder**.

| Defect | Why the gate was green |
|---|---|
| The corpus builder followed a **link out of the tree**: a junction (creatable on Windows without any privilege) inside the project was indexed as if it were part of it — a private document, or a `secrets.py` with an API key, reached the model and the sealed cache | The *agent* read gate refuses exactly this (ADR 0033) — but the corpus builder is a **second reader** with its own policy: no link rule, no secret-name rule, and a directory list that **omitted** two entries the shared kernel refuses |
| Every image under `Docs/` was indexed **twice** (the two extension lists overlapped and the docs directories are inside the project root) — the corpus and the sealed cache held it twice, and a retrieval could return it twice | Nothing asserted the corpus was a *set*; the duplicated chunk was a valid chunk |
| The cache hash and the scan were **two different walks**: the hash covered every file under the docs directories, the scan only the document types — so a stray file invalidated the cache for nothing | The hash was only ever compared to itself |

ADR 0038 closes the class: the corpus builder now uses the **shared kernel**
(`path_policy`: the blocked directories, the secret names, the real-target
rule), every file is walked **once** by the walk that owns it, and the cache
hash is computed over **exactly** the indexed set. Mutant 105 → 110.

It also found a defect in **the mutation gate itself**. Two mutants were
reported as "survived" that a manual check killed: Python validates a `.pyc`
by the source's mtime **in whole seconds** and its size, so a mutation that
keeps the size (`"1"` → `"0"`) and is restored within the same second left a
bytecode cache still holding the **mutated** code — the next import loaded the
mutant, a test that should have failed passed, and the tree kept mutated code
after the run. The gate now drops the mutated module's bytecode cache before
and after the run and executes the suite with `PYTHONDONTWRITEBYTECODE=1`, so
what is measured is the mutated *source* and nothing mutated is left behind.
Two new cases pin that.

### The sixth review: the layers just changed

Five review rounds in a row found defects in the layer that had *just* been
fixed, so the sixth round re-read the three things that had just changed (the
corpus boundary, the command grammar, the row-bound at-rest state). It found
**two defects in the reviewer's own code**:

| Defect | Fix |
|---|---|
| The corpus walk still **descended into** a junction pointing outside the tree and refused the files one by one — so a junction at a volume root made the walk traverse the whole disk | The same "the real target stays inside" rule is now applied at the **directory** level (do not descend) as well as the file level (do not index) |
| The operator's "seal the plaintext state" action could **break** an already-sealed state: it wrote the new `user_version` over content it had not re-sealed, so the file claimed the new format and could no longer be read | The action now **refuses** a state that is already sealed and names the row-binding migration. No path writes a version it did not actually write |

Mutant 110 → 111.

### The seventh review: the tools' *option* surface

Six rounds in a row found defects in the layer that had *just* been fixed, so
the seventh round read the **options** of the allow-listed binaries — the
surface no round had touched. It found one **P1** of the same class as the LSP
read gate (ADR 0035) and the corpus builder (ADR 0038):

```
$ ruff check ../outside
I001 [*] Import block is un-sorted or un-formatted
 --> outside\secret.py:1:1
  |
1 | import os
2 | TOKEN = "leaked-secret-value"
```

`ruff check <dir>` **prints the source of every file it finds** — a read
primitive outside the workspace, behind a command approval rather than a file
approval. The same held for the options that read or write a path the sandbox
cannot confine, and they hide *behind the verb* (`ruff check --config <file>`,
`pytest --junitxml=<file>`, `git diff --no-index <a> <b>`, `git commit
--template=<file>`).

The fix: the path positionals of `python`/`pytest`/**`ruff`** must stay inside
the workspace, and a per-binary table of *dangerous options* is refused
**everywhere** on the line — `pytest` is no longer a special case, it is the
same table and the same rule. Mutant 111 → 112.

### Remaining tracked gaps (ADR 0038)

- **The corpus is still a copy of the user's documents.** ADR 0028 seals the
  *copy*; encrypting the document tree itself is the user's and the
  operator's business (volume encryption).

The other ADR 0038 gap — `llm/` and `internal/` were read but not measured — is
closed by ADR 0039 (8 + 3 mutants, above).

## Follow-up on the 2026-10-21 review (ADR 0037)

A fourth architectural review ran against the source, not this page, and
re-ran every gate before reading the code. **Every gate was green** — and the
review found **four** defects of one class: *a rule written as a list, and the
list has a hole somewhere else*.

| Defect | Why the gate was green |
|---|---|
| The sandbox accepted **any** token starting with `-` as "just a flag", so the verb was never checked: `git -C <dir> clean -fdx`, `git -ccore.sshCommand=<cmd>`, `pytest -p <mod>`, `pytest -o addopts=-p <mod>`, `ctest --build-and-test --test-command <cmd>`, and `git clone <url> <path outside the workspace>` all passed the allow-list | The gate looked at the **first** token; a leading option hid the verb, and the tests only ever used a verb-first command |
| The write deny-list was an **enumeration** of policy trees, so `agents/`, `adapters/persistence/`, `config/`, `settings.py`, `path_policy.py` and `exec_policy.py` — the rules the agent is judged by — were still writable | Every earlier revision of the list named the files it knew about; a list that is extended by hand goes stale |
| The classification gate compared `tool_max >= principal` — the **wrong way round**: the least trusted caller was let through and the most trusted was refused, and every tool carried the same ceiling, so the gate compared against a constant | The test **asserted** the inverted behaviour, so the mutation gate confirmed the code as written, not as intended |
| The at-rest state was bound to the **field** (`task.message`), not the **row**: a ciphertext copied to another task opened and was served as that task's content, and `owner`/`classification`/`status` were plain text — so a write-access attacker could retarget a task to another tenant, downgrade its classification or mark it completed | The sealed-regime test only replaced a column with *plaintext*; it never moved a value between rows nor edited an identity column |

ADR 0037 closes the **class**, not the instances:

- **The command surface is a grammar, not a filter.** The verb is the first
  positional token; only declared leading options may precede it; the options
  that can execute or import (`-p`, `--pyargs`, `-o`, `--override-ini`,
  `--build-and-test`, …) are refused anywhere on the line; a `git clone`/`init`
  destination is confined to the workspace. The isolation gate now defaults to
  **on**: an allow-list is not an OS boundary, so the local runtime is a
  decision an operator has to write down.
- **The write deny-list is a rule.** The platform's own source is not
  writable — the whole package, not a list of trees — and a new module cannot
  be added to the platform without being denied with it.
- **Clearance is derived from the granted scopes**, and a tool's data ceiling
  comes from the tool's own contract. The comparison is
  `clearance(principal) >= ceiling(tool)`; an unknown label is the *most*
  protective value, never the least.
- **The at-rest state is bound to its row.** The associated data carries the
  row id, and each row carries a sealed identity bundle beside its plain-text
  columns — the store compares them and refuses a row that disagrees. A state
  file that predates this is **refused**, and the operator re-binds it once
  with `scripts/migrate_state_row_binding.py` (never implicitly: a legacy file
  and a substituted file are the same bytes).
- **One decision, one place.** The tool success rule lives in the domain
  (`tool_result_succeeded`): the application layer used to re-spell half of it
  and drop `return_code`, so a failed command was reported as a success. The
  duck-typed store fallback is detected by asking the **signature** instead of
  catching the `TypeError` the call raises — which also swallowed a `TypeError`
  raised *inside* the store and then called it a second time. The attachment
  gate opens only a **regular file** (a FIFO would block the worker thread on
  a path the request body chose), and its "sealed envelope" branch is
  documented as a *hint* rather than a proof of provenance — the AEAD in the
  unseal step is what refuses a planted file.

### Remaining tracked gaps (ADR 0037)

- **`cmake -P`, `ctest --build-and-test` and `pytest` execute code by
  design.** No allow-list makes them harmless; the container is their real
  boundary — which is why the isolation gate is on by default, and why the
  old claim ("an allow-list is an OS boundary") is written down as wrong.
- **The platform's own tests are still writable.** An approved write can
  change a test so that it stops pinning a guard. The trust anchor for that
  layer is **CI**, which re-runs the gates on the committed tree.
- **Whole-file rollback**: an attacker with write access can still restore an
  older sealed file of their own. Row binding does not stop a rollback; that
  needs an anchor outside the file.
- **The upgrade step needs an operator**: a state that predates the row
  binding does not start until
  `python AIAssistant/scripts/migrate_state_row_binding.py` has been run once.
  The desktop client should run it automatically when it sees that message —
  a C++-side change and outside this package.
- **Plugin provenance is now closed.** `PluginRegistry.registration_scope`
  binds every registration to the identity instantiated by the loader; a
  spec that claims another enabled plugin id is refused. The owning regression
  test and mutant make this a closed control rather than a documented gap.


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

### ADR 0035 follow-up status

- **The two LSP gaps are closed by ADR 0039.** `go_to_definition` and
  `find_references` are read-only research tools in the canonical
  `TOOL_SPECIALISTS` table; the supervisor projection therefore authorises
  them. `lsp_client` resolves executables through the shared fail-closed
  resolver before either tool can run. ADR 0013 §6 now records both the
  historical refusal and this later activation instead of claiming the old
  state is current.
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

### The last control that was still mock-only — now closed (ADR 0036)

The **sealed-regime read** of the task store was exercised for real, but the
*signal* that says "this store has migrated" was `PRAGMA user_version` — which
lives **inside the file** an attacker with write access can also edit. A
forged version made the store re-run its one-time migration, and the
migration sealed the substituted rows. So the "leave it plaintext" downgrade
was closed (ADR 0034) while the "forge the version" one was not.

**Both are closed now.** ADR 0036 removes the implicit migration entirely: a
store configured with a key **never** accepts content it cannot authenticate —
neither the task store (it refuses to open) nor the pending store (it refuses
to read). The one-time migration became an explicit operator action,
`scripts/seal_plaintext_state.py`, which seals the state under the *same* key
the application opens it with, in one transaction each, and refuses to act
without a configured key.

The reason the application cannot make this decision itself is the point: **a
legacy file and a substituted file are the same bytes**. Nothing inside the
file can tell them apart, so only the operator can — by saying so explicitly.

The cost is a real upgrade step (a state that predates encryption must be
sealed once, by hand), and it is written down rather than hidden. The gain is
that an attacker with write access to the state file now gets **nothing**: they
can still delete it, but they cannot make the application serve their payload.

### Remaining tracked gaps (ADR 0036)

- **The upgrade step needs an operator.** A deployment whose local state
  predates at-rest encryption will not start until
  `python AIAssistant/scripts/seal_plaintext_state.py` has been run once. The
  desktop client should run it automatically when it sees that exact message —
  that is a C++-side change and outside this package.
- **A short secret in prose has no shape to recognise.** The value rule needs
  a long enough value; a four-character password in a sentence is invisible
  to any pattern matcher. Written down rather than hidden.
- **An attacker with write access to the state file can still delete it.** That
  is availability, not integrity — and it is the limit of every at-rest design
  without an external trust anchor, as ADR 0027 recorded.


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
  `A2A_ENABLED`/`A2A_REMOTE_AGENTS`/card settings at module scope. ADR 0059
  later removed the unused optional-SDK probe and enable switch entirely;
  only the discovery target list is resolved on first use, so importing the
  adapter has no environment-dependent side effects.
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

- **Unattended rotation and KMS integration.** The at-rest key can be
  rotated now — `scripts/rekey_storage.py` re-encrypts the task store in one
  transaction, re-seals the pending payload atomically, deletes the derived
  RAG cache, and refuses to write anything unless the old key actually opens
  the state (ADR 0030). Windows desktop custody now uses DPAPI (ADR 0061).
  What remains is a generic KMS integration and rotation without an operator
  in the loop.
- **The source documents are still the user's own files.** ADR 0028 seals the
  *copy* the assistant creates; encrypting the original document tree is the
  user's and the operator's business (volume encryption).
- **Volume-level encryption stays an operator concern.** Field-level encryption
  covers the listed content columns, the cache artifacts and the chat
  attachments; BitLocker/LUKS, volume ACLs and backup encryption remain
  deployment responsibilities.
- **The POSIX default key file relies on the account's filesystem ACL**
  (owner-only mode 0600). Windows uses user-scope DPAPI. A POSIX deployment
  that needs stronger custody sets `AI_STORAGE_KEY_FILE` to a KMS-mounted
  secret instead.
- **The Postgres and Redis checkpoint backends** keep their published
  constructor signature (`from_conn_string` takes no `serde`), so their
  state is protected by the database's own encryption rather than by the
  platform key. The SQLite backend — the one that lands in the app's own
  data directory — is sealed (ADR 0032).
## Numbered signals

The `SCORECARD #N` labels below are the historical identifiers used in code
comments (`observability/metrics.py`, `application/coordination.py`,
`evals/run_model_in_loop_eval.py`) and in `README.md`; they resolve to
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

- Every row is a measurement with the command that reproduces it. Linux runs
  the complete lint/compile/test/coverage/type/import/mutation gate. Windows,
  the shipping platform, independently runs the full suite with coverage,
  import contracts, mypy and the platform-sensitive security subset. The
  model eval workflow currently runs on Linux; rows do not claim a Windows
  rerun where the workflow does not provide one (ADR 0032/0051).
- Coverage and mutation are *outputs* of the suite, not targets set by hand:
  `pyproject.toml` owns the configuration (`branch = true`, `fail_under = 100`,
  `exclude_also`), so a local run and CI measure the same thing.
- The mutation **count** is a gate too (`MIN_MUTANTS`): deleting a mutant is
  a decision, and the gate turns it into a visible one (ADR 0032).
- This page is **tracked**, so the numbers are readable by anyone who clones
  the repository — a scorecard nobody else can see pins nothing (ADR 0032).
- The evaluation scripts exit non-zero on a failing **or not-run** case, so a
  green eval cannot come from an unchecked path.
