# ADR 0032 — Closing the last mile: the hardening the scorecard claimed

- Status: Accepted
- Date: 2026-10-21
- Supersedes: none

## Context

An independent architectural review ran against the source, not against
`Docs/SCORECARD.md`. It re-ran every gate and then read the code behind them.

The gates were real: lint, types, the nine import contracts, 1 975 tests,
100.00 % statement *and* branch coverage, 67/67 mutants killed. The
architecture was sound in its core — the domain was load-bearing, the tool
gateway was a genuine single chokepoint, the compare-and-set was a real
TOCTOU fix, the at-rest AEAD was correct.

But the review found that the *evidence* and the *hardening* had gaps that
the scorecard's own wording did not cover. Nine of them, in three groups.

**Group A — the evidence was not in the repository.**

`Docs/SCORECARD.md` — the file that holds "100.00 %", "1 975 tests", "67/67"
— was listed in `.gitignore` (`/AIAssistant/Docs/*`). So were nine ADRs
(0026–0031). A reviewer, another developer or CI could not see any of it.
ADR 0015 says *"a claim must be pinned or removed"*; a scorecard nobody else
can read pins nothing.

The mutation gate also had no floor: deleting a mutant was indistinguishable
from passing it. The suite stayed green, the coverage floor stayed at 100 %,
and the guard silently stopped being guarded.

**Group B — the hardening covered part of the surface, not all of it.**

| Finding | Why it mattered |
|---|---|
| `/docs`, `/redoc`, `/openapi.json` are *app-level* FastAPI routes | The transport guard protects the routers; it never reaches them. A peer that could reach the port received the full route schema. |
| The Agent Card advertised `request.base_url` | That is derived from the `Host` header, so `Host: evil.example` made this server publish a card pointing at the attacker. |
| Tool results were logged verbatim | The file handler is at DEBUG, so the document a tool read landed in a plaintext log — the same content the at-rest cipher protects. The user prompt and the raw model output were logged too, on one path but not the other. |
| The RAG reset had two owners with two different field sets | One cleared the pipeline but not the encoder; the other cleared the encoder but not the pipeline. `/health` could report zero chunks while retrieval still served the previous corpus. |
| Task state and its event log were two transactions | A crash between them left a task `completed` with no terminal event, and an SSE subscriber waited for an event that would never arrive. The adapter already did it in one transaction — the port could not *ask* for it. |
| The LangGraph checkpoint store was plaintext | It holds the system prompt, the user's messages and the tool results. `.env.example` claimed `Cache/` was sealed. |
| The write deny-list named two policy files | The transport policy, the A2A scope map, the tool catalogue and the approval logic were all writable. |
| The at-rest interop vector was pasted by hand on both sides | Re-pinning it on one side left two different vectors, each still correct on its own — and the break surfaced as "the assistant cannot see the image". |

**Group C — the delivery did not measure what the scorecard claimed.**

CI ran on Linux; the scorecard was measured on Windows, the platform the
product ships on. Two of the sandbox's security guards are `skipUnless(win32)`
and therefore **never executed in CI**. No action was pinned to a commit SHA.
`ruff` was installed unpinned and was in neither lock. The Docker image
installed the *ranges* rather than the hash-pinned lock, pulled a CUDA wheel
into a CPU-only image, and probed `/readyz` for a container that ships no
model weights — so it was permanently unhealthy.

## Decision

The scorecard's rule is applied to the scorecard itself, and the gaps in the
hardening are closed.

### 1. The evidence is tracked and the counts are gates

* `Docs/SCORECARD.md` is un-ignored and tracked; ADRs 0026–0031 are
  committed.
* `scripts/mutation_check.py` declares `MIN_MUTANTS` and fails when the
  declared count drops below it. Deleting a mutant is now a visible,
  failing decision instead of a silent one.

### 2. The hardening covers the whole surface

* **The OpenAPI surface is opt-in.** `create_app` passes
  `docs_url`/`redoc_url`/`openapi_url` only when the deployment asked for
  them (`AGENT_EXPOSE_API_DOCS` / `[api] expose_docs`). A
  `TrustedHostMiddleware` is installed when the deployment declares a host
  allow-list.
* **The Agent Card advertises the configured URL.** `build_a2a_router` takes
  a `card_base_url` that the composition root injects from the settings; the
  request's `Host` header is never consulted.
* **The log states what happened, not what was read.** A new domain helper
  `describe_tool_result` summarises a result by shape and size. The user
  prompt and the model output go through `scrub_pii` on *both* the local and
  the remote leg.
* **The RAG reset has one owner.** `RagModule.reset_index_state()` is
  public and clears every retrieval handle under the lock; the HTTP admin
  route calls it instead of reaching into the handles.
* **Task state and its events are one unit of work.** `ports.TaskStore` now
  declares a single `commit(task, *, expected, events)`; the SQLite adapter
  implements it in one transaction. The duck-typed read-then-write fallback
  — the very race the compare-and-set exists to close — is gone.
* **The checkpoint store is sealed.** The SQLite checkpointer writes through
  LangGraph's `EncryptedSerializer`, backed by the platform's at-rest key, so
  one key protects the task store, the pending actions, the RAG cache, the
  checkpoints and the chat attachments.
* **The policy trees are not writable.** The deny-list covers the
  directories that hold the platform's own policy, matched on the *resolved*
  path so a symlink cannot walk around it.
* **The at-rest contract is checked from one side.** The format constants
  are read out of `src/core/AtRestCipher.cpp` and the reference vector out of
  `tests/TestAtRestCipher.cpp`, and compared with the Python ones. A change
  on either side fails the Python CI, which now triggers on the C++ files as
  well — so no cross-language trigger is needed.

### 3. The rules live in the layer that owns them

Six rules were implemented in a layer *above* the model they constrain, so
two layers each believed they owned them. They now live in `domain`:

* the classification lattice (`domain/security.py`);
* the capability→scope lattice (`domain/security.py`);
* the terminal task states (`domain/tasks.py`);
* the identity of a tool invocation (`domain/approvals.py`) — **one** owner
  with two named rules, the binding identity an approval grant is bound to
  and the retry identity the duplicate-call guard uses. The two used to
  disagree about the volatile per-retry keys, so a retry that only changed
  its `request_id` slipped past the guard.

Two ports were completed in the same pass, because both were *declared but
bypassed*:

* **`ports.LLMRuntime.model_lease`** — the drain contract (a run pins one
  backend for its whole length, so an admin reload cannot swap the model
  mid-run) was duck-typed with a `getattr` probe. A runtime that forgot the
  method silently lost the guarantee: no type error, no log. The port now
  states it, and the probe is gone.
* **`ports.ToolCatalog`** — the application layer depended on the *concrete*
  `plugins.registry.PluginRegistry`, a peer layer's class, through an
  import-linter exception. The catalogue is a port now; the HTTP adapter's
  own duck-typed duplicate of it is gone, and the import contract plus an
  AST guard forbid the concrete import from coming back.

### 4. Delivery measures what is claimed

* Every action in the Python workflows is pinned to a commit SHA; `ruff`
  comes from the hash-pinned lock.
* A **Windows job** runs the suite, the coverage gate, the import contracts
  and the type check on the platform the product ships on — so the two
  Windows-only security guards actually execute.
* The container installs its own hash-pinned, CPU-only lock
  (`requirements-docker.lock`), the base image is pinned by digest, the
  health probe checks liveness (the image ships no weights, so readiness
  would never pass), and a failed `useradd` fails the build instead of
  producing a green image that dies at `USER appuser`.

## Consequences

* The scorecard's numbers are readable by anyone, and the counts are gates.
* The hardening has no remaining "app-level route", "private member of the
  implementation" or "second owner of the same transition" in the paths the
  review walked.
* The at-rest story is now one key over one set of artifacts, and the
  plaintext channels that undercut it (the log, the checkpoint store) are
  closed.
* The cross-language contract is checked from the Python side, so the C++
  workflow no longer has to be triggered by a Python change.
* CI runs on the platform the product ships on.

### Remaining tracked gaps

These are decisions the repository deliberately does not make, with the reason
written down — not gate failures:

* **Unattended key rotation and KMS/DPAPI.** The key can be rotated
  (`scripts/rekey_storage.py`); calling a KMS directly and rotating without
  an operator in the loop is not implemented.
* **The source documents are the user's own files.** ADR 0028 seals the
  *copy* the assistant creates; encrypting the original tree is the user's
  and the operator's business.
* **Volume-level encryption** (BitLocker/LUKS, volume ACLs, backup
  encryption) stays an operator concern.
* **The key file relies on the account's filesystem ACL.** A deployment that
  needs stronger custody sets `AI_STORAGE_KEY_FILE` to a KMS-mounted secret.
* **The Postgres and Redis checkpoint backends** keep their published
  constructor signature, so their state is protected by the database's own
  encryption rather than by the platform key.
* **The two third-party C++ actions** (`jurplel/install-qt-action`,
  `lukka/run-vcpkg`) are still referenced by tag in the C++ workflow. They
  are outside this package's scope and were not pinned in this change.
