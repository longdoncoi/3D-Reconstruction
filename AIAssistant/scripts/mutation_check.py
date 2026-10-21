"""Mutation check for the load-bearing guards (ADR 0015 / 0011 / 0018 / 0020).

100% statement/branch coverage says the guards are *executed*, not that they
are *correct*: a test that asserts the wrong direction still covers the line.
This script inverts one decision at a time in each guard and runs the tests
that own it. A mutant is KILLED when the suite fails (the guard's behaviour is
pinned) and SURVIVES when it stays green (the test would not notice the guard
being disabled).

Each mutant is written into the real source file and restored in a ``finally``
that re-reads the file and verifies it came back byte-identical. A lock file
keeps two runs from overlapping, and its presence after a crash refuses to start
again until the source is restored by hand. Run it *alone* — never in the same
step as the suite, which reads exactly these files; CI runs it after the tests
in ``Build-Python-ci.yml``.

An anchor that no longer matches the source counts as STALE and fails the run:
the check may not go green by mutating code that is not there.

Usage (from AIAssistant/):  PYTHONPATH=src python scripts/mutation_check.py
Exit code 0 = every mutant killed and every anchor present; 1 otherwise.
"""
from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Mutant:
    """One deliberate logic inversion in one guard, with the tests that own it."""

    file: str
    original: str
    mutated: str
    tests: tuple[str, ...]
    label: str
    # Per-mutant budget (seconds) for the owning suite. The default fits every
    # suite that terminates; a mutant may only shorten it, never extend it.
    timeout_seconds: int = 600
    # Opt-in, one mutant only (see the recompose guard below): when breaking
    # the guard turns the owning suite from "failing" into "never finishing",
    # a timeout IS the detection — but only with a budget far above the
    # unmutated runtime, so a slow machine can never fake a kill.
    kill_on_timeout: bool = False


#: The number of guards this gate is contractually required to cover.
#:
#: Without a floor, *deleting* a mutant is indistinguishable from *passing*
#: it: the suite stays green, the coverage floor stays at 100 %, and the
#: guard silently stops being guarded. ADR 0032 makes the count a gate — a
#: new load-bearing decision adds a mutant in the same commit, and a removed
#: one lowers this number in the same commit, so the diff always shows it.
#:
#: Measured 2026-10-20: 67 mutants, 67 killed, 0 survived, 0 stale.
#: ADR 0032 adds seven more (the policy-tree write deny and its prefix match,
#: the OpenAPI surface, the log redaction, the compare-and-set, the
#: checkpoint seal, the adapter's CAS guard and the volatile-key rule).
#: ADR 0033 adds one more (the write target is resolved, so a symlink into a
#: policy tree is matched on the file it points at).
#: ADR 0034 adds five more (the secret-file read guard, the PATH scan that
#: never consults the current directory, the case-insensitive write deny,
#: the sealed-regime read refusal and the value-shaped egress check).
MIN_MUTANTS = 80

MUTANTS: tuple[Mutant, ...] = (
    # ── Sandbox allow-list (ADR 0011) ────────────────────────────────────────
    Mutant(
        "src/ai_assistant/tools/sandbox.py",
        "if executable.casefold() not in _ALLOWED:",
        "if False:",
        ("tests.unit.test_sandbox",),
        "executable allow-list removed (any binary runs)",
    ),
    Mutant(
        "src/ai_assistant/tools/sandbox.py",
        'if "/" in executable or "\\\\" in executable:',
        "if False:",
        ("tests.unit.test_sandbox",),
        "path-shaped executable accepted (/opt/attacker/git runs as git)",
    ),
    Mutant(
        "src/ai_assistant/tools/sandbox.py",
        'return "Sandbox: dangerous cmake -E subcommand blocked."',
        "return None",
        ("tests.unit.test_sandbox",),
        "cmake -E <arbitrary cmd> allowed",
    ),
    Mutant(
        "src/ai_assistant/tools/sandbox.py",
        'return "Sandbox: git -c config injection blocked."',
        "return None",
        ("tests.unit.test_sandbox",),
        "git -c core.sshCommand injection allowed",
    ),
    Mutant(
        "src/ai_assistant/tools/sandbox.py",
        "return f\"Sandbox: subcommand '{subcmd}' is not in the {executable} allow-list.\"",
        "return None",
        ("tests.unit.test_sandbox",),
        "unknown subcommand allowed",
    ),
    Mutant(
        "src/ai_assistant/tools/sandbox.py",
        "if head in _GIT_ESCAPE_FLAGS:",
        "if False:",
        ("tests.unit.test_sandbox",),
        "git escape flags (--upload-pack/--exec-path/…) allowed",
    ),
    Mutant(
        "src/ai_assistant/tools/sandbox.py",
        "if verb is not None and verb not in _GIT_BISECT_VERBS:",
        "if False:",
        ("tests.unit.test_sandbox",),
        "git bisect run (arbitrary verdict hook) allowed",
    ),
    Mutant(
        "src/ai_assistant/tools/sandbox.py",
        "if not value or _path_escapes_workspace(value, cwd):",
        "if False:",
        ("tests.unit.test_sandbox",),
        "cmake -P/--build pointed outside the workspace allowed",
    ),
    Mutant(
        "src/ai_assistant/tools/sandbox.py",
        "if resolved is None:",
        "if False:",
        ("tests.unit.test_sandbox",),
        "binary absent from PATH delegated to the OS search (cwd hijack)",
    ),
    Mutant(
        "src/ai_assistant/tools/sandbox.py",
        "if target_name.lower() in _DENIED_WRITE_NAMES or any(",
        "if False:",
        ("tests.unit.test_sandbox",),
        "write-deny on .env/policy files removed",
    ),
    Mutant(
        "src/ai_assistant/tools/sandbox.py",
        "if target_name.lower() in _DENIED_WRITE_NAMES or any(",
        "if target_name in _DENIED_WRITE_NAMES or any(",
        ("tests.unit.test_sandbox",),
        "the write deny-list compared case-sensitively (PYPROJECT.TOML "
        "replaces pyproject.toml on a case-insensitive filesystem)",
    ),
    Mutant(
        "src/ai_assistant/tools/sandbox.py",
        'for entry in (path or "").split(os.pathsep):',
        'for entry in (path or "").split(os.pathsep) + [os.curdir]:',
        ("tests.unit.test_sandbox",),
        "the PATH scan consults the current directory (a planted binary wins "
        "over the operator's own tool)",
    ),
    Mutant(
        "src/ai_assistant/tools/sandbox.py",
        "    if _denied_tree_hit(relative):",
        "    if False:",
        ("tests.unit.test_sandbox",),
        "write-deny on the policy trees removed (the agent may rewrite the "
        "transport policy, the A2A scope map and the approval logic)",
    ),
    Mutant(
        "src/ai_assistant/tools/sandbox.py",
        "        if parts[:len(denied_parts)] == denied_parts:",
        "        if False:",
        ("tests.unit.test_sandbox",),
        "the policy-tree match compares the wrong prefix (every policy file "
        "becomes writable again)",
    ),
    Mutant(
        "src/ai_assistant/domain/approvals.py",
        "    if ignore_volatile:",
        "    if False:",
        ("tests.unit.test_reason_node_guards", "tests.unit.test_approval_grants"),
        "the volatile per-retry keys are no longer filtered out of the retry "
        "fingerprint (a retry that only changed its request_id slips past the "
        "duplicate-call guard)",
    ),
    Mutant(
        "src/ai_assistant/bootstrap/app_factory.py",
        '        docs_url="/docs" if docs else None,\n'
        '        redoc_url="/redoc" if docs else None,\n'
        '        openapi_url="/openapi.json" if docs else None,',
        '        docs_url="/docs",\n'
        '        redoc_url="/redoc",\n'
        '        openapi_url="/openapi.json",',
        ("tests.unit.test_final_edges_b", "tests.integration.test_real_http_surface"),
        "the OpenAPI surface is published even when the deployment did not "
        "opt in (it is an app-level route, so the transport guard never "
        "reaches it)",
    ),
    Mutant(
        "src/ai_assistant/orchestration/nodes/tool.py",
        '        summary = describe_tool_result(result)',
        "        summary = str(result)",
        ("tests.unit.test_tool_node_guards", "tests.unit.test_tool_result_logging"),
        "the tool payload is written to the plaintext log again",
    ),
    Mutant(
        "src/ai_assistant/adapters/persistence/sqlite_store.py",
        "                if cursor.rowcount != 1:",
        "                if False:",
        ("tests.unit.test_final_edges_a", "tests.unit.test_task_service_cas_races"),
        "a lost compare-and-set still writes the event (the log announces a "
        "transition that did not happen)",
    ),
    Mutant(
        "src/ai_assistant/adapters/persistence/checkpointing.py",
        "    if cipher is None:\n        return None",
        "    if True:\n        return None",
        ("tests.unit.test_checkpointing",),
        "the LangGraph checkpoint store is written in the clear (the "
        "conversation is a plaintext copy of what the task store seals)",
    ),
    Mutant(
        "src/ai_assistant/tools/sandbox.py",
        '_FORBIDDEN_METACHAR = {"&&", "||", ";", "|", ">", "<", "`", "$(", "${"}',
        '_FORBIDDEN_METACHAR = {"&&", "||", ";", "|", ">", "<", "`", "$("}',
        ("tests.unit.test_sandbox",),
        "${VAR} expansion allowed (shell expansion behind the metachar gate)",
    ),
    Mutant(
        "src/ai_assistant/tools/sandbox.py",
        "if _path_escapes_workspace(argument, cwd):\n            return f\"Sandbox: {executable} script path must stay inside the workspace.\"",
        "if False:\n            return f\"Sandbox: {executable} script path must stay inside the workspace.\"",
        ("tests.unit.test_sandbox",),
        "python/pytest script outside the workspace allowed",
    ),
    # ── HITL approval gate ───────────────────────────────────────────────────
    Mutant(
        "src/ai_assistant/application/tools.py",
        "if spec.requires_approval and not self._grants.covers(",
        "if False and not self._grants.covers(",
        ("tests.unit.test_tool_gateway_guards", "tests.unit.test_deterministic_orchestrator"),
        "approval-required tools run without approval",
    ),
    Mutant(
        "src/ai_assistant/domain/governance.py",
        "if isinstance(value, str):\n        return scrub_pii(value)",
        "if False:\n        return scrub_pii(value)",
        ("tests.unit.test_governance_egress", "tests.unit.test_tool_event_wiring"),
        "secret pasted under a plain key reaches logs unmasked",
    ),
    # ── Contract: Qt history bound ───────────────────────────────────────────
    Mutant(
        "src/ai_assistant/agents/models.py",
        "history:             list[dict[str, Any]] = Field(default_factory=list, max_length=200)",
        "history:             list[dict[str, Any]] = Field(default_factory=list, max_length=20)",
        ("tests.contract.test_agent_contract_qt",),
        "history bound regressed 200 -> 20 (long sessions 422 again)",
    ),
    # ── Remote-leg opt-in (ADR 0018) ─────────────────────────────────────────
    Mutant(
        "src/ai_assistant/settings.py",
        'return values.get("AGENT_HYBRID_POLICY", "local_only").casefold()',
        'return values.get("AGENT_HYBRID_POLICY", "cloud_allowed").casefold()',
        ("tests.unit.test_inference",),
        "default hybrid policy flipped to cloud_allowed",
    ),
    # ── /health honesty ──────────────────────────────────────────────────────
    Mutant(
        "src/ai_assistant/adapters/http/health_routes.py",
        'status = "not_ready" if unready else ("degraded" if reasons else "ready")',
        'status = "ready"',
        ("tests.unit.test_http_health_routes",),
        "/health always claims ready",
    ),
    # ── RAG cache seal fail-closed (ADR 0028) ─────────────────────────────────
    # The cache holds a second copy of the user's document corpus, so it must
    # be sealed on disk and never loaded without being authenticated.
    Mutant(
        "src/ai_assistant/rag/index.py",
        "        path.write_bytes(cipher.seal_bytes(payload, context=context))",
        "        path.write_bytes(payload)",
        ("tests.unit.test_index",),
        "the cache artifacts are written to disk in the clear",
    ),
    Mutant(
        "src/ai_assistant/rag/index.py",
        "        if not is_sealed_metadata(meta):",
        "        if False:",
        ("tests.unit.test_index",),
        "a cache that was never sealed is loaded instead of rebuilt",
    ),
    Mutant(
        "src/ai_assistant/rag/index.py",
        '        "sealed": seal_metadata(),',
        '        "sealed": {},',
        ("tests.unit.test_index",),
        "the seal marker is not recorded (the cache can never be a hit)",
    ),
    Mutant(
        "src/ai_assistant/adapters/persistence/storage_crypto.py",
        "            return self._aead.decrypt(\n"
        "                raw[:_NONCE_BYTES], raw[_NONCE_BYTES:], _associated_data(context),\n"
        "            )\n"
        "        except (InvalidTag, ValueError) as error:",
        "            return self._aead.decrypt(\n"
        "                raw[:_NONCE_BYTES], raw[_NONCE_BYTES:], _associated_data(context),\n"
        "            )\n"
        "        except ValueError as error:",
        ("tests.unit.test_storage_crypto",),
        "a tampered binary envelope escapes as a raw crypto error",
    ),
    Mutant(
        "src/ai_assistant/adapters/persistence/storage_crypto.py",
        "        if not is_sealed_bytes(value):\n"
        '            message = f"{context} is not a sealed envelope"\n'
        "            raise StorageDecryptionError(message)",
        "        if not is_sealed_bytes(value):\n"
        "            return value",
        ("tests.unit.test_index", "tests.unit.test_storage_crypto"),
        "an unsealed binary artifact is accepted as plaintext (legacy cache loaded)",
    ),
    # ── Cancel that races the worker ─────────────────────────────────────────
    Mutant(
        "src/ai_assistant/application/coordination.py",
        'if previous is not None and previous.cancelled and previous.metadata.get("pending_cancel"):',
        "if False:",
        ("tests.unit.test_task_coordinator_lifecycle",),
        "pending cancel lost when the worker registers its run",
    ),
    Mutant(
        "src/ai_assistant/application/coordination.py",
        "if record.status in self._TERMINAL and not record.cancelled:",
        "if False:",
        ("tests.unit.test_task_coordinator_lifecycle",),
        "cancelling a finished run resurrects it",
    ),
    # ── Transport policy: the MCP mount fails closed (ADR 0008) ──────────────
    Mutant(
        "src/ai_assistant/adapters/http/security.py",
        "return not _is_loopback(host) and not policy.mcp_token",
        "return False",
        ("tests.unit.test_http_security",),
        "public bind without AI_MCP_TOKEN left exposed",
    ),
    # ── Classification survives persistence (ADR 0010) ───────────────────────
    Mutant(
        "src/ai_assistant/adapters/persistence/sqlite_store.py",
        "classification=DataClassification(classification),",
        "classification=DataClassification.INTERNAL,",
        ("tests.unit.test_final_edges_a",),
        "reload promotes a RESTRICTED task back to INTERNAL",
    ),
    # ── Remote egress refuses sensitive params (ADR 0002/0010) ───────────────
    Mutant(
        "src/ai_assistant/agents/graph_wiring.py",
        "if has_sensitive_content(params):",
        "if False:",
        ("tests.unit.test_runner",),
        "credential-bearing params dispatched to a remote specialist",
    ),
    # ── Step vocabulary trust boundary (ADR 0019) ────────────────────────────
    Mutant(
        "src/ai_assistant/domain/steps.py",
        "if kind not in get_args(StepType):",
        "if False:",
        ("tests.unit.test_step_boundary_validation",),
        "step vocabulary check removed (unknown discriminator accepted)",
    ),
    Mutant(
        "src/ai_assistant/adapters/persistence/pending_store.py",
        'if "steps" in value:',
        "if False:",
        ("tests.unit.test_step_boundary_validation",),
        "persisted step log accepted without the runtime gate",
    ),
    Mutant(
        "src/ai_assistant/adapters/orchestration/deterministic.py",
        "validated = validate_steps(steps)",
        "validated = list(steps)",
        ("tests.unit.test_step_boundary_validation",),
        "continuation steps accepted without the vocabulary gate",
    ),
    # ── Plugin enablement gate: metadata before module execution ─────────────
    Mutant(
        "src/ai_assistant/plugins/loader.py",
        "if not any(registry.allows(name) for name in _declared_names(entry_point)):",
        "if False:",
        ("tests.unit.test_plugins", "tests.unit.test_plugin_reentry"),
        "disabled plugin instantiated (its module code runs at bootstrap)",
    ),
    # ── Approval resume re-enters the session run lock ───────────────────────
    Mutant(
        "src/ai_assistant/agents/service.py",
        "with self._session_lock(request.session_id):\n            return self._approval_service.approve(request)",
        "return self._approval_service.approve(request)",
        ("tests.unit.test_session_serialization",),
        "approval resume runs outside the session lock (interleaves a run)",
    ),
    # ── Transport hardening: bounded, redirect-free A2A egress ───────────────
    Mutant(
        "src/ai_assistant/adapters/a2a_client.py",
        "if len(body) > _MAX_RESPONSE_BYTES:",
        "if False:",
        ("tests.unit.test_a2a_client",),
        "unbounded remote response buffered whole (memory DoS)",
    ),
    Mutant(
        "src/ai_assistant/adapters/a2a_client.py",
        "def redirect_request(self, req: Any, fp: Any, code: int, msg: str, headers: Any, newurl: str) -> None:\n        # The parameters are the handler contract; none of them are read here.\n        return None",
        "def redirect_request(self, req: Any, fp: Any, code: int, msg: str, headers: Any, newurl: str) -> None:\n        return super().redirect_request(req, fp, code, msg, headers, newurl)",
        ("tests.unit.test_a2a_client", "tests.unit.test_a2a_protocol_guards"),
        "A2A redirects followed (SSRF behind the endpoint allowlist) — on the "
        "dispatch leg and on the discovery leg, which share one opener",
    ),
    Mutant(
        "src/ai_assistant/adapters/a2a_protocol.py",
        "if not isinstance(base_url, str) or not _is_http_url(base_url):",
        "if False:",
        ("tests.unit.test_a2a_protocol_guards",),
        "non-HTTP discovery URL fetched (SSRF via discovery)",
    ),
    Mutant(
        "src/ai_assistant/adapters/a2a_protocol.py",
        "if len(raw) > _DISCOVERY_MAX_BYTES:",
        "if False:",
        ("tests.unit.test_a2a_protocol_guards",),
        "oversized Agent Card buffered whole (memory DoS via discovery)",
    ),
    Mutant(
        "src/ai_assistant/adapters/a2a_protocol.py",
        "                        url=base_url, card=card_data, skills=skills,",
        '                        url=str(card_data.get("url") or base_url), card=card_data, skills=skills,',
        ("tests.unit.test_a2a_protocol_guards",),
        "Agent Card controls the dispatch endpoint (allowlist bypass)",
    ),
    Mutant(
        "src/ai_assistant/tools/builtin/path_utils.py",
        "if is_secret_name(os.path.basename(abs_path)) or is_secret_name(os.path.basename(real_path)):",
        "if False:",
        ("tests.unit.test_file_tools",),
        "the secret-file read guard removed (the agent reads .env — the "
        "at-rest key and every bearer token — without approval)",
    ),
    Mutant(
        "src/ai_assistant/adapters/persistence/sqlite_store.py",
        "if require_sealed and not cipher.is_sealed(value):",
        "if False:",
        ("tests.unit.test_sqlite_store_crypto",),
        "an unsealed column served as authentic (AEAD downgrade: a "
        "substituted plaintext column is accepted instead of rejected)",
    ),
    Mutant(
        "src/ai_assistant/adapters/persistence/pending_store.py",
        "if legacy:",
        "if False:",
        ("tests.unit.test_pending_store_crypto",),
        "the legacy pending-action file is not re-sealed on load (the "
        "plaintext copy stays on disk)",
    ),
    Mutant(
        "src/ai_assistant/domain/governance.py",
        "    if isinstance(value, str):\n        return carries_credential_shape(value)",
        "    if False:\n        return carries_credential_shape(value)",
        ("tests.unit.test_governance_egress",),
        "a credential pasted into a free-text value is not detected (the "
        "user's own prompt leaves the trust boundary in clear)",
    ),
    Mutant(
        "src/ai_assistant/tools/builtin/vcs_tools.py",
        "if sub_rejection:",
        "if False:",
        ("tests.unit.test_vcs_tools",),
        "blocked git subcommand runs outside the sandbox gate",
    ),
    # ── Task-list pagination is clamped, never handed to SQL raw ─────────────
    Mutant(
        "src/ai_assistant/adapters/a2a_router.py",
        "return max(1, min(int(value), _MAX_TASK_PAGE))",
        "return int(value)",
        ("tests.unit.test_a2a_http_router",),
        "limit=-1 reaches SQLite as LIMIT -1 (whole task table)",
    ),
    # ── Request-body cap refuses before parsing ──────────────────────────────
    Mutant(
        "src/ai_assistant/bootstrap/app_factory.py",
        "if declared > self._max_bytes:",
        "if False:",
        ("tests.unit.test_http_security",),
        "declared oversize body accepted (no 413)",
    ),
    # ── Backend owns the llama-context serialization ─────────────────────────
    Mutant(
        "src/ai_assistant/llm/llama_cpp_backend.py",
        "with self._generation_lock:\n            return self._llm.create_chat_completion(",
        "if True:\n            return self._llm.create_chat_completion(",
        ("tests.unit.test_llm_backend",),
        "llama context no longer serialized (overlapping generations corrupt it)",
    ),
    # ── Lazy composition root (ADR 0020) ──────────────────────────────────────
    Mutant(
        "src/ai_assistant/bootstrap/server.py",
        "if name in _COMPOSED_NAMES:",
        "if False:",
        ("tests.integration.test_composition_root_bootstrap",),
        "__getattr__ never composes (composed names degrade to AttributeError)",
    ),
    Mutant(
        "src/ai_assistant/bootstrap/__init__.py",
        "raise AttributeError(msg) from None",
        "return None",
        ("tests.integration.test_composition_root_bootstrap",),
        "unknown bootstrap attribute returns None instead of raising",
    ),
    # Without the early return, the compose() → _register_optional_protocols()
    # → compose() back-edge never terminates: every cycle re-opens logs and
    # stores (~1.6s of IO), so the owning suite hangs instead of failing
    # (observed >600s unmutated-vs-mutated). The file runs ~25s unmutated, so
    # a 180s budget with kill_on_timeout cannot fake a kill on a slow
    # machine — a timeout here IS the guard being load-bearing.
    Mutant(
        "src/ai_assistant/bootstrap/server.py",
        "if _COMPOSED:\n"
        "            return\n"
        "        # Set before the body so the compose() → _register_optional_protocols()\n"
        "        # → compose() back-edge terminates instead of rebuilding. The public\n"
        "        # wrapper resets it if the body raises.\n"
        "        _COMPOSED = True",
        "_COMPOSED = True",
        ("tests.integration.test_composition_root_bootstrap",),
        "second compose() rebuilds the graph (duplicate handlers and routers)",
        timeout_seconds=180,
        kill_on_timeout=True,
    ),
    Mutant(
        "src/ai_assistant/bootstrap/server.py",
        "except BaseException:\n        _COMPOSED = False\n        raise",
        "except BaseException:\n        raise",
        ("tests.integration.test_composition_root_bootstrap",),
        "a failed compose leaves the flag set (half-built graph looks ready)",
    ),
    # ── A2A per-tenant task ownership ─────────────────────────────────────────
    Mutant(
        "src/ai_assistant/adapters/http/security.py",
        "if tenant is None:",
        "if False:",
        ("tests.unit.test_http_security", "tests.unit.test_a2a_http_router"),
        "an unknown A2A tenant token is accepted instead of 401",
    ),
    Mutant(
        "src/ai_assistant/application/tasks.py",
        "def get(self, task_id: str, *, owner: str = \"default\") -> AgentTask | None:\n"
        "        task = self._store.get(task_id)\n"
        "        if task is None or task.owner != owner:",
        "def get(self, task_id: str, *, owner: str = \"default\") -> AgentTask | None:\n"
        "        task = self._store.get(task_id)\n"
        "        if task is None:",
        ("tests.unit.test_task_lifecycle_guards", "tests.unit.test_a2a_http_router"),
        "a task can be read across tenant boundaries",
    ),
    Mutant(
        "src/ai_assistant/adapters/persistence/sqlite_store.py",
        "if context_id is not None and owner is not None:",
        "if False:",
        ("tests.unit.test_final_edges_a",),
        "the owner filter is dropped when listing tasks",
    ),
    Mutant(
        "src/ai_assistant/adapters/persistence/sqlite_store.py",
        "if retention_days <= 0:",
        "if False:",
        ("tests.unit.test_final_edges_a",),
        "a disabled retention window still prunes tasks",
    ),
    # ── Production gates & admission governor (ADR 0023/0024) ────────────────
    Mutant(
        "src/ai_assistant/tools/sandbox.py",
        'if runtime_settings.sandbox_require_isolation and runtime_settings.sandbox_runtime != "docker":',
        "if False:",
        ("tests.unit.test_sandbox",),
        "isolation gate removed (local runtime runs with server privileges)",
    ),
    Mutant(
        "src/ai_assistant/adapters/persistence/checkpointing.py",
        'if required and backend == "memory":',
        "if False:",
        ("tests.unit.test_checkpointing",),
        "required-checkpoint gate removed (silent MemorySaver fallback)",
    ),
    Mutant(
        "src/ai_assistant/adapters/persistence/checkpointing.py",
        "if saver is not None:\n            return saver\n    except Exception as error:",
        "if saver is not None:\n            return saver\n    except KeyboardInterrupt as error:",
        ("tests.unit.test_checkpointing",),
        "unexpected backend failure escapes before the gate decides",
    ),
    Mutant(
        "src/ai_assistant/adapters/http/rate_limit.py",
        "if used >= self._limit:",
        "if False:",
        ("tests.unit.test_rate_limit",),
        "per-peer budget never refuses (rate limit is decorative)",
    ),
    Mutant(
        "src/ai_assistant/agents/service.py",
        "if slots is not None and not slots.acquire(blocking=False):",
        "if False:",
        ("tests.unit.test_session_serialization",),
        "run budget removed (unbounded concurrent runs)",
    ),
    # ── At-rest encryption of local state (ADR 0027) ─────────────────────────
    # The encryption guards fail *closed* everywhere: a missing key, a wrong
    # key or a tampered row must abort the read instead of handing the caller
    # ciphertext or an empty store. Coverage cannot see that direction, so
    # each decision is inverted here.
    Mutant(
        "src/ai_assistant/adapters/persistence/storage_crypto.py",
        "    if explicit:\n"
        '        message = f"AI_STORAGE_KEY_FILE does not exist: {path}"\n'
        "        raise StorageKeyError(message)",
        "    if False:",
        ("tests.unit.test_storage_crypto",),
        "a missing explicit key file is replaced by a fresh key (existing state orphaned)",
    ),
    Mutant(
        "src/ai_assistant/adapters/persistence/storage_crypto.py",
        "            plaintext = self._aead.decrypt(\n"
        "                raw[:_NONCE_BYTES], raw[_NONCE_BYTES:], _associated_data(context),\n"
        "            )\n"
        "        except (InvalidTag, ValueError) as error:",
        "            plaintext = self._aead.decrypt(\n"
        "                raw[:_NONCE_BYTES], raw[_NONCE_BYTES:], _associated_data(context),\n"
        "            )\n"
        "        except ValueError as error:",
        ("tests.unit.test_storage_crypto",),
        "a tampered envelope escapes as a raw crypto error instead of a decryption failure",
    ),
    Mutant(
        "src/ai_assistant/adapters/persistence/storage_crypto.py",
        "        if len(key) != KEY_BYTES:\n"
        '            message = f"at-rest key must decode to exactly {KEY_BYTES} bytes"\n'
        "            raise StorageKeyError(message)",
        "        if False:",
        ("tests.unit.test_storage_crypto",),
        "an at-rest key of the wrong length is accepted",
    ),
    Mutant(
        "src/ai_assistant/adapters/persistence/storage_crypto.py",
        "    if _create_key_file(path, key):\n"
        "        return AesGcmFieldCipher(key)\n"
        "    # Another process won the O_EXCL race; its key is the one on disk.\n"
        "    return AesGcmFieldCipher(_read_key_file(path))",
        "    if _create_key_file(path, key):\n"
        "        return AesGcmFieldCipher(key)\n"
        "    return AesGcmFieldCipher(key)",
        ("tests.unit.test_storage_crypto",),
        "the process that lost the key-creation race keeps its own key (processes disagree)",
    ),
    Mutant(
        "src/ai_assistant/adapters/persistence/storage_crypto.py",
        "    _fsync_directory(path.parent)\n"
        "    return True",
        "    return True",
        ("tests.unit.test_storage_crypto",),
        "the key file's directory entry is never flushed (a crash orphans every row)",
    ),
    Mutant(
        "src/ai_assistant/adapters/persistence/pending_store.py",
        "                if self._cipher is None:\n"
        "                    if is_sealed(raw):",
        "                if self._cipher is None:\n"
        "                    if False:",
        ("tests.unit.test_pending_store_crypto",),
        "sealed pending actions are silently dropped instead of failing closed",
    ),
    Mutant(
        "src/ai_assistant/adapters/persistence/pending_store.py",
        "                if self._cipher is not None:\n"
        '                    raw = self._cipher.seal(raw, context="pending.actions")',
        "                if False:\n"
        '                    raw = self._cipher.seal(raw, context="pending.actions")',
        ("tests.unit.test_pending_store_crypto",),
        "pending actions are written to disk as plaintext",
    ),
    Mutant(
        "src/ai_assistant/adapters/persistence/sqlite_store.py",
        "    if cipher is None:\n"
        "        if is_sealed(value):",
        "    if cipher is None:\n"
        "        if False:",
        ("tests.unit.test_sqlite_store_crypto",),
        "sealed task state is served as ciphertext instead of failing closed",
    ),
    Mutant(
        "src/ai_assistant/adapters/persistence/sqlite_store.py",
        "            if cipher is not None and version < 3:\n"
        "                self._seal_legacy_rows(cipher)\n"
        '                self._conn.execute("PRAGMA user_version = 3")',
        "            if False:\n"
        "                self._seal_legacy_rows(cipher)\n"
        '                self._conn.execute("PRAGMA user_version = 3")',
        ("tests.unit.test_sqlite_store_crypto",),
        "legacy plaintext task rows are never migrated to ciphertext",
    ),
    Mutant(
        "src/ai_assistant/adapters/persistence/sqlite_store.py",
        "    if value is None or is_sealed(value):\n"
        "        return value",
        "    if value is None:\n"
        "        return value",
        ("tests.unit.test_sqlite_store_crypto",),
        "an already-sealed column is sealed a second time (state becomes unreadable)",
    ),
    # ── At-rest key rotation (ADR 0030) ──────────────────────────────────────
    # The rotation is the only path that rewrites sealed state. It has to
    # fail closed in two directions: a value the old key cannot open must
    # abort the whole rotation, and a rotation that aborts must leave the
    # state exactly as it was — half of it re-encrypted would be readable
    # with neither key.
    Mutant(
        "src/ai_assistant/adapters/persistence/storage_crypto.py",
        "    except StorageDecryptionError:\n"
        "        new.open(value, context=context)\n"
        "        return value, False",
        "    except StorageDecryptionError:\n"
        "        return value, False",
        ("tests.unit.test_storage_rekey",),
        "the rotation rewrites a value the old key cannot open (wrong key accepted)",
    ),
    Mutant(
        "src/ai_assistant/adapters/persistence/sqlite_store.py",
        "        with self._lock, self._conn:\n"
        "            for task_id, message, result, error, metadata in self._conn.execute(",
        "        with self._lock:\n"
        "            for task_id, message, result, error, metadata in self._conn.execute(",
        ("tests.unit.test_storage_rekey",),
        "the rotation is not one transaction (an aborted rotation leaves half the state re-encrypted)",
    ),
    Mutant(
        "scripts/rekey_storage.py",
        "    report = _inspect(state_dir, old, new)\n"
        "    if report.failures:",
        "    report = _inspect(state_dir, old, new)\n"
        "    if False:",
        ("tests.unit.test_storage_rekey",),
        "the rotation starts without checking that the old key opens the state",
    ),
)


#: Marks an in-flight run. Its absence means "the tree is safe to mutate": a
#: lock left behind by a killed process has to be cleared by hand *after*
#: restoring the source, which is exactly the conversation we want to force.
LOCK_PATH = BASE / "scripts" / ".mutation_check.lock"


def _acquire_lock() -> None:
    """Refuse to overlap another run (or a crashed one still holding it)."""
    try:
        handle = os.open(LOCK_PATH, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        message = (
            f"mutation_check: {LOCK_PATH} exists — another run is active, or a "
            "previous one was killed mid-mutation. Verify the source is intact "
            "(git status / git diff), restore it if not, then delete the lock. "
            "Never run this alongside the test suite."
        )
        raise SystemExit(message) from None
    os.write(handle, str(os.getpid()).encode("ascii"))
    os.close(handle)


def _release_lock() -> None:
    LOCK_PATH.unlink(missing_ok=True)


def run_mutant(mutant: Mutant) -> str:
    """Mutate one guard, run its tests, restore the file verbatim.

    Returns ``"KILLED"``, ``"SURVIVED"`` or ``"STALE"`` — the last when the
    anchor no longer matches the source, which fails the run rather than
    passing it silently. A suite that times out is a crash, unless the mutant
    explicitly opts into ``kill_on_timeout`` (its guard turns "fail" into
    "never finish", and the budget is set far above the unmutated runtime).
    """
    target = BASE / mutant.file
    original = target.read_text(encoding="utf-8")
    if original.count(mutant.original) != 1:
        print(f"  STALE {mutant.label}: anchor not unique/absent in {mutant.file}")
        return "STALE"
    try:
        target.write_text(original.replace(mutant.original, mutant.mutated), encoding="utf-8")
        env = {**os.environ, "PYTHONPATH": "src"}
        try:
            outcome = subprocess.run(
                [sys.executable, "-m", "unittest", *mutant.tests, "-q"],
                cwd=BASE, capture_output=True, text=True, timeout=mutant.timeout_seconds, env=env,
            )
        except subprocess.TimeoutExpired:
            if not mutant.kill_on_timeout:
                raise
            print(f"  TIMEOUT {mutant.label}: suite never finished in "
                  f"{mutant.timeout_seconds}s — the guard's absence hangs it")
            return "KILLED"
    finally:
        target.write_text(original, encoding="utf-8")
        if target.read_text(encoding="utf-8") != original:
            message = (
                f"{mutant.file} was not restored byte-for-byte; restore it "
                "before running anything against this tree."
            )
            raise RuntimeError(message)
    return "KILLED" if outcome.returncode != 0 else "SURVIVED"


def floor_violation(mutant_count: int) -> str | None:
    """The reason the declared mutant count is below the floor, if it is.

    Without a floor, *deleting* a mutant is indistinguishable from *passing*
    it: the suite stays green, the coverage floor stays at 100 %, and the
    guard silently stops being guarded. ADR 0032 makes the count a gate — a
    new load-bearing decision adds a mutant in the same commit, and a removed
    one lowers this number in the same commit, so the diff always shows it.
    """
    if mutant_count < MIN_MUTANTS:
        return (
            f"{mutant_count} mutants declared, {MIN_MUTANTS} required — "
            "a guard was removed without a decision"
        )
    return None


def main() -> int:
    survivors: list[str] = []
    stale: list[str] = []
    _acquire_lock()
    try:
        for index, mutant in enumerate(MUTANTS, start=1):
            verdict = run_mutant(mutant)
            print(
                f"[{index:2d}/{len(MUTANTS)}] {verdict:9s} "
                f"{mutant.file.split('/')[-1]}: {mutant.label}"
            )
            entry = f"{mutant.file}: {mutant.label} (tests: {', '.join(mutant.tests)})"
            if verdict == "SURVIVED":
                survivors.append(entry)
            elif verdict == "STALE":
                stale.append(entry)
    finally:
        _release_lock()
    killed = len(MUTANTS) - len(survivors) - len(stale)
    print(
        f"\nMutants: {len(MUTANTS)}  killed: {killed}  "
        f"survived: {len(survivors)}  stale: {len(stale)}"
    )
    for entry in survivors:
        print(f"  SURVIVED -> {entry}")
    for entry in stale:
        print(f"  STALE    -> {entry} (anchor no longer matches the source)")
    if len(MUTANTS) < MIN_MUTANTS:
        # ADR 0032: the count itself is a gate. A mutant that disappeared
        # without a decision is exactly the drift this gate exists to catch.
        print(f"  FLOOR    -> {floor_violation(len(MUTANTS))}")
        return 1
    return 1 if survivors or stale else 0


if __name__ == "__main__":
    sys.exit(main())
