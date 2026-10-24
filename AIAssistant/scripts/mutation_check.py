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
#: ADR 0035 adds six more (the path-shaped program name, the case-insensitive
#: blocked-directory rule, the path confinement itself, the LSP server
#: resolution, the fail-closed git resolution and the attachment gate).
#: The command-surface hardening adds eleven more (the pre-verb option
#: allow-list, the git destination confinement, the attached git config
#: spelling, the pytest plugin/import options, the ctest option allow-list,
#: the two write deny-tree guards, the classification comparison, the
#: clearance derivation, the unknown classification label and the isolation
#: gate's default).
#: The row-bound at-rest state adds three more (the row-bound associated
#: data, the identity verification and the refusal of a state file that
#: predates it).
#: The single-rule cleanup adds four more (the success rule, the cursor
#: detection, the cursor use and the regular-file gate on the attachment).
#: The corpus boundary adds five more (the link rule, the secret names, the
#: shared blocked directories, the extension lists and the cache hash).
#: The sixth review adds one more (the sealing action refuses a state that is
#: already sealed).
#: The seventh review adds one more (the binaries whose path positionals are
#: confined).
#: The eighth review adds four more: one model lease per entry point that
#: runs the graph (``execute``, ``approve``, the desktop ack) plus the chat
#: route's inference span (ADR 0040).
#: The tenth review adds ten more: the owner-enforced reload window (guard and
#: the startup caller), the fail-closed scope map, the missing-policy and
#: missing-catalogue bootstrap refusals, the graceful stop (event, watcher and
#: admin route) and the public RAG surface (property and health read).
MIN_MUTANTS = 160

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
        "return f\"Sandbox: subcommand '{token}' is not in the {executable} allow-list.\"",
        "return None",
        ("tests.unit.test_sandbox",),
        "unknown subcommand allowed",
    ),
    Mutant(
        "src/ai_assistant/tools/sandbox.py",
        "if head in _GIT_ESCAPE_FLAGS or argument == \"-c\" or argument.startswith(\"-c\"):",
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
        "src/ai_assistant/exec_policy.py",
        'for entry in (path or "").split(os.pathsep):',
        'for entry in (path or "").split(os.pathsep) + [os.curdir]:',
        ("tests.unit.test_sandbox",),
        "the PATH scan consults the current directory (a planted binary wins "
        "over the operator's own tool)",
    ),
    Mutant(
        "src/ai_assistant/exec_policy.py",
        "if not name or os.path.isabs(name) or os.path.dirname(name):",
        "if False:",
        ("tests.unit.test_sandbox",),
        "a path-shaped program name is resolved instead of refused (the "
        "allow-list constrains the name, not the path)",
    ),
    Mutant(
        "src/ai_assistant/path_policy.py",
        "    return name.lower() in _BLOCKED_DIR_NAMES",
        "    return name in _BLOCKED_DIR_NAMES",
        ("tests.unit.test_path_policy",),
        "the blocked-directory rule compares case-sensitively (.GIT is "
        "the same directory as .git on a case-insensitive filesystem)",
    ),
    Mutant(
        "src/ai_assistant/path_policy.py",
        "        canonical_path(target).relative_to(canonical_path(root_path))",
        "        pass",
        ("tests.unit.test_path_policy",),
        "the path confinement check removed (a junction out of the root is "
        "read as if it were inside it)",
    ),    Mutant(
        "src/ai_assistant/path_policy.py",
        "        if os.path.commonpath([lexical, lexical_root]) != lexical_root:",
        "        if False:",
        ("tests.unit.test_path_policy",),
        "the spelling half of the confinement check removed (a path that "
        "only looks like it leaves the root is accepted)",
    ),
    Mutant(
        "src/ai_assistant/tools/lsp_client.py",
        "    resolved_bin = resolve_tool_binary(server_bin)",
        "    resolved_bin = shutil.which(server_bin)",
        ("tests.unit.test_lsp_client",),
        "the LSP server is resolved with the platform lookup again (the "
        "current directory wins over PATH on Windows)",
    ),
    Mutant(
        "src/ai_assistant/tools/builtin/vcs_tools.py",
        '    binary = resolve_executable("git", os.environ.get("PATH", ""))',
        '    binary = "git"',
        ("tests.unit.test_vcs_tools",),
        "the git tools fall back to a bare name (the OS searches the "
        "current directory first on Windows)",
    ),
    Mutant(
        "src/ai_assistant/adapters/http/chat_routes.py",
        "        if resolve_within(root, path) is not None:",
        "        if False:",
        ("tests.unit.test_http_chat_routes",),
        "a client-named attachment is opened from anywhere on the host",
    ),
    Mutant(
        "src/ai_assistant/tools/tool_contract.py",
        "    if override is None and canonical not in READ_ONLY_TOOLS:",
        "    if False:",
        ("tests.unit.test_tool_contract",),
        "an unknown tool silently receives the read-only default instead of "
        "failing bootstrap (the review's unlisted-tool finding)",
    ),
    Mutant(
        "src/ai_assistant/tools/sandbox.py",
        "    if _denied_tree_hit(relative):\n"
        '        return {"error": "Write target is denied by sandbox policy (policy tree)."}',
        "    if False:\n"
        '        return {"error": "Write target is denied by sandbox policy (policy tree)."}',
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
        "if _path_escapes_workspace(argument, cwd):\n            return f\"Sandbox: {executable} path must stay inside the workspace.\"",
        "if False:\n            return f\"Sandbox: {executable} path must stay inside the workspace.\"",
        ("tests.unit.test_sandbox",),
        "a path positional outside the workspace is allowed (ruff check <dir> "
        "prints the source lines of every file it finds)",
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
        "        with self._hold_session(request.session_id):\n"
        "            slots = self._acquire_run_slot()\n"
        "            try:\n"
        "                with self._model_lease():\n"
        "                    return self._approval_service.approve(request)\n"
        "            finally:\n"
        "                self._release_run_slot(slots)",
        "        slots = self._acquire_run_slot()\n"
        "        try:\n"
        "            with self._model_lease():\n"
        "                return self._approval_service.approve(request)\n"
        "        finally:\n"
        "            self._release_run_slot(slots)",
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
        "if is_secret_name(os.path.basename(rel_path)) or is_secret_name(real_basename):",
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
    # ADR 0040 pinned the model lease to every entry point that runs the
    # graph; ADR 0024's budget now follows the same rule. One mutant per
    # resume path — leaving either outside the cap is the exact defect the
    # 2026-10-23 follow-up closed — plus one for the shared release.
    Mutant(
        "src/ai_assistant/agents/service.py",
        "            slots = self._acquire_run_slot()\n"
        "            try:\n"
        "                with self._model_lease():\n"
        "                    return self._approval_service.approve(request)",
        "            slots = None\n"
        "            try:\n"
        "                with self._model_lease():\n"
        "                    return self._approval_service.approve(request)",
        ("tests.unit.test_session_serialization",),
        "an approval resume bypasses the global run budget",
    ),
    Mutant(
        "src/ai_assistant/agents/service.py",
        "            slots = self._acquire_run_slot()\n"
        "            try:\n"
        "                with self._model_lease():\n"
        "                    return self._ui_action_service.ui_action_result(request)",
        "            slots = None\n"
        "            try:\n"
        "                with self._model_lease():\n"
        "                    return self._ui_action_service.ui_action_result(request)",
        ("tests.unit.test_session_serialization",),
        "a desktop-ack resume bypasses the global run budget",
    ),
    Mutant(
        "src/ai_assistant/agents/service.py",
        "        if slots is not None:\n            slots.release()",
        "        if False:\n            slots.release()",
        ("tests.unit.test_session_serialization",),
        "a run slot is never returned (the budget leaks until 503)",
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
        "                if self._has_unsealed_content(cipher):",
        "                if False:",
        ("tests.unit.test_sqlite_store_crypto",),
        "a plaintext task store is accepted and migrated implicitly (the "
        "forge-the-schema-version attack)",
    ),
    Mutant(
        "src/ai_assistant/adapters/persistence/pending_store.py",
        "                elif not is_sealed(raw):",
        "                elif False:",
        ("tests.unit.test_pending_store_crypto",),
        "a plaintext pending-action file is accepted (the attacker's own "
        "pending approval is served to the user)",
    ),
    Mutant(
        "scripts/migrate_state_row_binding.py",
        "    try:\n"
        "        cipher.open(value, context=context)\n"
        "    except StorageCryptoError:\n"
        "        pass\n"
        "    else:\n"
        "        return value  # already row-bound: the script is idempotent",
        "    try:\n"
        "        cipher.open(value, context=context)\n"
        "    except StorageCryptoError:\n"
        "        pass",
        ("tests.unit.test_migrate_state_row_binding",),
        "an already-sealed value is re-sealed (the state becomes unreadable)",
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
        "        rekeyed = 0\n"
        "        with self._lock, self._conn:\n"
        "            for (task_id, message, result, error, metadata, row_identity, owner, classification,",
        "        rekeyed = 0\n"
        "        with self._lock:\n"
        "            for (task_id, message, result, error, metadata, row_identity, owner, classification,",
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
    # ── The command surface: a leading option is not "just a flag" ───────────
    # The allow-list used to accept *any* token that started with ``-`` as an
    # ordinary flag, so the verb was never checked: ``git -C <dir> clean
    # -fdx`` retargeted it, ``git -ccore.sshCommand=<cmd>`` injected config,
    # ``pytest -p <mod>`` imported code and ``ctest --build-and-test
    # --test-command <cmd>`` executed one. The pre-verb surface is an
    # allow-list now, and each of these guards has its own mutant.
    Mutant(
        "src/ai_assistant/tools/sandbox.py",
        "            if token in flags or head in flags:",
        "            if True:",
        ("tests.unit.test_sandbox",),
        "an unknown leading flag is accepted (the verb can be hidden behind it)",
    ),
    Mutant(
        "src/ai_assistant/tools/sandbox.py",
        "        if _looks_like_path(argument) and _path_escapes_workspace(argument, cwd):",
        "        if False:",
        ("tests.unit.test_sandbox",),
        "a git clone/init destination outside the workspace is accepted",
    ),
    Mutant(
        "src/ai_assistant/tools/sandbox.py",
        'argument == "-c" or argument.startswith("-c")',
        "False",
        ("tests.unit.test_sandbox",),
        "the attached git config spelling (-ccore.sshCommand=…) is accepted",
    ),
    Mutant(
        "src/ai_assistant/tools/sandbox.py",
        "        if head in dangerous:",
        "        if False:",
        ("tests.unit.test_sandbox",),
        "an option that reads or writes a path outside the workspace is "
        "accepted (ruff --config, pytest --junitxml, git diff --no-index, …)",
    ),
    Mutant(
        "src/ai_assistant/tools/sandbox.py",
        '        if token not in _ALLOWED_FLAGS["ctest"] and head not in _ALLOWED_FLAGS["ctest"]:',
        "        if False:",
        ("tests.unit.test_sandbox",),
        "ctest may build and run a command (--build-and-test --test-command)",
    ),
    # ── The write deny-list is a rule, not an enumeration ────────────────────
    Mutant(
        "src/ai_assistant/tools/sandbox.py",
        '    if _denied_tree_hit(relative):\n'
        '        return {"error": "Directory target is denied by sandbox policy (policy tree)."}',
        '    if False:\n'
        '        return {"error": "Directory target is denied by sandbox policy (policy tree)."}',
        ("tests.unit.test_sandbox",),
        "a directory is created inside the platform's own source",
    ),
    # ── The clearance gate: the principal's clearance covers the tool ────────
    Mutant(
        "src/ai_assistant/application/tools.py",
        "if not classification_at_least(principal.classification, spec.maximum_classification):",
        "if not classification_at_least(spec.maximum_classification, principal.classification):",
        ("tests.unit.test_tool_gateway_guards",),
        "the classification gate runs the other way round (the least trusted "
        "caller is let through and the most trusted is refused)",
    ),
    Mutant(
        "src/ai_assistant/domain/security.py",
        "    if scopes & _ELEVATED_SCOPES:",
        "    if False:",
        ("tests.unit.test_tool_gateway_guards",),
        "the clearance no longer follows the granted scopes",
    ),
    Mutant(
        "src/ai_assistant/bootstrap/container.py",
        "}.get(label.casefold(), DataClassification.REGULATED)",
        "}.get(label.casefold(), DataClassification.PUBLIC)",
        ("tests.unit.test_gateway_bootstrap_guards",),
        "an unknown classification label lowers a tool's data ceiling",
    ),
    Mutant(
        "src/ai_assistant/tools/sandbox.py",
        '_PATH_POSITIONAL_READERS = frozenset({"python", "pytest", "ruff"})',
        '_PATH_POSITIONAL_READERS = frozenset({"python", "pytest"})',
        ("tests.unit.test_sandbox",),
        "the binaries whose path positionals are confined lose that rule "
        "(ruff check <dir> prints the source lines of every file it finds)",
    ),
    # ── The isolation gate is on by default ──────────────────────────────────
    Mutant(
        "src/ai_assistant/settings.py",
        'values.get("AGENT_SANDBOX_REQUIRE_ISOLATION", "1") == "1"',
        'values.get("AGENT_SANDBOX_REQUIRE_ISOLATION", "0") == "1"',
        ("tests.unit.test_settings",),
        "the sandbox isolation gate is off by default again (the local runtime "
        "runs commands with the server's own privileges)",
    ),
    # ── The at-rest state is bound to its row ─────────────────────────────────
    Mutant(
        "src/ai_assistant/adapters/persistence/sqlite_store.py",
        '    return f"task.v2:{task_id}:{field}"',
        '    return f"task.v2:{field}"',
        ("tests.unit.test_sqlite_store_crypto",),
        "the sealed value is not bound to its row (a value can be moved to "
        "another task and be served as that task's content)",
    ),
    Mutant(
        "src/ai_assistant/adapters/persistence/sqlite_store.py",
        "        if bundle.get(key) != actual:",
        "        if False:",
        ("tests.unit.test_sqlite_store_crypto",),
        "the row identity is not verified (the owner, the classification and "
        "the status can be forged)",
    ),
    Mutant(
        "src/ai_assistant/adapters/persistence/sqlite_store.py",
        "                if cipher is not None and has_rows:",
        "                if False:",
        ("tests.unit.test_sqlite_store_crypto",),
        "a state file that predates the row-bound sealing is opened instead "
        "of being refused",
    ),
    Mutant(
        "src/ai_assistant/adapters/persistence/sqlite_store.py",
        "                    if value is not None and cipher.is_sealed(value):",
        "                    if False:",
        ("tests.unit.test_sqlite_store_crypto",),
        "the sealing action re-seals a state that is already sealed (the file "
        "claims the new version while its content stays in the old format)",
    ),
    # ── One rule per decision, and the signature is asked, not caught ────────
    Mutant(
        "src/ai_assistant/application/tools.py",
        "        success = tool_result_succeeded(payload)",
        "        success = not bool(payload.get(\"error\")) and payload.get(\"success\") is not False",
        ("tests.unit.test_tool_gateway_guards",),
        "the application re-spells the success rule and drops the exit code "
        "(a failed command is reported as a success)",
    ),
    Mutant(
        "src/ai_assistant/ports.py",
        '    return "since" in parameters',
        "    return True",
        ("tests.unit.test_task_lifecycle_guards",),
        "every store is assumed to accept the event cursor (a store that does "
        "not is called with a keyword it does not know)",
    ),
    Mutant(
        "src/ai_assistant/application/tasks.py",
        "        if store_supports_cursor(store_events):",
        "        if False:",
        ("tests.unit.test_task_lifecycle_guards",),
        "the event cursor is never asked for (the whole log is re-read on "
        "every tick)",
    ),
    Mutant(
        "src/ai_assistant/adapters/http/chat_routes.py",
        "        if not stat.S_ISREG(os.stat(path).st_mode):",
        "        if False:",
        ("tests.unit.test_http_chat_routes",),
        "a client-named path is opened even when it is not a regular file "
        "(a FIFO blocks the worker thread)",
    ),
    # ── The corpus is a second reader: one rule, one owner ───────────────────
    Mutant(
        "src/ai_assistant/rag/index.py",
        "            if not _stays_inside(root, path):",
        "            if False:",
        ("tests.unit.test_rag_corpus_boundary",),
        "the corpus follows a link out of the tree (a file outside the "
        "project is indexed and reaches the model)",
    ),
    Mutant(
        "src/ai_assistant/rag/index.py",
        "            if is_secret_name(filename):",
        "            if False:",
        ("tests.unit.test_rag_corpus_boundary",),
        "the corpus reads a file whose name is a secret name",
    ),
    Mutant(
        "src/ai_assistant/rag/index.py",
        "    return is_blocked_dir(name) or name in EXCLUDED_DIRS",
        "    return name in EXCLUDED_DIRS",
        ("tests.unit.test_rag_corpus_boundary",),
        "the corpus keeps its own directory list and drops the shared "
        "blocked-directory rule",
    ),
    Mutant(
        "src/ai_assistant/rag/index.py",
        "    for fp in _indexable_files(paths.project_root, SCANNABLE_EXTS, exclude=docs_dirs):",
        "    for fp in _indexable_files(paths.project_root, SCANNABLE_EXTS):",
        ("tests.unit.test_rag_corpus_boundary",),
        "the docs directories are walked twice (an image under Docs/ is "
        "indexed twice)",
    ),
    Mutant(
        "src/ai_assistant/rag/index.py",
        "        for path in _indexable_files(docs_dir, DOC_EXTS):",
        "        for path in ():",
        ("tests.unit.test_rag_corpus_boundary",),
        "the cache hash does not cover the documents (a changed document does "
        "not invalidate the cache)",
    ),
    # ── One catalogue, one policy: plugin tools and fail-closed bridges ──────
    Mutant(
        "src/ai_assistant/bootstrap/container.py",
        "    if name != raw_name:",
        "    if False:",
        ("tests.unit.test_gateway_bootstrap_guards",),
        "an alias is converted into an execution spec, so one desktop action "
        "gets a second (read-only) policy row",
    ),
    Mutant(
        "src/ai_assistant/bootstrap/container.py",
        "        if entry.spec.plugin_id == BUILTIN_PLUGIN_ID:\n            continue",
        "        if False:\n            continue",
        ("tests.unit.test_plugin_tool_bridge",),
        "builtin tools are merged as plugin tools (the dynamic plugin row can "
        "grant a builtin the builtin specialists own)",
    ),
    Mutant(
        "src/ai_assistant/orchestration/supervisor.py",
        "    if tool is not None and tool in plugin_tools:",
        "    if False:",
        ("tests.unit.test_plugin_tool_bridge",),
        "a plugin tool falls through to supervisor_fallback and is refused "
        "even when the composition root published it",
    ),
    Mutant(
        "src/ai_assistant/orchestration/supervisor.py",
        "    policy = ALLOWED_TOOLS if allowed_tools is None else allowed_tools",
        "    policy = ALLOWED_TOOLS",
        ("tests.unit.test_plugin_tool_bridge",),
        "the dynamic plugin row is ignored by authorise (the plugin tool is "
        "visible to the model but can never execute)",
    ),
    Mutant(
        "src/ai_assistant/plugins/registry.py",
        "        self._allowed_plugins = frozenset(allowed_plugins) | {BUILTIN_PLUGIN_ID}",
        "        self._allowed_plugins = frozenset(allowed_plugins)",
        ("tests.unit.test_plugins",),
        "the builtin catalogue follows the third-party allow-list (an operator "
        "config can disable the product's own tools)",
    ),
    Mutant(
        "src/ai_assistant/plugins/registry.py",
        "        if self._active_plugin_id is not None and spec.plugin_id != self._active_plugin_id:",
        "        if False:",
        ("tests.unit.test_plugins",),
        "a plugin borrows another enabled plugin's id during registration "
        "(provenance is not bound to the loaded plugin)",
    ),
    Mutant(
        "src/ai_assistant/plugins/registry.py",
        "            if spec.side_effect in {SideEffect.WRITE, SideEffect.EXECUTE} and not spec.requires_approval:",
        "            if False:",
        ("tests.unit.test_plugins",),
        "a mutating plugin tool is registered without approval",
    ),
    Mutant(
        "src/ai_assistant/agents/session_locks.py",
        "                if entry.users == 0:\n                    self._entries.pop(session_id, None)",
        "                if False:\n                    self._entries.pop(session_id, None)",
        ("tests.unit.test_session_serialization",),
        "the session lock entry is never released (the process retains one "
        "lock per session id it has ever seen)",
    ),
    Mutant(
        "src/ai_assistant/agents/models.py",
        "    session_id:          str   = Field(default=\"\", max_length=128)",
        "    session_id:          str   = Field(default=\"\")",
        ("tests.unit.test_session_serialization",),
        "the execute session id is unbounded (the lock pool and checkpointer "
        "thread id retain an arbitrarily large key)",
    ),
    Mutant(
        "src/ai_assistant/adapters/mcp.py",
        "    if getattr(spec, \"requires_approval\", False):\n        return False",
        "    if False:\n        return False",
        ("tests.unit.test_mcp_adapter",),
        "MCP advertises an approval-gated tool it has no HITL flow to approve",
    ),
    Mutant(
        "src/ai_assistant/adapters/mcp.py",
        "    if canonical_tool_name(str(spec.name)) != spec.name:\n        return False",
        "    if False:\n        return False",
        ("tests.unit.test_mcp_adapter",),
        "an alias is advertised as a second MCP tool with no execution policy",
    ),
    # ── LLM egress and artifact integrity ───────────────────────────────────
    Mutant(
        "src/ai_assistant/llm/inference.py",
        '    return policy == "cloud_allowed" and data_classification.casefold() in {"public", "non_sensitive"}',
        '    return policy == "cloud_allowed" or data_classification.casefold() in {"public", "non_sensitive"}',
        ("tests.unit.test_inference",),
        "the privacy gate ORs the policy with the classification, so "
        "local_only still routes public data to a remote model",
    ),
    Mutant(
        "src/ai_assistant/llm/inference.py",
        '    if not cloud_allowed("non_sensitive"):',
        "    if False:",
        ("tests.unit.test_inference",),
        "remote completion skips the cloud-policy gate (bytes leave before "
        "the operator opted in)",
    ),
    Mutant(
        "src/ai_assistant/llm/inference.py",
        "        if len(body) > _MAX_RESPONSE_BYTES:",
        "        if False:",
        ("tests.unit.test_inference",),
        "the remote response size bound is removed (a degraded peer can make "
        "the client buffer without limit)",
    ),
    Mutant(
        "src/ai_assistant/llm/model_loader.py",
        "    if not expected_sha256:\n        return True",
        "    if False:\n        return True",
        ("tests.unit.test_llm_backend",),
        "an unpinned artifact is compared against the empty string and can "
        "never load (or, with the inverse policy, is treated as pinned)",
    ),
    Mutant(
        "src/ai_assistant/llm/model_loader.py",
        "        return _sha256_of(path).casefold() == expected_sha256.casefold()",
        "        return _sha256_of(path).casefold() == expected_sha256",
        ("tests.unit.test_llm_backend",),
        "the pinned SHA-256 comparison becomes case-sensitive (an uppercase "
        "pin rejects the same artifact)",
    ),
    Mutant(
        "src/ai_assistant/llm/model_loader.py",
        "        if _matches_pin(path, sha256):\n            return path",
        "        if False:\n            return path",
        ("tests.unit.test_llm_backend",),
        "an already-downloaded pinned artifact is discarded and downloaded "
        "again on every load",
    ),
    Mutant(
        "src/ai_assistant/llm/model_loader.py",
        "    if not _matches_pin(path, sha256):",
        "    if False:",
        ("tests.unit.test_llm_backend",),
        "a post-download hash mismatch loads unverified weights instead of "
        "raising ModelIntegrityError",
    ),
    Mutant(
        "src/ai_assistant/llm/model_loader.py",
        "    if model.mmproj_repo_id == model.repo_id:",
        "    if model.mmproj_repo_id != model.repo_id:",
        ("tests.unit.test_llm_backend",),
        "the vision projector inherits the model revision across repos (the "
        "wrong commit is downloaded) and stays unpinned in its own repo",
    ),
    # ── The internal framework boundary ─────────────────────────────────────
    Mutant(
        "src/ai_assistant/internal/langgraph_engine.py",
        "        return CompiledGraph(self._builder.compile(checkpointer=checkpointer))",
        "        return CompiledGraph(self._builder.compile())",
        ("tests.unit.test_langgraph_engine_wrapper",),
        "compile drops the checkpointer (resumable runs become stateless)",
    ),
    Mutant(
        "src/ai_assistant/internal/langgraph_engine.py",
        "        return cast(StateT, self._graph.invoke(state, config=config))",
        "        return cast(StateT, self._graph.invoke(state, config={}))",
        ("tests.unit.test_langgraph_engine_wrapper",),
        "invoke drops the caller's config (thread id and checkpoint settings "
        "never reach the compiled graph)",
    ),
    Mutant(
        "src/ai_assistant/internal/langgraph_engine.py",
        "        return self._graph.get_state(config)",
        "        return None",
        ("tests.unit.test_langgraph_engine_wrapper",),
        "get_state stops delegating (resume reads no checkpoint)",
    ),
    # ── One model lease per entry point that runs the graph (ADR 0040) ───────
    Mutant(
        "src/ai_assistant/agents/service.py",
        "                with self._model_lease():\n"
        "                    return self._execute_locked(",
        "                if True:\n"
        "                    return self._execute_locked(",
        ("tests.unit.test_agent_service_guards",),
        "the fresh run executes without a lease (a reload frees the model "
        "under a running graph)",
    ),
    Mutant(
        "src/ai_assistant/agents/service.py",
        "        with self._hold_session(request.session_id):\n"
        "            slots = self._acquire_run_slot()\n"
        "            try:\n"
        "                with self._model_lease():\n"
        "                    return self._approval_service.approve(request)\n"
        "            finally:\n"
        "                self._release_run_slot(slots)",
        "        with self._hold_session(request.session_id):\n"
        "            slots = self._acquire_run_slot()\n"
        "            try:\n"
        "                if True:\n"
        "                    return self._approval_service.approve(request)\n"
        "            finally:\n"
        "                self._release_run_slot(slots)",
        ("tests.unit.test_agent_service_guards",),
        "the approval resume executes without a lease (only fresh runs block "
        "the reload)",
    ),
    Mutant(
        "src/ai_assistant/agents/service.py",
        "        with self._hold_session(str(parked.get(\"session_id\", \"\"))):\n"
        "            slots = self._acquire_run_slot()\n"
        "            try:\n"
        "                with self._model_lease():\n"
        "                    return self._ui_action_service.ui_action_result(request)\n"
        "            finally:\n"
        "                self._release_run_slot(slots)",
        "        with self._hold_session(str(parked.get(\"session_id\", \"\"))):\n"
        "            slots = self._acquire_run_slot()\n"
        "            try:\n"
        "                if True:\n"
        "                    return self._ui_action_service.ui_action_result(request)\n"
        "            finally:\n"
        "                self._release_run_slot(slots)",
        ("tests.unit.test_agent_service_guards",),
        "the desktop-ack resume executes without a lease (only fresh runs "
        "block the reload)",
    ),
    Mutant(
        "src/ai_assistant/adapters/http/chat_routes.py",
        "        with llm_module.model_lease() as backend:\n"
        "            if backend is None:",
        "        if True:\n"
        "            backend = llm_module.llm\n"
        "            if backend is None:",
        ("tests.unit.test_http_chat_routes",),
        "chat inference runs outside the model lease (a reload frees the "
        "backend under the stream)",
    ),
    # ── One owner for the model, and a release that really releases (ADR 0041) ─
    Mutant(
        "src/ai_assistant/bootstrap/modules.py",
        "        had_backend = self.llm is not None\n        self.llm = None",
        "        had_backend = self.llm is not None\n        pass",
        ("tests.unit.test_bootstrap_coverage_gaps",),
        "release keeps the backend reference (VRAM stays resident while the "
        "route returns ok, the exact defect the single owner removed)",
    ),
    Mutant(
        "src/ai_assistant/bootstrap/modules.py",
        "            # Free the previous weights before the new download/constructor:\n"
        "            # two models must never be resident at once on a GPU.\n"
        "            with self.llm_lock:\n"
        "                had_backend = self._clear_backend_locked()\n"
        "            if had_backend:\n"
        "                release_ml_memory()",
        "            # Free the previous weights before the new download/constructor:\n"
        "            # two models must never be resident at once on a GPU.\n"
        "            with self.llm_lock:\n"
        "                had_backend = self._clear_backend_locked()\n"
        "            if False:\n"
        "                release_ml_memory()",
        ("tests.unit.test_bootstrap_coverage_gaps",),
        "a reload drops the old reference without asking the runtime to free "
        "it (both models stay resident during the swap)",
    ),
    Mutant(
        "src/ai_assistant/adapters/http/health_routes.py",
        "        if llm_module.llm is None:\n            return {\"data\": []}",
        "        if False:\n            return {\"data\": []}",
        ("tests.unit.test_http_health_routes",),
        "a released model is still advertised by /v1/models (clients route "
        "requests to weights that no longer exist)",
    ),
    Mutant(
        "src/ai_assistant/adapters/http/health_routes.py",
        "            \"model\": llm_module.active_model_desc if loaded else \"\",",
        "            \"model\": llm_module.active_model_desc,",
        ("tests.unit.test_http_health_routes",),
        "/health advertises the previous model after release (one endpoint "
        "says unloaded while the other names the model)",
    ),
    # ── Admission happens before the worker thread exists (ADR 0041) ─────────
    Mutant(
        "src/ai_assistant/application/tasks.py",
        "        if not self._try_acquire_slot():\n"
        "            return self._reject_at_capacity(task)",
        "        if False:\n"
        "            return self._reject_at_capacity(task)",
        ("tests.unit.test_task_lifecycle_guards",),
        "a submit at capacity creates a worker anyway (the burst pays for the "
        "thread before the executor answers 503)",
    ),
    Mutant(
        "src/ai_assistant/application/tasks.py",
        "        if not self._try_acquire_slot():\n"
        "            raise TaskCapacityError(",
        "        if False:\n"
        "            raise TaskCapacityError(",
        ("tests.unit.test_task_lifecycle_guards",),
        "a resume at capacity starts a second worker (the admission bound "
        "covers submits but not resumes)",
    ),
    Mutant(
        "src/ai_assistant/application/tasks.py",
        "                self._release_slot()\n"
        "                with self._worker_lock:",
        "                if False:\n"
        "                    self._release_slot()\n"
        "                with self._worker_lock:",
        ("tests.unit.test_task_lifecycle_guards",),
        "a finished worker never returns its slot (capacity is permanently "
        "lost until restart)",
    ),
    Mutant(
        "src/ai_assistant/adapters/a2a_router.py",
        "    except TaskCapacityError as error:\n"
        "        raise HTTPException(\n"
        "            status_code=429, detail=str(error), headers={\"Retry-After\": \"1\"},\n"
        "        ) from error",
        "    except TaskCapacityError as error:\n"
        "        raise HTTPException(\n"
        "            status_code=409, detail=str(error), headers={\"Retry-After\": \"1\"},\n"
        "        ) from error",
        ("tests.unit.test_a2a_http_router",),
        "capacity exhaustion is reported as a permanent conflict instead of "
        "retryable backpressure",
    ),
    Mutant(
        "src/ai_assistant/adapters/a2a_router.py",
        "            return {\"jsonrpc\": \"2.0\", \"id\": request_id,\n"
        "                    \"error\": {\"code\": -32000, \"message\": str(error)}}",
        "            return {\"jsonrpc\": \"2.0\", \"id\": request_id,\n"
        "                    \"error\": {\"code\": -32602, \"message\": str(error)}}",
        ("tests.unit.test_a2a_http_router",),
        "a capacity refusal is reported as invalid params (the client retries "
        "the request instead of backing off)",
    ),
    # ── The tenth review: owner-enforced swaps, fail-closed defaults, public
    #    RAG surface and a graceful stop (ADR 0042) ────────────────────────────
    Mutant(
        "src/ai_assistant/bootstrap/modules.py",
        "        with self._model_condition:\n"
        "            if not self._reloading:",
        "        with self._model_condition:\n"
        "            if False:",
        ("tests.unit.test_bootstrap_coverage_gaps",),
        "load/release is callable outside model_reload(): a route can free "
        "the backend while a run holds a lease",
    ),
    Mutant(
        "src/ai_assistant/bootstrap/server.py",
        "    with runtime_modules.llm.model_reload():\n"
        "        runtime_modules.llm.load_model()",
        "    runtime_modules.llm.load_model()",
        ("tests.integration.test_composition_root_bootstrap",),
        "startup loads the model outside the drain window (the owner refuses "
        "it; bootstrap regresses to a direct primitive call)",
    ),
    Mutant(
        "src/ai_assistant/domain/security.py",
        "    if capability_scopes is None:\n"
        "        return READ_ONLY_SCOPES",
        "    if capability_scopes is None:\n"
        "        return frozenset({\"project.read\", \"project.write\", \"project.execute\"})",
        ("tests.unit.test_final_edges_a",),
        "a host that injects no scope map grants the trusted write/execute "
        "scopes to every embedded caller",
    ),
    Mutant(
        "src/ai_assistant/bootstrap/container.py",
        "    if policy_value is None:\n"
        "        # A missing label is not \"read-only\": a mutating tool whose policy was\n"
        "        # never authored would otherwise inherit the weakest scope and could\n"
        "        # be driven by a read-only principal. Unknown labels already fail in\n"
        "        # ``_side_effect``; missing ones must fail here for the same reason.\n"
        "        msg = f\"Tool {name!r} has no execution policy\"\n"
        "        raise ValueError(msg)",
        "    if policy_value is None:\n"
        "        policy_value = \"read_only\"",
        ("tests.unit.test_gateway_bootstrap_guards",),
        "a tool with no authored policy silently becomes read-only (a "
        "mutating tool inherits the weakest scope)",
    ),
    Mutant(
        "src/ai_assistant/tools/factory.py",
        "    if not config_file.exists():\n"
        "        # A server that starts without a catalogue advertises itself as ready\n"
        "        # but can execute nothing: the agent has no tools, and every request\n"
        "        # fails far from the cause. A missing required catalogue is a\n"
        "        # configuration error, not an empty-but-valid registry.\n"
        "        msg = f\"Tool catalogue not found: {config_file}\"\n"
        "        raise ConfigurationError(msg)",
        "    if not config_file.exists():\n"
        "        return registry",
        ("tests.unit.test_final_edges_b",),
        "a missing tool catalogue degrades into an empty registry (the server "
        "reports ready while it can execute nothing)",
    ),
    Mutant(
        "src/ai_assistant/bootstrap/modules.py",
        "        self.shutdown_requested.set()",
        "        pass",
        ("tests.unit.test_bootstrap_coverage_gaps",),
        "/admin/shutdown never signals the serving process (the stop is "
        "requested but the event stays clear)",
    ),
    Mutant(
        "src/ai_assistant/bootstrap/server.py",
        "    shutdown_requested.wait()\n"
        "    server.should_exit = True",
        "    shutdown_requested.wait()\n"
        "    server.should_exit = False",
        ("tests.integration.test_main_entry_points",),
        "the shutdown watcher never asks uvicorn to stop (graceful exit "
        "silently becomes a no-op)",
    ),
    Mutant(
        "src/ai_assistant/adapters/http/admin_routes.py",
        "            request_shutdown()",
        "            pass",
        ("tests.unit.test_http_admin_routes",),
        "/admin/shutdown answers ok without asking the process to stop",
    ),
    Mutant(
        "src/ai_assistant/bootstrap/modules.py",
        "        return self._reranker is not None",
        "        return True",
        ("tests.unit.test_bootstrap_coverage_gaps",),
        "the public reranker fact always reports active (health lies about the "
        "retrieval pipeline)",
    ),
    Mutant(
        "src/ai_assistant/adapters/http/health_routes.py",
        "            \"reranker\": rag_module.reranker_active,",
        "            \"reranker\": False,",
        ("tests.unit.test_http_health_routes",),
        "/health never reports an active reranker",
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


def clear_bytecode_cache(target: Path) -> None:
    """Drop the cached bytecode of the file the gate is about to mutate.

    Python validates a ``.pyc`` by the source's **mtime (whole seconds)** and
    its **size**. A mutation that keeps the size — ``"1"`` → ``"0"`` — and is
    restored within the same second therefore leaves a *stale* bytecode cache
    that still holds the mutated code: the next import loads the mutant, and
    a test that should have failed passes. The symptom is a mutant that
    "survived" for no reason, and — worse — a tree that keeps mutated code
    after the gate is done.

    The gate must not leave mutated code behind, neither on disk nor in the
    bytecode cache.
    """
    cache = target.parent / "__pycache__"
    if not cache.is_dir():
        return
    for cached in cache.glob(f"{target.stem}.*.pyc"):
        try:
            cached.unlink()
        except OSError:
            pass


def subprocess_env() -> dict[str, str]:
    """The environment the mutant's suite runs in.

    ``PYTHONDONTWRITEBYTECODE=1`` is not a nicety: the suite must not *write*
    a ``.pyc`` of the mutated module, and the cache is dropped before the
    run, so the import in the test process is the mutated source and not a
    stale bytecode of the original.
    """
    return {**os.environ, "PYTHONPATH": "src", "PYTHONDONTWRITEBYTECODE": "1"}


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
        # The mutated module must be the one the suite imports: a stale
        # ``.pyc`` (same size, same second) would import the *original* code
        # and report the mutant as survived. The cache is dropped and the
        # suite runs without writing bytecode, so the mutation is what is
        # measured and nothing mutated is left behind.
        clear_bytecode_cache(target)
        env = subprocess_env()
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
        clear_bytecode_cache(target)
        if target.read_text(encoding="utf-8") != original:
            message = (
                f"{mutant.file} was not restored byte-for-byte; restore it "
                "before running anything against this tree."
            )
            raise RuntimeError(message)
    if outcome.returncode == 0:
        # A survivor is the most expensive kind of answer: it means the suite
        # did not notice. Print *what the suite did* — a suite that ran zero
        # tests, or one that never imported the mutated module, looks exactly
        # like a suite that ran and agreed with the mutation.
        print(f"    (the suite stayed green: {outcome.stdout.strip()[-600:]!r})")
        return "SURVIVED"
    return "KILLED"


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
