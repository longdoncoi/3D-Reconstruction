# ADR 0017: complexity gate at 15, TRY/N lint breadth, boost files split

Date: 2026-10-12

Status: Accepted

## Context

ADR 0016 closed the coverage climb at 100/100. Three structural items from
the 10/10 checklist remained: the complexity gate still sat at 20 with seven
functions between 16 and 20, `test_coverage_boost*.py` held 1377 lines under
four batch names, and ruff stopped at `E/W/F/I/BLE/S/RUF/C90`.

## Decisions

1. **Complexity gate 20 → 15.** Six functions were decomposed along existing
   seams, each verified by the untouched suite at 100% before and after:
   - `build_chat_router`: attachment scan → `_scan_attachments_for_image`.
   - `run_langgraph_agent`: grant probe → `_approval_cover_probe` (kept the
     `gateway` local name the boundary test pins).
   - `plan_node`: prior-plan reuse/preserve/skip → `_reuse_prior_plan`.
   - `constrained_completion`: transport response decode →
     `_record_and_decode_content` (also removed a duplicated decode).
   - `tool_node`: RAG synthesis, completion announcement and the two
     verified-reflection shortcuts → three helpers, the two reflection arms
     unified behind a membership flag.
   - `build_index_from_scratch`: image batches → `_embed_image_batches`.
   - `tool_list_directory`: recursive walk → `_walk_directory_recursive`.
   The TRY-driven `try/else` rewrite pushed `build_admin_router` over, so
   `reload_rag` moved to `_rebuild_rag_index` as well.
2. **Boost files split verbatim.** Each `TestCase` moved whole into the topic
   file that owns its module (`test_llm_backend.py`, `test_a2a_client.py`,
   `test_task_lifecycle_guards.py`, `test_agent_service_guards.py`,
   `test_completion_envelopes.py`, …); no batch file remains. Coverage held
   at 100/100 throughout — moves were verified per file.
3. **Lint select gains `N` and `TRY`.** `N` violations were all external-API
   mirrors (A2A wire fields, faiss doubles, the entry-point filename) and
   carry reasoned `# noqa`s. `TRY300/301/400/401` were applied mechanically;
   `TRY003`'s 60 message hoists were done by AST script and verified by the
   message-asserting tests. `TRY004` stays `noqa`d where the exception type
   is contract: `discover()` catches `ValueError`, the decoders raise
   `RuntimeError`.
4. **Deferred with numbers, not silence.** `ANN` (1383), `D` (1807) and
   `PTH` (139) remain off: full annotation, full docstring coverage and the
   `os.path`→`pathlib` migration are each a project of their own, and
   `ANN401`/`PTH109`-class rules would fight real constraints
   (`Any` at untyped boundaries, `os.path.relpath` cross-drive semantics).
   They are future work, not debt — nothing is suppressed to pretend
   otherwise.

## Consequences

- `max-complexity = 15` and `select` including `N`/`TRY` are enforced by
  `ruff check .` in CI next to the coverage gate.
- `ARCHITECTURE.md` documents the lint set and the split.
