# AIAssistant — Nền tảng AI Agent cục bộ

Máy chủ AI phía sau plugin Assistant của ứng dụng Qt: một nền tảng agent theo
kiểu **policy-first**, phục vụ HTTP tại `127.0.0.1:8080` theo mặc định. Nó chạy
một LLM **cục bộ** qua `llama-cpp-python`, một chỉ mục RAG trên tài liệu dự án,
vòng lặp agent ReAct với các lời gọi công cụ bị chặn bởi chính sách phê duyệt
(HITL), cùng các bề mặt MCP và A2A để những tiến trình khác tiếp cận đúng lõi
thực thi đó. Không gì rời khỏi máy của bạn trừ khi một chính sách cho phép.

File README này là **tài liệu tổng** của thư mục: nó thay cho `ARCHITECTURE.md`
và `INDEX.md` trước đây (đã được gộp vào đây và xoá), nên mọi chi tiết — kiến
trúc, quy tắc runtime, hợp đồng API, lệnh kiểm thử, mục lục ADR — đều nằm trong
một chỗ duy nhất.

## Layout

Toàn bộ source của máy chủ nằm trong `src/ai_assistant/` (package production
duy nhất); `config/` cấp thư mục này là **hồ sơ triển khai** TOML, không phải
package mã nguồn.

```text
AIAssistant/
├─ src/ai_assistant/                  # ── PACKAGE SẢN PHẨM DUY NHẤT ──
│  │
│  ├─ __init__.py                     # Ghi chú package + version; import-light
│  ├─ ports.py                        # Shared kernel: Protocol của toàn hệ thống
│  │                                  #   (LLMBackend, ToolExecutionGateway, ToolGateway,
│  │                                  #    AgentOrchestrator, TaskStore, LLMRuntime,
│  │                                  #    ApprovalGrantIssuer, RagContextPort,
│  │                                  #    PendingActionPort, AgentTaskExecutor, LockLike)
│  ├─ settings.py                     # CHỦ SỞ HỮU duy nhất của môi trường process (ADR 0001/0014):
│  │                                  #   đọc .env, áp mặc định, ArchitectureSettings,
│  │                                  #   TransportTokens (AI_*_TOKEN), agent_bind_host/port
│  ├─ exec_policy.py                  # Kernel policy cho executable/argv được phép
│  ├─ path_policy.py                  # Kernel policy chuẩn hoá và giới hạn đường dẫn
│  ├─ image_utils.py                  # Trợ giúp ảnh chung cho hai bên vision và retrieval (module lá)
│  │
│  ├─ domain/                         # ── LỚP DOMAIN: thuần, không import framework nào ──
│  │  ├─ __init__.py                  # Hợp đồng domain framework-independent
│  │  ├─ agents.py                    # AgentDecisionKind + AgentDecision (kết quả điều phối)
│  │  ├─ approvals.py                 # Chính sách cấp phép HITL: grant gắn với đúng một lời gọi công cụ
│  │  ├─ context_budget.py            # Chính sách ngân sách context-window (CONTEXT_TOKEN_MARGIN…)
│  │  ├─ errors.py                    # Exception có kiểu của nền tảng
│  │  ├─ governance.py                # Quản trị dữ liệu: scrub PII, redact_value, SENSITIVE_KEYS
│  │  ├─ memory.py                    # Value/policy cho semantic, episodic, procedural memory
│  │  ├─ parallel.py                  # Hợp đồng nhóm bước có thể fan-out an toàn
│  │  ├─ pending.py                   # Chính sách chung cho các mục phê duyệt/hành động đang treo
│  │  ├─ security.py                  # DataClassification, Principal, current_principal, bind_principal
│  │  ├─ specialists.py               # Từ vựng chuyên gia: TRANSFER_TARGETS (một nguồn đặt tên duy nhất)
│  │  ├─ steps.py                     # Từ vựng log bước quan sát được (ADR 0019): Step/StepType
│  │  ├─ tasks.py                     # TaskStatus + AgentTask (chuẩn bị cho A2A)
│  │  └─ tools.py                     # ToolSpec + ToolRequest/ToolResult, SideEffect, tiêu chí "kết quả thành công"
│  │
│  ├─ application/                    # ── LỚP USE-CASE: điều phối domain qua port ──
│  │  ├─ __init__.py                  # Use-case layer
│  │  ├─ agent_runs.py                # Vòng đời task A2A bền vững trên engine điều phối hoán đổi được
│  │  ├─ approvals.py                 # Grant dùng một lần backing tool gateway (ADR 0009)
│  │  ├─ coordination.py              # TaskRecord + TaskCoordinator (vòng đời & đo thời gian hủy)
│  │  ├─ memory.py                    # MemoryService: recall/consolidate qua port
│  │  ├─ tasks.py                     # TaskService: chạy task trong worker thread, snapshot contextvars
│  │  ├─ tools.py                     # ToolExecutionService — CỔNG THỰC THI VÀ CHÍNH SÁCH DUY NHẤT
│  │  └─ validation.py                # Validator JSON-Schema con tối giản ở mọi biên giới protocol công cụ
│  │
│  ├─ agents/                         # ── LỚP AGENT: dịch vụ agent + vòng lặp runner ──
│  │  ├─ __init__.py                  # Agents package
│  │  ├─ service.py                   # AgentService (facade): execute/cancel/approve/ui_action_result,
│  │  │                               #   khóa RLock theo session cho toàn bộ lần chạy
│  │  ├─ approval_service.py          # ApprovalService: vòng đời phê duyệt HITL + resume LangGraph
│  │  ├─ ui_action_service.py         # UIActionService: tiếp tục hành động desktop sau ACK của Qt
│  │  ├─ runner.py                    # runner.run_langgraph_agent — vòng lặp quyết định sản xuất duy nhất
│  │  ├─ graph_wiring.py              # build_run_graph: callback + lời gọi local_agent_graph (ADR 0013)
│  │  ├─ response.py                  # Ánh xạ kết quả + vòng đời trace (pending/completed response)
│  │  ├─ context.py                   # AgentRunContext: bó wiring dùng chung cho 3 service (ADR 0012)
│  │  ├─ deps.py                      # RunDeps: dataclass đóng băng các dependency inject (ADR 0013)
│  │  ├─ completion.py                # Adapter completion: constrained/structured completion + grammar cache
│  │  ├─ catalog.py                   # Catalogue specialist/tool được inject
│  │  ├─ prompts.py                   # Build system prompt — không gọi LLM, không I/O
│  │  ├─ models.py                    # AgentExecuteRequest/AgentApproveRequest/… (schema Pydantic)
│  │  ├─ ids.py                       # Trợ giúp định danh dùng chung cho lớp agent
│  │  ├─ session_locks.py             # RLock bounded theo session
│  │  ├─ short_term_memory.py         # Buffer hội thoại bounded theo session
│  │  └─ pending_store.py             # MemoryPendingStore — mặc định của chính lớp agent (ADR 0003)
│  │
│  ├─ orchestration/                  # ── LỚP ORCHESTRATION: đồ thị LangGraph ──
│  │  ├─ __init__.py                  # Re-export AgentState; import vào đây KHÔNG kéo theo graph
│  │  ├─ graph.py                     # Dây đồ thị: node, edge, entry point (truyền AgentState làm constructor arg)
│  │  ├─ state.py                     # AgentState TypedDict + CONTEXT_EXHAUSTED_MESSAGE + has_step_type()
│  │  ├─ supervisor.py                # Supervisor: delegate/authorise/verify_result (bảng _TOOL_ROUTE_RULES)
│  │  ├─ prompts.py                   # System prompt của Planner, Critic, Reasoner
│  │  ├─ helpers.py                   # Trợ giúp dùng chung cho các node
│  │  ├─ nodes/
│  │  │  ├─ __init__.py               # Node sub-package
│  │  │  ├─ plan.py                   # Node plan — sinh kế hoạch JSON cho yêu cầu dài
│  │  │  ├─ plan_reflect.py           # Node plan_reflect — phản biện sau giai đoạn plan
│  │  │  ├─ reason.py                 # Node reason — dispatch ReAct (các handler theo từng kết cục)
│  │  │  ├─ reason_prepare.py         # Chuẩn bị context cho node reason (gói gọn prompt)
│  │  │  ├─ reflect.py                # Node reflect — người phản biện độc lập, tất định
│  │  │  ├─ tool.py                   # Node tool — điều phối gọi công cụ, đếm lỗi, circuit breaker
│  │  │  └─ summarize.py              # Node summarize — cô đọng quan sát trước khi quay lại reason
│  │  └─ specialists/
│  │     ├─ __init__.py               # Specialist agents package
│  │     ├─ chatbot.py                # Chuyên gia hội thoại: lịch sử chat + context RAG tuỳ chọn
│  │     └─ code.py                   # Chuyên gia code: chính sách & context cho nhiệm vụ repo
│  │
│  ├─ tools/                          # ── LỚP TOOLS: danh mục công cụ (không thực thi) ──
│  │  ├─ __init__.py                  # Tools package
│  │  ├─ registry.py                  # ToolRegistry — DANH MỤC, không thể thực thi (ADR 0003)
│  │  ├─ definition.py                # ToolDefinition — mục danh mục có thể thực thi
│  │  ├─ factory.py                   # Xây registry từ cấu hình + handler đăng ký
│  │  ├─ schema.py                    # Đọc schema công cụ + sinh grammar cho constrained decoding
│  │  ├─ validation.py                # Validate đầu vào công cụ theo schema tại gateway
│  │  ├─ tool_contract.py             # Ánh xạ và tương thích hợp đồng công cụ
│  │  ├─ sandbox.py                   # Sandbox policy-first cho công cụ phá huỷ (run_command…)
│  │  ├─ verified_write.py            # Commit write/create qua directory handle đã xác minh
│  │  ├─ action_manifest.py           # Hợp đồng hành động desktop có phiên bản (chia sẻ với C++)
│  │  ├─ aliases.py                   # Alias tên tool tương thích
│  │  ├─ lsp_client.py                # Client LSP cho công cụ của Coding Agent
│  │  └─ builtin/
│  │     ├─ __init__.py               # builtin tools package
│  │     ├─ file_tools.py             # Đọc/ghi file
│  │     ├─ code_tools.py             # Phân tích cấu trúc & mã nguồn
│  │     ├─ desktop_tools.py          # Hành động desktop (điều khiển UI Qt)
│  │     ├─ rag_tools.py              # Công cụ truy vấn RAG (rag_search)
│  │     ├─ search_tools.py           # Tìm kiếm/khám phá trong repo
│  │     ├─ system_tools.py           # Thực thi hệ thống qua sandbox
│  │     ├─ transfer_tools.py         # Chuyển giao specialist (transfer_to_*)
│  │     ├─ vcs_tools.py              # Công cụ VCS / trạng thái dự án (git_diff…)
│  │     └─ path_utils.py             # An toàn đường dẫn dùng chung cho builtin tools
│  │
│  ├─ rag/                            # ── LỚP RAG: nạp tài liệu, nhúng, chỉ mục, truy xuất ──
│  │  ├─ __init__.py                  # RAG package
│  │  ├─ loaders.py                   # Loader tài liệu (md/docx/pdf/email…) + chunking ngữ nghĩa
│  │  ├─ embeddings.py                # Adapter mô hình nhúng (huggingface, image-aware)
│  │  ├─ index.py                     # Xây/dựng cache/lưu chỉ mục FAISS + BM25
│  │  ├─ retrievers.py                # Retriever sản xuất trên FAISS đã cache + bảng BM25
│  │  ├─ retrieval.py                 # Pipeline truy xuất + định dạng context
│  │  ├─ cache_seal.py                # Định dạng cache RAG đã niêm phong (ADR 0028)
│  │  ├─ domain.py                    # Model domain của RAG (chunk, hit…)
│  │  └─ nlp_utils.py                 # Tiện ích NLP phục vụ RAG
│  │
│  ├─ llm/                            # ── LỚP LLM: backend cục bộ & điều khiển mô hình ──
│  │  ├─ __init__.py                  # LLM orchestration + định dạng prompt
│  │  ├─ contracts.py                 # Hợp đồng LLMBackend/LockLike (re-export từ ports)
│  │  ├─ llama_cpp_backend.py         # Backend llama.cpp (load model, create_completion)
│  │  ├─ model_loader.py              # Tải mô hình theo registry + download_if_missing
│  │  ├─ hardware.py                  # Quản lý phần cứng/bộ nhớ (giải phóng VRAM)
│  │  ├─ inference.py                 # Chọn backend inference + completion từ xa
│  │  ├─ defaults.py                  # Mặc định runtime LLM dùng chung (temperature, max_tokens)
│  │  └─ prompts.py                   # Xây prompt + quản lý lịch sử hội thoại
│  │
│  ├─ observability/                  # ── LỚP OBSERVABILITY: log sự kiện, metrics, trace ──
│  │  ├─ __init__.py                  # Observability package
│  │  ├─ tool_events.py               # Dòng audit cho cổng thực thi duy nhất (logger `agent.tools`)
│  │  ├─ metrics.py                   # Prometheus: agent_recovery_seconds, p95_latency_by_node()
│  │  ├─ tracing.py                   # OpenTelemetry: span(), traceparent, join trace từ xa
│  │  └─ langsmith_integration.py     # Tích hợp LangSmith (tùy chọn)
│  │
│  ├─ adapters/                       # ── LỚP ADAPTERS: cách ly protocol ──
│  │  ├─ __init__.py                  # Triển khai port của application layer
│  │  ├─ a2a.py                       # Re-export tương thích ngược (ADR 0005)
│  │  ├─ a2a_models.py                # Schema request/response A2A 0.3 (Pydantic)
│  │  ├─ a2a_payloads.py              # Build payload, map state, redaction dữ liệu nhạy cảm
│  │  ├─ a2a_router.py                # Router FastAPI A2A: /message:send, /tasks/*, /a2a/tasks/*
│  │  ├─ a2a_client.py                # TrustedA2AClient — transport remota duy nhất, allowlist-only
│  │  ├─ a2a_protocol.py              # RemoteAgentRegistry + A2ARouter (khám phá và định tuyến A2A)
│  │  ├─ mcp.py                       # McpToolServer — máy chủ công cụ MCP
│  │  ├─ http/
│  │  │  ├─ __init__.py               # HTTP adapters package
│  │  │  ├─ health_routes.py          # /health /livez /readyz /metrics /v1/models
│  │  │  ├─ chat_routes.py            # /v1/chat/completions (JSON)
│  │  │  ├─ agent_routes.py           # /v1/agent/execute|cancel|approve|ui-action-result
│  │  │  ├─ admin_routes.py           # /admin/* (reload-model, reload-rag, shutdown…)
│  │  │  ├─ rate_limit.py             # Admission governor HTTP per-peer
│  │  │  ├─ security.py               # TransportPolicy: peer loopback, Origin, bearer (ADR 0008)
│  │  │  └─ _types.py                 # Kiểu cấu trúc cho các dependency được tiêm vào router
│  │  ├─ orchestration/
│  │  │  ├─ __init__.py               # Adapter package cho engine điều phối
│  │  │  ├─ langgraph.py              # LangGraphAgentOrchestrator — adapter sản xuất
│  │  │  └─ deterministic.py          # DeterministicAgentOrchestrator — adapter tất định cho test/embed
│  │  └─ persistence/
│  │     ├─ __init__.py               # Persistence adapters package
│  │     ├─ checkpointing.py          # Checkpointer LangGraph hoán đổi được, fallback an toàn
│  │     ├─ pending_store.py          # PendingActionStore bền vững (file-backed) cho phê duyệt
│  │     ├─ memory_store.py           # Long-term memory SQLite đã seal
│  │     ├─ sqlite_store.py           # TaskStore lưu SQLite
│  │     └─ storage_crypto.py         # AES-256-GCM field cipher cho state at-rest (ADR 0027)
│  │
│  ├─ plugins/                        # ── LỚP PLUGINS: nạp plugin bên thứ ba ──
│  │  ├─ __init__.py                  # Export Plugin, PluginRegistry, ToolPlugin
│  │  ├─ contracts.py                 # Protocol Plugin/ToolPlugin
│  │  ├─ registry.py                  # PluginRegistry + RegisteredTool
│  │  └─ loader.py                    # Nạp qua entry-point group `ai_assistant.plugins`,
│  │                                  #   gate allowlist đọc METADATA trước khi load()
│  │
│  ├─ config/                         # ── LỚP CONFIG: cấu hình khả năng chạy ──
│  │  ├─ __init__.py                  # Typed settings + loaders
│  │  ├─ paths.py                     # Giải quyết đường dẫn & thư mục dữ liệu (AppPaths)
│  │  ├─ logging.py                   # Setup log không có tác dụng phụ lúc import
│  │  ├─ model_registry.py            # Model registry đọc config/models.toml (GGUF, mmproj)
│  │  └─ rag_config.py                # Cấu hình RAG (chunk, top-k, embedding)
│  │
│  ├─ internal/                       # ── INTERNAL: wrapper framework, không import layer nào ──
│  │  ├─ __init__.py                  # Internal infrastructure adapters
│  │  └─ langgraph_engine.py          # LangGraphEngine — wrapper cấp thấp LangGraph
│  │
│  └─ bootstrap/                      # ── BOOTSTRAP: composition root, bị import bởi không ai ──
│     ├─ __init__.py                  # DI container + app factory
│     ├─ server.py                    # Composition root: compose() lười (ADR 0020/0022), state process-wide
│     ├─ lifecycle.py                 # FastAPI lifespan: discovery/prune lúc start, đóng store lúc stop
│     ├─ cli.py                       # Entry point: main() + watcher graceful stop (uvicorn)
│     ├─ platform_patch.py            # Shim pipe-reset của Windows proactor (ADR 0046)
│     ├─ server_manager.py            # Supervisor mà Qt khởi chạy (in `[SUCCESS]`, tail log)
│     ├─ app_factory.py               # Biên FastAPI: middleware, lifespan, trace header
│     ├─ modules.py                   # LlmModule/RagModule/RuntimeModules — instance state, root tiêm
│     ├─ container.py                 # PlatformContainer + build_container (DI container)
│     └─ runtime.py                   # PlatformToolGateway — adapter thực thi do root tiêm
│
├─ config/                            # Hồ sơ triển khai TOML (base + profile + agents + models + plugins)
│  ├─ base.toml                       # runtime (enable_mcp/a2a), CORS allowed_origins
│  ├─ desktop.toml · server.toml      # Hồ sơ theo AI_ASSISTANT_PROFILE (mặc định: desktop)
│  ├─ agents.toml                     # Khả năng & scope least-privilege của từng specialist
│  ├─ models.toml                     # Registry GGUF (repo_id/filename/vision/mmproj + commit/SHA-256)
│  └─ plugins.toml                    # Allowlist plugin được phép nạp
├─ evals/                             # Eval tất định, offline (không gọi LLM thật)
│  ├─ run_tool_eval.py                # Envelope tool-call: golden + adversarial (expect_error)
│  ├─ run_agentic_eval.py             # Chính sách supervisor + gateway (grant/injection/rollback)
│  ├─ run_rag_framework_eval.py        # RAGAS precision/recall + DeepEval retrieval-F1 (offline)
│  ├─ run_model_in_loop_eval.py       # Chấm planner thật (cần backend; không có thì SKIP)
│  ├─ rag_retrieval_cases.json         # Golden set song ngữ cho hybrid retrieval
│  ├─ tool_call_cases.json            # Bộ case của eval tool-call (16 case)
│  └─ *_report.json                   # Report sinh ra (git-ignored)
├─ scripts/
│  ├─ mutation_check.py               # Đảo từng quyết định trọng yếu; test sở hữu phải fail
│  ├─ generate_sbom.py                # CycloneDX: distribution từ lock + artifact GGUF (ADR 0023)
│  ├─ rekey_storage.py                # Xoay khoá at-rest: --verify / --dry-run / xoay (ADR 0030)
│  ├─ migrate_state_row_binding.py    # Ràng buộc state cũ theo dòng: --verify / re-bind (ADR 0037)
│  └─ verify_models.py                # Pre-flight OK/MISSING/MISMATCH cho GGUF đã tải (ADR 0023)
├─ tests/
│  ├─ unit/                           # Hành vi từng module (concurrency, approval, streaming…)
│  ├─ integration/                    # Composition root, orchestration, A2A e2e, guard ranh giới AST
│  └─ contract/                       # Hợp đồng Qt ↔ server và multi-agent
├─ Docs/adr/                          # 64 Architecture Decision Record (0001–0064)
├─ Docs/archive/                      # Tài liệu đã hạ cấp (giữ để tham chiếu)
├─ Docs/RUNBOOK.md                    # SLO, cảnh báo, quy trình production (ADR 0024)
├─ Dockerfile                         # Image non-root (không kèm trọng số; mount Models/ + data)
├─ .env.example                       # Mẫu biến môi trường (sandbox, checkpoint, governor, observability)
├─ pyproject.toml                     # ruff · mypy strict · coverage fail_under=100 · import-linter
├─ requirements.txt                   # Dependency runtime (llama-cpp-python cài tay)
├─ requirements.lock                  # Pin phiên bản + SHA-256 (`uv --generate-hashes`, ADR 0023)
├─ requirements-test.lock             # Pin hash cho bộ test — CI cài bằng `--require-hashes`
├─ requirements-test.txt              # Dependency cho bộ test/eval offline
├─ requirements-eval.lock             # Pin hash riêng cho RAGAS/DeepEval và dependency tương thích
└─ requirements-eval.txt              # Dependency trực tiếp của RAG framework gate
```

### Giải thích theo tầng

Sơ đồ dưới đây là hợp đồng, không phải gợi ý:

```text
Qt / HTTP / MCP / A2A
         │
      adapters ────────── cách ly protocol: route HTTP, máy chủ MCP, A2A,
         │                persistence, shim điều phối
 agents · application · orchestration · tools · rag · llm · observability ·
 plugins ──────────────── service, use-case và thực thi chính sách
         │
      config ──────────── đường dẫn, log, cấu hình RAG/LLM
         │
      domain ──────────── entity, chính sách (thuần, không import gì)
      shared kernel ───── ports.py · settings.py · image_utils.py

      bootstrap ───────── composition root: import mọi layer, bị import bởi
                           không ai (lười, ADR 0020)
      internal ─────────── wrapper LangGraph: không import layer ai_assistant
                           nào (state schema được truyền vào, không import)
```

Mỗi mũi tên được thực thi **hai lần**: bằng hợp đồng `[tool.importlinter]` trong
`pyproject.toml` và bằng các guard AST trong
`tests/integration/test_architecture_boundaries.py`. Các quy tắc package mà
import contract không diễn đạt được: `internal/` không import layer nào, `llm`
không import `rag`, và `agents/` không import adapter nào — persistence chỉ tới
được qua `PendingActionPort` được tiêm vào.

* **`domain`** thuần và tĩnh nhất: entity, chính sách, từ vựng (`Step`,
  `ToolSpec`, `Principal`, `TRANSFER_TARGETS`). Không module nào ở đây biết
  FastAPI, LangGraph hay llama.cpp tồn tại.
* **`application`** giữ các use-case: `ToolExecutionService` là **cổng thực thi
  duy nhất** (mọi đường đi — vòng lặp agent, phê duyệt HITL, hành động desktop,
  worker A2A — đều đi qua nó và đều để lại một dòng audit), `TaskService` điều
  phối task dài, `TaskCoordinator` đo vòng đời và thời gian phục hồi khi hủy.
* **`agents`** sở hữu dịch vụ agent và vòng lặp quyết định; **`orchestration`**
  sở hữu đồ thị LangGraph và các node; **`tools`** chỉ là *danh mục* (một registry
  không thể thực thi); **`rag`**, **`llm`**, **`observability`** là các capability
  có thể thay thế.
* **`adapters`** là nơi duy nhất biết transport: route FastAPI, router A2A,
  client A2A có allowlist, máy chủ MCP, checkpointer và store bền vững. Không
  module nào ở tầng trong gọi thẳng adapter — chúng nhận port qua tiêm.
* **`bootstrap`** là composition root duy nhất: nó import mọi tầng và không bị
  ai import. `compose()` chạy đúng một lần theo thứ tự phụ thuộc; import module
  này **chỉ định nghĩa nó** (không mở file log, không đọc config, không import
  framework — FastAPI/LangGraph/llama.cpp/MCP tải bên trong `compose()`, ADR 0020
  và ADR 0022). Các handle runtime (`LlmModule`, `RagModule`, `ChatbotAgentHolder`
  trong `bootstrap/modules.py`) là **instance state** do `compose()` dựng một lần
  và tiêm vào router/service. Handler và registry giữ **instance state**;
  state cấp process duy nhất còn lại là `_COMPOSED` cùng các handle mà lifespan
  đọc qua `bootstrap.server` (ADR 0046) — một singleton *có chủ sở hữu và có
  test*, không phải state mutable rải rác trong module.
* **`internal/`** chứa wrapper framework: `LangGraphEngine` là nơi duy nhất
  chạm LangGraph, và nó nhận `AgentState` như một đối số constructor nên không
  tạo thành vòng lặp import với `orchestration`.

### Quy tắc runtime trọng yếu

* **Một cổng thực thi duy nhất.** `ToolExecutionService` là nơi duy nhất một
  công cụ chạy, và composition root tiêm cho nó event sink: mọi thực thi đều
  được ghi thành **một dòng JSON đã redact** trên logger `agent.tools`
  (`observability/tool_events.py`).
* **Cache RAG được niêm phong, không được tin — và không đọc được** (ADR 0028).
  `Cache/chunks.pkl` chứa **nguyên văn** tài liệu người dùng đã index (kèm ảnh
  base64), `bm25.pkl` chứa corpus đã token hoá, `faiss_index.bin` chứa
  embedding. Cả ba giờ được **mã hoá** bằng **đúng key** của ADR 0027 (một key
  cho mọi thứ app ghi ra đĩa), nên một bản sao volume dữ liệu không còn đọc
  được corpus. Index FAISS được **serialize trong RAM** rồi mới seal — không
  bao giờ chạm đĩa ở dạng rõ. Cache không xác thực được (sai key, bị sửa,
  chưa seal, cache của version cũ) là **miss** → dựng lại, không bao giờ
  unpickle payload chưa xác thực.
* **State cục bộ được mã hoá at-rest và ràng buộc theo dòng** (ADR 0027/0037).
  `adapters/persistence/storage_crypto.py` bọc **theo field** bằng AES-256-GCM:
  envelope `enc:v1:<base64(nonce||ciphertext)>`, nonce random 96-bit, và AAD
  mang cả id dòng lẫn tên field (`task.v2:<id>:<field>`). Ciphertext không thể
  chuyển sang cột hoặc dòng khác mà vẫn xác thực. `compose()` nạp key
  **một lần** (fail-closed) rồi tiêm vào `SqliteTaskStore` và
  `PendingActionStore`; đọc state đã seal mà thiếu/sai key thì **abort**, không
  bao giờ trả ciphertext lên tầng trên. Key lấy từ `AI_STORAGE_KEY` →
  `AI_STORAGE_KEY_FILE` → tự sinh một lần ở `%LOCALAPPDATA%`/`$XDG_CONFIG_HOME`
  (**ngoài** `APP_DATA_DIR`, nên backup volume không mang theo key). Cột chỉ mô tả
  (id, status, owner, timestamp) vẫn plaintext để SQL filter/retention chạy,
  nhưng identity bundle đã seal mới là nguồn sự thật và được đối chiếu mỗi lần đọc.
* **Hủy job được đo đúng chỗ runner dừng.** `TaskCoordinator.cancel()` đóng dấu
  `cancelled_at`, `finish()` báo delta qua observer mà composition root tiêm vào
  histogram Prometheus `agent_recovery_seconds`.
* **Đồng bộ hoá theo session, không toàn cục.** `AgentService.execute` giữ
  `RLock` theo session trong suốt lần chạy, nên hai lần chạy của cùng một
  session không thể chia sẻ `thread_id` checkpointer; còn `cancel` không giữ
  khóa session nên vẫn chạy được trong khi một run đang giữ nó. Hai đường *tiếp
  tục* run (phê duyệt HITL và ACK desktop) tự re-enter đúng khóa đó.
* **Đo latency theo node ngay trong tiến trình.** `span()` ghi mỗi quan sát vào
  cửa sổ có giới hạn (512/node) và `p95_latency_by_node()`
  (`observability/metrics.py`) trả về `{node: p95}` — tín hiệu SCORECARD #2 mà
  không cần vòng đi PromQL.
* **Trace agent có input/output rõ ràng.** Năm node `agent.plan`,
  `agent.plan_reflect`, `agent.reason`, `agent.tool` và `agent.reflect` dùng
  chung một OpenTelemetry span + LangSmith run cho mỗi lần chạy. LangSmith giữ
  object theo ngữ nghĩa node; OTel ghi cùng payload dưới
  `agent.node.input.*`/`agent.node.output.*`. Dữ liệu được redact trước khi
  export và preview lớn được giới hạn. Cùng contract này áp dụng cho
  `agent.session`, `agent.llm.completion`, local/A2A tool và `agent.tool_batch`;
  OTel nhận cùng metadata và output đã enrich (`outcome`, `duration_ms`) như
  LangSmith, không còn hai context lồng nhau với payload khác nhau.
* **Context trace vượt biên giới tiến trình.** Ra: `TrustedA2AClient.submit`
  gắn header W3C `traceparent`. Vào: `TraceContextMiddleware` đọc header trên
  mọi request và `attach_inbound_traceparent()` join trace của caller làm remote
  parent, trong khi `TaskService._start` chụp `contextvars` của người gửi vào
  thread worker A2A. Header thiếu/sai hoặc SDK không import được là no-op: truyền
  context có thể thất bại, còn request thì không.
* **`settings.py` là chủ sở hữu duy nhất của môi trường process** (ADR 0001,
  0014): nó đọc `.env`, áp mặc định làm ồn library và expose mọi toggle dạng
  accessor được gọi on-demand.
* **Cổng coverage và complexity nằm trong `pyproject.toml`** (ADR 0014), không
  nằm trên dòng lệnh CI: branch coverage bật với sàn 100 ở mức nền 100 đo được,
  và `C90` giới hạn mọi hàm ở complexity 15 — không hàm nào mang `# noqa: C901`.
* **Breadth lint là `E/W/F/I/BLE/S/RUF/C90/N/TRY`** (ADR 0014, mở rộng ADR 0016):
  `N` ghim vài tên phản chiếu API bên ngoài, `TRY` ngăn `return` trong `try`.
* **Mọi transport và chiến lược điều phối đều tới công cụ qua
  `PlatformToolGateway` được tiêm** (`bootstrap/runtime.py`); không còn service
  locator cấp module nào.
* **Vòng lặp LangGraph (`agents/runner.py`) là vòng lặp quyết định sản xuất duy
  nhất.** A2A dùng lại nó qua port `AgentOrchestrator`; lớp application không
  import LangGraph và không giữ vòng lặp riêng. Bản fallback tất định là một
  adapter (`adapters/orchestration/deterministic.py`) do test/embed tiêm vào.
* **LangGraph cô lập sau wrapper engine** (`internal/langgraph_engine.py`) và
  adapter checkpointer (`adapters/persistence/checkpointing.py`); LangSmith nằm
  sau `observability/langsmith_integration.py`. Guard AST làm đỏ CI nếu module
  khác import trực tiếp các framework này (ADR 0012).
* **Một chính sách context-window, một ngôi nhà:** `domain/context_budget.py`
  sở hữu ngưỡng từ chối (`CONTEXT_TOKEN_MARGIN`) và budget completion, cả hai
  consumer đều gọi nó — route HTTP và vòng lặp ReAct không thể bất đồng ý về
  cùng một phép tính.
* **Chính sách HITL thuộc domain** (`domain/approvals.py`):
  `approval_covers_invocation`, `derive_approval_scope`, `build_approval_preview`.
  Node reason tư vấn chính sách đó nhưng không giữ quy tắc phê duyệt riêng
  (ADR 0009).
* **A2A sở hữu vòng đời task bền vững.** Transport remota duy nhất là
  `adapters/a2a_client.TrustedA2AClient` (allowlist-only); allowlist là tập hợp
  triển khai từ `settings.a2a_trusted_endpoints()` — không cấu hình gì thì mọi
  điều phối remota fail closed (ADR 0012). Supervisor chỉ chạm A2A qua đảo ngược
  phụ thuộc: `compose()` tạo một `RemoteAgentRegistry`, tiêm cùng instance đó
  vào `A2ARouter` và vào `functools.partial(delegate, remote_lookup=…)` của
  `AgentService`; không còn lookup toàn tiến trình.
* **Từ vựng bàn giao là dữ liệu registry, không phải luồng điều khiển:**
  `TRANSFER_TARGETS` trong `domain/specialists.py` là nơi đặt tên duy nhất cho
  mọi handoff; routing theo công cụ là bảng `_TOOL_ROUTE_RULES` được `_route`
  đọc (trùng khớp đầu tiên thắng; thứ tự tuple là thứ tự ưu tiên).

## Các design pattern Agentic AI

Trợ lý được hiện thực như **một đồ thị điều phối duy nhất trên một mô hình local
với các specialist logic**. Cách này giữownership rõ ràng và các bước kiểm tra
độc lập mà không phải nhân bối cảnh mô hình cho từng vai trò.

| Giai đoạn / pattern | Hiện thực | Bằng chứng trong mã nguồn |
| --- | --- | --- |
| Vì sao cần pattern | Hợp đồng công cụ tất định, least-privilege và ranh giới phê duyệt tường minh. | `domain/tools.py`, `application/tools.py` |
| Vòng lặp ReAct | `reason → tool → reflect → reason` với số vòng lặp có giới hạn. | `agents/runner.py`, `orchestration/nodes/{reason,tool,reflect}.py` |
| Phản biện (reflection) | Một người phản biện tất định, độc lập, rà mọi kết quả công cụ hoàn tất và nạp lỗi vào lượt suy luận kế tiếp. | `orchestration/nodes/reflect.py`, bước `reflection`, sự kiện audit `tool_reflected` |
| Dùng công cụ | Validate Pydantic/JSON-Schema, constrained decoding, sandbox và MCP — một hợp đồng chung. | `tools/validation.py`, `agents/completion.py`, `tools/sandbox.py`, `adapters/mcp.py` |
| Lập kế hoạch | Yêu cầu dài (không phải lệnh UI ngắn) nhận một kế hoạch JSON cô đọng trước khi thực thi; lệnh UI ngắn bỏ qua bước này. | `orchestration/nodes/plan.py` |
| Bộ nhớ | Buffer short-term theo session (bounded, merge với history client gửi lên) và long-term SQLite (semantic/episodic/procedural) recall bằng keyword + recency; chỉ ghi sau một run `completed` — kể cả run hoàn tất qua approve/desktop-ack, vì quyết định consolidate nằm ở `AgentRunContext` chứ không ở một entry point. | `domain/memory.py`, `application/memory.py`, `agents/context.py`, `agents/short_term_memory.py`, `adapters/persistence/memory_store.py` (ADR 0043) |
| Song song trong workflow | Planner khai báo `parallel_groups`; node tool fan-out các bước `rag_search` độc lập **bên trong node** (không đổi topology), merge thành cặp call/result có `batch_id`/`branch_index`; mọi nhóm không chứng minh được hợp lệ rơi về tuần tự. | `orchestration/nodes/tool.py`, `orchestration/helpers.py`, `domain/parallel.py` (ADR 0044) |
| Đa agent | Supervisor gán một specialist logic, phát ra bàn giao, và verification độc lập với execution. | `orchestration/supervisor.py`, `orchestration/specialists/`, `domain/specialists.py` |
| Sẵn sàng sản xuất | Timeout/giới hạn vòng lặp, cổng phê duyệt, log audit, tracing/Prometheus tuỳ chọn, endpoint health, backend checkpoint, lint CI và eval tất định. | `observability/*`, `adapters/persistence/checkpointing.py`, `.github/workflows/*` |

**Chính sách vận hành:**

* Vòng lặp LangGraph là **engine sản xuất duy nhất** — công tắc `USE_LANGGRAPH_AGENT`
  đã bị xoá (ADR 0013/0014), nên resume cũng chạy nó bất kể nhánh nào.
* `AGENT_OBSERVABILITY=1` bật `/metrics`; scrape từ hệ thống giám sát của bạn và
  cảnh báo khi lỗi công cụ hoặc latency tăng.
* `AGENT_CHECKPOINT_BACKEND=memory` là mặc định an toàn cho desktop. Chỉ cấu
  hình Postgres/Redis khi cần khôi phục session bền vững đa tiến trình.
* **Cổng production là opt-in, fail closed (ADR 0023/0024).**
  `AGENT_CHECKPOINT_REQUIRED=1` từ chối backend memory (raise thay vì rơi về
  `MemorySaver` im lặng); `AGENT_SANDBOX_REQUIRE_ISOLATION` **mặc định là `1`**
  nên từ chối runtime local (chỉ chạy tool trong docker `--cap-drop ALL` +
  `no-new-privileges`) — allow-list không phải là một OS boundary, nên đường
  local phải là một quyết định được viết ra
  (`AGENT_SANDBOX_REQUIRE_ISOLATION=0` cho desktop một người dùng, ADR 0037);
  `AGENT_MAX_CONCURRENT_RUNS` và `AGENT_RATE_LIMIT_PER_MINUTE` bật governor
  admission (hết slot → 503, vượt quota → 429 + `Retry-After`; `/livez`,
  `/readyz` được miễn). Budget run áp cho **mọi** entry point chạy graph —
  `execute`, `approve` và desktop-ack — nên một burst resume không thể vượt
  trần như trước. Deployment nhiều người dùng xem
  [`Docs/RUNBOOK.md`](Docs/RUNBOOK.md) (SLO, cảnh báo,
  quy trình chỉnh quota/backup/rollback).
* Người phản biện là tất định **bằng thiết kế**: không tốn một lời gọi mô hình
  thêm nào, và ngăn một kết quả công cụ thất bại bị âm thầm coi là sự thật.
* **Bộ nhớ (ADR 0043).** `AGENT_MEMORY_ENABLED=0` tắt cả hai nửa (buffer
  short-term và long-term store); `AGENT_MEMORY_STM_MAX_MESSAGES` (40) và
  `AGENT_MEMORY_LTM_TOP_K` (5) chỉnh cửa sổ buffer và số record recall;
  `AGENT_MEMORY_RETENTION_DAYS` (90) / `AGENT_MEMORY_RETENTION_MAX_MB` (64) là
  hạn mức prune của store SQLite. Secret vẫn đi qua vault: một record còn chứa
  dữ liệu nhạy cảm bị **loại** trước khi ghi, không được che.
* **Song song trong workflow (ADR 0044).** `AGENT_PARALLEL_MAX` (mặc định 4,
  clamp cứng `[1, 8]`) là độ rộng fan-out của **một** nhóm bước. Chỉ nhóm
  `rag_search` không cần approval mới chạy song song; nhóm chứa `direct`,
  `application_action`, `model_selected` hoặc một tool cần grant giữ nguyên
  đường tuần tự — fail-closed theo nhóm, không "nhánh nào tự quyết".

## Các quyết định & phân rã trọng yếu

* **Dịch vụ agent (ADR 0004).** `AgentService` (facade) + `ApprovalService`
  (vòng đời phê duyệt HITL) + `UIActionService` (tiếp tục hành động desktop),
  mỗi service nhận đúng bộ dependency được tiêm và test được độc lập.
  `AgentRunContext` (`agents/context.py`, ADR 0012) sở hữu bó wiring chung một
  lần, thay cho ~78 dòng lặp lại ở ba call-site; `RunDeps` (`agents/deps.py`,
  ADR 0013) gom dependency của chính runner — từ 46 tham số còn 21.
* **Node reason (ADR 0013).** Từ một hàm 353 dòng/19 nhánh, `reason_node` thành
  chuỗi dispatch 78 dòng; mỗi kết cục mà mô hình có thể sinh ra có một hàm riêng
  (`_handle_step_answer`, `_handle_final_answer`, `_handle_validation_error`,
  `_handle_forbidden_transfer`, `_guard_step_action_mismatch`,
  `_guard_duplicate_failed_call`, `_dispatch_tool_call`).
* **Adapter A2A (ADR 0005).** `adapters/a2a.py` (310 dòng) được tách thành
  `a2a_models.py` (schema), `a2a_payloads.py` (payload + redaction),
  `a2a_router.py` (route) và `a2a.py` (re-export tương thích).
* **An toàn kiểu (ADR 0007).** `LLMRuntime` dùng kiểu cụ thể
  (`llm: LLMBackend | None`, `llm_lock: LockLike`) thay cho `Any`, nên cả lớp
  agent được type-check tĩnh. `LLMBackend` thuộc `ai_assistant.ports`;
  `LockLike` là protocol cấu trúc thỏa mãn cả `threading.Lock` và `RLock`.
* **Từ vựng bước là hợp đồng domain (ADR 0019).** `Step`/`StepType`/
  `has_step_type` nằm ở `domain.steps`; `validate_steps` kiểm lại danh sách bước
  đã persist so với `get_args(StepType)`. Không request HTTP/A2A inbound nào
  mang trường `steps` — có test ghim điều đó.
* **Cứng hoá transport.** ADR 0048/0051 — peer loopback + allow-list `Origin` +
  bearer **bắt buộc** trên mọi bề mặt đặc quyền. A2A inbound dùng
  `AI_A2A_TOKEN` (single-tenant) hoặc `AI_A2A_TENANTS`; discovery/dispatch
  outbound dùng `AI_A2A_OUTBOUND_TOKEN` (fallback tường minh sang token
  single-tenant). Outbound mặc định bắt buộc HTTPS; HTTP chỉ được phép khi
  development đặt rõ `AI_A2A_ALLOW_INSECURE_HTTP=1`. Thiếu credential hoặc
  transport an toàn thì adapter từ chối trước khi chạm mạng.
  ADR 0009 — phê duyệt là capability dùng một lần, gắn với đúng
  lời gọi (`approval_fingerprint` + grant store). ADR 0010 — domain sở hữu một
  chính sách redaction duy nhất. ADR 0011 — sandbox lệnh từ chối code interpreter
  nội tuyến (`-c`, `-m`, `-`) và shell substitution.
* **Mã hoá at-rest cho state cục bộ** (ADR 0027, ADR 0037). Prompt, đối số tool, kết quả
  model và event log không còn nằm plaintext trên đĩa. Key được **tiêm** từ
  composition root (không có global), đọc thiếu/sai key thì **fail closed**,
  và row plaintext cũ được seal đúng **một lần** nhờ `PRAGMA user_version = 3`.
  Từ ADR 0037, state còn được **ràng buộc theo dòng**: AAD mang id của dòng
  (`task.v2:<id>:<field>`) nên một giá trị không chép sang dòng khác được, và
  mỗi dòng mang một **identity bundle đã seal** (`owner`, `classification`,
  `status`) nên các cột quyết định *ai được đọc dòng này* không còn là
  plaintext có thể sửa. File đã seal bằng định dạng cũ được từ chối mở và
  người vận hành chạy một lần `scripts/migrate_state_row_binding.py`.
* **Cache RAG dùng chung key đó** (ADR 0028). `Cache/chunks.pkl` là **bản sao
  thứ hai của corpus tài liệu** (kèm ảnh base64) — nên nó được seal như mọi
  thứ khác, **một key cho tất cả**. HMAC cũ bị bỏ (GCM đã tự xác thực), index
  FAISS được serialize trong RAM trước khi seal, và cache không xác thực
  được thì là **miss** → dựng lại, không bao giờ unpickle payload chưa xác thực.
* **Di sản đã xoá, không phải được tài liệu hoá** (ADR 0006/0012/0013/0014).
  Package `modules/` và shim `legacy/` không còn tồn tại; mọi production import
  đi thẳng `ai_assistant.*`. Hợp đồng import-linter và guard AST giữ `modules`
  không thể hồi sinh.

## Diện tích API

| Bề mặt | Endpoint |
|--------|----------|
| Health probes (public) | `/health`, `/livez`, `/readyz` |
| Metrics (admin bearer) | `/metrics` |
| Chat (tương thích OpenAI) | `/v1/models`, `/v1/chat/completions` (JSON hoặc SSE khi `stream=true`) |
| Agent | `/v1/agent/execute` (JSON hoặc SSE), `/v1/agent/cancel`, `/v1/agent/approve`, `/v1/agent/ui-action-result` |
| Admin | `/admin/release-vram`, `/admin/reload-model`, `/admin/reload-rag`, `/admin/reload-agent`, `/admin/shutdown` |
| A2A | `/.well-known/agent.json`, `/message:send`, `/tasks/*` (`:cancel`, `:resume`, `:subscribe`), `/a2a/tasks/*` (tương thích) |
| MCP | `/mcp` (chỉ mount khi bật **và** có `AI_MCP_TOKEN`) |

**A2A.** Bề mặt chính là `POST /message:send`, `GET /tasks/{id}`, `GET /tasks`,
`POST /tasks/{id}:cancel`, `POST /tasks/{id}:subscribe`; discovery nằm ở
`/.well-known/agent.json`. Agent Card quảng bá giao diện REST của nó và yêu cầu
header `A2A-Version: 0.3` khi client gửi header phiên bản. Phê duyệt và ACK của
Qt làm **tạm dừng** task bền vững ở trạng thái `input-required`; chúng chỉ tiếp
tục qua extension được khai báo
`urn:3d-reconstruction:a2a:approval-resume:v1` tại `POST /tasks/{id}:resume`;
ngữ cảnh thực thi được persist nội bộ và không bao giờ lộ trong artifact.
Mọi route A2A inbound yêu cầu bearer; client outbound dùng cùng hợp đồng cho cả
Agent Card discovery lẫn task submit (`AI_A2A_OUTBOUND_TOKEN`).

**MCP.** `adapters/mcp.py` là adapter bên ngoài; thực thi agent **không** gọi
`/mcp` nội bộ. Thiếu `AI_MCP_TOKEN` thì endpoint không được mount, kể cả trên
loopback (fail closed, ADR 0048).

**Plugins.** Tool tương thích builtin được đăng ký trong lúc bootstrap. Plugin
bên thứ ba dùng entry-point group `ai_assistant.plugins` và chỉ được nạp khi
`plugin_id` của nó được bật tường minh trong `config/plugins.toml`. Cổng allowlist
đọc **metadata** của package trước khi `entry_point.load()` chạy — vì load một
entry point thực thi module của nó, nên một distribution đã cài nhưng bị tắt
không bao giờ có cơ hội chạy code lúc bootstrap.

## Cài đặt

```powershell
# Runtime (llama-cpp-python cài riêng — xem dưới)
pip install -r AIAssistant/requirements.txt

# Bản hash-pinned đã resolve (ADR 0023, khuyến nghị cho production):
# --require-hashes bắt pip xác minh SHA-256 của từng artifact tải về
pip install --require-hashes -r AIAssistant/requirements.lock

# Bộ test/eval offline (cùng cơ chế hash; CI cài đúng lock này)
pip install --require-hashes -r AIAssistant/requirements-test.lock

# Gate RAGAS/DeepEval (evaluation-only, không đưa vào runtime)
pip install --require-hashes -r AIAssistant/requirements-eval.lock

# Chế độ phát triển: cài package ở chế độ editable
pip install -e AIAssistant
```

`llama-cpp-python` cố ý không pin trong `requirements.txt`: hãy cài bản khớp
với cấu hình của bạn (wheel CUDA hoặc CPU thuần) theo phần đầu file đó. Trọng số
mô hình **không** nằm trong repo — chúng nằm trong thư mục `Models/` mà
`ai_assistant.config.paths` resolve (mặc định `<repo>/Models`, git-ignored) và
được tải về ở lần dùng đầu từ registry trong
[`config/models.toml`](config/models.toml). Mọi GGUF đều được pin theo commit
Hugging Face + SHA-256; kiểm tra trước khi chạy server:

```powershell
python AIAssistant/scripts/verify_models.py          # OK/MISSING/MISMATCH, exit 1 nếu lệch pin
python AIAssistant/scripts/generate_sbom.py --output sbom.json   # CycloneDX: lock (version + SHA-256) + model artifacts
python AIAssistant/scripts/rekey_storage.py --verify  # Kiểm tra khoá at-rest có mở được state không (ADR 0030)
```

Biến runtime đặt trong `AIAssistant/.env` (git-ignored); bắt đầu từ
[`.env.example`](.env.example), file mô tả chính sách sandbox, backend checkpoint,
observability/OTLP, chính sách bảo mật hybrid và các token transport.

Hồ sơ triển khai là TOML nhiều lớp: `config/base.toml` → `config/<profile>.toml`
→ `plugins.toml` → `agents.toml`, trong đó `<profile>` là `AI_ASSISTANT_PROFILE`
(mặc định `desktop`). `config/models.toml` là model registry do
`config/model_registry.py` đọc.

## Sử dụng

```powershell
cd AIAssistant

# Chạy thủ công phải sở hữu token; desktop Qt tự sinh và truyền token riêng.
$env:AI_ADMIN_TOKEN = "replace-with-a-secret"
$env:AI_AGENT_TOKEN = "replace-with-a-different-secret"

# Loopback, cổng 8080; AGENT_HOST / AGENT_PORT đổi địa chỉ bind.
python src/ai_assistant/bootstrap/server.py
# cùng một thứ khi package đã nằm trên path (pip install -e . hoặc PYTHONPATH=src):
#   python -m ai_assistant.bootstrap.server

# Ứng dụng desktop không gọi hai lệnh trên: nó khởi chạy supervisor — tiến trình
# quản lý con, chỉ khởi động lại khi model được yêu cầu khác với model đang chạy,
# in sentinel [SUCCESS] mà dock Qt neo vào, rồi tail log
python src/ai_assistant/bootstrap/server_manager.py 0
```

```bash
# Server có sống không?
curl http://127.0.0.1:8080/health

# Chạy một nhiệm vụ agent (agent sẽ xin phê duyệt trước khi write_file / run_command)
curl -X POST http://127.0.0.1:8080/v1/agent/execute \
  -H "Authorization: Bearer $AI_AGENT_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"task": "Liệt kê các file trong src/", "session_id": "demo", "language": "en"}'
```

Container:

```powershell
# Lần đầu: Compose tự đọc AIAssistant/.env để nội suy và truyền runtime config.
Copy-Item AIAssistant/.env.example AIAssistant/.env
# Sửa AI_ADMIN_TOKEN và AI_AGENT_TOKEN trong .env thành hai secret khác nhau.

docker compose -f AIAssistant/compose.yaml config
docker compose -f AIAssistant/compose.yaml up --build -d
docker compose -f AIAssistant/compose.yaml ps
docker compose -f AIAssistant/compose.yaml logs -f ai-assistant
```

Compose chỉ publish `127.0.0.1:8080` mặc định (đổi cổng host bằng
`AI_ASSISTANT_PORT`, đổi địa chỉ bằng `AI_ASSISTANT_BIND_IP`). Service chạy profile
`server`, checkpoint SQLite bền vững, root filesystem chỉ đọc, không capability và
dùng hai named volume riêng:

* `3d-reconstruction-ai_ai_data`: model, cache RAG, task store và log;
* `3d-reconstruction-ai_ai_secrets`: storage key tự sinh một lần. Phải backup volume
  này cùng `ai_data`; mất key đồng nghĩa state đã seal không thể phục hồi.

Image không đóng gói trọng số. Lần khởi động đầu có thể tải model đã pin vào
`ai_data`, vì vậy healthcheck có `start_period` dài. Có thể chép model có sẵn bằng
`docker compose -f AIAssistant/compose.yaml cp AIAssistant/Models/. ai-assistant:/app/data/AIAssistant/Models/`
trước khi dùng.

Image Compose tham chiếu wheel CPU `llama-cpp-python` chính thức cho
`linux/amd64`, khóa theo URL và SHA-256 trong `requirements-llama-cpu.lock`.
Deployment CUDA/ROCm/Vulkan phải dùng Dockerfile dẫn xuất thay wheel này bằng
wheel accelerator tương ứng; không cài chồng một backend floating lúc container
khởi động.

`run_command` bị tắt trong service chính: mount Docker socket sẽ trao quyền tương
đương root trên host và đường dẫn workspace trong container không ánh xạ đúng vào
daemon host. Các tool không thực thi process vẫn hoạt động. Muốn chạy command tool
trong production cần một sandbox worker/daemon tách riêng thay vì mount
`/var/run/docker.sock` vào API service.

```powershell
# Dừng nhưng giữ state/key
docker compose -f AIAssistant/compose.yaml down

# Xoá cả state và key (không thể phục hồi nếu chưa backup)
docker compose -f AIAssistant/compose.yaml down --volumes
```

Nếu chỉ cần build image thủ công, build từ gốc repo để tiền tố `AIAssistant/` phân
giải đúng: `docker build -f AIAssistant/Dockerfile -t ai-assistant .`.

## Kiểm thử

```powershell
cd AIAssistant
ruff check .
python -m mypy src/ai_assistant
python -m compileall -q evals src
$env:PYTHONPATH = "src"
lint-imports --no-logo
python -m coverage run -m tests.run_suite
python -m coverage report
python scripts/mutation_check.py
python evals/run_rag_framework_eval.py
```

Các cổng là **cấu hình**, không phải cờ dòng lệnh: `pyproject.toml` sở hữu breadth
lint (`E/W/F/I/BLE/S/RUF/C90/N/TRY`, complexity ≤ 15), thiết lập mypy strict và
sàn coverage theo branch (`fail_under = 100`), nên lần chạy cục bộ và CI đo cùng
một thứ. Theo lần đo ngày 2026-11-17 (các con số đo được được ghi lại trong
`Docs/SCORECARD.md` — **đã theo git** từ ADR 0032): **100.00%** statement *và*
branch toàn package (12 286 statement / 3 440 branch, 2 727 test);
inventory mutation hiện có **363/363** killed, 0 survived, 0 stale trong sandbox
biệt lập của ADR 0056/0058/0059/0060/0061/0062/0063.
Số mutant tối thiểu cũng là một cổng
(`MIN_MUTANTS`): xoá một mutant là một quyết định, và cổng biến nó thành
quyết định *nhìn thấy được*. Từ 2026-10-28, cổng còn có sàn **theo từng file**
(`PINNED_DECISION_FILES`): một file mang quyết định load-bearing phải khai báo
ít nhất một mutant, nên tổng số không thể che việc một file mất mutant cuối.

Nhưng một bảng cổng xanh **không phải** là một hệ thống an toàn. Vòng review
2026-10-21 thứ ba (ADR 0034) tìm ra tám lỗi mà **mọi cổng đều không thấy** —
vì từng lỗi nằm đúng ở chỗ mà test chỉ chứng minh được *qua mock*: control
có thật, test có thật, nhưng test **không tái hiện được tấn công**. Đáng chú ý:
`.env` (khoá at-rest + mọi bearer token) đọc được **không cần approval** vì
block-list là *extension* mà dotfile không có extension; `shutil.which` trên
Windows tìm CWD trước PATH nên binary agent cài được chạy thay binary thật;
deny-list so khớp phân biệt hoa thường nên `PYPROJECT.TOML` ghi đè file thật;
`open()` trả plaintext chưa seal nên cột bị thay thế vẫn được phục vụ như
thật. Cả tám đã đóng và ghim bằng test hồi quy *chạy thật* — và SCORECARD giờ
có mục riêng ghi lại control nào **chỉ** được kiểm bằng mock, để lần review sau
không phải đoán.

Vòng audit thứ ba (ADR 0035) chạy lại trên **các lớp vòng trước chưa chạm tới**
và tìm thêm **năm lỗi cùng lớp** — trong đó một lỗi là *anh em* của một quy
tắc ADR 0034 vừa sửa: quy tắc *thư mục bị chặn* so phân biệt hoa thường trong
khi quy tắc *tên file bí mật* ngay cạnh đó thì không, nên `.GIT/config`
(remote URL + credential helper) đọc được không cần approval; và
`lsp_client` vẫn dùng `shutil.which` — **chỗ thứ hai** giải chương trình, nơi
ADR 0034 chỉ sửa chỗ thứ nhất.

Hai mươi lỗi cùng một lớp thì *đếm số lỗi đã sửa* là một con số vô nghĩa:
lớp lỗi vẫn còn, và thực thể thứ hai mươi mốt chỉ là một dòng code nữa. Vì vậy
ADR 0035 đóng **lớp**, không đóng từng thực thể: hai module kernel
(`path_policy`, `exec_policy`) làm chủ sở hữu duy nhất của quy tắc đường dẫn và
quy tắc giải chương trình; mọi so sánh tên đều không phân biệt hoa thường và
được kiểm theo **dạng tính chất** (duyệt chính tập quy tắc, nên thêm một tên
tự đòi hành vi đúng); và một **guard tĩnh** quét AST toàn bộ `src/ai_assistant`
fail build nếu bất kỳ module nào gọi `shutil.which` — vì lỗi đó **không nhìn
thấy được lúc chạy**: mock cũ trả về một tên *tương đối*, đúng hình dạng mà
một CWD-hit thật tạo ra.

CI chạy cổng trên **cả Linux lẫn Windows** (`Build-Python-ci.yml`): sản phẩm
ship trên Windows, và hai guard bảo mật của sandbox chỉ chạy ở đó
(`skipUnless(win32)`) — một CI chỉ có Linux sẽ báo xanh trong khi nền tảng
người dùng thật chưa từng được kiểm tra (ADR 0032).

Vòng review thứ tư (ADR 0037) chạy lại toàn bộ cổng rồi đọc source, và tìm
ra **bốn lỗi cùng một lớp**: *một quy tắc viết thành một danh sách, và danh
sách hở chỗ khác*. Sandbox coi **mọi** token bắt đầu bằng `-` là "cờ thường"
nên **verb thật chưa bao giờ được kiểm** (`git -C <dir> clean -fdx`,
`pytest -p <mod>`, `ctest --build-and-test --test-command <cmd>`); deny-list
ghi của nền tảng là một *bảng liệt kê* nên `agents/`, `adapters/persistence/`
và hai lá kernel `path_policy`/`exec_policy` vẫn ghi được; cổng classification
so **ngược chiều** (principal yếu hơn thì qua, mạnh hơn thì bị chặn) và mọi
tool mang cùng một trần nên nó chỉ là phép so sánh với hằng số; và state
at-rest ràng buộc theo **cột** chứ không theo **dòng**, nên một giá trị chép
sang dòng khác vẫn mở ra, còn `owner`/`classification`/`status` — cột quyết
định *ai được đọc dòng này* — là plaintext không xác thực. ADR 0037 đóng lớp:
bề mặt lệnh thành **ngữ pháp** (verb là token vị trí đầu tiên, cờ dẫn là
allow-list, cờ import/execute bị từ chối ở mọi vị trí) và cổng isolation
**mặc định bật**; deny-list thành **luật** ("nguồn của nền tảng không ghi
được"); clearance **suy ra từ scope được cấp** và so sánh đúng chiều; và
state at-rest **ràng buộc theo dòng** với một identity bundle đã seal, file
cũ bị từ chối và người vận hành re-bind một lần bằng
`scripts/migrate_state_row_binding.py`. Mutant **88 → 105**: mỗi quyết định
mới có một mutant **trong cùng commit**, theo đúng luật của repo.

Vòng review thứ năm (ADR 0038) đi vào các lớp **chưa ai rà sâu** — `rag/`,
`llm/`, `internal/`, `config/`, `observability/` — và tìm ra ba lỗi cùng lớp,
tất cả ở nơi mà **người đọc thứ hai** của hệ cây thư mục đứng: trình dựng
corpus của RAG. Nó có chính sách riêng — không có luật link, danh sách thư
mục *thiếu* hai mục mà kernel chung chặn, không có luật tên bí mật — nên một
**junction** (trên Windows tạo được không cần quyền đặc biệt) bên trong dự án
được index như thể nó thuộc dự án: một tài liệu riêng tư, hay một
`secrets.py` chứa API key, đi thẳng vào mô hình và vào cache đã seal. Ảnh
dưới `Docs/` bị index **hai lần**, và lần quét tính hash cache **khác** lần
quát index. ADR 0038 đóng lớp: corpus builder dùng **kernel chung** (`path_policy`),
mỗi file được quét **đúng một lần** bởi lần quét sở hữu nó, và hash cache
hỏi trên **đúng** tập file đã index. Mutant **105 → 110**.

Vòng review thứ sáu (ADR 0038) làm đúng điều mà năm vòng trước đã lặp lại:
**rà lại chính code vừa viết**. Nó tìm ra hai lỗi trong code của chính tác
giả — lần quét vẫn đi *vào* junction trỏ ra ngoài rồi mới từ chối từng file
(duyệt cả ổ đĩa), và hành động "seal state" có thể **phá** một state đã seal
(bằng cách ghi version mới lên nội dung cũ). Cả hai đã sửa theo hướng
fail-closed: cùng một luật áp ở cấp thư mục lẫn cấp file, và không còn đường
nào ghi một version mà nó không thực sự đã viết. Mutant **110 → 111**.

Vòng review thứ bảy rà **bề mặt tuỳ chọn** của các binary trong allow-list —
thứ chưa vòng nào chạm tới — và tìm ra một lỗi P1 cùng lớp: `ruff check
<dir>` **in ra source của mọi file** nó tìm thấy, tức một đường đọc file ngoài
workspace chỉ với phê duyệt *lệnh*. Cùng kiểu, `ruff --config <file>` (ruff
echo nội dung file khi không parse được), `pytest --junitxml=<file>` (ghi ra
ngoài), `git diff --no-index <a> <b>` (in hai file bất kỳ), `git commit
--template=<file>`. Đã sửa: path positional của `python`/`pytest`/`ruff` phải
nằm trong workspace, và **một bảng tuỳ chọn nguy hiểm theo từng binary** bị
từ chối ở *mọi vị trí* trên dòng lệnh — `pytest` không còn là trường hợp
đặc biệt nữa. Mutant **111 → 112**.

Vòng review 2026-10-22 (ADR 0039) hỏi ba câu mà các vòng trước chưa hỏi: tool
của plugin có đi cùng catalogue/chính sách với builtin không, các cầu nối
policy/alias/MCP có fail-closed không, và state theo session có bị giữ vô hạn
không. Cả ba đều có lỗi trong khi mọi gate xanh: plugin tool *executable* nhưng
*vô hình* với agent; policy có ba nguồn (một alias `app_action_*` mang policy
`read_only` cho action `desktop_ack`, nhãn lạ rơi về `read_only`); và 10 000
session id để lại 10 000 `RLock`. ADR 0039 đóng lớp: một vocabulary/một bảng
quyền trong `domain/specialists.py`, một bảng alias duy nhất, plugin tool được
merge vào catalogue với row động, mọi nhãn lạ fail-closed, và `SessionLockPool`
reference-count để entry biến mất khi người dùng cuối rời đi. Mutant
**112 → 134**.

Vòng review kế tiếp (ADR 0040) rà *ai thực sự giữ model lease* — thứ ADR 0032
mới gắn vào `execute`. `approve`, `ui_action_result` và route
`/v1/chat/completions` đều dùng model mà không giữ lease, nên
`/admin/reload-model` có thể `del` backend dưới một resume hoặc một stream đang
chạy. ADR 0040 đóng lớp: mọi entry point chạy graph mở cùng một lease, chat giữ
lease từ lúc dựng prompt (vision flag) đến token cuối và dùng backend do lease
trả về; fast-path 503 vẫn còn nhưng không còn là bảo đảm. Mutant **134 → 138**.

* `tests/unit/` — hành vi từng module, gồm cả các ca biên về concurrency, phê
  duyệt và streaming.
* `tests/integration/` — composition root, orchestration, A2A streaming end-to-end
  và các guard ranh giới layer bằng AST.
* `tests/integration/test_real_http_surface.py` — **bề mặt HTTP thật**: compose
  đồ thị đối tượng thật trong một interpreter mới, chỉ thay backend mô hình
  rồi bắn một request HTTP thật qua router thật. Mọi test router khác dùng
  double, nên *binding* giữa composition root và router chưa từng được
  chạy — đúng loại lỗi mà `/v1/chat/completions` chết trong production đã bắt
  (ADR 0029). Test này **đỏ** khi binding bị hỏng (ADR 0032).
* `tests/contract/` — hợp đồng Qt ↔ server và wiring multi-agent.
* `tests/unit/test_at_rest_cipher_contract.py` — hợp đồng chéo ngôn ngữ cho
  envelope at-rest: hằng số format đọc từ `src/core/AtRestCipher.cpp` và
  vector tham chiếu đọc từ `tests/TestAtRestCipher.cpp` phải khớp bên Python.
  Nhờ vậy một vector chỉ được ghim lại ở một bên sẽ làm CI Python đỏ — không
  cần trigger chéo ngôn ngữ (ADR 0032).
* `scripts/mutation_check.py` — đảo từng quyết định trọng yếu (allowlist sandbox,
  phê duyệt HITL, `/health`, niêm phong cache, mã hoá at-rest, các cuộc đua cancel…) và **bắt
  buộc** test sở hữu phải fail; CI chạy lại con số này mỗi lần push (mutant recompose
  được ghim bằng timeout 180s — ADR 0022). Con số là một đo lường, không phải mục
  tiêu: nó tái lập được khi cài `faiss-cpu`, `rank-bm25`, `mcp` và `cryptography`
  (`requirements-test.lock`).
* `evals/` — ba eval tất định, offline chạy trong CI trên mọi push/PR chạm
  `AIAssistant/**` (`run_tool_eval.py` cho envelope tool-call — 16 case gồm
  adversarial: tool bịa, tham số lạ, type không coerce, JSON cụt;
  `run_agentic_eval.py` cho chính sách supervisor + gateway thật — 12 nhóm
  scenario, 27 scenario/136 case; A2A là transport nội bộ luôn được kiểm tra,
  còn case LangSmith tuỳ chọn báo *not run* và exit khác 0 khi thiếu; gồm grant
  dùng-một-lần, rollback/cancel, scope và classification; ADR 0025).
  `run_rag_framework_eval.py` chạy `hybrid_retrieve` thật trên golden set song
  ngữ; RAGAS 0.4.3 chấm ID-based context precision/recall và DeepEval 4.3.3
  chạy retrieval-F1 qua `LLMTestCase`/`assert_test`. Gate không gọi judge/API
  ngoài, cài từ `requirements-eval.lock` có SHA-256 và xuất report artifact.
  `run_model_in_loop_eval.py` chạy tay và cần
  backend thật, có `--baseline` để gate tụt điểm so với report cũ. Cả bốn
  script đều tự đặt `src/` lên `sys.path`, và một case **không chạy được** sẽ
  được báo là *not run* và exit khác 0.
* Coverage không thể phát hiện một test asserted sai chiều, nên các guard trọng
  yếu (allowlist sandbox, phê duyệt HITL, chính sách remota, `/health`, niêm
  phong cache, các cuộc đua cancel, cổng ranh giới bước ADR 0019) còn được
  mutation-check.

`.github/workflows/Build-Python-ci.yml` chạy các cổng lint/type/test/coverage;
`agent-evals.yml` chạy policy eval và một job riêng tải, xác minh SHA-256 rồi
chấm pinned 3B model. Job model thật chấm ma trận cân bằng 32 case
Việt/Anh × `direct`/`rag_search`/`application_action`/`web_search` và 16 case
freshness đi qua classifier production; ngưỡng lần lượt là model coverage 0.9,
accuracy tổng 0.9, accuracy từng slice 0.75 và freshness 0.9. Lần đo
2026-11-12 trên artifact đúng SHA đạt **32/32 planner + 16/16 freshness
(100%)**, coverage 100% và cả 8 slice đều 4/4 (ADR 0053/0056).

## Tài liệu & ADR

| Tài liệu | Vai trò |
|----------|---------|
| `README.md` (file này) | Điểm vào duy nhất: layout, kiến trúc, API, cài đặt, kiểm thử |
| `Docs/SCORECARD.md` | Các metric đo được (coverage, mutation, lint, type, eval) — **đã theo git** từ ADR 0032 |
| [`Docs/RUNBOOK.md`](Docs/RUNBOOK.md) | Vận hành production: SLO, cảnh báo, quota, checkpoint, luân chuyển pin model |
| [`Docs/adr/`](Docs/adr/) | 64 Architecture Decision Record (0001–0064) |
| [`Docs/archive/`](Docs/archive/) | Tài liệu đã hạ cấp, giữ lại để tham chiếu lịch sử |

> `Docs/` chỉ chứa `adr/`, `archive/`, `RUNBOOK.md` và `SCORECARD.md`. Các bản
> tài liệu vận hành/MCP/roadmap cũ đã bị xoá thay vì để thành link chết
> (ADR 0015 — một claim không được test nào ghim thì bị xoá, không phải được
> nhắc lại).

Các quyết định trọng yếu, theo thứ tự:
[0001](Docs/adr/0001-platform-boundaries.md) ranh giới nền tảng,
[0002](Docs/adr/0002-mcp-a2a-trust-and-execution.md) niềm tin & thực thi MCP/A2A,
[0003](Docs/adr/0003-single-execution-core.md) lõi thực thi duy nhất,
[0004](Docs/adr/0004-agent-service-subservices.md) phân rã dịch vụ agent,
[0005](Docs/adr/0005-a2a-protocol-adapter-package.md) gói adapter A2A,
[0006](Docs/adr/0006-legacy-modules-removed.md) xoá module di sản,
[0007](Docs/adr/0007-llm-runtime-type-safety.md) an toàn kiểu runtime LLM,
[0008](Docs/adr/0008-transport-trust-policy.md) chính sách tin transport,
[0009](Docs/adr/0009-single-use-approval-grants.md) grant phê duyệt dùng một lần,
[0010](Docs/adr/0010-sensitive-data-redaction-policy.md) redaction dữ liệu nhạy cảm,
[0011](Docs/adr/0011-command-sandbox-policy.md) sandbox lệnh,
[0012](Docs/adr/0012-cut-cycle-enforce-allowlist-retire-legacy.md) bật allowlist, nghỉ hưu legacy,
[0013](Docs/adr/0013-rundeps-reason-decomposition-dead-code.md) `RunDeps`, phân rã reason, dọn dead code,
[0014](Docs/adr/0014-coverage-gate-environment-owner-complexity.md) cổng coverage, chủ sở hữu env, complexity,
[0015](Docs/adr/0015-unenforced-claims-pinned-or-removed.md) claim không được ghim thì bị gỡ,
[0016](Docs/adr/0016-coverage-gate-raised-to-100.md) nâng sàn coverage lên 100,
[0017](Docs/adr/0017-complexity-15-try-n-lint-boost-split.md) complexity 15, lint try/except,
[0018](Docs/adr/0018-hybrid-policy-enforcement.md) thực thi chính sách hybrid,
[0019](Docs/adr/0019-step-vocabulary-domain-boundary.md) từ vựng bước, ranh giới domain,
[0020](Docs/adr/0020-lazy-composition-root.md) composition root lười,
[0021](Docs/adr/0021-hardening-batch-10-10.md) lô cứng hoá 10/10,
[0022](Docs/adr/0022-true-lazy-strict-10-10.md) root lười đúng nghĩa, strict 10/10,
[0023](Docs/adr/0023-artifact-pinning-production-gates.md) pin artifact & cổng production fail-closed,
[0024](Docs/adr/0024-admission-governor.md) governor admission (rate limit, run budget, SLO),
[0025](Docs/adr/0025-eval-expansion.md) eval mở rộng (adversarial, gateway, multilingual, baseline),
[0026](Docs/adr/0026-a2a-registry-owned-by-composition-root.md) A2A registry & remote lookup thuộc composition root,
[0027](Docs/adr/0027-at-rest-encryption-local-state.md) mã hoá at-rest cho state cục bộ,
[0028](Docs/adr/0028-sealed-rag-cache.md) cache RAG cũng được niêm phong,
[0029](Docs/adr/0029-prompt-builder-as-a-port.md) prompt builder là một port riêng,
[0030](Docs/adr/0030-at-rest-key-rotation.md) xoay khoá at-rest,
[0031](Docs/adr/0031-sealed-chat-attachments.md) file đính kèm chat cũng được niêm phong,
[0032](Docs/adr/0032-closing-the-last-mile.md) đóng nốt mile cuối của hợp đồng 10/10,
[0033](Docs/adr/0033-one-spelling-of-a-path.md) cổng ghi so trên *một* cách viết của đường dẫn,
[0034](Docs/adr/0034-the-controls-the-gates-cannot-see.md) đóng những control mà cổng kiểm không nhìn thấy,
[0035](Docs/adr/0035-one-rule-one-owner.md) đóng *lớp* lỗi: một quy tắc, một chủ sở hữu,
[0036](Docs/adr/0036-a-store-with-a-key-never-accepts-what-it-cannot-authenticate.md) một store có khoá không bao giờ chấp nhận nội dung nó không xác thực được,
[0037](Docs/adr/0037-closing-the-command-surface-the-policy-package-and-the-row-bound-state.md) đóng command surface và state ràng buộc theo dòng,
[0038](Docs/adr/0038-the-corpus-builder-is-the-second-reader.md) corpus builder là reader thứ hai,
[0039](Docs/adr/0039-plugin-tools-first-class-fail-closed-bridges-bounded-session-state.md) plugin tool hạng nhất và session state hữu hạn,
[0040](Docs/adr/0040-one-model-lease-per-entry-point-that-runs-the-graph.md) mỗi entry point chạy graph giữ một model lease,
[0041](Docs/adr/0041-one-model-owner-and-admission-before-threads.md) một model owner và admission trước khi tạo thread,
[0042](Docs/adr/0042-the-owner-enforces-the-invariant.md) chính owner thực thi invariant,
[0043](Docs/adr/0043-memory-short-and-long-term.md) bộ nhớ ngắn hạn và dài hạn,
[0044](Docs/adr/0044-parallel-tool-batch-in-the-tool-node.md) batch tool song song có kiểm soát,
[0045](Docs/adr/0045-the-agent-surface-is-a-port-too.md) agent surface cũng là một port,
[0046](Docs/adr/0046-split-the-composition-root.md) tách composition root,
[0047](Docs/adr/0047-close-the-fail-open-boundaries.md) đóng các biên fail-open,
[0048](Docs/adr/0048-authenticated-loopback-and-release-model-gate.md) loopback có xác thực và model release gate,
[0049](Docs/adr/0049-authenticated-shutdown-in-server-manager.md) shutdown có xác thực,
[0050](Docs/adr/0050-balanced-model-in-loop-release-gate.md) model-in-loop cân bằng,
[0051](Docs/adr/0051-end-to-end-authenticated-a2a-and-observable-http-boundaries.md) A2A xác thực end-to-end và HTTP boundary quan sát được,
[0052](Docs/adr/0052-web-grounded-external-knowledge.md) tìm kiếm web có nguồn cho kiến thức bên ngoài cần độ mới/kiểm chứng,
[0053](Docs/adr/0053-owned-sse-workers-and-production-path-freshness-eval.md) owner/drain cho SSE worker và model eval đi đúng đường freshness production,
[0054](Docs/adr/0054-owned-tool-deadlines-and-drain-before-close.md) deadline tool có owner và drain mọi worker trước khi đóng shared store,
[0055](Docs/adr/0055-runtime-path-parity-web-grounding-hardening-and-shutdown-bounds.md) đồng bộ path runtime, cứng hoá web grounding và chặn đóng worker,
[0056](Docs/adr/0056-isolated-mutation-workers-and-teardown-ownership.md) cô lập mutation worker và làm rõ quyền sở hữu tài nguyên teardown,
[0057](Docs/adr/0057-checkpoint-retention-is-a-runtime-contract.md) biến cửa sổ retention checkpoint thành contract runtime fail-closed,
[0058](Docs/adr/0058-verified-file-handles-and-executable-resource-leak-gate.md) xác minh file handle sau khi mở, chặn hard-link alias và biến resource leak thành gate thực thi được,
[0059](Docs/adr/0059-close-container-memory-and-concurrency-trust-gaps.md) đóng các lỗ trust của Docker peer, metadata continuation, memory prompt, reload đồng thời, state path, A2A và STM owner,
[0060](Docs/adr/0060-bounded-lsp-and-durable-worker-handoff.md) deadline LSP thật, correlation JSON-RPC, teardown process đầy đủ và handoff durable từ task sang worker,
[0061](Docs/adr/0061-windows-dpapi-storage-key-custody.md) custody khóa local Windows bằng DPAPI, migration plaintext nguyên tử và fail-closed,
[0062](Docs/adr/0062-verified-write-handles-and-ambient-container-sandbox.md) commit write/create qua handle đã xác minh và sandbox compose dùng chính container hiện tại làm OS boundary.
[0063](Docs/adr/0063-transactional-bootstrap-secure-egress-and-release-artifact-gates.md) bootstrap giao dịch có rollback, A2A HTTPS mặc định, image đủ contract và CI audit dependency/container,
[0064](Docs/adr/0064-verified-rag-bounded-replanning-and-selective-token-streaming.md) RAG có provenance/answerability, workflow replan có budget và token streaming chỉ cho prose hiển thị.

Cài đặt trong lúc phát triển: `pip install -e AIAssistant`. Cấu hình triển khai
nằm trong `config/*.toml`; các ADR ràng buộc nằm trong `Docs/adr/`.
