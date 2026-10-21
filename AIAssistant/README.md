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
│  ├─ image_utils.py                  # Trợ giúp ảnh chung cho hai bên vision và retrieval (module lá)
│  │
│  ├─ domain/                         # ── LỚP DOMAIN: thuần, không import framework nào ──
│  │  ├─ __init__.py                  # Hợp đồng domain framework-independent
│  │  ├─ agents.py                    # AgentDecisionKind + AgentDecision (kết quả điều phối)
│  │  ├─ approvals.py                 # Chính sách cấp phép HITL: grant gắn với đúng một lời gọi công cụ
│  │  ├─ context_budget.py            # Chính sách ngân sách context-window (CONTEXT_TOKEN_MARGIN…)
│  │  ├─ errors.py                    # Exception có kiểu của nền tảng
│  │  ├─ governance.py                # Quản trị dữ liệu: scrub PII, redact_value, SENSITIVE_KEYS
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
│  │  ├─ prompts.py                   # Build system prompt — không gọi LLM, không I/O
│  │  ├─ models.py                    # AgentExecuteRequest/AgentApproveRequest/… (schema Pydantic)
│  │  ├─ ids.py                       # Trợ giúp định danh dùng chung cho lớp agent
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
│  │  ├─ action_manifest.py           # Hợp đồng hành động desktop có phiên bản (chia sẻ với C++)
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
│  │  │  ├─ chat_routes.py            # /v1/chat/completions (JSON + SSE)
│  │  │  ├─ agent_routes.py           # /v1/agent/execute|cancel|approve|ui-action-result
│  │  │  ├─ admin_routes.py           # /admin/* (reload-model, reload-rag, shutdown…)
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
│     ├─ server.py                    # Entry point: compose() lười (ADR 0020/0022), main(), uvicorn
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
│  ├─ run_model_in_loop_eval.py       # Chấm planner thật (cần backend; không có thì SKIP)
│  ├─ tool_call_cases.json            # Bộ case của eval tool-call (16 case)
│  └─ *_report.json                   # Report sinh ra (git-ignored)
├─ scripts/
│  ├─ mutation_check.py               # Đảo từng quyết định trọng yếu; test sở hữu phải fail
│  ├─ generate_sbom.py                # CycloneDX: distribution từ lock + artifact GGUF (ADR 0023)
│  ├─ rekey_storage.py                # Xoay khoá at-rest: --verify / --dry-run / xoay (ADR 0030)
│  └─ verify_models.py                # Pre-flight OK/MISSING/MISMATCH cho GGUF đã tải (ADR 0023)
├─ tests/
│  ├─ unit/                           # Hành vi từng module (concurrency, approval, streaming…)
│  ├─ integration/                    # Composition root, orchestration, A2A e2e, guard ranh giới AST
│  └─ contract/                       # Hợp đồng Qt ↔ server và multi-agent
├─ Docs/adr/                          # 31 Architecture Decision Record (0001–0031)
├─ Docs/archive/                      # Tài liệu đã hạ cấp (giữ để tham chiếu)
├─ Docs/RUNBOOK.md                    # SLO, cảnh báo, quy trình production (ADR 0024)
├─ Dockerfile                         # Image non-root (không kèm trọng số; mount Models/ + data)
├─ .env.example                       # Mẫu biến môi trường (sandbox, checkpoint, governor, observability)
├─ pyproject.toml                     # ruff · mypy strict · coverage fail_under=100 · import-linter
├─ requirements.txt                   # Dependency runtime (llama-cpp-python cài tay)
├─ requirements.lock                  # Pin phiên bản + SHA-256 (`uv --generate-hashes`, ADR 0023)
├─ requirements-test.lock             # Pin hash cho bộ test — CI cài bằng `--require-hashes`
└─ requirements-test.txt              # Dependency cho bộ test/eval offline
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
  và tiêm vào router/service — không còn class-level hay module-global mutable
  state.
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
* **State cục bộ được mã hoá at-rest** (ADR 0027). `adapters/persistence/storage_crypto.py`
  bọc **theo field** bằng AES-256-GCM: envelope `enc:v1:<base64(nonce||ciphertext)>`,
  nonce random 96-bit cho từng field, và **tên field làm associated data** nên
  ciphertext bị chuyển sang cột khác sẽ fail authentication. `compose()` nạp key
  **một lần** (fail-closed) rồi tiêm vào `SqliteTaskStore` và
  `PendingActionStore`; đọc state đã seal mà thiếu/sai key thì **abort**, không
  bao giờ trả ciphertext lên tầng trên. Key lấy từ `AI_STORAGE_KEY` →
  `AI_STORAGE_KEY_FILE` → tự sinh một lần ở `%LOCALAPPDATA%`/`$XDG_CONFIG_HOME`
  (**ngoài** `APP_DATA_DIR`, nên backup volume không mang theo key). Cột chỉ mô tả
  (id, status, owner, timestamp) vẫn plaintext để SQL filter/retention/tenant
  còn chạy trên chỉ mục thật.
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
  `MemorySaver` im lặng); `AGENT_SANDBOX_REQUIRE_ISOLATION=1` từ chối runtime
  local (chỉ chạy tool trong docker `--cap-drop ALL` + `no-new-privileges`);
  `AGENT_MAX_CONCURRENT_RUNS` và `AGENT_RATE_LIMIT_PER_MINUTE` bật governor
  admission (hết slot → 503, vượt quota → 429 + `Retry-After`; `/livez`,
  `/readyz` được miễn). Desktop không đặt gì thì hành vi không đổi; deployment
  nhiều người dùng xem [`Docs/RUNBOOK.md`](Docs/RUNBOOK.md) (SLO, cảnh báo,
  quy trình chỉnh quota/backup/rollback).
* Người phản biện là tất định **bằng thiết kế**: không tốn một lời gọi mô hình
  thêm nào, và ngăn một kết quả công cụ thất bại bị âm thầm coi là sự thật.

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
* **Cứng hoá transport.** ADR 0008 — peer loopback + allow-list `Origin` + bearer
  tuỳ chọn theo từng bề mặt, A2A remota không thể bật khi thiếu credential:
  `AI_A2A_TOKEN` (single-tenant) hoặc mapping `AI_A2A_TENANTS` (mỗi token sở
  hữu task riêng). ADR 0009 — phê duyệt là capability dùng một lần, gắn với đúng
  lời gọi (`approval_fingerprint` + grant store). ADR 0010 — domain sở hữu một
  chính sách redaction duy nhất. ADR 0011 — sandbox lệnh từ chối code interpreter
  nội tuyến (`-c`, `-m`, `-`) và shell substitution.
* **Mã hoá at-rest cho state cục bộ** (ADR 0027). Prompt, đối số tool, kết quả
  model và event log không còn nằm plaintext trên đĩa. Key được **tiêm** từ
  composition root (không có global), đọc thiếu/sai key thì **fail closed**,
  và row plaintext cũ được seal đúng **một lần** nhờ `PRAGMA user_version = 3`.
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
| Health & metrics | `/health`, `/livez`, `/readyz`, `/metrics` |
| Chat (tương thích OpenAI) | `/v1/models`, `/v1/chat/completions` (JSON và SSE) |
| Agent | `/v1/agent/execute` (JSON hoặc SSE), `/v1/agent/cancel`, `/v1/agent/approve`, `/v1/agent/ui-action-result` |
| Admin | `/admin/release-vram`, `/admin/reload-model`, `/admin/reload-rag`, `/admin/reload-agent`, `/admin/shutdown` |
| A2A | `/.well-known/agent.json`, `/message:send`, `/tasks/*` (`:cancel`, `:resume`, `:subscribe`), `/a2a/tasks/*` (tương thích) |
| MCP | `/mcp` (chỉ mount khi bật và hợp lệ; loopback-only trừ khi có opt-in tường minh) |

**A2A.** Bề mặt chính là `POST /message:send`, `GET /tasks/{id}`, `GET /tasks`,
`POST /tasks/{id}:cancel`, `POST /tasks/{id}:subscribe`; discovery nằm ở
`/.well-known/agent.json`. Agent Card quảng bá giao diện REST của nó và yêu cầu
header `A2A-Version: 0.3` khi client gửi header phiên bản. Phê duyệt và ACK của
Qt làm **tạm dừng** task bền vững ở trạng thái `input-required`; chúng chỉ tiếp
tục qua extension được khai báo
`urn:3d-reconstruction:a2a:approval-resume:v1` tại `POST /tasks/{id}:resume`;
ngữ cảnh thực thi được persist nội bộ và không bao giờ lộ trong artifact.

**MCP.** `adapters/mcp.py` là adapter bên ngoài; thực thi agent **không** gọi
`/mcp` nội bộ. Nếu bind host vươn ra mạng mà thiếu `AI_MCP_TOKEN`, endpoint bị
từ chối mount (fail closed, ADR 0008).

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

# Chạy máy chủ trực tiếp (loopback, cổng 8080; AGENT_HOST / AGENT_PORT đổi địa chỉ bind)
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
  -H "Content-Type: application/json" \
  -d '{"task": "Liệt kê các file trong src/", "session_id": "demo", "language": "en"}'
```

Container:

```powershell
# Build từ gốc repo để tiền tố AIAssistant/ phân giải đúng
docker build -f AIAssistant/Dockerfile -t ai-assistant .
docker run -p 8080:8080 -v ${PWD}/AIAssistant/Models:/app/AIAssistant/Models ai-assistant
```

Image chạy bằng user không phải root và đẩy dữ liệu có thể thay đổi (cache RAG,
task store, log) vào `APP_DATA_DIR=/app/data`; image không kèm trọng số, nên
hãy mount chúng.

## Kiểm thử

```powershell
cd AIAssistant
ruff check .
python -m mypy src/ai_assistant
python -m compileall -q evals src
PYTHONPATH=src lint-imports --no-logo
PYTHONPATH=src python -m coverage run -m unittest discover -s tests -p "test_*.py"
python -m coverage report
PYTHONPATH=src python scripts/mutation_check.py
```

Các cổng là **cấu hình**, không phải cờ dòng lệnh: `pyproject.toml` sở hữu breadth
lint (`E/W/F/I/BLE/S/RUF/C90/N/TRY`, complexity ≤ 15), thiết lập mypy strict và
sàn coverage theo branch (`fail_under = 100`), nên lần chạy cục bộ và CI đo cùng
một thứ. Theo lần đo ngày 2026-10-21 (các con số đo được được ghi lại trong
`Docs/SCORECARD.md` — **đã theo git** từ ADR 0032): **100.00%** statement *và*
branch toàn package (9 220 statement / 2 436 branch, 2 093 test), và **87/87**
kill trong mutation check. Số mutant tối thiểu cũng là một cổng
(`MIN_MUTANTS`): xoá một mutant là một quyết định, và cổng biến nó thành
quyết định *nhìn thấy được*.

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
* `evals/` — hai eval tất định, offline chạy trong CI trên mọi push/PR chạm
  `AIAssistant/**` (`run_tool_eval.py` cho envelope tool-call — 16 case gồm
  adversarial: tool bịa, tham số lạ, type không coerce, JSON cụt;
  `run_agentic_eval.py` cho chính sách supervisor + gateway thật — 25
  scenario/114 case gồm grant dùng-một-lần, rollback/cancel, scope và
  classification; ADR 0025). `run_model_in_loop_eval.py` chạy tay và cần
  backend thật, có `--baseline` để gate tụt điểm so với report cũ. Cả ba
  script đều tự đặt `src/` lên `sys.path`, và một case **không chạy được** sẽ
  được báo là *not run* và exit khác 0.
* Coverage không thể phát hiện một test asserted sai chiều, nên các guard trọng
  yếu (allowlist sandbox, phê duyệt HITL, chính sách remota, `/health`, niêm
  phong cache, các cuộc đua cancel, cổng ranh giới bước ADR 0019) còn được
  mutation-check.

`.github/workflows/Build-Python-ci.yml` chạy các cổng lint/type/test/coverage;
`agent-evals.yml` chạy các eval. Không workflow nào load mô hình.

## Tài liệu & ADR

| Tài liệu | Vai trò |
|----------|---------|
| `README.md` (file này) | Điểm vào duy nhất: layout, kiến trúc, API, cài đặt, kiểm thử |
| `Docs/SCORECARD.md` | Các metric đo được (coverage, mutation, lint, type, eval) — bản local, không theo git |
| [`Docs/RUNBOOK.md`](Docs/RUNBOOK.md) | Vận hành production: SLO, cảnh báo, quota, checkpoint, luân chuyển pin model |
| [`Docs/adr/`](Docs/adr/) | 31 Architecture Decision Record (0001–0031) |
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
[0035](Docs/adr/0035-one-rule-one-owner.md) đóng *lớp* lỗi: một quy tắc, một chủ sở hữu.

Cài đặt trong lúc phát triển: `pip install -e AIAssistant`. Cấu hình triển khai
nằm trong `config/*.toml`; các ADR ràng buộc nằm trong `Docs/adr/`.
