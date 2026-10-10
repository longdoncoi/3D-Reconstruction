# Agentic AI design patterns

The assistant is implemented as a single local-model orchestration graph with
logical specialists. This preserves clear ownership and independent checks
without duplicating the model context for every role.

| Roadmap stage | Implementation | Evidence |
| --- | --- | --- |
| Why patterns | Deterministic tool contracts, least privilege and explicit approval boundaries. | `modules/tool_contract.py`, `modules/multi_agent.py` |
| ReAct loop | `reason -> tool -> reflect -> reason` with a bounded iteration count. | `LangGraphAgent.py` |
| Reflection | A deterministic independent critic reviews every completed tool result and feeds failures into the next reasoning turn. | `reflect_result`, `reflection` UI step and `tool_reflected` audit event |
| Tool use | Pydantic validation, constrained decoding, sandboxed tools and MCP expose one shared contract. | `modules/tool_contract.py`, `modules/mcp_server.py` |
| Planning | Long non-UI requests receive a compact JSON plan before execution; short UI commands skip it. | `LocalAgentGraph._plan` |
| Multi-agent | Supervisor assigns a logical specialist, emits a handoff, and verification remains independent of execution. | `modules/multi_agent.py`, `LocalAgentGraph._reason` |
| Production ready | Timeouts/iteration limits, approval gates, audit logs, optional tracing/Prometheus, health endpoint, checkpoint backends, CI lint and deterministic evals. | `modules/observability.py`, `modules/checkpointing.py`, workflows |

## Operating policy

- `USE_LANGGRAPH_AGENT=1` enables the graph (default).
- `AGENT_OBSERVABILITY=1` enables `/metrics`; scrape it from the deployment
  monitoring system and alert on increased tool errors or latency.
- `AGENT_CHECKPOINT_BACKEND=memory` is the safe desktop default. Configure a
  Postgres or Redis backend only when durable, multi-process session recovery
  is required.
- The reflection critic is deterministic by design. It costs no extra model
  call and prevents a failed tool result from being silently treated as fact.

## Verification

Run from `AIAssistant/`:

```powershell
python -m unittest test_multi_agent.py test_agent_reflection.py test_langgraph_multi_agent.py
ruff check .
python -m compileall -q LangGraphAgent.py modules evals
```
