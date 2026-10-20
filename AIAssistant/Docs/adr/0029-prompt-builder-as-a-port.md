# ADR 0029 — Prompt builder là một collaborator riêng, không phải method của LLM runtime

## Bối cảnh

`/v1/chat/completions` **chết** trong production, và không một cổng đo được nào
báo động.

`ChatbotAgent` (đặt ở `orchestration/specialists/chatbot.py`) gọi
`self._llm._is_character_query(...)`, `self._llm.build_vision_messages(...)`,
`self._llm.build_text_messages(...)` và
`self._llm._strip_reference_citations_for_character_answer(...)`. Nhưng
`self._llm` là `LlmModule` — một **handle runtime** chỉ giữ backend, lock và
lease. Nó không có một trong bốn method đó. Method chúng nằm ở
`DefaultPromptBuilder` (`llm/prompts.py`), và `DefaultPromptBuilder` **không
được khởi tạo ở bất kỳ đâu trong `src/`** — nó chỉ xuất hiện trong test.

Hệ quả: mọi request chat đi tới `chatbot_agent.build_messages(...)` đều ném
`AttributeError: 'LlmModule' object has no attribute '_is_character_query'` →
500. Kiểm chứng trực tiếp:

```
$ PYTHONPATH=src python -c "...ChatbotAgent(llm, rag).build_messages(...)"
AttributeError: 'LlmModule' object has no attribute '_is_character_query'
```

**Vì sao mọi cổng vẫn xanh?** Vì mọi test đều mock. `test_chatbot_security_guards.py`
dựng `MagicMock()` cho "llm" — mock đáp ứng *mọi* thuộc tính, nên nó giấu
đúng cái sai lệch cần bắt. Coverage 100% statement *và* branch, 64/64 mutant
killed, eval exit 0 — và một endpoint chết. Đây là bằng chứng rõ nhất cho
luật ADR 0015: một claim phải được **ghim bằng test**, và "test xanh" không
thay thế cho "test kiểm tra đúng thứ".

Nguyên nhân gốc: một lần refactor tách `LlmModule` (handle) khỏi
`DefaultPromptBuilder` (dựng prompt) nhưng **chỉ sửa một phía** — phía
composition root. Phía specialist vẫn gọi tên method của object cũ.

## Quyết định

1. **`PromptBuilder` là một port thật, không phải một protocol chết.**
   `llm/contracts.py` đã khai báo `PromptBuilder` với đúng bốn method mà
   specialist cần (`build_messages`, `build_vision_messages`,
   `is_character_query`, `strip_citations`). Nay nó **được dùng**: specialist
   nhận nó qua constructor và chỉ gọi qua port.
2. **`ChatbotAgent` nhận ba collaborator thật:** handle LLM (cho
   `is_vision_model`), RAG runtime, và prompt builder. Không có đường suy
   diễn "tự tạo nếu thiếu" — thiếu dependency thì hỏng lúc **dựng**, không
   phải lúc chạy.
3. **Tên method theo port, không theo lịch sử.** `is_character_query` /
   `build_messages` / `strip_citations` — bỏ tiền tố `_` và bỏ
   `build_text_messages` (tên cũ chưa từng nằm trong port).
4. **Composition root sở hữu việc dựng.** `compose()` tạo
   `DefaultPromptBuilder(chars_per_token=CHARS_PER_TOKEN)` và tiêm vào
   `RuntimeModules` → `ChatbotAgentHolder` → `ChatbotAgent`. `LlmModule` quay
   lại đúng vai trò: một handle runtime.
5. **Test phải dựng object thật.** `tests/integration/test_chatbot_agent.py`
   có thêm `RealWiringTests`: dựng `LlmModule` thật + `DefaultPromptBuilder`
   thật + `ChatbotAgentHolder` thật (đúng đường của composition root) và
   gọi `build_messages`. Test này **đỏ** với code cũ — đó là điều kiện để
   nó có giá trị.

## Hệ quả

- `/v1/chat/completions` hoạt động lại với wiring thật.
- `DefaultPromptBuilder` hết trạng thái "dead code trong test".
- `PromptBuilder` hết trạng thái "protocol không ai dùng".
- Chi phí: một tham số constructor ở ba chỗ. Đổi lại: một lớp lỗi wiring
  (đúng loại lỗi mà mock giấu) không còn khả năng.
- **Bài học được ghi lại, không chỉ sửa code:** mock là công cụ để *kiểm tra
  hành vi*, không phải để *thay thế* collaborator. Test ở đây đã mock cả
  object mà lỗi nằm ở. Từ nay, mỗi collaborator mới phải có ít nhất một
  case dựng object thật.

## Không giải quyết

- **Route chat vẫn chưa có test end-to-end với backend thật** (cần model).
  Test mới dừng ở tầng specialist + wiring, đủ để bắt đúng lớp lỗi này.
- Việc các route khác (`/v1/agent/execute`) có dùng đúng collaborator hay
  không được kiểm tra riêng; lần review sau sẽ phải soi từng route.
