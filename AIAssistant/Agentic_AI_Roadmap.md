# Agentic AI / Multi-Agent Roadmap

## Mục tiêu

Xây dựng một AI Assistant duy nhất cho ứng dụng: trả lời hội thoại, tra cứu
RAG, gọi MCP tool và thực hiện workflow desktop có kiểm soát. Không tách Chat
Mode và Agent Mode ở giao diện.

## Giai đoạn 1 — Unified Assistant (đã bắt đầu)

- Một endpoint Agent cho mọi tin nhắn; LLM tự quyết định trả lời trực tiếp hay
  gọi tool.
- Truyền lịch sử hội thoại, file đính kèm và ngôn ngữ giao diện vào request.
- MCP Streamable HTTP là ranh giới thực thi tool; Qt vẫn là nơi xác nhận action
  giao diện và các thao tác có tác động.
- Duy trì approval cho ghi file, chạy lệnh và action có rủi ro.

**Tiêu chí hoàn thành:** chat thường, lệnh UI, RAG và tool MCP dùng chung một
session; không có role nội bộ nào đi vào payload LLM.

## Giai đoạn 2 — Điều phối Agentic

Thêm `Supervisor` (LangGraph) với trạng thái phiên rõ ràng: mục tiêu, kế hoạch,
tool calls, quan sát, giới hạn vòng lặp và kết quả. Supervisor chỉ được phép:

1. Trả lời trực tiếp cho câu hỏi đơn giản.
2. Giao việc cho một specialist.
3. Yêu cầu xác nhận người dùng trước hành động rủi ro.

Mỗi tool cần JSON Schema, timeout, audit log, idempotency key và policy quyền
truy cập riêng.

## Giai đoạn 3 — Multi-Agent theo chuyên môn

| Agent | Trách nhiệm | Quyền tool |
| --- | --- | --- |
| Supervisor | Phân loại ý định, lập kế hoạch, tổng hợp | Chỉ điều phối |
| RAG/Research | Tìm tài liệu, đọc source, trích dẫn | read/search/rag |
| Desktop Workflow | Điều khiển Viewer, Reconstruction, AI Processor | application_action qua Qt ACK |
| Code Agent | Đề xuất/chỉnh mã, chạy kiểm thử | read mặc định; write/run cần approval |
| Verification Agent | Kiểm chứng kết quả, test và policy | read/validate/test an toàn |

Các agent không gọi trực tiếp lẫn nhau; Supervisor truyền context tối thiểu và
nhận structured result. Điều này giúp giới hạn context, kiểm toán được và tránh
vòng lặp giữa agent.

## Giai đoạn 4 — Độ tin cậy và vận hành

- Đo success rate theo tool, thời gian phản hồi, tỷ lệ approval/reject và lỗi
  schema.
- Evals cố định cho: hội thoại VI/EN, RAG, lệnh UI, thao tác bị từ chối và lỗi
  MCP/network.
- Checkpoint theo session để resume workflow sau khi server khởi động lại.
- Chỉ thêm agent mới khi một nhóm task có metric và ownership rõ ràng.

## Thứ tự ưu tiên đề xuất

1. Hoàn thiện test endpoint Unified Assistant và regression 422.
2. Chuẩn hoá MCP schemas và tool policy.
3. Thêm Supervisor + Verification Agent.
4. Tách RAG/Workflow/Code specialist khi eval chứng minh cần thiết.
