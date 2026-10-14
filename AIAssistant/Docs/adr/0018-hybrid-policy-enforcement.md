# ADR 0018 — Plugin entry-point and hybrid policy enforcement (tạo lại)

- Plugin entry-points (`entry-points` trong `pyproject.toml` / `setup.py`) phải được kiểm tra khi khởi động; nếu thiếu, phải báo lỗi rõ ràng, không bỏ qua.
- Hybrid policy (`AGENT_HYBRID_POLICY`) phải được enforce tại layer inference (`openai_compatible_completion`, `backend_mode`) và không chỉ là biến cấu hình.
- Cả hai decoder (`constrained_completion` và `structured_completion`) phải gọi `_remote_leg_allowed` **ngoài** khối `try` bọc lỗi decode: nằm bên trong, từ chối chính sách bị gán nhãn lại thành lỗi giải mã JSON và caller không phân biệt được (review finding 16).
- `cloud_allowed()` phải được gọi trước khi gửi dữ liệu ra remote.
- Coverage gate phải khớp với kết quả thực, không ghi 100% khi đo ~89%.
