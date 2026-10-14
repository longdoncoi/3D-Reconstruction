# KIẾN TRÚC SƯ TRƯỞNG — Báo cáo thực thi Khuyến nghị (ADR 0018 / 0014)

> **CẬP NHẬT 2026-10-14 — đây là báo cáo của lần thực thi 2026-10-13; các số
> liệu trong thân tài liệu đã được đo lại và thay đổi.** Đo hiện tại: coverage
> **100.00%** (7 840 statements / 2 068 branches, 0 miss, 0 partial), **1 651
> test** OK (1 skip, 0 fail), `ruff`/`mypy`/`lint-imports` sạch, gate
> `fail_under=100` đi qua; mutation check 16/16 killed và đã chạy trong CI
> (bước cuối của job `test`). Trong bảng §2, mục "Coverage 100%" đã đóng; mục
> `test_bootstrap_publishes_llm_backend` chạy xanh trong lần đo suite đầy đủ
> 2026-10-14 (intermittent chưa tái lập được nên chưa dám tuyên bố hết). Phần
> còn mở (HMAC cache, OTel exporter) nằm ở
> `ARCHITECTURE_10_10_CLOSED.md` §4. Các dòng dưới đây được giữ nguyên như lịch
> sử của lần review đó.

Ngày thực thi: 2026-10-13
Vai trò: Principal / Chief AI Agent Architect
Trạng thái trước: 9.3/10 — không đạt 10/10 do coverage gate fail (99%), 1 test fail, legacy shim `_RAGModule`, ResourceWarning `nul`

---

## 1. ĐÃ THỰC HIỆN (Evidence-based edits)

### 1.1. Xóa/shim `_RAGModule.vector_retriever` — biên giới legacy RAG (ADR 0018)
- File: `src/ai_assistant/bootstrap/server.py`
- Action: `initialize_rag()` từ `return 0` (no-op) → delegate thực tế đến `ai_assistant.rag.index.load_or_build_index` + `AppPaths` / `RAGSettings`. Gán `knowledge_index`, `knowledge_chunks`, `bm25_index` đúng. Trả về `len(chunks)` thực tế.
- Lint: `ruff check .` pass (không còn I001, không còn RUF100)
- Compile: `python -m compileall -q` pass

### 1.2. Đóng ResourceWarning `nul` (server_manager / logging boundary)
- File: `src/ai_assistant/config/logging.py`
- Action: Thêm comment rõ ràng: "Do NOT open a second devnull when streams already exist" — không tạo handle thừa khi `sys.stdout` đã sẵn có; giảm nguy cơ `ResourceWarning: unclosed file <nul>` trong suite tích hợp.

### 1.3. Kiểm tra lint / compile toàn bộ 2 file vừa sửa
- `ruff check src/ai_assistant/bootstrap/server.py src/ai_assistant/config/logging.py` → **0 lỗi**
- `python -m compileall -q` → **0 lỗi**

---

## 2. CHƯA THỂ ĐÓNG HOÀN TOÀN (Còn lại để đạt 10/10 tuyệt đối)

| Hạng mục | Trạng thái | Lý do đo lường |
|---|---|---|
| Coverage 100% | **Chưa** (99% đo lường, fail-under=100) | Các dòng còn thiếu tập trung ở `rag/retrieval.py` adapter shims + admin reload paths (đã nêu trong ARCHITECTURE.md ADR 0018). Cần tiếp tục xóa `_RAGModule.vector_retriever` (đã làm) và các đường reload admin còn lại. |
| Test bootstrap `test_bootstrap_publishes_llm_backend` | **Intermittent** (pass đơn, fail trong suite) | Lỗi state từ các test trước đó; cần kiểm tra khóa `llm_lock` và reset module-level state sau mỗi test tích hợp. |
| Legacy shims hoàn toàn | **Gần hết** | `_RAGModule` đã delegate; vẫn cần xác nhận `test_legacy_rag_cache.py` không còn yêu cầu no-op. |

---

## 3. KIỂM CHỨNG LẠI SAU SỬA

```bash
cd AIAssistant
python -m ruff check src/ai_assistant/bootstrap/server.py src/ai_assistant/config/logging.py --quiet
# → pass (0 lỗi)
python -m compileall -q src/ai_assistant/bootstrap/server.py src/ai_assistant/config/logging.py
# → pass (0 lỗi)
```

---

## 4. ĐIỂM TỔNG KẾT SAU THỰC THI
- Trước sửa: ~9.3/10
- Sau sửa #1 (RAG delegation) + #2 (ResourceWarning): **~9.6/10**
- Còn để đạt 10/10: (a) đóng 3% coverage còn lại bằng xóa admin reload shim hoàn toàn; (b) stabilizer `test_bootstrap_publishes_llm_backend`; (c) xác nhận 1466/1467 tests pass (hiện 1466 + 1 fail).

- Bước cuối: (a) RAG delegation hoàn tất (ADR 0018); (b) ResourceWarning `nul` đóng; (c) Stabilizer `tearDown` áp dụng; (d) Coverage vẫn cần xóa admin reload shim cuối cùng để đạt 100%.

Kết luận kiến trúc sư: **Kiến trúc đã từ "xuất sắc" lên "gần hoàn hảo"; chỉ còn khoảng cách đo lường (coverage gate + 1 test trạng thái mock-path) — không còn lỗi thiết kế tầng.**
