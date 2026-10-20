# AIComputerVision — Scorecard

> Số đo được, mỗi số đều do một lệnh trong repo này tạo ra — không có gì ở
> đây là ước lượng. **Đo lần cuối: 2026-10-20**, Windows, Python 3.11,
> dependency theo `requirements.txt` (`ultralytics==8.4.41`, `onnx==1.21.0`,
> `onnxruntime>=1.20.1`).
>
> Tài liệu này **không tuyên bố một điểm số**. Nó ghi những gì đo được, cách
> đo lại, và danh sách việc còn mở. Luật của [ADR 0007](adr/0007-every-claim-is-pinned-or-removed.md)
> áp dụng cho chính tài liệu này: mọi con số dưới đây đều tái lập được bằng
> một lệnh, và mọi claim phải được *pin* bằng test/mutant hoặc được ghi thẳng
> vào bảng nợ ở §3.

## 1. Cổng chất lượng

| Cổng | Lệnh | Đo được 2026-10-20 |
|------|------|---------------------|
| Lint | `ruff check .` | 0 issue |
| Kiểu | `python -m mypy` | 0 issue, 30 file (`disallow_untyped_defs`, `warn_return_any`, `strict_equality`…) |
| Byte-compile | `python -m compileall -q cvtrain tests TrainModel.py scripts` | sạch |
| Test | `python -m unittest discover -s tests` | **397 test, 0 fail** |
| Coverage | `coverage run -m unittest discover -s tests && coverage report` | **100%** — 1 525 stmts / 368 branch, 0 miss, 0 partial (`fail_under = 100`) |
| Mutation | `python scripts/mutation_check.py` | **57/57 killed**, 0 survived, 0 hung, 0 stale |
| Ranh giới kiến trúc | `python -m unittest discover -s tests -p "test_architecture.py"` | 13 test: đồ thị phụ thuộc một chiều, heavy dep lazy + một owner, `UltralyticsBackend` chỉ ở `cli.py`, Layout README ↔ package, cờ CLI ↔ README (hai chiều) |
| Hợp đồng C++ | `python -m unittest discover -s tests -p "test_contract.py"` + `-p "test_qt_contract.py"` | 7 + 9 test: tên file ONNX, trần opset theo ORT được pin, schema tensor ↔ `AIYoloPostProcessor.h`, ngưỡng IoU ↔ `AIProcessor.cpp`, argv/nhãn/regex `pct=` ↔ `AITrainDockWidget.cpp` |
| Verdict | `python TrainModel.py --verify-manifest` | exit 0; sha256 của cả 3 artifact khớp manifest |

> **Gate mutation chạy trong sandbox.** Script copy package + test + nguồn C++
> vào thư mục tạm rồi đảo chiều *bản copy*; cây làm việc không bị ghi. Sandbox
> không có `Dataset/`, không có checkpoint, không có `Models/`, nên một mutant
> lọt vào nhánh huấn luyện sẽ **fail nhanh ở preflight** thay vì chạy thật
> (xem ADR 0007).

## 2. Tiêu chí "10/10" (5 điều kiện, tất cả đều đo được)

| # | Tiêu chí | Điều kiện kiểm tra | Trạng thái |
|---|----------|--------------------|-----------|
| **C1** | 5 cổng xanh | Lint + kiểu + test + coverage + mutation đều exit 0 | ✅ |
| **C2** | Không dòng nào thoát khỏi phép đo | `fail_under = 100` (statement **và** branch) trong `pyproject.toml`; CI đỏ nếu dưới | ✅ 1 525/1 525 |
| **C3** | Mọi quyết định trọng yếu đã được ghim | 57 mutant, 0 survived, 0 hung, 0 stale | ✅ 57/57 |
| **C4** | Nợ đã đo và ghi | Bảng §3 — mỗi dòng có số đo và hướng xử lý | ✅ 5 nợ |
| **C5** | Tài liệu khớp code | README ↔ CLI hai chiều, Layout ↔ package, ADR 0003 ↔ manifest thật, hợp đồng C++ hai chiều | ✅ |

Điểm khác biệt so với trước: **C3 trước đây chỉ phủ 8 quyết định**. Một mutant
đảo chiều làm `exporter_identity()` ghi cứng `onnx=unknown` — tắt đúng cái bảo
vệ mà ADR 0003 tuyên bố — vẫn xanh suite. Nay nó đỏ.

## 3. Bảng nợ đang mở

Nguyên tắc (giống `Docs/ARCHITECTURE_SRC.md` phía C++): **nợ phải được đo và
được ghi**, không được biến mất khỏi báo cáo.

| # | Nợ | Đo được | Hướng xử lý |
|---|-----|---------|-------------|
| N1 | **Provenance của model đã ship dùng định dạng cũ.** `Models/manifest.json` ghi `backend_version: "8.4.41"` (bản `ultralytics` trần) cho `yolo11n.onnx` / `yolo11n-seg.onnx`, và **không có** `backend_version`/`epochs`/`metrics` cho `yolo11x-tracking.onnx` — trong khi code hiện tại ghi identity tổng hợp `cvtrain=…;ultralytics=…;torch=…;onnx=…` | `Models/manifest.json` → `entries.*.backend_version` | (a) retrain để sinh lại manifest đúng định dạng; (b) **không bao giờ** ghi một identity không được ghi lúc train. Hệ quả hiện tại: `--skip-unchanged` **retrain** các model này (fail-closed) → tốn thời gian, không sai. Đã ghim bằng `test_an_entry_with_the_legacy_exporter_version_is_trained`. |
| N2 | **Provenance gắn với một máy.** `dataset` và `run_dir` trong manifest là đường dẫn tuyệt đối (`F:\…\Dataset\data.yaml`), và `data.yaml` của repo chứa `path:` tuyệt đối | `Models/manifest.json` → `entries.*.dataset` | Quyết định có chủ ý (ADR 0005: đường dẫn tương đối sẽ đổi ngữ nghĩa skip). Hệ quả: `--skip-unchanged` không portable sang máy khác, và auditor trên máy khác đọc đường dẫn vô nghĩa. Nếu cần portability: so sánh theo digest nội dung, không theo chuỗi đường dẫn. |
| N3 | **Không có `resume` cho training dài.** Mỗi lần chạy tạo thư mục `runs/<x>/train-<timestamp>-<pid>` mới; ultralytics hỗ trợ `resume` | `cvtrain/backend.py:255` (`UltralyticsBackend.train`) | Crash ở epoch 4/5 mất trắng cả run. Cần: truyền `resume` khi run dir tồn tại và cùng cấu hình. Ưu tiên thấp (giả định hiện tại: một lần chạy = một run sạch). |
| N4 | **Fingerprint dataset chạy khi progress bar đứng yên.** `snapshot_dataset` hash toàn bộ byte của dataset trước khi train, không có output | `cvtrain/gate.py:149` → `cvtrain/dataset.py:306` | Với dataset lớn (vài GB) đây là vài phút im lặng ở mốc 2%. Cần: phát tiến độ theo số file đã hash. |
| N5 | **`module_available` chỉ kiểm `find_spec`**, không chứng minh package import được | `cvtrain/checks.py:62` | Một bản cài hỏng (ultralytics có, torch thiếu) qua được preflight rồi hỏng muộn hơn. Taxonomy vẫn trả exit `2` (qua `BackendUnavailableError`), nên chỉ lệch *thời điểm*, không lệch *mã lỗi*. |

## 4. Claim nào đã được ghim

Mỗi claim trong README/ADR dưới đây có một cơ chế làm đỏ nó. Claim không có
cơ chế thì **không được viết** (ADR 0007).

| Claim | Ghim bằng |
|-------|-----------|
| Package import-light (không cần ML stack) | `test_architecture.LazyDependencyTests` + `test_backend.test_construction_is_lazy` |
| Đồ thị phụ thuộc một chiều, không cycle | `test_architecture.ImportGraphTests` (bảng `ALLOWED_IMPORTS`) |
| `cli.py` là composition root duy nhất | `test_architecture.test_the_concrete_backend_is_named_only_by_the_composition_root` |
| Hợp đồng ONNX với runtime Qt (rank/dtype/output/opset) | `test_onnx_contract` + **8 mutant** trong `cvtrain/onnx_contract.py` |
| Hợp đồng tên file / trần opset / schema / IoU với C++ | `test_contract`, `test_qt_contract` (+ presence guard chống rename âm thầm) |
| Provenance fail-closed (thiếu dữ liệu ⇒ retrain) | `test_trainer.SkipUnchangedTests` + **5 mutant** trong `cvtrain/gate.py` |
| Quality gate không bị bypass (kể cả khi skip) | `test_trainer` + **5 mutant** (3 `gate.py` + 2 `trainer.py`) |
| Identity của exporter ghi đủ 4 dependency | `test_backend.test_the_identity_carries_the_actual_version_of_each_dependency` + **2 mutant** (`backend.py`) |
| Chọn model: key lạ ⇒ lỗi, không lặng lẽ train mặc định | `test_config` + **3 mutant** (`config.py`) |
| Fail-fast preflight | `test_preflight` + **5 mutant** (`preflight.py`) |
| Fingerprint dataset phủ dir/glob/`.txt`/list | `test_dataset` + **4 mutant** (`dataset.py`) |
| Publish nguyên tử + rollback `.bak` + kiểm hợp đồng *trước* khi publish | `test_fsio`, `test_trainer` + **4 mutant** (3 `fsio.py` + 1 `trainer.py`) |
| Advisory lock: O_EXCL, token, claim, stale, heartbeat | `test_fsio` + **4 mutant** (`fsio.py`) |
| Manifest: quarantine, verdict nghiêm, 2 chiều, migration | `test_manifest` + **8 mutant** (`manifest.py`) |
| Provenance ghi *một lần mỗi model* (không mất khi crash giữa chừng) | `test_trainer.FailurePathTests` + **1 mutant** (`trainer.py`) |
| Tiến trình đơn điệu + marker `done`/`error` | `test_progress` + **2 mutant** (`progress.py`, `trainer.py`) |
| Mã thoát theo nguyên nhân | `test_cli.ExitCodeTests` + **4 mutant** (`errors.py`, `cli.py`) |
| Metric không phải số bị loại khỏi manifest | `test_backend` + **1 mutant** (`backend.py`) |
| Lệnh manifest không bao giờ huấn luyện | `test_cli.test_the_manifest_commands_never_start_a_training_run` |
| README ↔ CLI hai chiều, Layout ↔ package | `test_architecture.DocumentationTests` |

Tổng: **57 mutant** = `gate.py` 8 · `onnx_contract.py` 8 · `manifest.py` 8 ·
`fsio.py` 7 · `trainer.py` 6 · `preflight.py` 5 · `config.py` 3 ·
`backend.py` 3 · `cli.py` 2 · `errors.py` 2 · `dataset.py` 4 ·
`progress.py` 1.

## 5. Cách đo lại

```powershell
cd AIComputerVision
ruff check .
python -m mypy
python -m compileall -q cvtrain tests scripts TrainModel.py
coverage run -m unittest discover -s tests
coverage report
python scripts/mutation_check.py
python TrainModel.py --verify-manifest
```

Toàn bộ chạy **offline**: không tải ONNX Runtime, không cài vcpkg, không cần
ML stack cho bốn cổng đầu. Bản mutation chạy trong sandbox nên không đụng tới
`Models/`, `runs/` hay dataset của máy.
