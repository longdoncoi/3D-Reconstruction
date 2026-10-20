# AIComputerVision — Scorecard

> Số đo được, mỗi số đều do một lệnh trong repo này tạo ra — không có gì ở
> đây là ước lượng. **Đo lần cuối: 2026-10-21**, Windows, Python 3.11,
> dependency theo `requirements.txt` (`ultralytics==8.4.41`, `onnx==1.21.0`,
> `onnxruntime>=1.20.1`), GPU NVIDIA RTX 3050 (chỉ dùng cho lần retrain ở N1 —
> bốn cổng đầu chạy offline, không cần GPU).
>
> Tài liệu này **không tuyên bố một điểm số**. Nó ghi những gì đo được, cách
> đo lại, và danh sách việc còn mở. Luật của [ADR 0007](adr/0007-every-claim-is-pinned-or-removed.md)
> áp dụng cho chính tài liệu này: mọi con số dưới đây đều tái lập được bằng
> một lệnh, và mọi claim phải được *pin* bằng test/mutant hoặc được ghi thẳng
> vào bảng nợ ở §3.

## 1. Cổng chất lượng

| Cổng | Lệnh | Đo được 2026-10-21 |
|------|------|---------------------|
| Lint | `ruff check .` | 0 issue |
| Kiểu | `python -m mypy` | 0 issue, 30 file (`disallow_untyped_defs`, `warn_return_any`, `strict_equality`…) |
| Byte-compile | `python -m compileall -q cvtrain tests TrainModel.py scripts` | sạch |
| Test | `python -m unittest discover -s tests` | **420 test, 0 fail** |
| Coverage | `coverage run -m unittest discover -s tests && coverage report` | **100%** — 1 593 stmts / 392 branch, 0 miss, 0 partial (`fail_under = 100`) |
| Mutation | `python scripts/mutation_check.py` | **67/67 killed**, 0 survived, 0 hung, 0 stale |
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
| **C2** | Không dòng nào thoát khỏi phép đo | `fail_under = 100` (statement **và** branch) trong `pyproject.toml`; CI đỏ nếu dưới | ✅ 1 593/1 593 |
| **C3** | Mọi quyết định trọng yếu đã được ghim | 67 mutant, 0 survived, 0 hung, 0 stale | ✅ 67/67 |
| **C4** | Nợ đã đo và ghi | Bảng §3 — mỗi dòng có số đo, cách đóng và cơ chế ghim | ✅ **6 đóng / 0 đang mở** |
| **C5** | Tài liệu khớp code | README ↔ CLI hai chiều, Layout ↔ package, ADR 0003/0008 ↔ manifest thật, hợp đồng C++ hai chiều | ✅ |

Điểm khác biệt so với trước: **C3 trước đây chỉ phủ 8 quyết định**. Một mutant
đảo chiều làm `exporter_identity()` ghi cứng `onnx=unknown` — tắt đúng cái bảo
vệ mà ADR 0003 tuyên bố — vẫn xanh suite. Nay nó đỏ. Năm nợ ở §3 được đóng
bằng quyết định + cơ chế ghim, không bằng lời.

## 3. Bảng nợ

Nguyên tắc (giống `Docs/ARCHITECTURE_SRC.md` phía C++): **nợ phải được đo và
được ghi**, không được biến mất khỏi báo cáo. Đóng nợ = một quyết định được
ghi thành ADR + một cơ chế làm đỏ nó (test hoặc mutant) — không phải một dòng
ghi chú.

| # | Nợ | Đo được | Cách đóng | Trạng thái |
|---|-----|---------|-----------|------------|
| N1 | **Provenance của model đã ship dùng định dạng cũ.** `Models/manifest.json` ghi `backend_version: "8.4.41"` (bản `ultralytics` trần) cho `yolo11n.onnx` / `yolo11n-seg.onnx`, và **không có** `backend_version`/`epochs`/`metrics` cho `yolo11x-tracking.onnx` — trong khi code ghi identity tổng hợp `cvtrain=…;ultralytics=…;torch=…;onnx=…` | `Models/manifest.json` → `entries.*.backend_version` | **Retrain bằng build hiện tại** (ADR 0008) — identity luôn do *run sinh ra artifact* ghi, không bao giờ dựng lại sau. Đo lại sau retrain (2026-10-21): cả **3** mục đều mang `backend_version = cvtrain=1.0.0;ultralytics=8.4.41;torch=2.7.0+cu128;onnx=1.21.0` + `epochs/imgsz/batch/device` + `trained_at` + `metrics`; `metrics/mAP50(B)` của det và seg **giữ nguyên 0.975** ⇒ không suy giảm chất lượng; tracking được đo lần đầu (0.975) thay vì "không rõ". Phép thử quyết định: `TrainModel.py --yes --skip-unchanged --det --seg --track` → **cả 3 mục "already matches this invocation; skipping"** (trước retrain: cả 3 đều bị train lại). Ghim bằng `test_an_entry_with_the_legacy_exporter_version_is_trained` | ✅ Đóng (2026-10-21) |
| N2 | **Provenance gắn với một máy.** `dataset` và `run_dir` trong manifest là đường dẫn tuyệt đối, và quy tắc skip *so sánh chuỗi đường dẫn* | `Models/manifest.json` → `entries.*.dataset` | **So sánh theo digest nội dung, không theo đường dẫn** (ADR 0008). Đường dẫn vẫn được *ghi* (provenance) nhưng không được *so sánh*: `dataset_sha256` + `dataset_content_sha256` mới là bằng chứng "cùng một dataset". Ghim bằng `test_the_same_dataset_behind_another_path_is_not_retrained` | ✅ Đóng (2026-10-21) |
| N3 | **Không có `resume` cho training dài.** Mỗi lần chạy tạo thư mục `runs/<x>/train-<timestamp>-<pid>` mới; crash ở epoch 4/5 mất trắng cả run | `cvtrain/backend.py` (`UltralyticsBackend.train`) | **`--resume`** (ADR 0008): lấy `weights/last.pt` mới nhất của model và train với `resume=True`, nên framework khôi phục đúng siêu tham số và bộ đếm epoch của run bị gián đoạn. Mặc định tắt. **Kiểm chứng thật (2026-10-21):** một run `yolo11x` bị giết ở epoch 5/5 → `--resume` in ra `Resuming training … from epoch 5 to 5 total epochs` và chỉ train nốt epoch cuối, không train lại từ đầu. Ghim bằng 3 test ở `test_backend.TrainTests` + 2 mutant | ✅ Đóng (2026-10-21) |
| N4 | **Fingerprint dataset chạy khi progress bar đứng yên.** `snapshot_dataset` hash toàn bộ byte của dataset trước khi train, không có output | `cvtrain/gate.py` → `cvtrain/dataset.py` | **Fingerprint báo tiến độ** (ADR 0008): `on_file(done, total)` đi từ `dataset.py` lên tới reporter, và giao thức progress có thêm pha `dataset` (2–8 %). Ghim bằng `test_the_fingerprint_reports_progress_for_every_file` + 2 mutant | ✅ Đóng (2026-10-21) |
| N5 | **`module_available` chỉ kiểm `find_spec`**, không chứng minh package import được | `cvtrain/checks.py` | **Backend tự chứng minh khả năng import** (ADR 0008): `TrainingBackend.check_importable` đưa verdict vào *cùng báo cáo preflight*, nên stack hỏng là lỗi preflight chết (exit 2) chứ không phải bất ngờ về sau. Import nằm trong adapter → đồ thị phụ thuộc không đổi (ADR 0001). Ghim bằng `test_a_stack_that_cannot_be_imported_fails_in_preflight` + 2 mutant | ✅ Đóng (2026-10-21) |
| N6 | **`--device auto` không tái lập được.** Manifest ghi `device: "auto"` — nghĩa là "để framework chọn" — nên một lần chạy trên GPU và một lần chạy trên CPU đều ghi giống nhau, dù artifact khác nhau | `Models/manifest.json` → `entries.*.device` | **Pipeline tự phân giải device trước khi train** (ADR 0009): `TrainingBackend.resolve_device` trả về một device *cụ thể* (`auto` → `cuda:0` nếu runtime có GPU, không thì `cpu`; yêu cầu tường minh là câu trả lời của chính nó), framework được *báo* dùng device nào thay vì tự chọn, và provenance ghi **cả hai**: `device` (yêu cầu) + `device_resolved` (sự thật). Quy tắc skip so sánh cả hai; mục không có `device_resolved` ⇒ *unprovable* ⇒ retrain. Đo lại (2026-10-21): `--check` in `Device: cuda:0 (requested auto)`. Ghim bằng 4 test (`test_backend.DeviceResolutionTests`, `test_trainer.DeviceResolutionTests`) + 4 mutant | ✅ Đóng (2026-10-21) |

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
| Fingerprint dataset phủ dir/glob/`.txt`/list | `test_dataset` + **5 mutant** (`dataset.py`) |
| Dataset so sánh theo **digest**, không theo đường dẫn | `test_trainer.test_the_same_dataset_behind_another_path_is_not_retrained` |
| Run bị gián đoạn thì tiếp tục được | `test_backend.TrainTests` (3 test) + **2 mutant** (`backend.py`) |
| Fingerprint báo tiến độ (dải `dataset` 2–8 %) | `test_dataset` + `test_progress` + **2 mutant** (`dataset.py`, `progress.py`) |
| Preflight **chứng minh** stack import được | `test_trainer` + `test_backend.ImportabilityTests` + **2 mutant** (`trainer.py`, `backend.py`) |
| Publish nguyên tử + rollback `.bak` + kiểm hợp đồng *trước* khi publish | `test_fsio`, `test_trainer` + **4 mutant** (3 `fsio.py` + 1 `trainer.py`) |
| Advisory lock: O_EXCL, token, claim, stale, heartbeat | `test_fsio` + **4 mutant** (`fsio.py`) |
| Manifest: quarantine, verdict nghiêm, 2 chiều, migration | `test_manifest` + **8 mutant** (`manifest.py`) |
| Provenance ghi *một lần mỗi model* (không mất khi crash giữa chừng) | `test_trainer.FailurePathTests` + **1 mutant** (`trainer.py`) |
| Tiến trình đơn điệu + marker `done`/`error` | `test_progress` + **2 mutant** (`progress.py`, `trainer.py`) |
| Mã thoát theo nguyên nhân | `test_cli.ExitCodeTests` + **4 mutant** (`errors.py`, `cli.py`) |
| Metric không phải số bị loại khỏi manifest | `test_backend` + **1 mutant** (`backend.py`) |
| Lệnh manifest không bao giờ huấn luyện | `test_cli.test_the_manifest_commands_never_start_a_training_run` |
| README ↔ CLI hai chiều, Layout ↔ package | `test_architecture.DocumentationTests` |

Tổng: **67 mutant** = `gate.py` 9 · `backend.py` 8 · `manifest.py` 8 ·
`onnx_contract.py` 8 · `trainer.py` 8 · `fsio.py` 7 · `dataset.py` 5 ·
`preflight.py` 5 · `config.py` 3 · `cli.py` 2 · `errors.py` 2 · `progress.py` 2.

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
