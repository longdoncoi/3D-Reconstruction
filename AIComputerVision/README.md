# AIComputerVision

Pipeline huấn luyện và xuất bản mô hình YOLO mà plugin **AI Processor** của ứng
dụng Qt sử dụng. Pipeline nhận một dataset đã gán nhãn và biến thành các file
ONNX trong [`Models/`](Models/), đồng thời ghi lại nguồn gốc (provenance) của
chúng trong [`Models/manifest.json`](Models/manifest.json).

Toàn bộ tài liệu trong file này đều bằng tiếng Việt; các thuật ngữ kỹ thuật
(provenance, preflight, quality gate…) giữ nguyên tên gốc để khớp với log và mã
nguồn.

## Layout

```text
AIComputerVision/
├─ TrainModel.py          # Entry point mà ứng dụng desktop gọi (composition root ở mức tiến trình)
├─ cvtrain/               # Toàn bộ triển khai của pipeline (package)
│  ├─ __init__.py         # Đánh dấu package + __version__ (import-light: không kéo theo ML stack)
│  ├─ cli.py              # Composition root: argparse, ánh xạ tham số → TrainConfig, dispatch, mã thoát
│  ├─ config.py           # Model contract + đường dẫn/mặc định (models_dir, runs_dir, weights_path…)
│  ├─ checks.py           # Ngôn ngữ chung: Check / PreflightReport (mọi module dùng chung)
│  ├─ dataset.py          # Đọc và kiểm tra data.yaml + các đường dẫn train/val mà nó trỏ tới
│  ├─ errors.py           # Enum ExitCode + phân cấp lỗi ngoại lệ có chủ đích (CvTrainError)
│  ├─ fsio.py             # Primitive hệ thống tệp: sha256, ghi nguyên tử, lease lock (host+PID), retention
│  ├─ preflight.py        # Kiểm tra môi trường và siêu tham số fail-fast trước khi huấn luyện
│  ├─ gate.py             # Chính sách xuất bản: already_current (--skip-unchanged) + enforce_quality_gate
│  ├─ metrics.py          # "Số nào là một số đo?" — một vị từ dùng chung cho backend và gate
│  ├─ backend.py          # Protocol TrainingBackend + adapter UltralyticsBackend (lazy import)
│  ├─ trainer.py          # Điều phối: preflight → train → gate → export → publish → verify
│  ├─ manifest.py         # Schema manifest (TypedDict), đọc/ghi/xác minh, migration + cách ly file hỏng
│  ├─ onnx_contract.py    # Hợp đồng tensor ONNX sau export (rank/dtype/output/opset + smoke inference ORT)
│  ├─ progress.py         # Giao thức [PROGRESS] mà dock Qt parse để dựng progress bar
│  └─ logging_setup.py    # Định dạng log ra stdout
├─ Models/                # *.onnx xuất bản (Git LFS) + manifest.json + advisory lock
├─ runs/                  # Log huấn luyện của Ultralytics (git-ignored)
├─ Docs/adr/              # 15 Architecture Decision Record cho các quyết định dưới đây
├─ Docs/SCORECARD.md      # Số đo các gate + bảng nợ đang mở (nguồn sự thật của "10/10")
├─ scripts/               # Công cụ kiểm chứng (mutation check) — ngoài package
├─ requirements.txt       # Dependency runtime đã pin phiên bản
├─ requirements-test.txt  # Dependency của *cổng chất lượng* (đều pin chính xác)
├─ requirements-test.lock # Bản hash-pinned của file trên — CI cài bằng --require-hashes
├─ pyproject.toml         # Cổng ruff / mypy / coverage cho thư mục này
└─ tests/                 # Unit test offline (không cần trọng số mô hình)
```

> **Lưu ý về tính chính xác:** khối Layout trên chỉ liệt kê đúng các module
> trong `cvtrain/` + `TrainModel.py`; `tests/test_architecture.py` máy hoá điều
> này — nếu thêm/bớt file `.py` mà không cập nhật Layout, test sẽ đỏ. Danh sách
> các file test được giải thích ngay dưới đây.

### Giải thích từng module trong `cvtrain/`

| File | Vai trò |
|------|---------|
| `__init__.py` | Đánh dấu package và khai báo `__version__` (hiện `1.0.0`) cùng `generator_name()` — một chỗ duy nhất sinh chuỗi `cvtrain X.Y.Z` dập vào manifest. Cố ý *import-light*: không import `ultralytics`/`yaml`, nên `--help`, lint, type-check và unit test chạy được khi chưa cài ML stack. |
| `cli.py` | **Composition root** của package: dựng parser argparse, ánh xạ đối số → `TrainConfig`, chọn mode (`--check`/`--dry-run`/`--write-manifest`/`--verify-manifest`/`--prune-runs`) hoặc chạy huấn luyện, dịch lỗi thành mã thoát. Đây là module duy nhất được phép nhắc tới lớp cụ thể `UltralyticsBackend`. |
| `config.py` | Model contract (map `--det`/`--seg`/`--track` → file `.pt`/`.onnx`) và cách phân giải đường dẫn: `models_dir`, `runs_dir`, `weights_path`. Preflight, trainer và digest provenance đều resolve checkpoint qua `TrainConfig.weights_path` nên ba nơi này không thể lệch nhau. |
| `checks.py` | Từ vựng kiểm tra dùng chung: `Check` (tên + ok + thông điệp) và `PreflightReport`. Không module nào phải tự nghĩ lại cách báo lỗi. |
| `dataset.py` | Đọc `data.yaml` bằng PyYAML (lazy), kiểm tra khóa `train`/`val`/`names`, resolve đường dẫn tương đối/tuyệt đối/glob theo đúng ngữ nghĩa của Ultralytics (kể cả khoá `path:` cấp trên). |
| `errors.py` | `ExitCode` (`0/1/2/3/4/130`) khai báo **một lần** và phân cấp `CvTrainError` (`StorageError`, `ManifestIntegrityError`, `QualityGateError`, `ExportContractError`, `ConcurrentRunError`, `ResumeError`…). Lỗi dự kiến được log thành một dòng actionable; lỗi còn lại giữ trọn traceback. |
| `fsio.py` | Primitive hệ thống tệp dùng chung: `sha256_file`, ghi nguyên tử (staging + `os.replace` + fsync thư mục), sao chép có backup `.bak`, **advisory lock** `Models/.cvtrain.lock`, và chính sách retention `prune_run_directories`. Lock là một **lease có chủ**: file ghi `host` + `pid`, heartbeat mỗi 30 s làm tươi mtime; chủ cùng máy còn sống thì lock **không bao giờ** bị đánh cắp dù quá hạn, chủ đã chết thì lock thành stale ngay, còn không chứng minh được (máy khác, lock cũ, thiếu quyền) thì mới dùng tuổi lease. Run cũng có lease riêng `.cvtrain-alive`: làm tươi mỗi epoch, **claim lại bằng PID hiện tại khi `--resume`** (marker cũ còn ghi tiến trình đã bị giết), và xoá khi train xong; tuổi không đọc được thì coi như còn sống. |
| `preflight.py` | Kiểm tra fail-fast: môi trường (`env.*`) và siêu tham số (`param.*`) — `epochs`, `imgsz`, `batch`, `workers`, `opset` (cửa sổ `MIN_OPSET=11..MAX_OPSET=21`), `device` (grammar + mọi CUDA index so với `torch.cuda.device_count()`), từng ngưỡng `--min-metric`. Nhận danh sách dependency từ backend thay vì hard-code framework. |
| `gate.py` | Chính sách (không phải luồng điều khiển): `already_current` trả lời "ONNX đã publish có đúng những gì lần gọi này sẽ build không?" (quy tắc `--skip-unchanged`, và artifact của một run `--resume` **không bao giờ** được coi là tái lập được); `enforce_quality_gate` trả lời "artifact có đủ tốt để giữ không?" (mục Quality gate). |
| `metrics.py` | Module lá cho đúng **một** câu hỏi: "giá trị này có phải số đo không?" — loại `bool`, `NaN`/`Infinity` và số tràn `float`, giữ `numbers.Real` (kể cả scalar của numpy). Backend dùng khi đọc kết quả framework, gate dùng khi đọc manifest do tay người sửa; nhờ nó hai bên không thể bất đồng về cùng một con số, và `gate` không phải import adapter. |
| `backend.py` | Protocol `TrainingBackend` (`plan`/`prepare`/`train`/`export` + `required_modules`) và adapter `UltralyticsBackend` — nơi duy nhất được import `ultralytics`, và chỉ khi huấn luyện thực sự chạy. `plan()` chốt **file bắt đầu** (checkpoint gốc, hoặc `last.pt` khi resume) *trước* khi provenance hash; `--resume` fail-closed với checkpoint thiếu/đã xong/không có train_args/dataset khác; `RunFacts` đọc budget, số epoch đã chạy và device từ chính trainer. `exporter_identity()` ghi cả `cvtrain`/`ultralytics`/`torch`/`onnx`/`onnxslim` vào provenance, vì cả năm đều có thể đổi graph được xuất (`onnxslim` là simplifier mặc định của Ultralytics; `onnxruntime` chỉ kiểm hợp đồng nên không nằm trong identity). |
| `trainer.py` | Điều phối “điều gì xảy ra”: preflight → plan → huấn luyện từng model → quality gate → export ONNX → **kiểm tra hợp đồng tensor** → publish nguyên tử (kèm backup) → ghi manifest → xác minh lại. Manifest ghi **sự thật của run** (`epochs` là budget hiệu lực, `epochs_completed` là số epoch đã chạy, `resume_from` khi tiếp tục run khác), không ghi lời gọi. Không bao giờ import `ultralytics`. |
| `manifest.py` | Schema manifest dạng `TypedDict`, đọc/ghi (`schema_version` + migration `_MIGRATIONS`), xác minh hai chiều, và cách ly manifest không parse được thành `manifest.json.corrupt-<timestamp>`. |
| `onnx_contract.py` | Hợp đồng tensor mà `AIProcessor.cpp` giả định: một input NCHW float32, output detection rank-3, hai output segmentation (một rank-4 proto), opset ≤ runtime được pin, và một smoke inference bằng chính ONNX Runtime trước khi publish. |
| `progress.py` | Giao thức stdout `[PROGRESS] pct=… stage=… key=…` — một *hợp đồng dây* với chương trình khác (dock Qt), tách khỏi `logging_setup.py` vì không phải “sở thích log”. Có guard thế tiến: marker có giá trị lùi sẽ bị giữ lại kèm warning. |
| `logging_setup.py` | Định dạng log stdout thống nhất (một dòng, có timestamp/level) cho toàn bộ pipeline. |

### Giải thích các thư mục/file còn lại

| Thành phần | Nội dung |
|-----------|----------|
| `TrainModel.py` | Entry point mà `AITrainDockWidget` của Qt gọi (`python TrainModel.py …`). Chỉ khoảng vài dòng: bật logging rồi ủy quyền cho `cvtrain.cli.main()`. |
| `Models/` | Sản phẩm: `yolo11n.onnx`, `yolo11n-seg.onnx`, `yolo11x-tracking.onnx` (Git LFS), `manifest.json` (provenance), `Models/.cvtrain.lock` (advisory lock, git-ignored) và `Models/*.onnx.bak` (một thế hệ khôi phục, git-ignored). |
| `runs/` | Log huấn luyện mà Ultralytics ghi ra (git-ignored). `--runs-dir` di chuyển nó; `--prune-runs` dọn run cũ (xem mục *Dọn run cũ*). |
| `Docs/adr/` | 16 ADR: 0001 đồ thị phụ thuộc một chiều + composition root; 0002 protocol `TrainingBackend`; 0003 manifest provenance + `--skip-unchanged` fail-closed; 0004 advisory lock, publish nguyên tử, quality gate; 0005 lỗi lưu trữ có kiểu, sở hữu lock, tiến trình đơn điệu; 0006 dependency có điều kiện, input fail-closed, heartbeat lock; 0007 mọi claim phải được *pin* bằng test/mutant hoặc bị xóa (+ sửa đổi 2026-10-21); 0008 đóng nợ đo được: provenance theo digest, `--resume`, fingerprint có tiến độ, preflight chứng minh import được; 0009 device được phân giải *trước* khi train (`auto` là yêu cầu, `device_resolved` là sự thật); 0010 retention là *chính sách* (`--prune-runs`), không phải tai nạn; 0011 quy tắc fail-closed ở cả hai phía, provenance ghi số epoch framework thực sự chạy, và mutation set phủ hết mọi claim; 0012 chuỗi công cụ có hash, cổng chạy trên cả hai nền tảng, và `--resume` là hợp đồng hai tầng; 0013 `--resume` là một continuation phải chứng minh được, manifest ghi run đã xảy ra (budget + completed + resume_from), và liveness là lease có chủ (host + PID); 0014 lease đổi chủ khi resume, `onnxslim` là provenance, preflight import toàn bộ stack, "tuổi không rõ" được giữ, và stack thật chạy trên Windows; 0015 yêu cầu và sự thật có nhà riêng (sự thật không bao giờ được ghi/so dưới tên yêu cầu; `device_resolved` vẫn được so như fact), `export_args` thuộc danh tính artifact, `--batch -1` không bao giờ skip, và lần chạy lại default-device có kiểm chứng skip với framework thật trong nightly; 0016 device fact viết theo cách của framework, multi-GPU không bị thu về GPU chính, và mọi CUDA index được kiểm tra trước khi hỏi. |
| `Docs/SCORECARD.md` | Số đo của từng gate kèm ngày đo, bảng nợ đang mở, và quy tắc "claim nào đã được ghim". Một nơi duy nhất để trả lời "package này đã ổn chưa". |
| `scripts/mutation_check.py` | Đảo chiều từng quyết định trọng yếu một lần và bắt suite phải đỏ (ADR 0007). Danh sách mutant *là* hợp đồng: một guard không có mutant là một guard chưa ai ghim. `--files` chạy một *tập con* (job Windows chỉ quét lại các quyết định phụ thuộc nền tảng) nhưng **không** hạ được floor. |
| `requirements.txt` | Dependency runtime đã pin: `ultralytics==8.4.41`, `PyYAML==6.0.2`, `onnx==1.21.0`, `onnxslim==0.1.91`, `onnxruntime==1.20.1` (đúng bằng `ONNXRUNTIME_VERSION` trong `CMakeLists.txt` — engine hợp đồng phải là engine mà ứng dụng ship), `tensorboard==2.15.2` (chỉ cần khi `--tensorboard`). `torch` dùng khoảng `>=2.7,<3`: bản CPU/CUDA chia sẻ cùng số version và được chọn bằng index cài đặt. |
| `requirements-test.txt` / `requirements-test.lock` | Dependency của **cổng chất lượng**: `ruff`, `mypy`, `coverage` cùng `PyYAML`/`onnx`/`onnxruntime` mà hợp đồng export dùng. Mọi pin *chính xác*; `lock` là bản hash-pinned của file trên (lệnh sinh lại nằm ở `Docs/SCORECARD.md` §5), CI cài bằng `pip install --require-hashes` (ADR 0012). Một ruff/mypy mới có thể đổi *kết luận* của một gate mà không ai duyệt thay đổi đó — nên nó phải được pin. |
| `pyproject.toml` | Cấu hình ruff (`E/W/F/I/BLE/S/RUF`), mypy (strict, `files = ["cvtrain", "tests", "scripts", "TrainModel.py"]`) và coverage (branch, `fail_under = 100`). Không phải package cài đặt được. |

### Giải thích từng file test trong `tests/`

| File | Nội dung kiểm thử |
|------|-------------------|
| `test_architecture.py` | Buộc đúng phần Kiến trúc ở trên: bản đồ import (`ast`), dependency nặng chỉ import lazy và chỉ bởi owner duy nhất, chỉ `cli.py` nhắc `UltralyticsBackend`, docstring của mọi module, **và** cả Layout lẫn danh sách cờ dòng lệnh của chính README này. |
| `test_backend.py` | Adapter `UltralyticsBackend` thật, chạy trên package `ultralytics` giả: resolve checkpoint, gắn callback, đối số export, nhánh thiếu dependency; `TrainPlan`/`RunFacts`; `--resume` fail-closed (thiếu/đã xong/không train_args/dataset mất/dataset khác byte) và log giá trị checkpoint ghi đè; budget + completed + `resume_from`; metric `bool`/`NaN`/`Infinity`/tràn `float`. |
| `test_ci_contract.py` | Buộc **hai cấu trúc CI**: cổng chất lượng phải chạy trên `windows-latest` (nơi sản phẩm ship) chứ không chỉ trên Linux, và các quyết định phụ thuộc nền tảng phải được mutation-check lại trên Windows. Ngoài ra buộc job *real-dependency* còn phải huấn luyện → cắt → `--resume` trong thư mục `ResumeRuns` riêng, assert budget/completed/`resume_from` của run được tiếp tục, và phải chạy cho pull request (ADR 0012, 0013). |
| `test_cli.py` | wiring argparse, ánh xạ đối số → `TrainConfig` và mã thoát — `main()` là nơi duy nhất biến thất bại thành mã process. Cả các mode (`--check`/`--dry-run`/`--write-manifest`/`--verify-manifest`/`--prune-runs`) và việc chúng không bao giờ huấn luyện. |
| `test_config.py` | Model contract (map spec, tên export không trùng), cách resolve lựa chọn, mặc định `TrainConfig`, override đường dẫn `--models-dir`/`--weights-dir`. |
| `test_contract.py` | Ràng buộc hai chiều với `src/include/AIConstants.h` (tên file ONNX), `MAX_OPSET` với ONNX Runtime được pin trong `CMakeLists.txt`, các hằng schema ONNX với `AIYoloPostProcessor.h`, và ngưỡng IoU của tracker với `AIProcessor.cpp`. |
| `test_dataset.py` | Kiểm tra `data.yaml`: khóa bắt buộc, di chuyển bằng `path:`, tham chiếu tuyệt đối/tương đối/glob, số class cho hợp đồng ONNX, và fingerprint nội dung (đổi byte cùng kích thước phải đổi digest). |
| `test_fsio.py` | Primitive hệ thống tệp: digest, ghi nguyên tử, sao chép có backup, toàn bộ quy tắc sở hữu của advisory lock (từ chối lock đang sống, đánh cắp lock của chủ đã chết ngay cả khi lease còn mới, không đánh cắp lock của chủ còn sống dù lease cũ, fallback theo tuổi khi không probe được, heartbeat làm tươi lease, platform probe đúng nền tảng), lease `.cvtrain-alive`, và chính sách retention (giữ run mới nhất, giữ run có chủ còn sống, marker của máy khác vẫn tính là hoạt động, không đụng thư mục không phải của pipeline). |
| `test_manifest.py` | Tính toàn vẹn và provenance của manifest: digest, migration `schema_version`, cách ly manifest hỏng, xác minh hai chiều, và từ chối ghi `NaN`/`Infinity` (JSON không hợp lệ). |
| `test_onnx_contract.py` | Hợp đồng ONNX sau export: input/output rank–dtype–channel theo role, opset, file hỏng, smoke inference thật và mọi nhánh từ chối của validator. |
| `test_preflight.py` | Kiểm tra môi trường và siêu tham số — các check `env.*` và `param.*` mà một lần chạy có thể fail. |
| `test_progress.py` | Giao thức `[PROGRESS]`: các dải %, thứ tự stage, guard đơn điệu, marker kết thúc `done`/`error`, slot của model bị skip. |
| `test_qt_contract.py` | argv, nhãn checkbox và regex `pct=` mà `AITrainDockWidget.cpp` dùng có khớp với thứ CLI chấp nhận và phát ra hay không. |
| `test_trainer.py` | Chạy trọn pipeline trên backend giả: preflight, plan, dry-run, hỏi đáp, quality gate, độ bền của manifest, tiến trình — không GPU, không dataset, không cài ML package; provenance ghi budget/completed/`resume_from` và artifact của run `--resume` không bao giờ được skip. |
| `test_toolchain_contract.py` | Buộc **chuỗi công cụ của gate** phải được pin: mọi requirement là pin chính xác (ngoại lệ duy nhất là `torch`, có lý do đã ghi), `onnxruntime` phải đúng bằng `ONNXRUNTIME_VERSION` mà `CMakeLists.txt` pin, lock phải mang digest cho từng gói, và CI phải cài lock bằng `--require-hashes` chứ không cài gói trôi (ADR 0012). |

## Kiến trúc

Phụ thuộc trỏ về một chiều, từ tầng điều phối xuống tầng nguyên thuỷ:

```text
TrainModel.py → cli ──► trainer ──┬─► preflight ─┬─► dataset
   (composition root,              ├─► manifest   └─► checks
    tiêm backend vào)              ├─► onnx_contract ─► preflight
                                   ├─► gate ─┐
                                   └─► backend┤
                                              ├─► fsio ──► errors
                                              └─► config
```

`fsio` nằm trên `errors` vì advisory lock ném ra `ConcurrentRunError` có kiểu
(xem mục Manifest); mọi nguyên thuỷ khác đều là lá. Lý do đằng sau các quyết
định trọng yếu được ghi thành các ADR ngắn gọn dưới [`Docs/adr/`](Docs/adr/):
[0001](Docs/adr/0001-one-way-dependency-graph-and-composition-root.md) đồ thị
phụ thuộc một chiều được thực thi + composition root duy nhất,
[0002](Docs/adr/0002-training-backend-protocol.md) protocol `TrainingBackend`,
[0003](Docs/adr/0003-provenance-manifest-and-skip-unchanged.md) manifest
provenance + quy tắc `--skip-unchanged` fail-closed,
[0004](Docs/adr/0004-advisory-lock-and-publication-policy.md) advisory lock,
publish nguyên tử và quality gate,
[0005](Docs/adr/0005-typed-storage-lock-ownership-monotonic-progress.md) lỗi
lưu trữ có kiểu, sở hữu lock và tiến trình đơn điệu,
[0006](Docs/adr/0006-conditional-dependencies-fail-closed-inputs-lock-heartbeat.md)
dependency có điều kiện, input fail-closed và heartbeat lock,
[0007](Docs/adr/0007-every-claim-is-pinned-or-removed.md) mọi claim phải được
*pin* bằng test/mutant hoặc bị xóa,
[0008](Docs/adr/0008-closing-the-measured-debt.md) đóng nợ đo được: provenance
theo digest nội dung, `--resume`, fingerprint có tiến độ và preflight chứng minh
stack import được,
[0009](Docs/adr/0009-resolve-the-device-before-training.md) device được phân giải
trước khi train — `auto` là yêu cầu, `device_resolved` là sự thật,
[0010](Docs/adr/0010-retention-is-a-policy-not-an-accident.md) retention là
chính sách (`--prune-runs`), không phải tai nạn,
[0011](Docs/adr/0011-fail-closed-on-both-sides-and-the-mutation-set-covers-every-claim.md)
fail-closed ở cả hai phía và mutation set phủ mọi claim,
[0012](Docs/adr/0012-a-hash-pinned-toolchain-the-shipping-platform-and-a-two-layer-resume-contract.md)
chuỗi công cụ hash-pinned, nền tảng ship và `--resume` hai tầng,
[0013](Docs/adr/0013-resume-is-provenance-effective-facts-and-a-lease-is-evidence-of-life.md)
resume là provenance, effective facts và lease,
[0014](Docs/adr/0014-the-lease-changes-hands-the-simplifier-is-provenance-and-the-real-stack-runs-on-windows.md)
lease đổi chủ khi resume, `onnxslim` là provenance và stack thật chạy trên
Windows,
[0015](Docs/adr/0015-the-request-and-the-fact-have-separate-homes.md)
manifest tách *yêu cầu* khỏi *sự thật* (không ghi/so sánh sự thật dưới tên yêu cầu), `export_args` thuộc danh tính artifact, và lần chạy lại default-device phải
skip,
[0016](Docs/adr/0016-the-device-fact-is-canonical-and-every-index-is-checked.md)
device fact viết theo cách của framework, multi-GPU không bị thu về GPU chính,
và mọi CUDA index được kiểm tra trước khi hỏi.

Sơ đồ trên là một **hợp đồng**, không phải gợi ý: `tests/test_architecture.py`
parse `cvtrain/*.py` bằng `ast` và làm đỏ build khi có import nằm ngoài bản đồ
đó — một module import “hàng xóm” mà nó không được biết, một file mới chưa
khai báo phụ thuộc trong bảng đó, stack nặng (`ultralytics`, `torch`, `yaml`) bị chạm
từ nơi khác ngoài owner duy nhất hoặc bị import ở cấp module thay vì lazy, hay
lớp cụ thể `UltralyticsBackend` được nhắc tới ngoài `cli.py`. Kiến trúc không
thể mục nát trong im lặng chừng nào suite còn xanh.

`cli.main` là **composition root** của package: nó là module duy nhất nhắc tới
lớp `TrainingBackend` cụ thể, và trao instance đó cho `trainer.run(cfg,
backend=…)` — với tham số *bắt buộc*. Vì vậy điều phối không có mặc định để
bịa ra và cũng không có adapter nào để import.

`trainer` sở hữu “điều gì xảy ra” và không bao giờ import `ultralytics`. Mọi
truy cập framework đi qua protocol `TrainingBackend` (`prepare`/`train`/
`export`), mà production gắn với `UltralyticsBackend` — adapter mỏng chỉ import
ML stack khi huấn luyện thực sự bắt đầu. Model đã huấn luyện là
`TrainedModel[H]`, generic trên handle framework mờ đục `H`: pipeline chỉ trả
nó lại cho `export` của đúng backend đó, không bao giờ tự inspect. Ba hệ quả:

* `--help`, `--check`, `--dry-run`, lint, type-check và unit suite đều chạy
  được khi không cài dependency ML nào;
* toàn bộ pipeline được điều khiển bởi backend giả trong `tests/test_trainer.py`,
  nên preflight/dry-run/hỏi đáp/quality gate/độ bền manifest/tiến trình đều
  được xác minh offline;
* `preflight` nhận danh sách dependency từ backend (`backend.required_modules(cfg)`)
  thay vì hard-code một framework — nên một tính năng tùy chọn có thể đòi một
  package tùy chọn: `--tensorboard` fail ngay ở preflight (exit `2`) thay vì
  huấn luyện một giờ rồi không ghi nổi logfile nào.

Kiểm tra được chia ba hướng để không module nào có hai lý do phải đổi: `checks`
sở hữu từ vựng `Check`/`PreflightReport` mọi cộng tác viên dùng chung, `dataset`
sở hữu tài liệu `data.yaml`, và `preflight` sở hữu quy tắc môi trường/siêu tham
số và tổng hợp phần còn lại. `progress` tách khỏi `logging_setup` vì lý do
tương tự: đó là hợp đồng dây với chương trình khác, không phải sở thích log.

Hai *phán quyết* mà một lần chạy tuân theo là chính sách, không phải luồng
điều khiển, nên nằm trong `gate.py` chứ không trong điều phối: `already_current`
quyết định ONNX đã publish có phải đúng thứ lần gọi này sẽ build không (quy tắc
`--skip-unchanged`, mục Manifest), và `enforce_quality_gate` quyết định artifact
có đủ tốt để giữ không (mục Quality gate). `trainer` hỏi phán quyết và tuân theo;
`manifest` ghi lại provenance mà các phán quyết đó so sánh.

## Cài đặt

```powershell
pip install -r AIComputerVision/requirements.txt
```

Ứng dụng desktop khởi chạy `python` từ `PATH`. Muốn dùng interpreter/venv cụ
thể, đặt biến môi trường `RECONSTRUCTION_PYTHON` tới đường dẫn đầy đủ của nó
trước khi mở app.

Dataset nằm ở `Dataset/data.yaml` cấp gốc repo (git-ignored). File phải định
nghĩa `train`, `val` và `names`; `--check` xác minh điều này trước khi huấn
luyện — kể cả việc các thư mục mà `train`/`val` trỏ tới có tồn tại trên đĩa
hay không (đường dẫn tương đối resolve theo thư mục chứa YAML, hoặc theo khoá
`path:` cấp trên khai báo tường minh, đúng như Ultralytics).

Các checkpoint gốc (`yolo11n.pt`, `yolo11n-seg.pt`, `yolo11x.pt`) nằm trong
`AIComputerVision/` theo mặc định và được Ultralytics tải về ở lần dùng đầu.
`--weights-dir` di chuyển chúng, đúng như cách `--models-dir` và `--runs-dir`
di chuyển output — preflight, trainer và digest provenance đều resolve checkpoint
qua `TrainConfig.weights_path`, nên ba chỗ đó không thể bất đồng ý. SHA-256 của
checkpoint được ghi cùng mỗi mục manifest và được kiểm lại trước khi
`--skip-unchanged` tin một artifact, vì checkpoint bị thay là một mô hình khác
bất chấp tên file.

## Sử dụng

```powershell
# Huấn luyện các model mặc định (detection + segmentation)
python AIComputerVision/TrainModel.py --yes

# Chỉ huấn luyện model tracking
python AIComputerVision/TrainModel.py --track --yes

# Kiểm tra môi trường và in ra kế hoạch mà không huấn luyện
python AIComputerVision/TrainModel.py --check
python AIComputerVision/TrainModel.py --dry-run

# Chạy lại idempotent: chỉ huấn luyện các model ONNX không còn khớp
python AIComputerVision/TrainModel.py --skip-unchanged --yes

# Tiếp tục run bị gián đoạn (lấy weights/last.pt mới nhất của model)
python AIComputerVision/TrainModel.py --resume --yes

# Ghi đè siêu tham số
python AIComputerVision/TrainModel.py --det --epochs 50 --batch 8 --imgsz 640 --seed 42

# Trỏ vào dataset riêng, bật log TensorBoard và để framework nói nhiều hơn
python AIComputerVision/TrainModel.py --data path/to/data.yaml --tensorboard --verbose --yes

# Chỉ xuất bản model vượt qua ngưỡng chất lượng (lặp được; xem mục Quality gate)
python AIComputerVision/TrainModel.py --yes --min-metric 'metrics/mAP50(B)=0.5' --min-metric 'metrics/mAP50-95(B)=0.3'

# Công cụ provenance
python AIComputerVision/TrainModel.py --write-manifest
python AIComputerVision/TrainModel.py --verify-manifest

# Dọn run cũ: giữ lại 1 run mới nhất cho mỗi model (xem mục Dọn run cũ)
python AIComputerVision/TrainModel.py --prune-runs
python AIComputerVision/TrainModel.py --prune-runs 3

# In version được dập vào manifest dạng `cvtrain X.Y.Z`
python AIComputerVision/TrainModel.py --version
```

`--skip-unchanged` nói về *việc có huấn luyện hay không*, không phải quyền
ghi đè: một lần chạy tương tác sẽ hỏi về *bất kỳ* model nào đã tồn tại trước
khi bắt đầu, rồi sau đó mới quyết định model đó có thực sự bị đụng tới hay
không — nên model không còn khớp vẫn phải được xác nhận trước khi bị thay (ứng
dụng desktop truyền `--yes` và không bao giờ hỏi). Xem mục Manifest để biết
chính xác model phải thỏa điều kiện gì thì mới bị bỏ qua.

`--resume` nói về *cách huấn luyện*: thay vì tạo run mới, nó lấy
`weights/last.pt` mới nhất của model trong `runs/<model>/` rồi tiếp tục từ đó —
kèm cả bộ siêu tham số của run bị gián đoạn (framework đọc lại từ checkpoint),
nên một lần chạy bị ngắt ở epoch 4/5 không phải làm lại từ đầu.

`--resume` **fail-closed**: không có checkpoint, checkpoint đã chạy xong
(`epoch == -1`), checkpoint không mang `train_args`, checkpoint trỏ tới dataset
không còn trên đĩa, dataset của lần gọi không fingerprint được, hoặc dataset
khác byte với dataset đã huấn luyện — tất cả đều là `ResumeError` (exit `2`),
không bao giờ âm thầm bắt đầu run mới. Lý do: người vận hành đã yêu cầu *tiếp
tục* một thí nghiệm; một run mới có thể ghi đè đúng artifact họ đang cố giữ.
Dataset được so sánh theo **byte**, nên chuyển project sang đường dẫn khác vẫn
resume được, còn đổi dữ liệu thì bị từ chối với cả hai đường dẫn trong thông
điệp. Siêu tham số khác (imgsz/batch/seed/device/epochs) không phải lỗi: run
tiếp tục với giá trị của checkpoint, và pipeline log từng giá trị bị checkpoint
ghi đè để dòng lệnh không âm thầm bị bỏ qua. Mặc định là `--resume` **tắt**:
một run sạch mới là run tái lập được.

`--check`, `--dry-run`, `--write-manifest`, `--verify-manifest` và `--prune-runs` là
năm *mode* và loại trừ lẫn nhau (lỗi dùng sai, exit `2`, nếu truyền hai cái):
trước đây kết hợp chúng khiến một cái âm thầm thắng, đáng kể nhất là
`--write-manifest --verify-manifest` vốn sẽ ghi lại manifest rồi “xác minh”
chính file vừa ghi. Không truyền cái nào thì pipeline huấn luyện.

Mọi siêu tham số được preflight kiểm tra — `--epochs`, `--imgsz`, `--batch`,
`--workers`, `--opset`, `--device` và mọi ngưỡng `--min-metric` — nên giá trị
không thể cho ra model dùng được sẽ bị từ chối theo tên
(`[FAIL] param.opset: 99 (must be 11..21; pass --opset N)`) thay vì fail bên
trong framework rất lâu sau đó. Trần opset không phải tuỳ hứng: ONNX Runtime
1.20.1 — bản mà `CMakeLists.txt` pin cho ứng dụng Qt — chỉ hỗ trợ tới opset 21,
và một artifact được dập cao hơn sẽ chỉ phát nổ khi có người cố load nó.

Mã thoát được khai báo **một lần** trong `cvtrain/errors.py` (`ExitCode`) và
không bao giờ viết như số nguyên trần ở nơi khác: `0` thành công, `1` thất bại
bất ngờ, `2` preflight/manifest thất bại (kể cả file mà pipeline phải hash,
đo lường hoặc *ghi* — `StorageError`, "this storage is not ready"), `3` quality
gate thất bại, `4` export không thoả hợp đồng tensor của Qt (`ExportContractError`),
`130` bị ngắt. Lỗi dự kiến (`CvTrainError`) được log thành một
dòng actionable duy nhất; mọi thứ khác giữ nguyên traceback đầy đủ. Dock Qt báo
cái số đó nguyên văn và chỉ so nó với zero, nên thêm mã mới là một thay đổi tương
thích.

## Quality gate (cổng chất lượng)

`--min-metric NAME=VALUE` (lặp được) từ chối xuất bản một model có metric được
báo cáo thấp hơn `VALUE`. Kiểm tra chạy *giữa* huấn luyện và xuất bản: model
không vượt ngưỡng không bao giờ tới được `Models/`, artifact đã xuất bản trước
đó vẫn nằm nguyên chỗ nó (vẫn nguyên vẹn, vẫn xác minh được), và lần chạy exit
`3` với cả hai con số trong thông điệp. Tên metric là những gì framework báo cáo;
yêu cầu một cái tên không ai báo cáo là một thất bại, và thông điệp lỗi liệt kê
những gì *đã* được báo cáo để một lỗi gõ chính tả dễ thấy ngay.

Cổng này cũng áp dụng cho model mà `--skip-unchanged` quyết định giữ lại, dựa
trên metric được ghi khi model đó được huấn luyện — bỏ qua không bao giờ là con
đường vòng quanh cổng. Ngưỡng dành cho metric dạng “càng lớn càng tốt” trong
khoảng `0..1` (precision, recall, mAP, fitness), không dành cho loss.

## Dọn run cũ (retention)

`runs/<model>/` tích tụ mỗi lần huấn luyện — một run `yolo11x` để lại
checkpoint nặng hàng trăm MB — nên không có chính sách thì thư mục này chỉ
lớn dần. `--prune-runs` là chính sách đó, và là **mode riêng**: nó không bao
giờ huấn luyện và không bao giờ chạm vào `Models/` (artifact đã xuất bản và
provenance của nó không phải việc của lệnh này).

```powershell
python AIComputerVision/TrainModel.py --prune-runs      # giữ 1 run mới nhất mỗi model
python AIComputerVision/TrainModel.py --prune-runs 3    # giữ 3 run mới nhất mỗi model
```

Ba quy tắc làm cho thao tác xoá này an toàn, và mỗi quy tắc đều được ghim:

- **Run mới nhất của *mỗi* model luôn sống sót** — nên `--resume` vẫn tìm
  thấy checkpoint, và vẫn còn một thế hệ để quay lui.
- **Run còn đang chạy không bao giờ bị xoá, và "còn sống" là một lease có
  chủ.** Mỗi run mang `.cvtrain-alive` ghi `host` + `pid`; backend làm tươi nó
  ở mỗi epoch và xoá nó khi train trả về (một run đã xong không phải là run
  đang sống chỉ vì process còn tồn tại). Pruner giữ run nếu chủ cùng máy được
  chứng minh còn sống, dù cửa sổ 24 giờ đã qua; với run trên máy khác (không
  probe được PID), mtime của marker được tính là hoạt động, và ngưỡng 24 giờ
  vẫn là fallback. Hướng an toàn của một thao tác xoá là *giữ*.
- **Chỉ thư mục run do pipeline này tạo mới bị xoá** — thư mục có tên khác
  trong `runs/` không phải của ta.

Một thư mục không đọc được hoặc không xoá được là `StorageError` (exit `2`),
không phải traceback; các run đã xoá thì vẫn xoá (dọn dẹp không xoá được
không phải là rollback).

## Hợp đồng mô hình (C++ ↔ Python)

Tên file xuất bản là một hợp đồng với `src/include/AIConstants.h` (hằng số nằm ở
đó; `AppConstants.h` chỉ `#include` nó). Thay đổi bên nào cũng phải được phản
chiếu sang bên kia; `tests/test_contract.py` làm đỏ CI nếu hai bên lệch nhau.

| Lựa chọn | Trọng số gốc     | File xuất bản         | Vai trò       |
|----------|------------------|-----------------------|---------------|
| `--det`  | `yolo11n.pt`     | `yolo11n.onnx`        | detection     |
| `--seg`  | `yolo11n-seg.pt` | `yolo11n-seg.onnx`    | segmentation  |
| `--track`| `yolo11x.pt`     | `yolo11x-tracking.onnx` | tracking detector |

Khi không truyền cờ lựa chọn nào, `--det` và `--seg` được huấn luyện (hành vi
di sản). Trọng số gốc bị thiếu sẽ được Ultralytics tải về ở lần dùng đầu.

Ngoài tên file, artifact còn phải khớp **hợp đồng tensor** mà
`src/services/ai/AIProcessor.cpp` đọc: một input NCHW float32; detection và
tracking có đúng một output rank-3 dạng `[1, 4+classes, N]`; segmentation có
hai output (một rank-3 `[1, 4+classes+32, N]`, một proto rank-4 `[1, 32, h, w]`,
thứ tự không quan trọng vì C++ nhận diện theo rank); opset không vượt runtime
được pin. `cvtrain/onnx_contract.py` kiểm tra cấu trúc bằng `onnx.checker`, rồi
chạy một smoke inference thật bằng ONNX Runtime với đúng kích thước input trước
khi publish. Artifact không đạt bị chặn với exit `4`, và model đang chạy tốt
trong `Models/` không bị thay thế.

`--track` là **detector + tracker phía Qt**: file ONNX chỉ chứa trọng số phát
hiện; việc gán ID theo IoU nằm trong `AIProcessor::runTracking`, và manifest ghi
lại thuật toán/ngưỡng dưới khóa `tracker` để provenance không ám chỉ rằng graph
tự tracking.

## Giao thức tiến trình (progress protocol)

`TrainModel.py` in ra một dòng cho mỗi mốc để dock Qt có thể điều khiển progress
bar:

```text
[PROGRESS] pct=2 stage=preflight
[PROGRESS] pct=5 stage=dataset
[PROGRESS] pct=42 stage=train key=seg epoch=3/5
[PROGRESS] pct=96 stage=export key=seg
[PROGRESS] pct=96 stage=skip key=track
[PROGRESS] pct=100 stage=manifest
[PROGRESS] pct=100 stage=done
```

Mọi lần chạy kết thúc bằng một marker tận cùng — `stage=done` khi thành công,
hoặc `stage=error` (giữ percentage đã công bố cuối cùng) khi thất bại — nên
thanh tiến trình không bao giờ treo chờ một dòng sẽ không bao giờ tới. Marker
có thể kéo thanh lùi sẽ bị giữ lại kèm warning thay vì được công bố (thông điệp
diagnostic không mang token `pct=`, nên regex của chính dock không thể tua nó
lùi). Model bị `--skip-unchanged` bỏ qua sẽ công bố `stage=skip` tại percentage
mà một lần export hoàn tất sẽ chiếm, nên một lần chạy mà mọi model đều bị bỏ qua
vẫn leo đơn điệu tới dải manifest. Phía Qt
(`AITrainDockWidget::onTrainProcessOutput`) chỉ parse `pct=`; các trường còn
lại mang tính thông tin.

Dải phần trăm: preflight 0–2 %, **dataset 2–8 %**, các model 8–97 %, manifest
97–100 %. Dải `dataset` là dải riêng vì `--skip-unchanged` chứng minh dữ liệu
bằng cách đọc *mọi byte* của nó: trên dataset lớn đó là vài phút, và nếu
thanh đứng yên ở 2 % thì người vận hành không phân biệt được "đang băm" với
"treo". Nay mỗi file đã băm sẽ công bố một mốc trong dải đó.

## Manifest & provenance

Mỗi model xuất bản được ghi vào `Models/manifest.json` kèm SHA-256, kích thước,
đường dẫn dataset *và digest của `data.yaml`*, digest nội dung của mọi file
mà `train`/`val` tham chiếu — kể cả directory, glob, file đơn, listing `.txt`
và split dạng list — mọi siêu tham số, thiết bị tính toán, digest của checkpoint
gốc, **composite exporter identity** (`cvtrain`, `ultralytics`, `torch`, `onnx`,
`onnxslim`)
đã tạo ra nó, và (best-effort) các metric. Metric chỉ được ghi khi thực sự là
một số đo hữu hạn: `bool`, `NaN`, `Infinity` và số tràn `float` bị loại
(`cvtrain/metrics.py`), và manifest được ghi với `allow_nan=False` — một tài
liệu mà JSON reader nghiêm ngặt từ chối không phải là provenance. Các digest
được chụp **trước khi**
`backend.train` chạy: file dataset hay checkpoint bị sửa giữa lần chạy không
bao giờ được gán nhầm cho artifact, và lần sau `--skip-unchanged` sẽ huấn luyện
lại thay vì tin một mục thiếu `dataset_content_sha256`. Mục `tracking` còn mang khối
`tracker` (`algorithm`, `iou_threshold`, `scope`, `implementation`) vì phần
association nằm ở Qt chứ không ở graph. Header của lần chạy
(`last_run`) ghi lại lời gọi đã tạo ra nó — tham số, hạng device, phiên bản
interpreter và ngưỡng cổng — và `updated_at` được làm mới trên mọi lần ghi
provenance, nên header không bao giờ cũ hơn các mục bên dưới. `--verify-manifest`
tính lại các digest, thứ giúp phát hiện file bị cắt cụt hoặc vô tình bị thay.
Xác minh là hai chiều: một mục mà file của nó biến mất sẽ được báo cáo, và một
file ONNX nằm trong `Models/` không có mục cũng bị báo cáo.

Một lần huấn luyện sẽ xác minh lại chính các model vừa xuất bản và thoát với mã
`2` khi kiểm tra đó fail — đúng mã mà `--verify-manifest` trả về, được ném ra
qua cùng `ManifestIntegrityError` — nên một artifact mà pipeline không thể bảo
chứng không bao giờ được báo cáo như một lần chạy sạch. Kiểm tra cuối lần chạy
được giới hạn trong các model mà lần gọi này chịu trách nhiệm
(`verify_manifest(only=…)`): một ONNX lạc loài đã nằm sẵn trong `Models/` không
phải vấn đề của lần chạy này, trong khi lệnh `--verify-manifest` độc lập giữ
nguyên cuộc quét hai chiều đầy đủ. ONNX lạc loài được *báo cáo* một cách có chủ
đích thay vì được nhận làm con nuôi: `--write-manifest` chỉ dựng lại các mục
cho các model trong hợp đồng, vì bảo chứng cho một ONNX không rõ nguồn sẽ trao
cho nó một xác minh sạch mà không có provenance nào phía sau. Lệnh này cũng
xoá các mục mà file tương ứng đã biến mất — quét đĩa mô tả hiện tại, không giữ
lại bằng chứng ma. Cách xử lý ONNX lạc loài là xoá file hoặc xuất bản lại từ
pipeline. Manifest không parse được sẽ bị cách ly thành
`manifest.json.corrupt-<timestamp>` trước khi trả về một bộ khung, để lần ghi kế
tiếp không thể lặng lẽ xoá mất kỷ lục duy nhất về những gì đang nằm trên đĩa.
Đường khôi phục đó thuộc về các lần chạy, không thuộc về các phán quyết:
`--verify-manifest` đọc ở chế độ strict và báo tài liệu không đọc được dưới dạng
exit `2`, nêu rõ lý do — mà không đổi tên nó, vì lệnh này được mô tả là *đọc*
`Models/`.

**Đường dẫn dataset được ghi, nhưng không được so sánh.** Quy tắc
`--skip-unchanged` so sánh *digest của `data.yaml`* và *fingerprint nội dung của
mọi file mà `train`/`val` chạm tới* — đó mới là bằng chứng "cùng một dataset".
Chuỗi đường dẫn chỉ là một sự thật về *một máy*: cùng một dataset, copy sang
thư mục khác (hoặc checkout ở một đường dẫn tuyệt đối khác), vẫn là cùng một
thí nghiệm, và một phán quyết phụ thuộc vào *nơi* dữ liệu nằm sẽ không sống sót
qua một lần checkout trên máy khác (ADR 0008).

**Device: yêu cầu và sự thật là hai trường, không được lẫn nhau.** `device` là
thứ *được yêu cầu* (`auto` = "để framework chọn"), lấy nguyên từ
`TrainConfig.device_class`; `device_resolved` là thứ *đã chạy* (`cuda:0`, `cpu`…),
do pipeline phân giải **trước khi train** (ADR 0009) và được viết theo **cách
viết của framework**: `--device 0` là `cuda:0`, `--device 0,1` là
`cuda:0,cuda:1` (`canonical_device`, ADR 0016). Ultralytics viết lại `args.device`
trong lúc train (thành `"0"` trên GPU, `"cpu"` trên CPU): đó là *argument của
framework*, không phải yêu cầu, nên nó **không được ghi vào `device`**. Nó chỉ
là một nguồn cho `device_resolved` — và là nguồn **ưu tiên**, vì nó giữ nguyên
danh sách nhiều GPU mà `trainer.device` đã thu về GPU chính. So sánh fact với
yêu cầu bằng chuỗi thô làm `--device 0` / `0,1` retrain vô hạn (P1; ADR 0016).

**Mọi CUDA index được kiểm tra trước khi hỏi.** `check_device` không chỉ hỏi
"CUDA có sẵn không": nó đọc `torch.cuda.device_count()` và từ chối index
`>= count` (kèm dải hợp lệ trong thông điệp), từ chối danh sách trộn CUDA với
`cpu`/`mps`, và biến một `torch` không đọc được (`import`, `is_available()`,
`device_count()` ném lỗi) thành preflight đỏ thay vì traceback. Nhờ vậy
`--device 5` trên máy một GPU dừng *trước* lời nhắc retrain, không phải sau khi
đã xác nhận (ADR 0016).

**Manifest tách yêu cầu khỏi sự thật, và không ghi sự thật vào tên của yêu
cầu.** Trường không hậu tố (`imgsz`, `batch`, `seed`, `opset`, `workers`,
`device`) là *điều lần chạy sau sẽ yêu cầu lại*; trường `*_resolved`
(`imgsz_resolved`, `batch_resolved`, `device_resolved`) là *điều framework đã
làm*. `gate` so sánh nhóm yêu cầu: phát lại cùng một yêu cầu sẽ tái lập cùng
một phân giải (nó là hàm thuần của yêu cầu và backend identity), còn ghi phân
giải vào trường được so sánh thì một lần chạy `--batch 16` bị máy ép xuống 8 sẽ
không bao giờ khớp lại. Hai sự thật vẫn được so sánh vì không suy ra được từ
yêu cầu: `epochs` là **budget hiệu lực** của checkpoint khi `--resume` (giải
thích ở dưới), và `device_resolved` — `auto` không cho biết máy đã chạy CPU hay
GPU, nên một run CPU không được coi là cùng artifact với run GPU.
`imgsz_resolved` và `batch_resolved` chỉ để audit.

**`--batch -1` không bao giờ được bỏ qua.** Framework chọn batch từ bộ nhớ máy
*trong lúc train*, nên trước khi train không ai biết lần này sẽ là bao nhiêu;
hai lần cùng ghi `-1` không chứng minh chúng dùng cùng batch. "Không chứng minh
được" thì huấn luyện lại, đúng như một cấu hình chưa phân giải device.

**`export_args` là một phần của danh tính artifact.** Cùng weights nhưng opset
khác (hoặc `simplify`/`dynamic`/`half`/`int8`/`nms` khác, hoặc `imgsz` khác) là
một artifact khác. Pipeline truyền tường minh cả bộ cờ thay vì phó mặc default
của package, ghi nguyên mapping vào `export_args` và so sánh nguyên mapping
trước khi skip; entry không có trường này bị huấn luyện lại. `imgsz` phải là bội
của 32 — preflight từ chối trước khi train, vì framework sẽ làm tròn và contract
ONNX so sánh shape tĩnh với `--imgsz` sau khi đã train xong.

**Manifest ghi run *đã xảy ra*.** `epochs` là **budget hiệu lực** — khi
`--resume`, Ultralytics khôi phục toàn bộ tham số từ checkpoint và bỏ qua
`--epochs`, nên budget là con số của checkpoint (và entry có `resume_from` không
bao giờ được skip). `epochs_completed` là số epoch **thực sự hoàn thành** (cộng
dồn qua các lần resume): early stopping có thể kết thúc budget 100 epoch ở epoch
40, và budget với số đã hoàn thành không được gộp làm một. `resume_from` chỉ có
mặt khi artifact được tạo bởi một run tiếp tục run khác — và chính sự có mặt đó
khiến `--skip-unchanged` **không bao giờ** bỏ qua artifact (một lần chạy mới sẽ
không tái lập nó). Các **sự thật** khác (`imgsz_resolved`, `batch_resolved`,
`dataset`) được đọc từ `trainer.args` (sau train), rồi `ckpt["train_args"]`,
rồi mới đến giá trị pipeline đã phân giải; `device_resolved` cũng đọc
`trainer.args.device` trước (giữ nguyên danh sách nhiều GPU), rồi
`trainer.device`, rồi giá trị pipeline đã phân giải, và được chuẩn hóa về cách
viết của framework. Sự thật không
nguồn nào chứng minh được thì **để trống** (`epochs_completed` vắng mặt) thay vì
bịa ra. Các **yêu cầu** (`imgsz`, `batch`, `seed`, `device`…) thì lấy nguyên từ
`TrainConfig`, vì đó chính là điều lần chạy sau sẽ hỏi lại. Ghim bởi
`test_trainer.EffectiveEpochProvenanceTests`, `test_backend` và các mutant tương
ứng (ADR 0013/0015).

Tài liệu mang theo `schema_version`. Khi đọc, tài liệu cũ hơn (schema `0`: được
ghi trước khi trường tồn tại) được migrate tới bằng `cvtrain/manifest.py::
_MIGRATIONS`; tài liệu được ghi bởi một `cvtrain` **mới hơn** được đọc nguyên
trạng và cố ý *không* bị hạ version, để một build cũ không thể xoá mất version
mà người đọc tương lai cần cho phép migrate của riêng họ. Các trường mục *tùy
chọn* mới không làm bump schema: hình dạng đã là `total=False`, và một consumer
cần trường đó (quy tắc skip coi sự vắng mặt của nó là “provenance chúng ta
không có” và fail closed bằng cách huấn luyện lại).

Xuất bản một export sẽ thay ONNX cũ tại chỗ và giữ lại một thế hệ của nó ở
`<name>.onnx.bak`. File đó chỉ là artifact khôi phục: nó không có mục manifest,
`--verify-manifest` bỏ qua nó, và ứng dụng Qt không bao giờ load nó.
`--skip-unchanged` cũng từ chối tin nó — một model chỉ bị bỏ qua khi *tất cả*
các điều kiện sau đúng với file đã publish, để artifact được `--write-manifest`
nhận về (mà không ghi provenance) bị huấn luyện lại:

* digest của nó khớp với những gì manifest đã ghi;
* artifact **không** mang `resume_from`: một run `--resume` bắt đầu từ
  checkpoint của run khác và không theo bộ tham số của lần gọi này, nên nó
  không phải thứ một lần chạy sạch sẽ tạo ra — kể cả khi mọi trường được ghi
  tình cờ khớp;
* `epochs`, `imgsz`, `seed`, `opset`, `batch`, `workers` và `device` khớp
  **chính xác** với lần gọi này (đây là các *yêu cầu*; giá trị framework phân
  giải được ghi riêng ở `*_resolved` và không tham gia so sánh);
* `export_args` khớp nguyên vẹn: cùng weights nhưng opset/`simplify`/`dynamic`/
  `half`/`int8`/`nms`/`imgsz` khác là artifact khác, và entry thiếu trường này
  là provenance không đủ để tin;
* `--batch -1` **không bao giờ** được skip: batch do framework chọn từ bộ nhớ
  máy trong lúc train, nên lần chạy này sẽ dùng batch nào là điều không biết
  trước khi train;
* phiên bản backend đã ghi lại là phiên bản sẽ tạo ra nó ngay lúc này (upgrade
  framework làm đổi bản export);
* hạng device khớp (lần chạy CPU và CUDA với cùng tham số là hai artifact khác
  sau cùng một tên file; “unspecified” và `auto` là cùng một thiết bị);
* device **đã phân giải** cũng khớp (`device_resolved`) — và cả hai phía của
  phép so sánh này đều phải *biết*: một cấu hình chưa phân giải device không
  thể biết lần chạy này sẽ build ra gì, nên nó cũng bị huấn luyện lại. Phép so
  sánh dùng cách viết chuẩn của framework (`0` ≡ `cuda:0`, `0,1` ≡
  `cuda:0,cuda:1`), nên một entry cũ ghi theo cách của người vận hành vẫn là
  cùng một artifact (ADR 0016);
* `data.yaml` vẫn cho cùng một digest — riêng đường dẫn không phải là provenance,
  vì đổi tên class hoặc chuyển split sau một cái tên không đổi là một thí điểm
  khác;
* digest **nội dung** các file `train`/`val` vẫn khớp (đường dẫn tương đối +
  byte của từng file). Đây là lý do fingerprint hash byte chứ không chỉ
  `(tên, kích thước)`: thay một ảnh bằng pixel khác cùng độ dài vẫn là dataset
  khác, và `--skip-unchanged` không được phép bỏ qua nó. Chi phí O(dataset) mỗi
  lần kiểm tra là có chủ đích — đó là cái giá của một bằng chứng thật. Digest
  được ghi bởi thuật toán inventory cũ sẽ không khớp một lần sau khi nâng cấp,
  nên lần chạy đầu tiên huấn luyện lại thay vì tin nhầm (fail-closed);
* checkpoint gốc trên đĩa vẫn cho digest đã ghi khi artifact được huấn luyện —
  `yolo11n.pt` bị thay hoặc tải lại là một điểm xuất phát khác, và nếu nó biến
  mất thì không thể xác minh gì cả. Với artifact của run `--resume`, digest
  được ghi là của **file run thực sự bắt đầu** (`last.pt` của run bị gián
  đoạn), không phải checkpoint gốc mà spec đặt tên: hash file sau khi train
  sẽ mô tả một checkpoint mà chính run đó vừa ghi đè.

Các digest được so sánh sau cùng: chúng là phần tốn kém nhất, nên một sai khác ở
các siêu tham số rẻ hơn không bao giờ khiến ta phải trả giá cho việc hash một
checkpoint.

Pipeline được thiết kế cho **một người ghi** (ứng dụng desktop chạy một công
việc huấn luyện tại một thời điểm), và giả định đó giờ được *buộc* chứ không chỉ
được nêu: một lần chạy — và cả `--write-manifest`, cả `--verify-manifest` — giữ
một advisory lock (`Models/.cvtrain.lock`) trong toàn bộ cửa sổ đọc–huấn
luyện–ghi, nên lần `cvtrain` thứ hai nhắm vào cùng thư mục fail nhanh với
`ConcurrentRunError` (exit `2`) thay vì giành nhau `manifest.json`. Việc xác
minh được đưa vào một cách có chủ đích: nó hash các artifact mà một lần chạy
song song có thể đang thay, và “chờ lần chạy kia xong” là câu trả lời trung
thực ở chỗ mà một cuộc quét không lock sẽ báo một sai khác chỉ tồn tại do cuộc
đua. File lock mang pid+token, và việc giải phóng chỉ xoá nó khi nó vẫn còn của
chúng ta — một lần đánh cắp lock xảy ra giữa chừng không bao giờ bị mở ra khỏi
chủ mới của nó. File lock ghi cả `host` + `pid` và được làm tươi bởi heartbeat
mỗi 30 giây (và mỗi epoch qua `fsio.touch_lock`): một chủ **cùng máy** được
chứng minh còn sống thì lock không bao giờ bị đánh cắp dù lease đã cũ, còn chủ
đã chết thì lock thành stale **ngay** thay vì phải chờ `LOCK_STALE_SECONDS`
(24 giờ). Khi không probe được chủ (máy khác, lock của bản cũ, thiếu quyền),
tuổi lease quyết định — "không biết" không bao giờ bị đọc thành "đã chết".
Mọi thao tác ghi đều nguyên tử bất kể
thế nào, nên ngay cả một writer bên ngoài cũng không để lại file rách — và một
lỗi ghi mà OS từ chối (`Models/` chỉ-đọc, đĩa đầy, Windows sharing violation)
xuất hiện dưới dạng `StorageError` có kiểu (exit `2`) thay vì một traceback.
Ngoại lệ duy nhất là output của framework lẽ ra phải tồn tại mà không tồn tại:
đó là bug và giữ nguyên stack trace của nó.

## Kiểm thử

```powershell
cd AIComputerVision
pip install --require-hashes -r requirements-test.lock   # ruff + mypy + coverage, đã hash-pin
ruff check .
python -m mypy
coverage run -m unittest discover -s tests -v
coverage report
python scripts/mutation_check.py
```

> **Cổng chất lượng phải cài từ lock, không cài gói trôi** (ADR 0012): `ruff`,
> `mypy` và `coverage` *quyết định* xanh hay đỏ, nên một bản phát hành mới có
> thể đổi kết luận của chúng mà không ai duyệt thay đổi đó. `requirements-test.lock`
> là bản hash-pinned của `requirements-test.txt`; CI cài đúng lock đó bằng
> `--require-hashes`, nên pip từ chối bất kỳ artifact nào có digest không khớp.
> Lệnh sinh lại lock nằm ở [`Docs/SCORECARD.md`](Docs/SCORECARD.md) §5: đổi một
> phiên bản trong `requirements-test.txt` thì phải sinh lại lock và commit cả
> hai — nếu không, `test_toolchain_contract` sẽ đỏ.

Ruff, mypy và coverage được cấu hình bởi `pyproject.toml` trong thư mục này: ruff chọn
`E/W/F/I/BLE/S/RUF`, mypy chạy với bộ tuỳ chọn strict trên
`files = ["cvtrain", "tests", "scripts", "TrainModel.py"]` (nên lệnh `python -m mypy` trần
đã bao phủ package), và coverage theo branch với `fail_under = 100`. `cvtrain`
bị chặn ở **100% statement và branch coverage** — mọi dòng đều được diễn chạy,
nên code mới phải đi kèm test của nó.

Coverage chỉ chứng minh quyết định được *chạy*, không chứng minh nó đúng chiều.
`scripts/mutation_check.py` đảo từng quyết định trọng yếu **một lần** rồi bắt
suite phải đỏ. Danh sách 126 mutant trong script **là hợp đồng** (ADR 0007, mở
rộng bởi ADR 0011, 0013, 0014, 0015 và 0016): nó phủ fail-closed provenance, quality gate,
hợp đồng ONNX
dùng chung với runtime Qt, thứ tự publish, advisory lock, retention, giao thức
progress và taxonomy mã thoát — một guard không có mutant là một guard chưa ai
ghim. Mutant sống sót, mutant treo (`HUNG`) hoặc anchor đã cũ đều làm bước CI
thất bại. Cờ `--files` chạy một *tập con* mutant (job Windows quét lại các
quyết định phụ thuộc nền tảng) nhưng **không** hạ được floor: floor luôn được
kiểm tra trên toàn bộ tập mutant khai báo.

Script chạy trong **sandbox**: nó copy package + test + nguồn C++ vào thư mục
tạm rồi đảo chiều *bản copy*, nên cây làm việc không bị ghi, và một mutant lọt
vào nhánh huấn luyện sẽ fail nhanh ở preflight thay vì chạy thật trên dataset
của máy.

> **Quy tắc của ADR 0007:** mọi claim trong tài liệu này và trong các ADR phải
> được **pin** bằng test hoặc mutant, hoặc bị sửa thành điều đúng. Claim không
> pin được thì không được viết ra.
>
> Số đo cụ thể của từng cổng, ngày đo và bảng nợ đang mở nằm ở
> [`Docs/SCORECARD.md`](Docs/SCORECARD.md) — một nơi duy nhất để trả lời "package
> này đã ổn chưa".

Bảng các file test và nội dung từng file nằm ở mục [Layout](#layout) phía trên.
Ba điều đáng nói thêm:

* `tests/test_architecture.py` không chỉ buộc kiến trúc mà còn buộc **chính
  README này**: khối Layout phải liệt kê đúng các module của package, và mọi
  tuỳ chọn dòng lệnh được nhắc tới phải được CLI chấp nhận (và cả chiều ngược
  lại — CLI không được có cờ nào không được tài liệu hoá). Tài liệu sai
  thì test đỏ.
* `tests/test_contract.py` và `tests/test_qt_contract.py` buộc **hai chiều**
  với phía C++: tên file ONNX, trần opset theo ONNX Runtime được pin trong
  `CMakeLists.txt`, hằng schema tensor, ngưỡng IoU của tracker, argv/nhãn
  checkbox và regex `pct=` của dock Qt. Chúng còn có *presence guard*: nếu file
  C++ bị đổi tên, contract phải đỏ chứ không âm thầm tự tắt.
* Bộ suite offline giả lập framework ML, nên drift API trong `ultralytics` được
  pin chỉ có thể bắt được bằng cách huấn luyện thật. Workflow
  `.github/workflows/cvtrain-nightly.yml` chạy stack thật (CPU torch) trên một
  dataset tổng hợp tí hon — một epoch cho detection và segmentation, huấn
  luyện → ONNX → manifest → xác minh — hằng đêm, khi gọi tay, **và khi có pull
  request đụng vào package**. Một job thứ hai (`windows-train-smoke`) chạy
  đúng stack đó trên `windows-latest` (một epoch detection) vì sản phẩm ship
  Windows và các wheel torch/ONNX Runtime/onnxslim là native: cài được, load
  được và export được trên nền tảng đó phải là một verdict của CI, không phải
  niềm tin (ADR 0014). Tracking cố ý đứng ngoài nightly: `yolo11x` nhân
  ba ngân sách trong khi vẫn diễn chạy cùng một seam backend.

### CI: hai nền tảng, một chuỗi công cụ đã pin

Workflow `.github/workflows/Build-cvtrain-ci.yml` chạy cổng chất lượng trên
**`ubuntu-latest` và `windows-latest`** (ADR 0012). Lý do không phải là "cho
đủ số": sản phẩm ship trên Windows, và package này đụng vào ngữ nghĩa tệp tin
của Windows — advisory lock `O_EXCL`, publish nguyên tử, lỗi sharing violation
khi một file đang bị thay. Một CI chỉ chạy trên Linux sẽ báo xanh trong khi
nền tảng người dùng thật chưa từng được kiểm tra. Job Windows còn quét lại
mutation của `cvtrain/fsio.py` (21 mutant: advisory lock, lease/owner liveness,
publish nguyên tử, retention) — đúng những quyết định mà hành vi phụ thuộc
nền tảng.

`--resume` là hợp đồng **hai tầng**, và cả hai tầng đều có cơ chế làm đỏ:

| Tầng | Kiểm chứng | Chạy ở đâu |
|------|-----------|------------|
| Seam của pipeline (chọn checkpoint mới nhất, từ chối checkpoint không chứng minh được, truyền `resume=True`, không tạo thư mục run mới, ghi budget/completed/`resume_from`) | `test_backend.TrainTests` + `test_trainer.EffectiveEpochProvenanceTests` + **14 mutant** | offline, mọi lần |
| Hành vi của framework thật (checkpoint mang bộ đếm epoch và trạng thái optimizer được khôi phục; manifest ghi đúng budget 5 và completed 5) | workflow `cvtrain-nightly.yml`: huấn luyện → `kill -9` → `--resume` trong `ResumeRuns` riêng → đếm thư mục run → đọc manifest | nightly **và** pull request đụng package |

Nếu job *real-dependency* bị xoá hoặc chỉ chạy theo lịch, tripwire trong
`tests/test_ci_contract.py` sẽ đỏ — một claim không còn được kiểm chứng thì
không được viết ra (ADR 0007).
