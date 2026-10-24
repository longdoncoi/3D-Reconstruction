# AIComputerVision — Scorecard

> Số đo được, mỗi số đều do một lệnh trong repo này tạo ra — không có gì ở
> đây là ước lượng. **Đo lần cuối: 2026-10-24**, Windows, Python 3.11,
> dependency theo `requirements-test.lock` (`ruff==0.16.0`, `mypy==2.4.0`,
> `coverage==7.16.2`, `onnx==1.21.0`, `onnxruntime==1.20.1`, `numpy==1.26.4`),
> GPU NVIDIA RTX 3050 (chỉ dùng cho lần retrain ở N1 — bốn cổng đầu chạy
> offline, không cần GPU).
>
> Tài liệu này **không tuyên bố một điểm số**. Nó ghi những gì đo được, cách
> đo lại, và danh sách việc còn mở. Luật của [ADR 0007](adr/0007-every-claim-is-pinned-or-removed.md)
> áp dụng cho chính tài liệu này: mọi con số dưới đây đều tái lập được bằng
> một lệnh, và mọi claim phải được *pin* bằng test/mutant hoặc được ghi thẳng
> vào bảng nợ ở §3.

## 1. Cổng chất lượng

| Cổng | Lệnh | Đo được 2026-10-23 |
|------|------|---------------------|
| Lint | `ruff check .` | 0 issue |
| Kiểu | `python -m mypy` | 0 issue, 33 file (`disallow_untyped_defs`, `warn_return_any`, `strict_equality`…) |
| Byte-compile | `python -m compileall -q cvtrain tests TrainModel.py scripts` | sạch |
| Test | `python -m unittest discover -s tests` | **508 test, 0 fail** (499 trước ADR 0014) |
| Coverage | `coverage run -m unittest discover -s tests && coverage report` | **100%** — 1 919 stmts / 490 branch, 0 miss, 0 partial (`fail_under = 100`) |
| Mutation | `python scripts/mutation_check.py` | **107/107 killed**, 0 survived, 0 hung, 0 stale; `MIN_MUTANTS = 107` — xoá một mutant là gate đỏ |
| Ranh giới kiến trúc | `python -m unittest discover -s tests -p "test_architecture.py"` | 13 test: đồ thị phụ thuộc một chiều, heavy dep lazy + một owner, `UltralyticsBackend` chỉ ở `cli.py`, Layout README ↔ package, cờ CLI ↔ README (hai chiều) |
| Hợp đồng C++ | `python -m unittest discover -s tests -p "test_contract.py"` + `-p "test_qt_contract.py"` | 7 + 9 test: tên file ONNX, trần opset theo ORT được pin, schema tensor ↔ `AIYoloPostProcessor.h`, ngưỡng IoU ↔ `AIProcessor.cpp`, argv/nhãn/regex `pct=` ↔ `AITrainDockWidget.cpp` |
| **Chuỗi công cụ** (ADR 0012) | `python -m unittest discover -s tests -p "test_toolchain_contract.py"` | 9 test: mọi requirement là pin chính xác (ngoại lệ `torch` có lý do), `onnxruntime` = `ONNXRUNTIME_VERSION`, lock mang digest cho từng gói, CI cài lock bằng `--require-hashes` |
| **Hai nền tảng + resume hai tầng** (ADR 0012) | `python -m unittest discover -s tests -p "test_ci_contract.py"` | 10 test: cổng chạy trên `windows-latest`, mutation phụ thuộc nền tảng được quét lại trên Windows, job real-dependency còn huấn luyện → cắt → `--resume` và chạy cho pull request, và job thật thứ hai chạy trên Windows |
| Verdict | `python TrainModel.py --verify-manifest` | exit 0; sha256 của cả 3 artifact khớp manifest |

> **Cổng chất lượng được cài từ lock hash-pinned** (ADR 0012):
> `pip install --require-hashes -r requirements-test.lock`. `ruff`, `mypy` và
> `coverage` *quyết định* xanh hay đỏ, nên chúng phải là một thứ đã pin —
> cùng luật mà phía `AIAssistant` đã áp dụng. Lần đo này chạy trong một venv
> sạch cài đúng lock đó: kết quả y hệt.
>
> **CI chạy trên hai nền tảng.** `.github/workflows/Build-cvtrain-ci.yml` chạy
> cổng trên `ubuntu-latest` *và* `windows-latest` (sản phẩm ship Windows), và
> job Windows quét lại mutation của `cvtrain/fsio.py` (21 mutant — advisory
> lock, lease/owner liveness, publish nguyên tử, retention).

> **Gate mutation chạy trong sandbox.** Script copy package + test + nguồn C++
> vào thư mục tạm rồi đảo chiều *bản copy*; cây làm việc không bị ghi. Sandbox
> không có `Dataset/`, không có checkpoint, không có `Models/`, nên một mutant
> lọt vào nhánh huấn luyện sẽ **fail nhanh ở preflight** thay vì chạy thật
> (xem ADR 0007).
>
> **Số mutant cũng là một gate.** `MIN_MUTANTS = 107` (cùng quy tắc với phía
> Python service, ADR 0032, và gate C++): xoá một mutant mà không có quyết
> định thì suite vẫn xanh, coverage vẫn 100 %, và quyết định mà mutant đó ghim
> lặng lẽ mất người bảo vệ. Gate kiểm tra floor **trước** khi chạy sandbox nên
> vi phạm là đỏ ngay, không phải sau 20 phút.

> **Ba phép kiểm chứng độc lập** (không nằm trong bảng trên, nhưng là phần
> của cùng một câu hỏi — "artifact đã ship có thật sự thoả hợp đồng không?"):
> `import cvtrain` kéo **0** module nặng (`ultralytics`/`torch`/`onnx`/
> `onnxruntime`/`numpy`/`yaml`); `onnx_contract.validate_export` chạy trên
> **chính** 3 artifact trong `Models/` → cả 3 **OK** (kèm smoke inference bằng
> ONNX Runtime); `git status` sau mutation gate → **không có dòng nào** trong
> `AIComputerVision/`.

## 2. Tiêu chí "10/10" (5 điều kiện, tất cả đều đo được)

| # | Tiêu chí | Điều kiện kiểm tra | Trạng thái |
|---|----------|--------------------|-----------|
| **C1** | 5 cổng xanh | Lint + kiểu + test + coverage + mutation đều exit 0 | ✅ |
| **C2** | Không dòng nào thoát khỏi phép đo | `fail_under = 100` (statement **và** branch) trong `pyproject.toml`; CI đỏ nếu dưới | ✅ 1 919/1 919 |
| **C3** | Mọi quyết định trọng yếu đã được ghim | 107 mutant, 0 survived, 0 hung, 0 stale; **số mutant là một floor** (`MIN_MUTANTS = 107`) | ✅ 107/107 |
| **C4** | Nợ đã đo và ghi | Bảng §3 — mỗi dòng có số đo, cách đóng và cơ chế ghim | ✅ **14 đóng / 0 đang mở** |
| **C5** | Tài liệu khớp code | README ↔ CLI hai chiều, Layout ↔ package, ADR 0003/0008 ↔ manifest thật, hợp đồng C++ hai chiều | ✅ |

Điểm khác biệt so với trước: **C3 trước đây chỉ phủ 8 quyết định**. Một mutant
đảo chiều làm `exporter_identity()` ghi cứng `onnx=unknown` — tắt đúng cái bảo
vệ mà ADR 0003 tuyên bố — vẫn xanh suite. Nay nó đỏ. Tất cả các nợ ở §3 đều
được đóng bằng quyết định + cơ chế ghim, không bằng lời.

Lần rà soát kiến trúc 2026-10-21 (Kiến trúc sư trưởng) tìm thêm ba nợ và
đóng cả ba: **N7** (một quy tắc fail-closed chỉ fail-closed ở một phía),
**N8** (không có chính sách giữ lại cho `runs/`) và **N9** (provenance ghi số
epoch *yêu cầu* thay vì số epoch framework thực sự chạy). Cùng đợt đó,
mutation set được mở rộng từ 67 lên 80 để phủ hết các claim trước đây chỉ
được ghim bằng test — xem [ADR 0011](adr/0011-fail-closed-on-both-sides-and-the-mutation-set-covers-every-claim.md).

Lần đo lại 2026-10-22 xác nhận C1–C5 vẫn xanh (443 → **459 test** sau ADR
0012) và tìm thêm ba nợ mà *tiêu chí* không đo: **N10** (chuỗi công cụ của
gate không được pin), **N11** (cổng chỉ chạy trên Linux trong khi sản phẩm
ship Windows) và **N12** (`--resume` chỉ được kiểm chứng trong một job theo
lịch). Cả ba đã đóng — xem [ADR 0012](adr/0012-a-hash-pinned-toolchain-the-shipping-platform-and-a-two-layer-resume-contract.md).

Lần rà soát 2026-10-23 tìm thêm **N13**: bốn lỗi cùng một hình dạng — một
claim đúng trong tài liệu nhưng code làm khác — ở resume provenance, tính
fail-closed của `--resume`, liveness của lock/run, và vị từ metric. Cả bốn
đã đóng bằng [ADR 0013](adr/0013-resume-is-provenance-effective-facts-and-a-lease-is-evidence-of-life.md);
suite 459 → **499 test**, coverage vẫn 100 % (1 902/486), mutation set
80 → **101**.

Lần rà soát 2026-10-24 tìm thêm **N14**: năm chỗ cùng hình dạng đó còn sót lại
sau N13 — lease của run `--resume` không đổi chủ, identity của exporter bỏ sót
`onnxslim`, preflight chỉ chứng minh *một* package import được, "tuổi không rõ"
bị coi là già nhất khi prune, và job thật chỉ chạy trên Linux trong khi sản
phẩm ship Windows (kèm comment CI còn số cũ). Cả năm đóng bằng
[ADR 0014](adr/0014-the-lease-changes-hands-the-simplifier-is-provenance-and-the-real-stack-runs-on-windows.md);
suite 499 → **508 test**, coverage vẫn 100 % (1 919/490), mutation set
101 → **107**.

## 3. Bảng nợ

Nguyên tắc (giống `Docs/ARCHITECTURE_SRC.md` phía C++): **nợ phải được đo và
được ghi**, không được biến mất khỏi báo cáo. Đóng nợ = một quyết định được
ghi thành ADR + một cơ chế làm đỏ nó (test hoặc mutant) — không phải một dòng
ghi chú.

| # | Nợ | Đo được | Cách đóng | Trạng thái |
|---|-----|---------|-----------|------------|
| N1 | **Provenance của model đã ship dùng định dạng cũ.** `Models/manifest.json` ghi `backend_version: "8.4.41"` (bản `ultralytics` trần) cho `yolo11n.onnx` / `yolo11n-seg.onnx`, và **không có** `backend_version`/`epochs`/`metrics` cho `yolo11x-tracking.onnx` — trong khi code ghi identity tổng hợp `cvtrain=…;ultralytics=…;torch=…;onnx=…` | `Models/manifest.json` → `entries.*.backend_version` | **Retrain bằng build hiện tại** (ADR 0008) — identity luôn do *run sinh ra artifact* ghi, không bao giờ dựng lại sau. Đo lại sau retrain (2026-10-21): cả **3** mục đều mang `backend_version = cvtrain=1.0.0;ultralytics=8.4.41;torch=2.7.0+cu128;onnx=1.21.0` + `epochs/imgsz/batch/device` + `trained_at` + `metrics`; `metrics/mAP50(B)` của det và seg **giữ nguyên 0.975** ⇒ không suy giảm chất lượng; tracking được đo lần đầu (0.975) thay vì "không rõ". Phép thử quyết định: `TrainModel.py --yes --skip-unchanged --det --seg --track` → **cả 3 mục "already matches this invocation; skipping"** (trước retrain: cả 3 đều bị train lại). Ghim bằng `test_an_entry_with_the_legacy_exporter_version_is_trained` | ✅ Đóng (2026-10-21) |
| N2 | **Provenance gắn với một máy.** `dataset` và `run_dir` trong manifest là đường dẫn tuyệt đối, và quy tắc skip *so sánh chuỗi đường dẫn* | `Models/manifest.json` → `entries.*.dataset` | **So sánh theo digest nội dung, không theo đường dẫn** (ADR 0008). Đường dẫn vẫn được *ghi* (provenance) nhưng không được *so sánh*: `dataset_sha256` + `dataset_content_sha256` mới là bằng chứng "cùng một dataset". Ghim bằng `test_the_same_dataset_behind_another_path_is_not_retrained` | ✅ Đóng (2026-10-21) |
| N3 | **Không có `resume` cho training dài.** Mỗi lần chạy tạo thư mục `runs/<x>/train-<timestamp>-<pid>` mới; crash ở epoch 4/5 mất trắng cả run | `cvtrain/backend.py` (`UltralyticsBackend.train`) | **`--resume`** (ADR 0008): lấy `weights/last.pt` mới nhất của model và train với `resume=True`, nên framework khôi phục đúng siêu tham số và bộ đếm epoch của run bị gián đoạn. Mặc định tắt. **Kiểm chứng thật (2026-10-21):** một run `yolo11x` bị giết ở epoch 5/5 → `--resume` in ra `Resuming training … from epoch 5 to 5 total epochs` và chỉ train nốt epoch cuối, không train lại từ đầu. Ghim bằng `test_backend.TrainTests` (resume, fail-closed, dataset mismatch, checkpoint ghi đè) + `test_trainer` (artifact resume không bị skip) + **14 mutant** | ✅ Đóng (2026-10-21) |
| N4 | **Fingerprint dataset chạy khi progress bar đứng yên.** `snapshot_dataset` hash toàn bộ byte của dataset trước khi train, không có output | `cvtrain/gate.py` → `cvtrain/dataset.py` | **Fingerprint báo tiến độ** (ADR 0008): `on_file(done, total)` đi từ `dataset.py` lên tới reporter, và giao thức progress có thêm pha `dataset` (2–8 %). Ghim bằng `test_the_fingerprint_reports_progress_for_every_file` + 2 mutant | ✅ Đóng (2026-10-21) |
| N5 | **`module_available` chỉ kiểm `find_spec`**, không chứng minh package import được | `cvtrain/checks.py` | **Backend tự chứng minh khả năng import** (ADR 0008, mở rộng bởi ADR 0014): `TrainingBackend.check_importable` import **mọi** module `required_modules` khai báo (`ultralytics`, `yaml`, `onnx`, `onnxslim`, `onnxruntime`, và `tensorboard` khi được yêu cầu), đưa verdict vào *cùng báo cáo preflight*, nên stack hỏng là lỗi preflight chết (exit 2) chứ không phải bất ngờ ở bước export. Import nằm trong adapter → đồ thị phụ thuộc không đổi (ADR 0001). Ghim bằng `test_a_stack_that_cannot_be_imported_fails_in_preflight`, `test_a_declared_dependency_that_cannot_import_is_named` + **4 mutant** | ✅ Đóng (2026-10-21, mở rộng 2026-10-24) |
| N6 | **`--device auto` không tái lập được.** Manifest ghi `device: "auto"` — nghĩa là "để framework chọn" — nên một lần chạy trên GPU và một lần chạy trên CPU đều ghi giống nhau, dù artifact khác nhau | `Models/manifest.json` → `entries.*.device` | **Pipeline tự phân giải device trước khi train** (ADR 0009): `TrainingBackend.resolve_device` trả về một device *cụ thể* (`auto` → `cuda:0` nếu runtime có GPU, không thì `cpu`; yêu cầu tường minh là câu trả lời của chính nó), framework được *báo* dùng device nào thay vì tự chọn, và provenance ghi **cả hai**: `device` (yêu cầu) + `device_resolved` (sự thật). Quy tắc skip so sánh cả hai; mục không có `device_resolved` ⇒ *unprovable* ⇒ retrain. Đo lại (2026-10-21): `--check` in `Device: cuda:0 (requested auto)`. Ghim bằng 4 test (`test_backend.DeviceResolutionTests`, `test_trainer.DeviceResolutionTests`) + 4 mutant | ✅ Đóng (2026-10-21) |
| N7 | **Quy tắc fail-closed của phép so sánh device chỉ fail-closed ở MỘT phía.** `gate.already_current` so sánh `device_resolved` với điều kiện `cfg.device_resolved is not None and …` — nếu *cấu hình* chưa resolve thì phép so sánh bị **bỏ qua hẳn**, trong khi mọi "không chứng minh được" khác (thiếu digest dataset, thiếu digest checkpoint, thiếu digest artifact) đều ⇒ retrain. Đường sản xuất hiện tại **không** với tới (`trainer._execute` luôn resolve trước khi hỏi), nhưng đây đúng là bẫy cho người gọi kế tiếp — và một hàm public có docstring "fails closed" mà không fail-closed là mâu thuẫn tự thân | `cvtrain/gate.py` → `already_current` (nhánh so sánh `device_resolved`) | **Fail-closed ở CẢ HAI phía** (ADR 0011): một cấu hình chưa resolve không biết lần chạy này sẽ train trên máy nào, nên nó không thể khẳng định artifact đã publish là artifact mà lần chạy này sẽ build ⇒ retrain, giống hệt mọi "không chứng minh được" khác. Ghim bằng `test_an_unresolved_configuration_never_claims_a_model_is_current` + **2 mutant** (`gate.py`: bỏ phép so sánh, bỏ guard) | ✅ Đóng (2026-10-21) |
| N8 | **Không có chính sách giữ lại cho `runs/`.** Mỗi lần huấn luyện để lại một thư mục run (checkpoint `yolo11x` vài trăm MB); không có gì dọn, nên kích thước `runs/` là hàm của số lần đã train — một sự thật đúng, khám phá được, và không thuộc về ai. (`Models/*.onnx.bak` thì **có** giới hạn: một thế hệ mỗi model, nên không cần quyết định riêng.) | `AIComputerVision/runs/` | **Retention là một chính sách, không phải tai nạn** (ADR 0010): `--prune-runs` là một *mode* riêng — không bao giờ train, không bao giờ chạm `Models/`; giữ run mới nhất của *mỗi* model (nên `--resume` vẫn thấy checkpoint), **không bao giờ xoá run còn sống** (checkpoint `weights/last.pt` tươi = bằng chứng run còn chạy), và chỉ xoá thư mục do pipeline này tạo. Mỗi run còn mang lease `.cvtrain-alive` (host + PID) được backend làm tươi mỗi epoch và xoá khi train xong; pruner giữ run có chủ cùng máy còn sống dù đã quá 24 giờ, còn marker của máy khác được tính theo mtime. Ghim bằng `test_fsio.PruneRunDirectoriesTests` + **10 mutant** (`fsio.py` lease/owner/marker/prune × 9, `cli.py` × 1) | ✅ Đóng (2026-10-21) |
| N9 | **Provenance ghi số epoch *yêu cầu*, không phải số epoch framework thực sự chạy.** `manifest` ghi `epochs=cfg.epochs`, nhưng khi `--resume`, Ultralytics **không** dùng `--epochs` của dòng lệnh: `check_resume` khôi phục toàn bộ tham số từ checkpoint và danh sách override được phép **không** chứa `epochs`. Nên manifest mô tả một run chưa từng xảy ra — và `gate.already_current` *so sánh* trường này, nên một giá trị sai làm verdict `--skip-unchanged` tin vào một mệnh đề provenance không đúng | `Models/manifest.json` → `entries.*.epochs` | **Provenance ghi điều framework chạy** (ADR 0011, tách đôi bởi ADR 0013): adapter đọc budget từ `Trainer.epochs` và số epoch đã hoàn thành từ `Trainer.epoch + 1`, rồi ghi thành hai trường riêng (`epochs`, `epochs_completed`); số yêu cầu chỉ dùng khi framework không báo được ("không chứng minh được" thì *để trống*, không bịa ra con số). Ghim bằng `test_trainer.EffectiveEpochProvenanceTests` + `test_backend` (2 test) + **3 mutant** (`backend.py` × 2, `trainer.py` × 1) | ✅ Đóng (2026-10-21) |
| N10 | **Chuỗi công cụ của gate không được pin.** Job `cvtrain` cài `pip install ruff mypy coverage PyYAML==6.0.2 onnx==1.21.0 "onnxruntime>=1.20.1"` — `ruff`/`mypy`/`coverage` *quyết định* xanh hay đỏ, nên một bản phát hành mới có thể đổi kết luận của gate mà không ai duyệt thay đổi đó. `requirements.txt` cũng để `onnxruntime>=1.20.1` là dải, dù đó là engine hợp đồng export — engine mà ứng dụng Qt ship | `.github/workflows/Build-Python-ci.yml` (job `cvtrain`) + `AIComputerVision/requirements.txt` | **Lock hash-pinned cho toolchain của gate** (ADR 0012): `requirements-test.txt` (mọi pin *chính xác*) + `requirements-test.lock` (bản hash-pinned sinh bằng `uv pip compile … --generate-hashes`), CI cài bằng `pip install --require-hashes`. `onnxruntime` pin **chính xác** bằng `ONNXRUNTIME_VERSION` trong `CMakeLists.txt`. Stack của job real-dependency cũng được pin (`torch==2.7.0`, `torchvision==0.22.0`, `pillow==11.0.0`) để verdict của job đó không phụ thuộc ngày nó chạy. **Đo lại (2026-10-22):** lock cài sạch vào venv mới bằng `--require-hashes` (20 gói, mọi artifact của mỗi phiên bản đều có digest trong lock → cài được trên cả Linux lẫn Windows) và chạy lại **toàn bộ** cổng trong venv đó: ruff 0 · mypy 0 (32 file) · 459 test 0 fail · coverage 100 % (1 667/412). Ghim bằng `test_toolchain_contract.py` (9 test) + **3 phép đảo chiều** (mỗi phép đảo một quyết định trong file và bắt test tương ứng phải đỏ — xét dưới) | ✅ Đóng (2026-10-22) |
| N11 | **Cổng chỉ chạy trên Linux, trong khi sản phẩm ship Windows.** Job `cvtrain` chạy `runs-on: ubuntu-latest`; package này đụng ngữ nghĩa tệp tin của Windows — advisory lock `O_CREAT` + `O_EXCL`, publish nguyên tử (staging + `os.replace`), lỗi sharing violation khi một file đang bị thay. Một CI chỉ chạy trên Linux báo xanh trong khi nền tảng người dùng thật chưa từng được kiểm tra | `.github/workflows/Build-cvtrain-ci.yml` (job `gates`) | **Cổng chạy trên cả hai nền tảng** (ADR 0012): workflow riêng `.github/workflows/Build-cvtrain-ci.yml` với `matrix.os = [ubuntu-latest, windows-latest]` cho lint + kiểu + compileall + toàn bộ suite + coverage 100 %; job Windows còn quét lại mutation của `cvtrain/fsio.py` (advisory lock, lease/owner liveness, publish nguyên tử, retention), tức là đúng những quyết định mà hành vi phụ thuộc nền tảng. `scripts/mutation_check.py` có thêm `--files` cho việc này, nhưng **không** hạ được floor (floor luôn kiểm trên toàn bộ tập mutant khai báo) và phần tóm tắt luôn ghi rõ bao nhiêu/bao nhiêu mutant đã quét. **Đo lại (2026-10-22):** bộ suite chạy xanh trên Windows (nơi lần đo này diễn ra) với cùng lock mà CI sẽ dùng. Ghim bằng `test_ci_contract.py` (3 test) + **2 phép đảo chiều** | ✅ Đóng (2026-10-22) |
| N12 | **`--resume` chỉ được kiểm chứng trong một job theo lịch.** Đây là hành vi duy nhất backend giả không thể vay mượn: framework thật phải viết được checkpoint mang bộ đếm epoch + trạng thái optimizer và phải chấp nhận `resume=True`. Job `cvtrain-nightly.yml` làm đúng điều đó, nhưng nó chỉ chạy theo lịch → một thay đổi làm hỏng seam resume có thể lên `main` và chỉ lộ ra sáng hôm sau | `.github/workflows/cvtrain-nightly.yml` | **`--resume` là hợp đồng hai tầng, cả hai đều có cơ chế** (ADR 0012): *tầng 1* — seam của pipeline (chọn checkpoint mới nhất, truyền `resume=True`, không tạo thư mục run mới) đã ghim offline bằng `test_backend.TrainTests` + **14 mutant** (fail-closed, dataset mismatch, plan source, effective facts); *tầng 2* — hành vi framework thật: job real-dependency giờ chạy **cho pull request đụng vào package** (thêm trigger `pull_request` với path filter `AIComputerVision/**`), không chỉ theo lịch, và dùng `ResumeRuns` riêng với assertion `epochs=5`, `epochs_completed=5`, `resume_from`. Ghim bằng `test_ci_contract.py` (9 test) + **4 phép đảo chiều** (xoá assertion `Resuming …` → đỏ; bỏ trigger `pull_request` → đỏ; bỏ `ResumeRuns` → đỏ; bỏ `epochs_completed` → đỏ) | ✅ Đóng (2026-10-22) |

| N13 | **Resume, provenance, liveness và metric mô tả lời gọi thay vì run.** (a) manifest ghi `cfg.*` của lần gọi dù resume dùng tham số checkpoint, và digest là của checkpoint gốc thay vì file run thực sự bắt đầu; (b) `--resume` fail-open: không có checkpoint thì bắt đầu run mới, checkpoint đã xong/không có `train_args` vẫn đưa cho framework, dataset khác byte bị bỏ qua; (c) lock/run chỉ dùng mtime 24 h: crash cùng máy phải chờ hết cửa sổ, còn run đang sống có thể bị prune giữa epoch; (d) metric nhận `bool`/`NaN`/`Infinity` và có thể crash `OverflowError`; (e) artifact của run `--resume` vẫn được `--skip-unchanged` bỏ qua | `cvtrain/backend.py`, `trainer.py`, `gate.py`, `fsio.py`, `manifest.py`; `Models/manifest.json` | **ADR 0013**: `TrainPlan` chốt file bắt đầu *trước* snapshot; resume fail-closed và so sánh dataset theo byte; `RunFacts` ghi budget/completed/`resume_from`, digest là của file thực sự bắt đầu; `resume_from` chặn skip; lock/run lease ghi host + PID, heartbeat 30 s, chủ chết → stale ngay, chủ sống → không steal, không biết → tuổi lease; metric predicate ở `metrics.py` + `allow_nan=False`; nightly dùng `ResumeRuns` riêng và assert effective facts. Ghim bằng 499 test, coverage 100 % (1 902/486), **101 mutant** (21 mutant mới cho N13) | ✅ Đóng (2026-10-23) |

| N14 | **Năm claim an toàn còn đúng trong tài liệu nhưng sai trong code.** (a) Run `--resume` dùng lại thư mục run cũ nhưng `touch_run_marker` chỉ làm tươi mtime, không đổi owner — marker vẫn ghi PID đã chết, nên `--prune-runs` đọc run đang sống là "bỏ hoang" suốt epoch đầu (có thể xoá nó); (b) `exporter_identity` ghi `torch`/`onnx` nhưng bỏ sót `onnxslim` — simplifier mặc định của Ultralytics — nên đổi simplifier vẫn giữ nguyên `backend_version` và `--skip-unchanged` bỏ qua artifact không tái lập; (c) `check_importable` chỉ `import ultralytics`, còn `onnx`/`onnxruntime`/`onnxslim` chỉ được `find_spec` kiểm, nên một cài đặt có dist nhưng thiếu DLL qua preflight rồi chết ở export; (d) `_run_activity_mtime` trả `0.0` khi `stat` lỗi, tức "tuổi không rõ" bị xếp là già nhất và có thể bị xoá — ngược hướng an toàn của thao tác phá huỷ duy nhất; (e) job real-dependency chỉ chạy `ubuntu-latest` dù sản phẩm ship Windows, và comment CI còn ghi 443 test / 80 mutant / 11 mutant fsio | `cvtrain/fsio.py`, `cvtrain/backend.py`, `.github/workflows/*.yml` | **ADR 0014**: `claim_run_marker` ghi đè lease bằng PID hiện tại **trước** epoch đầu của run resume (trong khi `touch` vẫn không đổi chủ); identity ghi thêm `onnxslim`; `check_importable` import toàn bộ `required_modules` và biến cả `ImportError` lẫn lỗi nạp DLL thành verdict preflight; activity không đọc được ⇒ coi như vừa hoạt động (giữ); workflow có thêm job `windows-train-smoke` train + export + verify thật trên `windows-latest`; comment CI trỏ `Docs/SCORECARD.md` thay vì số cứng. Ghim bằng 508 test, coverage 100 % (1 919/490), **107 mutant** (+6 cho N14) và 1 phép đảo mới. **Hệ quả:** 3 artifact hiện có mang identity 4 trường; lần `--skip-unchanged` kế tiếp sẽ retrain chúng (fail-closed) — đúng giá của một trường provenance từng thiếu. | ✅ Đóng (2026-10-24) |

### 3.1 Phép đảo chiều của N10–N13 (bằng chứng "làm đỏ")

N10–N12 được ghim bằng **tripwire trên file của repo** (requirements, workflow), không
phải bằng mutant trong `cvtrain/` — cùng hình dạng với tripwire trần opset sẵn có
(`test_contract.py`). N13 có cả hai loại: các quyết định trong `cvtrain/` được ghim
bằng mutant, còn hai assertion mới của nightly job được ghim bằng tripwire + phép
đảo. Cơ chế làm đỏ là **đảo chiều quyết định trong file** và bắt test sở hữu nó phải
đỏ. Mười phép đảo đã chạy (bảy phép 2026-10-22, hai phép 2026-10-23, một phép
2026-10-24), mỗi phép khôi phục file byte-for-byte sau đó:

| # | Quyết định bị đảo | Test phải đỏ | Kết quả |
|---|-------------------|---------------|---------|
| 1 | `requirements.txt`: `onnxruntime==1.20.1` → `onnxruntime>=1.19` | `test_the_contract_engine_is_the_one_the_application_ships` | ✅ đỏ |
| 2 | Workflow: cài lock **không** `--require-hashes` | `test_requirements_files_are_installed_with_hash_verification` | ✅ đỏ |
| 3 | Workflow: cài `ruff mypy coverage` không pin | `test_no_floating_package_is_installed` | ✅ đỏ |
| 4 | Workflow: `matrix.os` bỏ `windows-latest` | `test_the_gates_run_on_every_required_runner` | ✅ đỏ |
| 5 | Workflow: job Windows quét full sweep thay vì tập con `fsio.py` | `test_the_platform_sensitive_decisions_are_mutation_checked_on_windows` | ✅ đỏ |
| 6 | Nightly: assertion `Resuming yolo11n.onnx` bị thay | `test_the_resume_seam_is_still_exercised_against_the_real_framework` | ✅ đỏ |
| 7 | Nightly: bỏ trigger `pull_request` | `test_the_check_runs_for_pull_requests_touching_the_package` | ✅ đỏ |
| 8 | Nightly: resume dùng `AIComputerVision/runs` thay vì `ResumeRuns` riêng | `test_the_resume_check_uses_its_own_runs_directory` | ✅ đỏ |
| 9 | Nightly: bỏ assertion `epochs_completed == 5` | `test_the_resume_check_asserts_the_effective_facts` | ✅ đỏ |
| 10 | Nightly: xoá job `windows-train-smoke` | `test_the_real_dependency_check_runs_on_the_product_platform` | ✅ đỏ |

> Bốn phép đảo đầu tiên **đã làm lộ một lỗi trong chính guard**: ban đầu chúng
> chỉ tìm chuỗi trong *cả file* (một dòng comment nhắc `--require-hashes` là đủ
> để làm guard xanh), và phép đảo #4 vẫn xanh vì job `mutation` cũng nhắc
> `windows-latest`. Guard được sửa để đọc *job* và *lệnh cài đặt* cụ thể, rồi
> bảy phép đảo được chạy lại từ đầu. Đây chính là lý do một guard phải được
> *thử đảo* chứ không được cho là đúng.

## 4. Claim nào đã được ghim

Mỗi claim trong README/ADR dưới đây có một cơ chế làm đỏ nó. Claim không có
cơ chế thì **không được viết** (ADR 0007, sửa đổi 2026-10-21).

| Claim | Ghim bằng |
|-------|-----------|
| Package import-light (không cần ML stack) | `test_architecture.LazyDependencyTests` + `test_backend.test_construction_is_lazy` |
| Đồ thị phụ thuộc một chiều, không cycle | `test_architecture.ImportGraphTests` (bảng `ALLOWED_IMPORTS`) |
| `cli.py` là composition root duy nhất | `test_architecture.test_the_concrete_backend_is_named_only_by_the_composition_root` |
| Hợp đồng ONNX với runtime Qt (rank/dtype/output/opset) | `test_onnx_contract` + **9 mutant** trong `cvtrain/onnx_contract.py` |
| Role lạ bị **từ chối**, không đoán bừa | `test_onnx_contract` + **1 mutant** (`onnx_contract.py`) |
| Hợp đồng tên file / trần opset / schema / IoU với C++ | `test_contract`, `test_qt_contract` (+ presence guard chống rename âm thầm) |
| Provenance fail-closed (thiếu dữ liệu ⇒ retrain) | `test_trainer.SkipUnchangedTests` + **5 mutant** trong `cvtrain/gate.py` |
| **Fail-closed ở cả hai phía** của phép so sánh device | `test_trainer.DeviceResolutionTests` + **2 mutant** (`gate.py`) |
| Yêu cầu device được **chuẩn hóa** (`"0, 1"` ≡ `"0,1"`, `""` ≡ `"auto"`) | `test_config` + **1 mutant** (`config.py`) |
| Device **đã phân giải** là cái được chuyển cho framework | `test_backend.DeviceResolutionTests` + **1 mutant** (`backend.py`) |
| Quality gate không bị bypass (kể cả khi skip) | `test_trainer` + **5 mutant** (3 `gate.py` + 2 `trainer.py`) |
| Identity của exporter ghi đủ **5** dependency (`torch`, `onnx`, `onnxslim`, `ultralytics`, `cvtrain`) | `test_backend.test_the_identity_carries_the_actual_version_of_each_dependency` + `test_a_moved_simplifier_changes_the_exporter_identity` + **3 mutant** (`backend.py`) |
| Chọn model: key lạ ⇒ lỗi, không lặng lẽ train mặc định | `test_config` + **3 mutant** (`config.py`) |
| Fail-fast preflight | `test_preflight` + **6 mutant** (`preflight.py`, gồm grammar `--device`) |
| Grammar `--min-metric` do CLI, ngữ nghĩa do preflight | `test_cli.MetricThresholdSyntaxTests` + **1 mutant** (`cli.py`) |
| Fingerprint dataset phủ dir/glob/`.txt`/list | `test_dataset` + **6 mutant** (`dataset.py`, gồm marker file đọc lỗi) |
| Dataset so sánh theo **digest**, không theo đường dẫn | `test_trainer.test_the_same_dataset_behind_another_path_is_not_retrained` |
| Run bị gián đoạn thì tiếp tục được, và `--resume` **fail-closed** (thiếu/đã xong/không train_args/dataset mất/dataset khác byte) | `test_backend.TrainTests` + **7 mutant** (`backend.py`) |
| Manifest ghi run đã xảy ra: budget, `epochs_completed`, `resume_from` | `test_trainer.EffectiveEpochProvenanceTests` + `test_backend` + **3 mutant** (`backend.py` × 2, `trainer.py` × 1) |
| Artifact của run `--resume` **không bao giờ** được `--skip-unchanged` bỏ qua | `test_trainer.test_a_resumed_artifact_is_never_skipped` + **1 mutant** (`gate.py`) |
| Digest nguồn là file run **thực sự bắt đầu**, không phải checkpoint gốc | `test_trainer.test_the_recorded_digest_is_the_file_training_started_from` + **2 mutant** (`backend.py`, `gate.py`) |
| Provenance tách **budget** (`epochs`) khỏi **số đã chạy** (`epochs_completed`) | `test_trainer.EffectiveEpochProvenanceTests` + `test_backend` + **3 mutant** (`backend.py` × 2, `trainer.py` × 1) |
| Fingerprint báo tiến độ (dải `dataset` 2–8 %) | `test_dataset` + `test_progress` + **2 mutant** (`dataset.py`, `progress.py`) |
| Preflight **chứng minh** toàn bộ stack khai báo import được (kể cả lỗi nạp DLL) | `test_trainer` + `test_backend.ImportabilityTests` + **4 mutant** (`trainer.py`, `backend.py` × 3) |
| Publish nguyên tử + rollback `.bak` + kiểm hợp đồng *trước* khi publish | `test_fsio`, `test_trainer` + **4 mutant** (3 `fsio.py` + 1 `trainer.py`) |
| Advisory lock: O_EXCL, token, claim, stale, heartbeat | `test_fsio` + **5 mutant** (`fsio.py`, gồm heartbeat best-effort) |
| Lock là **lease có chủ**: host + PID, chủ sống không bị steal, chủ chết stale ngay, không biết → tuổi | `test_fsio.LockOwnershipTests` + **5 mutant** (`fsio.py`) |
| Run lease `.cvtrain-alive`: làm tươi mỗi epoch, **claim lại khi resume**, xoá khi xong, marker máy khác tính theo mtime | `test_fsio.RunMarkerTests` + `test_fsio.PruneRunDirectoriesTests` + `test_backend.TrainTests` + **6 mutant** (`fsio.py` × 5, `backend.py` × 1) |
| **Retention**: giữ run mới nhất, không xoá run còn sống, không đụng thư mục lạ | `test_fsio.PruneRunDirectoriesTests` + `test_cli.PruneRunsCommandTests` + **10 mutant** (`fsio.py` lease/owner/marker/prune × 9 + `cli.py` × 1) |
| Manifest: quarantine, verdict nghiêm, 2 chiều, migration | `test_manifest` + **8 mutant** (`manifest.py`) |
| Provenance ghi *một lần mỗi model* (không mất khi crash giữa chừng) | `test_trainer.FailurePathTests` + **1 mutant** (`trainer.py`) |
| Tiến trình đơn điệu + marker `done`/`error` | `test_progress` + **2 mutant** (`progress.py`, `trainer.py`) |
| Mã thoát theo nguyên nhân | `test_cli.ExitCodeTests` + **4 mutant** (`errors.py`, `cli.py`) |
| Metric không phải số bị loại khỏi manifest, và JSON không bao giờ chứa `NaN`/`Infinity` | `test_backend` + `test_manifest` + **4 mutant** (`metrics.py` × 3 + `manifest.py` × 1) |
| Lệnh manifest không bao giờ huấn luyện | `test_cli.test_the_manifest_commands_never_start_a_training_run` |
| README ↔ CLI hai chiều, Layout ↔ package | `test_architecture.DocumentationTests` |
| Chuỗi công cụ của gate **đã pin** (không gói trôi) | `test_toolchain_contract.GateToolchainTests` + `GateWorkflowTests` (+ 3 phép đảo ở §3.1) |
| Engine hợp đồng export = engine ứng dụng ship | `test_toolchain_contract.test_the_contract_engine_is_the_one_the_application_ships` (+ 1 phép đảo) |
| Cổng chạy trên **nền tảng sản phẩm ship** | `test_ci_contract.PlatformCoverageTests` (+ 2 phép đảo) |
| Quyết định phụ thuộc nền tảng được mutation-check trên Windows | `test_ci_contract.test_the_platform_sensitive_decisions_are_mutation_checked_on_windows` (+ 1 phép đảo) |
| Seam `resume` của pipeline (chọn checkpoint, fail-closed, effective facts) | `test_backend.TrainTests` + **14 mutant** (`backend.py` × 10, `gate.py` × 2, `trainer.py` × 2) |
| Hành vi `resume` của **framework thật** vẫn được kiểm chứng, với `ResumeRuns` riêng và effective facts, và cho pull request | `test_ci_contract.RealDependencyCheckTests` (+ 4 phép đảo) |
| **Stack thật chạy trên nền tảng sản phẩm ship**: cài wheel native, train, export, verify manifest trên Windows | `cvtrain-nightly.yml` job `windows-train-smoke` + `test_ci_contract.test_the_real_dependency_check_runs_on_the_product_platform` (+ 1 phép đảo) |

Tổng: **107 mutant** = `fsio.py` 21 · `backend.py` 20 · `gate.py` 11 ·
`trainer.py` 10 · `manifest.py` 9 · `onnx_contract.py` 9 · `dataset.py` 6 ·
`preflight.py` 6 · `cli.py` 4 · `config.py` 4 · `metrics.py` 3 ·
`errors.py` 2 · `progress.py` 2.

> **Vì sao N10–N12 không có mutant.** Mutation set phủ các quyết định *bên
> trong* `cvtrain/`. Ba nợ đó là quyết định *về file của repo* (requirements,
> workflow), nên cơ chế ghim là tripwire đọc file + **phép đảo chiều** (§3.1),
> đúng hình dạng của tripwire trần opset sẵn có. Một guard mà *thêm* một quyết
> định vào mã nguồn package vẫn phải có mutant trong cùng commit (ADR 0007).

## 5. Cách đo lại

```powershell
cd AIComputerVision
pip install --require-hashes -r requirements-test.lock
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
`Models/`, `runs/` hay dataset của máy. Sandbox copy cả `src/`, các file
requirement và `.github/workflows/` — đúng những gì suite đọc — vì một file
mà suite đọc nhưng sandbox không copy sẽ biến guard thành **skip**, và một skip
không phải là một verdict.

> **Cổng được cài từ lock hash-pinned** (ADR 0012): `requirements-test.lock` là
> bản hash-pinned của `requirements-test.txt`, sinh bằng
> `uv pip compile AIComputerVision/requirements-test.txt --python-version 3.11
> --generate-hashes -o AIComputerVision/requirements-test.lock`. Không dùng chế
> độ universal: `numpy==1.26.4` (bản đã đo) không có nhánh cho Python ≥ 3.13, nên
> universal resolution không có nghiệm. Lock vẫn mang **mọi** digest của từng
> phiên bản trên PyPI (đã đối chiếu từng gói với `pypi.org/pypi/<name>/<version>/json`),
> nên cùng một file cài được trên cả `ubuntu-latest` lẫn `windows-latest`.
> Đổi một phiên bản trong `requirements-test.txt` thì phải sinh lại lock và
> commit cả hai — nếu không, `test_toolchain_contract` sẽ đỏ.

Ngoài ra, `cvtrain-nightly.yml` chạy **real-dependency smoke test** mỗi đêm
*và* cho mọi pull request đụng vào package: train + export thật (det + seg,
CPU), một lần **gián đoạn có chủ đích** để lại checkpoint, rồi `--resume` nối
tiếp run đó (xem N3) — đây là nơi duy nhất hành vi `resume` của framework thật
được kiểm chứng.
