import os
import importlib.util
import shutil
import subprocess
import sys
import webbrowser
from threading import Timer

# 1. Đảm bảo các gói cần thiết
def install_requirements():
    packages = ["ultralytics", "tensorboard"]
    for pkg in packages:
        if importlib.util.find_spec(pkg) is None:
            print(f"Installing {pkg}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

install_requirements()

from ultralytics import YOLO  # noqa: E402
from ultralytics.utils import SETTINGS  # noqa: E402

# Ép buộc bật TensorBoard trong cấu hình của YOLO
try:
    SETTINGS.update({"tensorboard": True})
    print("✅ Đã ép buộc bật TensorBoard trong YOLO Settings.")
except Exception as e:
    print(f"⚠️ Cảnh báo cấu hình: {e}")

# Lấy đường dẫn tuyệt đối
base_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(base_dir, ".."))
yaml_path = os.path.join(project_root, 'Dataset', 'data.yaml')
models_dir = os.path.join(base_dir, "Models")
os.makedirs(models_dir, exist_ok=True)

# Cấu hình đường dẫn LOG
RUNS_DIR = os.path.join(base_dir, "runs")

# ==========================================
# CẤU HÌNH TRAINING CHUNG (GLOBAL CONFIG)
# ==========================================
TRAIN_EPOCHS = 5
TRAIN_IMGSZ = 640
TRAIN_BATCH = 16
TIME_TO_WAIT = 12
# ==========================================

print("="*50)
print("   🚀 Ultralytics YOLO11 Training Script (Super Force Mode)")
print("="*50)

def launch_tensorboard():
    print("🧹 Dang quet don cac Server cu...")
    try:
        if os.name == 'nt': # Windows
            subprocess.run(["taskkill", "/F", "/IM", "tensorboard.exe"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass
    
    log_dir_clean = RUNS_DIR.replace("\\", "/")
    print("🌍 Dang khoi dong TensorBoard Server...")
    
    try:
        # Khoi chay TensorBoard
        subprocess.Popen([sys.executable, "-m", "tensorboard.main", "--logdir", log_dir_clean, "--port", "6006"], 
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, cwd=base_dir)
        
        Timer(TIME_TO_WAIT, lambda: webbrowser.open("http://localhost:6006")).start()
        print(f"🔗 Sau {TIME_TO_WAIT}s, mở trình duyệt tại http://localhost:6006.")
    except Exception as e:
        print(f"❌ không thể khởi động TensorBoard: {e}")

# Chạy TensorBoard
# launch_tensorboard()

common_kwargs = {
    "data": yaml_path,
    "epochs": TRAIN_EPOCHS,
    "imgsz": TRAIN_IMGSZ,
    "batch": TRAIN_BATCH,
    "workers": 0,
    "exist_ok": True, 
}

# Kiểm tra xem model đã tồn tại chưa
model_exists = os.path.exists(os.path.join(models_dir, "yolo11n.onnx")) or \
               os.path.exists(os.path.join(models_dir, "yolo11n-seg.onnx"))

if model_exists:
    if "--yes" in sys.argv:
        print("Đang chuẩn bị training lại (Auto Yes)...")
    else:
        while True:
            choice = input("Bạn có muốn Training lại model không? Y/N: ").strip().upper()
            if choice == 'Y':
                print("Đang chuẩn bị training lại...")
                break
            elif choice == 'N':
                print("Đã hủy quá trình training.")
                sys.exit(0)

print("\n[1/2] Training Object Detection Model (yolo11n)...")
model_det = YOLO('yolo11n.pt')
# Chạy training
model_det.train(project=os.path.join(RUNS_DIR, "detect"), name="train", **common_kwargs)

print("\nExporting Detection Model to ONNX...")
det_onnx = model_det.export(format='onnx', opset=12)
if os.path.exists(det_onnx):
    shutil.copy(det_onnx, os.path.join(models_dir, "yolo11n.onnx"))

print("\n[2/2] Training Segmentation Model (yolo11n-seg)...")
model_seg = YOLO('yolo11n-seg.pt')
model_seg.train(project=os.path.join(RUNS_DIR, "segment"), name="train", **common_kwargs)

print("\nExporting Segmentation Model to ONNX...")
seg_onnx = model_seg.export(format='onnx', opset=12)
if os.path.exists(seg_onnx):
    shutil.copy(seg_onnx, os.path.join(models_dir, "yolo11n-seg.onnx"))

print(f"\n[SUCCESS] Hoàn tất! Logs tại: {base_dir}")
print("="*50)
if "--yes" not in sys.argv:
    input("Nhấn Enter để thoát...")
