import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request

# ── Thông số server ────────────────────────────────────────────────────────────
HOST     = "127.0.0.1"
PORT     = 8080
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# APP_DATA_DIR được Qt truyền qua env — config.py dùng cùng biến này
# Nếu không có env var thì fallback về AIAssistant/ (chạy tay)
_app_data = os.environ.get("APP_DATA_DIR", "")
if _app_data:
    LOGS_DIR = os.path.join(_app_data, "AIAssistant", "logs")
else:
    LOGS_DIR = os.path.join(BASE_DIR, "logs")

# Timeout tối đa chờ server khởi động (giây) — load GGUF model có thể mất 2-3 phút
MAX_STARTUP_WAIT_SEC = 300  # 5 phút


def is_server_running() -> bool:
    """Kiểm tra xem AI server đã sẵn sàng qua /health endpoint."""
    try:
        req = urllib.request.Request(
            f"http://{HOST}:{PORT}/health", method="GET"
        )
        with urllib.request.urlopen(req, timeout=2) as resp:
            return resp.status == 200
    except Exception:
        return False


def get_latest_log_file():
    """Lấy file log mới nhất trong thư mục logs (hoặc None)."""
    if not os.path.isdir(LOGS_DIR):
        return None
    logs = [
        os.path.join(LOGS_DIR, f)
        for f in os.listdir(LOGS_DIR)
        if f.endswith(".log")
    ]
    return max(logs, key=os.path.getmtime) if logs else None


def tail_log_forever(log_file: str, from_start: bool = False):
    """Tail file log, in từng dòng mới ra stdout để Qt bắt được."""
    print(f"[ServerManager] Tailing log: {log_file}", flush=True)
    with open(log_file, "r", encoding="utf-8", errors="replace") as f:
        if not from_start:
            # Đọc từ cuối file để không replay log cũ
            f.seek(0, os.SEEK_END)
        while True:
            line = f.readline()
            if not line:
                time.sleep(0.15)
                continue
            print(line, end="", flush=True)


def wait_until_ready(server_process: subprocess.Popen | None = None) -> bool:
    """Poll HTTP /health cho đến khi server respond hoặc timeout."""
    deadline = time.monotonic() + MAX_STARTUP_WAIT_SEC
    elapsed  = 0
    while time.monotonic() < deadline:
        if is_server_running():
            return True
        if server_process is not None:
            exit_code = server_process.poll()
            if exit_code is not None:
                print(
                    f"[ServerManager] Server process exited early (code {exit_code}). "
                    "Check the server log for the traceback.",
                    flush=True,
                )
                return False
        time.sleep(2)
        elapsed += 2
        if elapsed % 10 == 0:
            print(
                f"[ServerManager] Đang chờ server khởi động... "
                f"({elapsed}/{MAX_STARTUP_WAIT_SEC}s)",
                flush=True,
            )
    return False


def shutdown_running_server() -> bool:
    """Yêu cầu AI Server đang chạy tự tắt qua /admin/shutdown.
    Trả về True nếu shutdown thành công (hoặc server đã không chạy).
    """
    try:
        req = urllib.request.Request(
            f"http://{HOST}:{PORT}/admin/shutdown",
            data=b"{}",
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            print(f"[ServerManager] Shutdown response: {body}", flush=True)
        return True
    except Exception as e:
        print(f"[ServerManager] Không thể gửi shutdown request: {e}", flush=True)
        return False


def get_active_model_index() -> int | None:
    """Lấy model index đang chạy từ /health endpoint. Trả về None nếu không lấy được."""
    try:
        req = urllib.request.Request(
            f"http://{HOST}:{PORT}/health", method="GET"
        )
        with urllib.request.urlopen(req, timeout=2) as resp:
            data = json.loads(resp.read().decode("utf-8", errors="replace"))
            return data.get("model_idx")
    except Exception:
        return None


def main():
    model_idx_str = sys.argv[1] if len(sys.argv) > 1 else "0"
    try:
        model_idx = int(model_idx_str)
    except ValueError:
        model_idx = 0

    print(
        f"[ServerManager] === Switch/Start model request: index={model_idx} ===",
        flush=True,
    )
    print(f"[ServerManager] LOGS_DIR = {LOGS_DIR}", flush=True)

    # ── Bước 1: Kiểm tra server đã chạy chưa ─────────────────────────────────
    already_running = is_server_running()
    if already_running:
        print(
            f"[ServerManager] AI Server đang chạy trên cổng {PORT}. "
            "Kiểm tra xem có cần đổi model không...",
            flush=True,
        )
        # Lấy model index đang chạy từ /health
        active_idx = get_active_model_index()
        print(
            f"[ServerManager] Model đang chạy: index={active_idx}, "
            f"model mới yêu cầu: index={model_idx}",
            flush=True,
        )

        if active_idx is not None and active_idx == model_idx:
            # Cùng model → không cần restart, dùng server hiện tại
            print(
                f"[ServerManager] Model index {model_idx} đã được load. "
                "Không cần khởi động lại.",
                flush=True,
            )
            print("[SUCCESS] AI Server started successfully", flush=True)
            # Không return — tiếp tục để tail log
        else:
            # Model khác hoặc không lấy được active_idx → phải shutdown và khởi động lại
            print(
                f"[ServerManager] Cần switch từ model={active_idx} sang model={model_idx}. "
                "Đang shutdown server hiện tại...",
                flush=True,
            )
            shutdown_running_server()

            # Chờ server thực sự dừng (tối đa 15 giây)
            deadline = time.monotonic() + 15
            while time.monotonic() < deadline:
                if not is_server_running():
                    break
                time.sleep(0.5)

            if is_server_running():
                print(
                    "[ServerManager] CẢNH BÁO: Server vẫn còn chạy sau khi shutdown. "
                    "Tiếp tục khởi động server mới...",
                    flush=True,
                )
            else:
                print("[ServerManager] Server cũ đã dừng.", flush=True)

            # Khởi động server mới với model mới
            already_running = False

    if not already_running:
        # ── Bước 2: Khởi động server ở chế độ detached ───────────────────────
        print(
            f"[ServerManager] Đang khởi động AI Server với model index={model_idx}...",
            flush=True,
        )
        server_script = os.path.join(BASE_DIR, "StartChatbotServer.py")

        # Truyền lại APP_DATA_DIR cho child process để config.py biết đúng đường dẫn
        env = os.environ.copy()

        # Detached + no console window — server sẽ sống sót khi Qt tắt
        flags = 0x00000008 | 0x00000200  # DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
        server_process = subprocess.Popen(
            [sys.executable, "-u", server_script, str(model_idx)],
            cwd=BASE_DIR,
            env=env,
            creationflags=flags,
            close_fds=True,
        )

        # ── Bước 3: Poll HTTP cho đến khi server sẵn sàng ────────────────────
        print("[ServerManager] Đang đợi server phản hồi...", flush=True)
        ready = wait_until_ready(server_process)

        if ready:
            print("[SUCCESS] AI Server started successfully", flush=True)
            sys.stdout.flush()
        else:
            print(
                f"[ServerManager] Lỗi: Server không phản hồi sau "
                f"{MAX_STARTUP_WAIT_SEC}s. Kiểm tra log để biết thêm.",
                flush=True,
            )
            sys.stdout.flush()
            return

    # ── Bước 4: Tail log liên tục ─────────────────────────────────────────────
    # Đợi tối đa 15s cho file log xuất hiện
    log_file = None
    for _ in range(15):
        log_file = get_latest_log_file()
        if log_file:
            break
        time.sleep(1)

    if log_file:
        try:
            # Nếu server đã chạy sẵn: bắt đầu từ cuối (tránh replay log cũ)
            # Nếu vừa khởi động: bắt đầu từ cuối (log đã được ghi trong quá trình wait)
            tail_log_forever(log_file, from_start=False)
        except KeyboardInterrupt:
            print("\n[ServerManager] Ngừng theo dõi log.", flush=True)
    else:
        print(
            "[ServerManager] Không tìm thấy file log — "
            f"LOGS_DIR={LOGS_DIR}. Server có thể đã lỗi khi khởi động.",
            flush=True,
        )
        # Giữ process sống để Qt không mất kết nối
        try:
            while True:
                time.sleep(5)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    # Đảm bảo stdout dùng UTF-8 cho Qt
    os.environ["PYTHONUTF8"] = "1"
    if sys.platform == "win32":
        try:
            import ctypes
            ctypes.windll.kernel32.SetConsoleOutputCP(65001)
        except Exception:
            pass

    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    main()
