import subprocess
import os
import sys
import time


def kill_port(port: int, wait: bool = True):
    """
    Kill ALL processes using a given port (any TCP state).
    Waits to confirm the port is actually free before returning.
    """
    killed = False
    try:
        result = subprocess.run(
            ["netstat", "-ano"],
            capture_output=True, text=True
        )
        pids_killed = set()
        for line in result.stdout.splitlines():
            # Match any line with our port — covers LISTENING, ESTABLISHED, TIME_WAIT
            if f":{port} " in line or f":{port}\t" in line:
                parts = line.strip().split()
                if parts:
                    pid = parts[-1]
                    if pid.isdigit() and int(pid) != os.getpid() and pid not in pids_killed:
                        print(f"  ↳ Killing PID {pid} on port {port}...", flush=True)
                        subprocess.run(
                            ["taskkill", "/F", "/PID", pid],
                            capture_output=True
                        )
                        pids_killed.add(pid)
                        killed = True

        if killed:
            print(f"✅ Port {port} cleared.", flush=True)
            if wait:
                # Wait until port is actually free (max 8s)
                for _ in range(8):
                    time.sleep(1)
                    check = subprocess.run(
                        ["netstat", "-ano"],
                        capture_output=True, text=True
                    )
                    still_in_use = any(
                        f":{port} " in ln or f":{port}\t" in ln
                        for ln in check.stdout.splitlines()
                    )
                    if not still_in_use:
                        break
    except Exception as e:
        print(f"[Warning] Could not kill port {port}: {e}", flush=True)


def run():
    print("\n🚀 Starting RAG Assistant (Full-Stack)...\n", flush=True)

    # ── Step 1: Clean up any leftover processes ──────────────────────────────
    print("🔧 Cleaning up old processes...", flush=True)
    kill_port(8080)
    kill_port(3000)
    # Extra safety — kill any stale uvicorn / vite child processes
    subprocess.run(["taskkill", "/F", "/IM", "uvicorn.exe"], capture_output=True)
    print("", flush=True)

    # ── Step 2: Start Backend ────────────────────────────────────────────────
    backend_process = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "backend.app.main:app",
         "--host", "127.0.0.1", "--port", "8080"],
        stdout=None,
        stderr=None
    )
    print("✅ Backend process started (PID: {})".format(backend_process.pid), flush=True)

    # ── Step 3: Wait for warmup ──────────────────────────────────────────────
    print("⏳ Waiting for engine warmup (14s)...", flush=True)
    time.sleep(14)

    if backend_process.poll() is not None:
        print("❌ Backend failed to start! Check the console output above.", flush=True)
        return

    print("✅ Backend is healthy on http://127.0.0.1:8080\n", flush=True)

    # ── Step 4: Start Frontend ───────────────────────────────────────────────
    frontend_process = subprocess.Popen(
        ["npm", "run", "dev"],
        cwd=os.path.join(os.getcwd(), "frontend"),
        shell=True,
        stdout=None,
        stderr=None
    )
    print("✅ Frontend starting → http://localhost:3000", flush=True)
    print("📌 Press Ctrl+C to gracefully stop both servers.\n", flush=True)

    # ── Step 5: Keep alive ───────────────────────────────────────────────────
    try:
        while True:
            time.sleep(2)
            if backend_process.poll() is not None:
                print("\n❌ Backend crashed. Stopping everything.", flush=True)
                frontend_process.terminate()
                break
    except KeyboardInterrupt:
        print("\n🛑 Stopping servers...", flush=True)
    finally:
        backend_process.terminate()
        try:
            frontend_process.terminate()
        except Exception:
            pass
        time.sleep(1)
        kill_port(8080, wait=False)   # Release port for next run
        print("✅ All servers stopped cleanly.", flush=True)


if __name__ == "__main__":
    run()
