import subprocess
import os
import sys
import time

def run():
    print("🚀 Starting RAG Assistant (Full-Stack)...")
    
    # 1. Start Backend (Uvicorn)
    # We use uvicorn as a module and point to the specific path
    backend_process = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "backend.app.main:app", "--host", "127.0.0.1", "--port", "8080"],
        stdout=None,
        stderr=None
    )
    print("✅ Backend starting on http://127.0.0.1:8080")

    # 2. Wait a bit for backend to initialize
    time.sleep(3)

    # 3. Start Frontend (Vite)
    # Using 'npm run dev' which is configured to stay alive
    frontend_process = subprocess.Popen(
        ["npm", "run", "dev"],
        cwd=os.path.join(os.getcwd(), "frontend"),
        shell=True
    )
    print("✅ Frontend starting...")

    try:
        while True:
            time.sleep(1)
            if backend_process.poll() is not None:
                print("❌ Backend process died. Exiting...")
                break
            if frontend_process.poll() is not None:
                print("❌ Frontend process died. Exiting...")
                break
    except KeyboardInterrupt:
        print("\n🛑 Stopping servers...")
    finally:
        backend_process.terminate()
        frontend_process.terminate()
        print("Done.")

if __name__ == "__main__":
    run()
