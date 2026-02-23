import subprocess
import os
import sys
import time

def start_project():
    # Root directory
    root_dir = os.path.dirname(os.path.abspath(__file__))
    
    print("🚀 Starting R_RAG Project...")
    
    # 1. Start Backend
    print("📦 Starting Backend (FastAPI on port 8080)...")
    backend_process = subprocess.Popen(
        ["python", "-m", "uvicorn", "backend.app.main:app", "--host", "127.0.0.1", "--port", "8080", "--reload"],
        cwd=root_dir,
        shell=True
    )
    
    # Give backend a moment to start
    time.sleep(3)
    
    # 2. Start Frontend
    print("🌐 Starting Frontend (Vite)...")
    frontend_dir = os.path.join(root_dir, "frontend")
    
    # Check if node_modules exists
    if not os.path.exists(os.path.join(frontend_dir, "node_modules")):
        print("⚠️ node_modules not found. Installing dependencies...")
        subprocess.run(["npm", "install"], cwd=frontend_dir, shell=True)
    
    frontend_process = subprocess.Popen(
        ["npm", "run", "dev"],
        cwd=frontend_dir,
        shell=True
    )
    
    print("\n✅ Both services are starting!")
    print("🔗 Backend: http://localhost:8080")
    print("🔗 Frontend: Look at terminal output for Vite URL (usually http://localhost:5173)")
    print("\nPress Ctrl+C to stop both services.")
    
    try:
        # Keep the script running
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Stopping services...")
        backend_process.terminate()
        frontend_process.terminate()
        print("Done.")

if __name__ == "__main__":
    start_project()
