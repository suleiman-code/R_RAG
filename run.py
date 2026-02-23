"""
Backend ka entry point — isko seed.py ki tarah chalate hain:
  conda run -n R_RAG uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
"""
import uvicorn

if __name__ == "__main__":
    uvicorn.run("backend.app.main:app", host="0.0.0.0", port=8000, reload=True)
