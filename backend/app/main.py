import os
import shutil
import sys
from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from .models import QueryRequest, QueryResponse

# Configure Loguru for Production
os.makedirs("logs", exist_ok=True)
logger.remove()
logger.add(sys.stdout, format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>", colorize=True)
logger.add("logs/backend.log", rotation="10 MB", retention="10 days", compression="zip")

# App banate hain
app = FastAPI(title="Full-Stack RAG API")

# CORS setup — Secured via Env
origins = os.getenv("ALLOWED_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# RAG instance — startup mein banayenge taake imports fail na ho
rag = None

@app.on_event("startup")
async def startup_event():
    """
    Server start hote hi RAGCore aur Qdrant collection setup hogi.
    Module level par nahi, yahan initialize karte hain.
    """
    global rag
    from .rag_core import RAGCore
    from .database import setup_hybrid_collection
    # Qdrant mein collection banao agar nahi hai
    setup_hybrid_collection()
    # RAG engine taiyar karo
    rag = RAGCore()
    logger.info("RAG engine ready!")

@app.get("/")
async def root():
    return {"message": "RAG Backend is fully functional!"}

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """
    PDF upload karke Qdrant mein index karta hai.
    """
    if rag is None:
        raise HTTPException(status_code=503, detail="RAG engine is not ready yet. Please wait a moment.")

    # Case-insensitive PDF check
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")

    # Safer temporary file path with UUID to avoid name collisions
    import uuid
    safe_name = f"temp_{uuid.uuid4().hex}_{file.filename}"
    temp_path = os.path.join(os.getcwd(), safe_name)
    
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        chunks_indexed = await rag.ingest_pdf(temp_path)
        logger.info(f"Successfully indexed {chunks_indexed} chunks from {file.filename}")
        return {"message": f"Successfully indexed {chunks_indexed} chunks.", "filename": file.filename}
    except Exception as e:
        logger.error(f"Index Error for {file.filename}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Sawal poochne ka endpoint — hybrid search karke Gemini se jawab milta hai.
    """
    if rag is None:
        raise HTTPException(status_code=503, detail="RAG engine is not ready yet.")
    try:
        response = await rag.generate_response(request.question)
        return response
    except Exception as e:
        import traceback
        err_msg = f"Query Error: {str(e)}\n{traceback.format_exc()}"
        print(err_msg)
        logger.error(err_msg)
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/clear")
async def clear_index():
    """
    Qdrant collection ko saaf karne ke liye.
    """
    if rag is None:
        raise HTTPException(status_code=503, detail="RAG engine is not ready yet.")
    try:
        await rag.clear_collection()
        return {"message": "Knowledge base cleared successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app.main:app", host="127.0.0.1", port=8080, reload=True)
