import os
import asyncio
from dotenv import load_dotenv

load_dotenv()

async def test_startup():
    print("Starting test...")
    from backend.app.database import setup_hybrid_collection
    print("Imported database.")
    
    # Test DB setup
    setup_hybrid_collection()
    print("Done DB setup.")

    from backend.app.rag_core import RAGCore
    print("Imported RAGCore.")
    
    rag = RAGCore()
    print("Initialized RAGCore.")

if __name__ == "__main__":
    asyncio.run(test_startup())
