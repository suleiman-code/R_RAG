import os
from qdrant_client import QdrantClient
from qdrant_client.http import models
from dotenv import load_dotenv

# Loading environment variables
# Environment variables load kar rahe hain (API keys etc.)
load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = "rag_hybrid_collection"

_client = None

def get_qdrant_client():
    global _client
    if _client is not None:
        return _client

    # Case 1: Local storage explicitly requested
    if QDRANT_URL == ":memory:" or not QDRANT_URL.startswith("http"):
        print("[Database] Using local storage (path=qdrant_db)")
        try:
            _client = QdrantClient(path="qdrant_db")
        except Exception as e:
            print(f"[Database] Local storage lock error: {e}")
            # Try to connect to existing instance or just fail gracefully
            raise e
        return _client
    
    # Case 2: Cloud connection with fallback
    print(f"[Database] Connecting to Qdrant at: {QDRANT_URL} (timeout=3s)")
    try:
        tmp_client = QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY,
            timeout=3,
            check_compatibility=False
        )
        # Test connection
        tmp_client.get_collections()
        print("[Database] Cloud connection successful!")
        _client = tmp_client
        return _client
    except Exception as e:
        print(f"[Database] Cloud connection failed: {e}. Falling back to local storage.")
        try:
            _client = QdrantClient(path="qdrant_db")
            return _client
        except Exception as e2:
            print(f"[Database] Severe Error: Could not start local storage either: {e2}")
            raise e2

def setup_hybrid_collection():
    client = get_qdrant_client()
    
    # Check if collection exists, if not create it
    try:
        exists = client.collection_exists(COLLECTION_NAME)
    except Exception as e:
        print(f"[Database] Error checking collection: {e}")
        return

    if not exists:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=3072, # Gemini Embedding size (actually 3072 in this env)
                distance=models.Distance.COSINE
            ),
            # Enabling sparse vectors for keyword search
            # Hybrid search ke liye sparse vectors enable kar rahe hain (keyword search ke liye)
            sparse_vectors_config={
                "text-sparse": models.SparseVectorParams(
                    index=models.SparseIndexParams(
                        on_disk=True
                    )
                )
            }
        )
        print(f"Collection {COLLECTION_NAME} created successfully.")
    else:
        print(f"Collection {COLLECTION_NAME} already exists.")
