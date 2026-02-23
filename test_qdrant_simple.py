import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http import models

load_dotenv()
url = os.getenv("QDRANT_URL")
api_key = os.getenv("QDRANT_API_KEY")

client = QdrantClient(url=url, api_key=api_key)
print("Testing simple Qdrant query...")
try:
    collections = client.get_collections()
    print("Collections:", collections)
    
    # Test a simple search if collection exists
    res = client.query_points(
        collection_name="rag_hybrid_collection",
        query=[0.1] * 3072,
        limit=1
    )
    print("Search result:", res)
except Exception as e:
    print("Qdrant Error:", str(e))
