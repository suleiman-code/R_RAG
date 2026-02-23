import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient

load_dotenv()

URL = os.getenv("QDRANT_URL")
API_KEY = os.getenv("QDRANT_API_KEY")

print(f"Testing Qdrant at: {URL}")
try:
    client = QdrantClient(url=URL, api_key=API_KEY, timeout=10)
    collections = client.get_collections()
    print("Successfully connected to Qdrant Cloud!")
    print(f"Collections: {collections}")
except Exception as e:
    print(f"Failed to connect: {e}")
