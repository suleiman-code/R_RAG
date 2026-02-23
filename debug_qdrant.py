import os
from qdrant_client import QdrantClient
from dotenv import load_dotenv

load_dotenv()

URL = os.getenv("QDRANT_URL")
API_KEY = os.getenv("QDRANT_API_KEY")

print(f"Connecting to {URL}...")
client = QdrantClient(url=URL, api_key=API_KEY)

try:
    collections = client.get_collections()
    print("Successfully connected!")
    print(f"Collections: {collections}")
except Exception as e:
    print(f"Failed to connect: {e}")
