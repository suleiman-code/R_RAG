from backend.app.database import get_qdrant_client, COLLECTION_NAME, setup_hybrid_collection
from qdrant_client.http import models

client = get_qdrant_client()
print(f"Deleting collection {COLLECTION_NAME}...")
try:
    client.delete_collection(COLLECTION_NAME)
    print("Deleted.")
except Exception as e:
    print(f"Error deleting: {e}")

print("Setting up fresh collection...")
setup_hybrid_collection()
print("Done! Collection recreated with 3072 dimensions.")
