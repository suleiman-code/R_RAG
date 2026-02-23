from backend.app.database import get_qdrant_client, COLLECTION_NAME

client = get_qdrant_client()
print(f"Deleting collection: {COLLECTION_NAME}...")
try:
    client.delete_collection(COLLECTION_NAME)
    print("Deleted successfully. The backend will recreate it with 3072 dimensions on next start.")
except Exception as e:
    print(f"Error: {e}")
