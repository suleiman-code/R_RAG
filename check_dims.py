from backend.app.database import get_qdrant_client, COLLECTION_NAME

client = get_qdrant_client()
collection_info = client.get_collection(COLLECTION_NAME)
print(f"Collection: {COLLECTION_NAME}")
print(f"Vector Dimensions: {collection_info.config.params.vectors.size}")
