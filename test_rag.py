try:
    print("Testing RAGCore import...")
    from backend.app.rag_core import RAGCore
    print("Import successful!")
    print("Initializing RAGCore...")
    rag = RAGCore()
    print("Initialization successful!")
except Exception as e:
    print(f"Error: {e}")
import traceback
traceback.print_exc()
