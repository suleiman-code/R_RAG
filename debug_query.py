import asyncio
from backend.app.rag_core import RAGCore

async def test():
    print("Testing RAGCore...")
    rag = RAGCore()
    try:
        print("Querying...")
        response = await rag.generate_response("What is this document about?")
        print("Response:", response)
    except Exception as e:
        import traceback
        print("Error during query:")
        print(traceback.format_exc())

if __name__ == "__main__":
    asyncio.run(test())
