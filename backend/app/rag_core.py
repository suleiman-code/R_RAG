import os
import uuid
import asyncio
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from qdrant_client.http import models

# LangChain text splitter — support for both old and new package layouts
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
from .database import get_qdrant_client, COLLECTION_NAME

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


class RAGCore:
    def __init__(self):
        print("[RAGCore] Initializing Gemini models...", flush=True)

        print("[RAGCore] Setting up Embeddings...", flush=True)
        # Dense Embeddings — Gemini ke saath semantic similarity
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            google_api_key=GEMINI_API_KEY
        )

        print("[RAGCore] Setting up Chat LLM...", flush=True)
        # LLM — Gemini Flash se jawab generate hoga
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=GEMINI_API_KEY
        )

        print("[RAGCore] Setting up Text Splitter...", flush=True)
        # Chunker — PDF ko 500 tokens ke tukron mein torega
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )

        print("[RAGCore] Getting Qdrant Client...", flush=True)
        # Qdrant client
        self.client = get_qdrant_client()

        # Sparse model — lazy load rakho taake startup hang na ho
        # (fastembed downloads model on first call — isko defer karte hain)
        self._sparse_model = None

        print("[RAGCore] Ready!", flush=True)

    def _get_sparse_model(self):
        """
        Sparse model ko pehli baar use pe load karta hai.
        Agar fastembed nahi milta (missing Rust/dependencies), toh None return karega.
        """
        if self._sparse_model is not None:
            return self._sparse_model

        try:
            print("[RAGCore] Loading sparse embedding model (first time only)...", flush=True)
            from fastembed import SparseTextEmbedding
            # Local cache directory path
            cache_path = os.path.join(os.getcwd(), "backend", ".fastembed_cache")
            if not os.path.exists(cache_path):
                os.makedirs(cache_path, exist_ok=True)
            
            # Model load karte waqt cache_dir specify kar rahe hain
            self._sparse_model = SparseTextEmbedding(
                model_name="prithivida/Splade_PP_en_v1",
                cache_dir=cache_path
            )
            print("[RAGCore] Sparse model loaded successfully.", flush=True)
            return self._sparse_model
        except (ImportError, Exception) as e:
            print(f"[RAGCore] WARNING: Could not load fastembed/sparse model: {str(e)}", flush=True)
            print("[RAGCore] Falling back to Dense-only search mode.", flush=True)
            self._sparse_model = "FAILED" # Marker to avoid retrying every time
            return None

    async def ingest_pdf(self, file_path: str) -> int:
        """
        PDF load→chunk→embed→Qdrant mein store.
        Batching dense embeddings to speed up processing.
        Returns: number of chunks indexed.
        """
        # Load PDF — LangChain ke zariye
        loader = PyPDFLoader(file_path)
        documents = await asyncio.get_event_loop().run_in_executor(None, loader.load)
        chunks = self.text_splitter.split_documents(documents)

        if not chunks:
            raise ValueError("No content found in PDF.")

        sparse_model = self._get_sparse_model()
        if sparse_model == "FAILED":
            sparse_model = None

        # Srf non-empty content walay chunks uthao
        valid_chunks = [c for c in chunks if c.page_content.strip()]
        if not valid_chunks:
            return 0
            
        texts = [c.page_content.strip() for c in valid_chunks]

        print(f"[RAGCore] Generating dense embeddings for {len(texts)} chunks...", flush=True)
        # Dense vectors (Gemini) — Batch process kar rahe hain
        dense_vectors = await asyncio.get_event_loop().run_in_executor(
            None, self.embeddings.embed_documents, texts
        )

        sparse_results = None
        if sparse_model:
            print("[RAGCore] Generating sparse embeddings...", flush=True)
            # Sparse vectors (SPLADE) — fastembed naturally batch supports
            sparse_results = list(sparse_model.embed(texts))

        points = []
        for i, text in enumerate(texts):
            point_id = str(uuid.uuid4())
            
            # Vector dictionary taiyar karo
            vectors = {"": dense_vectors[i]}
            if sparse_results:
                vectors["text-sparse"] = models.SparseVector(
                    indices=sparse_results[i].indices.tolist(),
                    values=sparse_results[i].values.tolist()
                )

            points.append(models.PointStruct(
                id=point_id,
                vector=vectors,
                payload={
                    "content": text,
                    "metadata": valid_chunks[i].metadata
                }
            ))

        # Batch upsert into Qdrant
        print(f"[RAGCore] Upserting {len(points)} points to Qdrant...", flush=True)
        self.client.upsert(collection_name=COLLECTION_NAME, points=points)
        return len(points)

    async def clear_collection(self):
        """
        Qdrant collection ke saaray points delete karta hai.
        """
        print(f"[RAGCore] Clearing collection {COLLECTION_NAME}...", flush=True)
        self.client.delete(
            collection_name=COLLECTION_NAME,
            points_selector=models.FilterSelector(
                filter=models.Filter()
            )
        )
        return True

    async def hybrid_search(self, query: str, limit: int = 5):
        """
        Hybrid search: dense (semantic) + sparse (keyword) fusion if sparse model is available.
        Fallback to dense-only query otherwise.
        """
        sparse_model = self._get_sparse_model()
        if sparse_model == "FAILED":
            sparse_model = None

        # Dense embed (hamesha chahiye)
        dense_query = await asyncio.get_event_loop().run_in_executor(
            None, self.embeddings.embed_query, query
        )

        if sparse_model:
            # Hybrid Search with RRF Fusion
            sparse_result = list(sparse_model.embed([query]))[0]
            results = self.client.query_points(
                collection_name=COLLECTION_NAME,
                prefetch=[
                    models.Prefetch(vector=dense_query, limit=limit * 2),
                    models.Prefetch(
                        vector=models.SparseVector(
                            indices=sparse_result.indices.tolist(),
                            values=sparse_result.values.tolist()
                        ),
                        using="text-sparse",
                        limit=limit * 2
                    )
                ],
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                limit=limit
            ).points
        else:
            # Simple Dense Search (Fallback)
            results = self.client.query_points(
                collection_name=COLLECTION_NAME,
                query=dense_query,
                limit=limit
            ).points

        return results

    async def generate_response(self, question: str) -> dict:
        """
        Full RAG pipeline: search context → construct prompt → Gemini jawab.
        """
        search_results = await self.hybrid_search(question)

        if not search_results:
            return {
                "answer": "I could not find relevant information in the uploaded documents. Please upload a PDF first.",
                "sources": []
            }

        # Context assemble karo
        context = "\n\n".join([r.payload["content"] for r in search_results])
        sources = list(set([
            f"Page {r.payload['metadata'].get('page', '?')}"
            for r in search_results
        ]))

        prompt = f"""You are 'Inference Logic', a sophisticated AI Assistant. Your goal is to provide accurate, concise, and professional answers based on the provided documents.

INSTRUCTIONS:
1. Use ONLY the provided context to answer the question.
2. If the answer is not in the context, politely state that you don't have enough information in the documents.
3. Use Markdown for formatting (bold, lists, tables) if it helps clarity.
4. If the question is a greeting or general, you can respond politely but remind the user to ask about the documents.

Context:
{context}

Question: {question}

Answer:"""

        # Gemini se jawab lo (blocking call — executor mein run karo)
        response = await asyncio.get_event_loop().run_in_executor(
            None, self.llm.invoke, prompt
        )

        return {
            "answer": response.content,
            "sources": sources
        }
