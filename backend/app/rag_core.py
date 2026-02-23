import os
import uuid
import asyncio
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from qdrant_client.http import models
from loguru import logger

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
        # LLM — Gemini 2.5 Flash for high performance and accuracy
        self.llm = ChatGoogleGenerativeAI(
            model="models/gemini-2.5-flash",
            google_api_key=GEMINI_API_KEY,
            temperature=0.1,
            top_p=0.9
        )

        print("[RAGCore] Setting up Text Splitter...", flush=True)
        # Chunker — Better context for relevancy (800 tokens)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100
        )

        print("[RAGCore] Getting Qdrant Client...", flush=True)
        # Qdrant client
        self.client = get_qdrant_client()

        # Sparse model — Load at startup to avoid delay during first query
        self._sparse_model = None
        self._sparse_model = self._get_sparse_model()

        logger.info("[RAGCore] Ready!")

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
        logger.info(f"Generating dense embedding for query: {query[:30]}...")
        dense_query = await asyncio.get_event_loop().run_in_executor(
            None, self.embeddings.embed_query, query
        )
        logger.info(f"Dense embedding generated. Dimension: {len(dense_query)}")

        try:
            if sparse_model:
                # Hybrid Search with RRF Fusion
                logger.info("Performing Hybrid Search with RRF...")
                sparse_result = list(sparse_model.embed([query]))[0]
                results = self.client.query_points(
                    collection_name=COLLECTION_NAME,
                    prefetch=[
                        models.Prefetch(vector=dense_query, limit=limit),
                        models.Prefetch(
                            vector=models.SparseVector(
                                indices=sparse_result.indices.tolist(),
                                values=sparse_result.values.tolist()
                            ),
                            using="text-sparse",
                            limit=limit
                        )
                    ],
                    query=models.FusionQuery(fusion=models.Fusion.RRF),
                    limit=limit
                ).points
                logger.info(f"Hybrid search returned {len(results)} points.")
            else:
                # Simple Dense Search (Fallback)
                logger.info("Performing Dense-only Search...")
                results = self.client.query_points(
                    collection_name=COLLECTION_NAME,
                    query=dense_query,
                    limit=limit
                ).points
                logger.info(f"Dense search returned {len(results)} points.")
            
            return results
        except Exception as e:
            logger.error(f"Search Error: {str(e)}")
            raise e

    async def generate_response(self, question: str) -> dict:
        """
        Full RAG pipeline: search context → construct prompt → Gemini jawab.
        """
        # Search for top 10 most relevant chunks
        search_results = await self.hybrid_search(question, limit=10)

        if not search_results:
            logger.warning(f"No context found for: {question}")
            return {
                "answer": "I'm sorry, I couldn't find any relevant information in the uploaded documents to answer this question.",
                "sources": []
            }

        # Deduplicate and Clean context for production quality
        context_parts = []
        for r in search_results:
            content = r.payload.get("content", "").strip()
            if content and content not in context_parts:
                context_parts.append(content)
        
        context = "\n\n---\n\n".join(context_parts)
        
        sources = sorted(list(set([
            f"Page {r.payload.get('metadata', {}).get('page', '?')}"
            for r in search_results if r.payload and 'metadata' in r.payload
        ])))

        prompt = f"""You are 'Inference Logic', a Senior Document Intelligence Assistant. 
Analyze the Context below and answer the Question with high precision.

TASK:
1. Use ONLY the provided Context. 
2. If the answer is not in the context, state that clearly.
3. Be professional, direct, and structured.
4. Mention sources when possible.

Context:
{context}

Question: {question}

Response:"""

        try:
            logger.info(f"Generating response for question: {question[:50]}...")
            # Use ainvoke for better async performance
            response = await self.llm.ainvoke(prompt)
            return {
                "answer": response.content,
                "sources": sources
            }
        except Exception as e:
            logger.error(f"LLM Generation Error: {str(e)}")
            raise e
