import os
import uuid
import asyncio
from dotenv import load_dotenv
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
from google.api_core.exceptions import ResourceExhausted
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
        # LLM — gemini-2.5-flash: best model confirmed on this API key
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=GEMINI_API_KEY,
            temperature=0.1,
            top_p=0.9
        )

        print("[RAGCore] Setting up Text Splitter...", flush=True)
        # Chunker — Optimized for complex docs and research papers
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=200 # Increased for better context continuity
        )

        print("[RAGCore] Getting Qdrant Client...", flush=True)
        # Qdrant client
        self.client = get_qdrant_client()

        # Sparse model — Load at startup, use flag to avoid retrying if failed
        self._sparse_model = None
        self._sparse_load_failed = False
        self._get_sparse_model()

        logger.info("[RAGCore] Ready!")

    def _get_sparse_model(self):
        """
        Loads the sparse embedding model once. Uses a flag to prevent re-trying
        when it has already failed (e.g., missing Rust dependency).
        """
        # Already loaded successfully
        if self._sparse_model is not None:
            return self._sparse_model
        
        # Already tried and failed
        if self._sparse_load_failed:
            return None

        try:
            print("[RAGCore] Loading sparse embedding model...", flush=True)
            from fastembed import SparseTextEmbedding
            cache_path = os.path.join(os.getcwd(), "backend", ".fastembed_cache")
            os.makedirs(cache_path, exist_ok=True)
            self._sparse_model = SparseTextEmbedding(
                model_name="prithivida/Splade_PP_en_v1",
                cache_dir=cache_path
            )
            print("[RAGCore] Sparse model loaded successfully.", flush=True)
            return self._sparse_model
        except Exception as e:
            print(f"[RAGCore] WARNING: Sparse model failed: {e}", flush=True)
            print("[RAGCore] Falling back to Dense-only search.", flush=True)
            self._sparse_load_failed = True
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
        # sparse_model is now always either a valid model or None

        # Srf non-empty content walay chunks uthao
        valid_chunks = [c for c in chunks if c.page_content.strip()]
        if not valid_chunks:
            return 0
            
        texts = [c.page_content.strip() for c in valid_chunks]

        print(f"[RAGCore] Generating dense embeddings for {len(texts)} chunks...", flush=True)
        # Dense vectors (Gemini) — Manual batching with cooldown to avoid Free Tier 429
        dense_vectors = []
        batch_size = 50
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            logger.info(f"Processing embedding batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}...")
            
            # Batch embeddings process
            batch_embeddings = await asyncio.get_event_loop().run_in_executor(
                None, self.embeddings.embed_documents, batch_texts
            )
            dense_vectors.extend(batch_embeddings)
            
            # Small cooldown between batches for Free Tier stability
            if i + batch_size < len(texts):
                await asyncio.sleep(2)

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

            # Extract filename for professional display
            filename = os.path.basename(file_path)
            enhanced_metadata = valid_chunks[i].metadata.copy()
            enhanced_metadata["filename"] = filename

            points.append(models.PointStruct(
                id=point_id,
                vector=vectors,
                payload={
                    "content": text,
                    "metadata": enhanced_metadata
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
        # sparse_model is now always either a valid model or None

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

    async def _expand_query(self, question: str) -> list[str]:
        """
        Generates 3 variations of the user question to improve search recall.
        Essential for complex research papers and professional docs.
        """
        expansion_prompt = f"""You are a Retrieval-Augmented Generation expert.
Given the User Question, generate 3 professional and semantically diverse variations of it.
The goal is to capture different technical aspects and potential keyword matches from a document.

User Question: {question}

Return exactly 3 variations, one per line. No numbers, no extra text."""
        try:
            response = await self.llm.ainvoke(expansion_prompt)
            variations = [v.strip() for v in response.content.split("\n") if v.strip()]
            # Keep the original and add top 2 variations
            return [question] + variations[:2]
        except Exception as e:
            logger.warning(f"Query expansion failed: {e}")
            return [question]

    @retry(
        wait=wait_exponential(multiplier=2, min=5, max=30),
        stop=stop_after_attempt(3),
        # Only retry on 429 Rate Limit errors — not on 400/404 errors
        retry=retry_if_exception_type(ResourceExhausted),
        reraise=True
    )
    async def _safe_invoke(self, prompt):
        """LLM call with auto-retry on 429 quota errors."""
        return await self.llm.ainvoke(prompt)

    async def generate_response(self, question: str) -> dict:
        """
        Full RAG pipeline: Multi-query expansion → Hybrid search → Gemini reasoning.
        """
        # Multi-query expansion for 100% accuracy coverage
        try:
            expanded_queries = await self._expand_query(question)
        except Exception:
            expanded_queries = [question]
        
        logger.info(f"Expanded query into {len(expanded_queries)} variations.")

        all_results = []
        for q in expanded_queries:
            results = await self.hybrid_search(q, limit=10)
            all_results.extend(results)
        
        # Deduplicate results by point ID
        seen_ids = set()
        unique_results = []
        for r in all_results:
            if r.id not in seen_ids:
                unique_results.append(r)
                seen_ids.add(r.id)
        
        search_results = unique_results[:20] # Keep top 20 after fusion

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
        
        # Source deduction with filename tracking
        source_set = set()
        for r in search_results:
            if r.payload and 'metadata' in r.payload:
                meta = r.payload['metadata']
                fname = meta.get('filename', 'Document')
                page = meta.get('page', '?')
                # Human-friendly source format
                source_set.add(f"{fname} (Page {page})")
        
        sources = sorted(list(source_set))

        # Professional prompt for Gemini 2.5 Flash - optimized for accuracy
        prompt = f"""You are 'Inference Logic', a High-Fidelity Document Intelligence System.
Your objective is to provide extremely accurate, factual, and analytical responses based on the provided technical context.

CRITICAL GUIDELINES:
1. ANALYSIS: Carefully scan all context parts. If information is contradictory, highlight the nuances.
2. SCOPE: Answer ONLY based on the provided context. If the data is missing, state: "The current document repository does not contain specific information regarding [X]."
3. STRUCTURE: Use professional headers and bullet points for complex data.
4. CITATION: If multiple documents are present, specify which document you are referencing.

CONTEXT DATA:
---
{context}
---

USER QUESTION: {question}

PROFESSIONAL RESPONSE:"""

        try:
            logger.info(f"Generating high-fidelity response for: {question[:50]}...")
            
            # Phase 1: Initial Draft Generation
            initial_response = await self._safe_invoke(prompt)
            
            # Phase 2: Verification Step (Self-Correction for 100% accuracy)
            verification_prompt = f"""You are a Fact-Checking Supervisor. 
Review the AI Response against the Original Context. 

AI RESPONSE:
{initial_response.content}

ORIGINAL CONTEXT:
{context}

TASK:
1. If the AI response contains any information NOT found in the context, remove it.
2. If the AI response is accurate, keep it as is.
3. Ensure the tone remains professional and precise.

Return only the final verified response."""
            
            verified_response = await self._safe_invoke(verification_prompt)
            
            return {
                "answer": verified_response.content,
                "sources": sources
            }
        except Exception as e:
            import traceback
            logger.error(f"High-Fidelity Generation Error: {str(e)}\n{traceback.format_exc()}")
            # Fallback to initial response if verification step failed
            if 'initial_response' in locals():
                return {"answer": initial_response.content, "sources": sources}
            raise e
