"""
RAG Pipeline for Gnani.ai Company Knowledge Assistant.
Implements:
1. Document loading from data/processed/
2. RecursiveCharacterTextSplitter chunking with metadata preservation
3. Local sentence-transformers embeddings (all-MiniLM-L6-v2)
4. FAISS vector index indexing, persistence, and loading
5. Application-level guardrails (security injection, greetings, off-topic filtering)
6. Relevance scoring and distance thresholds
7. Grounded LLM generation via Groq and LangChain
8. Source citation extraction with clickable links
"""

import json
import logging
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

from config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    COMPANY_NAME,
    DEFAULT_GROQ_MODEL,
    EMBEDDING_MODEL_NAME,
    FALLBACK_REFUSAL_MESSAGE,
    GROQ_API_KEY,
    GROQ_MODEL,
    PROCESSED_DOCS_DIR,
    RELEVANCE_DISTANCE_THRESHOLD,
    SYSTEM_PROMPT,
    TOP_K,
    VECTOR_STORE_DIR,
)

logger = logging.getLogger("GnaniRAG")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)


class LocalSentenceTransformerEmbeddings(Embeddings):
    """
    Lightweight, local SentenceTransformer embeddings wrapper for LangChain.
    Avoids deprecated pydantic_v1 dependencies and works completely offline.
    """

    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        self.model_name = model_name
        logger.info(f"Loading local embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.model.encode(
            texts, convert_to_numpy=True, show_progress_bar=False
        )
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        embedding = self.model.encode(
            [text], convert_to_numpy=True, show_progress_bar=False
        )
        return embedding[0].tolist()


# Singleton embedding instance for performance
_EMBEDDINGS_INSTANCE: Optional[LocalSentenceTransformerEmbeddings] = None


def get_embedding_model() -> LocalSentenceTransformerEmbeddings:
    global _EMBEDDINGS_INSTANCE
    if _EMBEDDINGS_INSTANCE is None:
        _EMBEDDINGS_INSTANCE = LocalSentenceTransformerEmbeddings()
    return _EMBEDDINGS_INSTANCE


class GnaniRAGPipeline:
    def __init__(self, groq_api_key: Optional[str] = None, groq_model: Optional[str] = None):
        self.api_key = groq_api_key or GROQ_API_KEY
        self.model_name = groq_model or GROQ_MODEL or DEFAULT_GROQ_MODEL
        self.embeddings = get_embedding_model()
        self.vector_store: Optional[FAISS] = None
        self._load_or_build_vector_store()

    def _load_or_build_vector_store(self) -> None:
        """
        Loads existing FAISS vector store from disk, or builds one from processed documents.
        """
        index_file = VECTOR_STORE_DIR / "index.faiss"
        pkl_file = VECTOR_STORE_DIR / "index.pkl"

        if index_file.exists() and pkl_file.exists():
            try:
                logger.info(f"Loading persisted FAISS index from {VECTOR_STORE_DIR}...")
                self.vector_store = FAISS.load_local(
                    str(VECTOR_STORE_DIR),
                    self.embeddings,
                    allow_dangerous_deserialization=True,
                )
                logger.info("FAISS index loaded successfully.")
                return
            except Exception as e:
                logger.warning(f"Failed to load existing FAISS index: {e}. Rebuilding...")

        self.build_vector_store()

    def build_vector_store(self) -> int:
        """
        Reads all JSON documents in data/processed/, splits into chunks,
        embeds them, and persists the FAISS vector store.
        Returns total number of chunks created.
        """
        doc_files = list(PROCESSED_DOCS_DIR.glob("*.json"))
        if not doc_files:
            logger.warning(
                f"No processed documents found in {PROCESSED_DOCS_DIR}. Run ingest_data.py first."
            )
            return 0

        documents: List[Document] = []
        for df in doc_files:
            try:
                with open(df, "r", encoding="utf-8") as f:
                    data = json.load(f)
                content = data.get("content", "")
                if not content or len(content.strip()) < 50:
                    continue

                metadata = {
                    "source": data.get("source_url", "https://www.gnani.ai"),
                    "title": data.get("title", "Gnani.ai Official"),
                    "source_type": data.get("source_type", "official_website"),
                    "retrieval_date": data.get("retrieval_date", ""),
                    "doc_id": data.get("id", df.stem),
                }
                documents.append(Document(page_content=content, metadata=metadata))
            except Exception as e:
                logger.error(f"Error loading {df}: {e}")

        if not documents:
            logger.warning("No valid documents to index.")
            return 0

        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

        chunked_docs = text_splitter.split_documents(documents)

        # Prepend contextual header (title and URL) to every chunk for optimal semantic retrieval
        for i, doc in enumerate(chunked_docs):
            doc.metadata["chunk_id"] = f"{doc.metadata.get('doc_id', 'doc')}_c{i}"
            title = doc.metadata.get("title", "")
            source = doc.metadata.get("source", "")
            doc.page_content = f"Document: {title}\nURL: {source}\n\n{doc.page_content}"

        logger.info(f"Generated {len(chunked_docs)} chunks from {len(documents)} documents.")

        # Build FAISS index
        self.vector_store = FAISS.from_documents(chunked_docs, self.embeddings)
        self.vector_store.save_local(str(VECTOR_STORE_DIR))
        logger.info(f"FAISS vector store saved to {VECTOR_STORE_DIR}.")
        return len(chunked_docs)

    def check_guardrails(self, query: str) -> Optional[Tuple[str, List[Dict]]]:
        """
        Evaluates input query against application-level safety and company-only rules.
        Returns (response_text, sources) if triggered, or None if safe to proceed.
        """
        q_clean = query.strip().lower()

        # 1. Security & Prompt Injection Defense
        injection_patterns = [
            r"ignore (all )?(previous|above|prior) (instructions|prompts|rules)",
            r"reveal (your |the )?(api.?key|system prompt|environment|credentials|secrets)",
            r"what is your system (prompt|instruction)",
            r"print (your |the )?(api.?key|env|groq_api_key)",
            r"bypass (all )?(restrictions|rules|filters)",
            r"act as (a |an )?(dan|unrestricted|jailbroken|pirate)",
            r"disclose (internal|confidential|secret)",
        ]
        for pattern in injection_patterns:
            if re.search(pattern, q_clean):
                return (
                    "I am the Gnani.ai Company Knowledge Assistant. I operate strictly under "
                    "official company guidelines and cannot disclose system prompts, credentials, "
                    "or bypass security boundaries. Please feel free to ask questions about Gnani.ai's "
                    "voice AI products, technologies, and services.",
                    [],
                )

        # 2. Friendly Greetings
        greeting_patterns = [
            r"^(hi|hello|hey|greetings|good (morning|afternoon|evening|day))[\.!\? ]*$",
            r"^(who are you|what can you do|how can you help me)[\.!\? ]*$",
        ]
        for pattern in greeting_patterns:
            if re.search(pattern, q_clean):
                return (
                    "Hello! I am the Gnani.ai Company Knowledge Assistant. I'm here to provide verified "
                    "information about Gnani.ai, including our voice AI platforms (Inya, Armour365, Assist365, "
                    "Aura365, Automate365), enterprise solutions, industry use cases, technologies, and official company updates.\n\n"
                    "How can I assist you with Gnani.ai today?",
                    [
                        {
                            "title": "Gnani.ai Official Website",
                            "url": "https://www.gnani.ai",
                        }
                    ],
                )

        # 3. Unsupported / Unverified Information (Prices, Salaries, Stock, Unverified Statistics)
        unsupported_patterns = [
            r"\b(price|pricing|cost|subscription fee|how much does .* cost|rate card)\b",
            r"\b(salary|salaries|compensation|pay scale)\b",
            r"\b(stock price|market cap|share price|quarterly profit|annual revenue|net worth)\b",
        ]
        for pattern in unsupported_patterns:
            if re.search(pattern, q_clean):
                return (
                    FALLBACK_REFUSAL_MESSAGE,
                    [],
                )

        # 4. Off-Topic & Unrelated Company / Domain Filtering
        off_topic_exact = [
            # Other companies
            r"\b(microsoft|google|apple|amazon|openai|tesla|facebook|netflix|meta)\b",
            # General coding / programming tutorials unrelated to Gnani
            r"\b(write a python (script|code|program)|create a react app|bubble sort|binary search|sql query to join)\b",
            # Sports / entertainment / general trivia
            r"\b(fifa|world cup|cricket score|recipe for|capital of france|who won the oscar|weather in)\b",
        ]
        # Only block if the question is NOT also asking about Gnani's relationship/comparison or Gnani context
        if not ("gnani" in q_clean or "inya" in q_clean or "armour365" in q_clean or "aura365" in q_clean):
            for pattern in off_topic_exact:
                if re.search(pattern, q_clean):
                    return (
                        "I am exclusively designed to answer questions about Gnani.ai, its voice AI products, "
                        "solutions, and company information. I cannot assist with general knowledge, programming "
                        "tutorials, entertainment, or unrelated companies. Please ask a question related to Gnani.ai!",
                        [],
                    )

        return None

    def retrieve(self, query: str, top_k: int = TOP_K) -> List[Tuple[Document, float]]:
        """
        Retrieves relevant chunks with similarity scores.
        """
        if self.vector_store is None:
            logger.error("Vector store is not initialized.")
            return []

        try:
            results = self.vector_store.similarity_search_with_score(query, k=top_k)
            return results
        except Exception as e:
            logger.error(f"Error during retrieval: {e}")
            return []

    def answer_query(self, query: str) -> Dict:
        """
        Full RAG workflow:
        1. Check guardrails
        2. Retrieve relevant chunks
        3. Score filtering
        4. LLM synthesis with Groq
        5. Return grounded answer and formatted citations
        """
        start_time = time.time()

        # Step 1: Guardrail Check
        guardrail_result = self.check_guardrails(query)
        if guardrail_result:
            answer_text, sources = guardrail_result
            is_fallback = (FALLBACK_REFUSAL_MESSAGE in answer_text)
            return {
                "answer": answer_text,
                "sources": sources,
                "chunks_retrieved": 0,
                "latency_seconds": round(time.time() - start_time, 2),
                "is_fallback": is_fallback,
            }

        # Step 2: Vector Retrieval
        results = self.retrieve(query, top_k=TOP_K)

        if not results:
            return {
                "answer": FALLBACK_REFUSAL_MESSAGE,
                "sources": [],
                "chunks_retrieved": 0,
                "latency_seconds": round(time.time() - start_time, 2),
                "is_fallback": True,
            }

        # Step 3: Relevance Distance Threshold Check
        # FAISS L2 distance: lower is more similar.
        best_score = results[0][1]
        logger.info(f"Top result distance score: {best_score:.4f} for query: '{query[:50]}'")

        filtered_results = [
            (doc, score)
            for doc, score in results
            if score <= RELEVANCE_DISTANCE_THRESHOLD
        ]

        if not filtered_results:
            logger.info("All retrieved chunks exceeded distance threshold.")
            return {
                "answer": FALLBACK_REFUSAL_MESSAGE,
                "sources": [],
                "chunks_retrieved": len(results),
                "latency_seconds": round(time.time() - start_time, 2),
                "is_fallback": True,
            }

        # Compile Context and Sources
        context_parts = []
        sources_dict = {}

        for doc, score in filtered_results:
            source_url = doc.metadata.get("source", "https://www.gnani.ai")
            title = doc.metadata.get("title", "Gnani.ai Official Document")
            sources_dict[source_url] = title

            snippet = (
                f"[Source: {title} | URL: {source_url}]\n"
                f"{doc.page_content.strip()}\n"
            )
            context_parts.append(snippet)

        context_text = "\n---\n".join(context_parts)

        # Step 4: Generation via Groq or Fallback Extractive Mode
        if not self.api_key:
            # When Groq API key is not configured, supply extracted official excerpts directly
            direct_summary = (
                "**[Notice: Free Groq API Key Not Configured]**\n\n"
                "To enable complete AI synthesis, please add your free Groq API key in the sidebar "
                "or configure `GROQ_API_KEY` in your `.env` file.\n\n"
                "**Verified Information from Gnani.ai Knowledge Base:**\n\n"
                f"{filtered_results[0][0].page_content.strip()[:600]}..."
            )
            return {
                "answer": direct_summary,
                "sources": [
                    {"title": title, "url": url}
                    for url, title in sources_dict.items()
                ],
                "chunks_retrieved": len(filtered_results),
                "latency_seconds": round(time.time() - start_time, 2),
                "is_fallback": False,
            }

        try:
            llm = ChatGroq(
                groq_api_key=self.api_key,
                model_name=self.model_name,
                temperature=0.1,
                max_retries=2,
            )

            prompt_user_content = f"""Context from Gnani.ai official documentation:
----------------------------------------
{context_text}
----------------------------------------

User Question: {query}

Instructions:
1. Answer the user question accurately based ONLY on the evidence provided above.
2. If the context does not contain sufficient facts to answer the question, respond EXACTLY with:
"{FALLBACK_REFUSAL_MESSAGE}"
3. Do not invent any pricing, metrics, customer names, or features not mentioned in the context.
4. Keep the answer professional, concise, and well-structured."""

            messages = [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=prompt_user_content),
            ]

            response = llm.invoke(messages)
            raw_answer = response.content.strip()

            # If the LLM declared lack of evidence, suppress citations
            if (
                FALLBACK_REFUSAL_MESSAGE.lower() in raw_answer.lower()
                or "don't have enough verified information" in raw_answer.lower()
            ):
                return {
                    "answer": FALLBACK_REFUSAL_MESSAGE,
                    "sources": [],
                    "chunks_retrieved": len(filtered_results),
                    "latency_seconds": round(time.time() - start_time, 2),
                    "is_fallback": True,
                }

            sources_list = [
                {"title": title, "url": url} for url, title in sources_dict.items()
            ]

            return {
                "answer": raw_answer,
                "sources": sources_list,
                "chunks_retrieved": len(filtered_results),
                "latency_seconds": round(time.time() - start_time, 2),
                "is_fallback": False,
            }

        except Exception as e:
            logger.error(f"Error during Groq LLM inference: {e}")
            # If API failure occurs, return friendly message without crashing
            return {
                "answer": (
                    f"An error occurred while communicating with the AI service: {str(e)}.\n\n"
                    "Please check your Groq API key and network connectivity."
                ),
                "sources": [
                    {"title": title, "url": url}
                    for url, title in sources_dict.items()
                ],
                "chunks_retrieved": len(filtered_results),
                "latency_seconds": round(time.time() - start_time, 2),
                "is_fallback": True,
            }


if __name__ == "__main__":
    logger.info("Rebuilding vector index from processed documents...")
    pipeline = GnaniRAGPipeline()
    total_chunks = pipeline.build_vector_store()
    logger.info(f"Rebuilt index successfully with {total_chunks} chunks.")
    test_query = "What is Armour365?"
    logger.info(f"Testing retrieval for: '{test_query}'")
    res = pipeline.retrieve(test_query, top_k=2)
    for doc, score in res:
        print(f"\n[Score: {score:.4f}] {doc.metadata['title']} ({doc.metadata['source']}):")
        print(doc.page_content[:200] + "...")
