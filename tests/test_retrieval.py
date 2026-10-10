"""
Automated tests for vector retrieval and FAISS indexing (rag_pipeline.py).
Tests embedding dimensions, metadata persistence, similarity search, and error tolerance.
"""

from pathlib import Path
import pytest
from langchain_core.documents import Document

from config import VECTOR_STORE_DIR
from rag_pipeline import GnaniRAGPipeline, LocalSentenceTransformerEmbeddings, get_embedding_model


def test_embedding_model_dimensions():
    model = get_embedding_model()
    vec = model.embed_query("Gnani.ai speech technology")
    assert len(vec) == 384, f"Expected 384 dimensions, got {len(vec)}"


def test_vector_store_loads_and_retrieves():
    pipeline = GnaniRAGPipeline()
    assert pipeline.vector_store is not None, "Vector store should be initialized"

    results = pipeline.retrieve("What is Gnani.ai?", top_k=3)
    assert len(results) > 0, "Should retrieve at least one result"

    doc, score = results[0]
    assert isinstance(doc, Document)
    assert "source" in doc.metadata
    assert "title" in doc.metadata
    assert score < 1.3, f"Expected good similarity score, got {score}"


def test_retrieval_product_specific_queries():
    pipeline = GnaniRAGPipeline()

    # Query Armour365
    armour_res = pipeline.retrieve("Armour365 voice biometrics authentication", top_k=3)
    armour_texts = " ".join([d.page_content.lower() for d, _ in armour_res])
    assert "armour" in armour_texts or "biometric" in armour_texts or "voice" in armour_texts

    # Query Banking / Insurance
    industry_res = pipeline.retrieve("AI solutions for banking and insurance", top_k=3)
    industry_texts = " ".join([d.page_content.lower() for d, _ in industry_res])
    assert "banking" in industry_texts or "insurance" in industry_texts or "agent" in industry_texts


def test_distance_threshold_filtering():
    pipeline = GnaniRAGPipeline()
    # Meaningless nonsense string should have high distance or yield fallback
    nonsense_query = "xyzkwjflasjfoiewjflskfjsldkjflsdkjf9238429384729384"
    res = pipeline.answer_query(nonsense_query)
    assert res["is_fallback"] is True
    assert "I’m sorry, I don’t have enough verified information" in res["answer"]
