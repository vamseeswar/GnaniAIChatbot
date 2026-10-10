"""
Automated tests for data ingestion pipeline (ingest_data.py).
Tests HTML cleaning, document validation, inventory integrity, and graceful error handling.
"""

import json
from pathlib import Path
import pytest
import requests

from config import DATA_DIR, INVENTORY_FILE, PROCESSED_DOCS_DIR
from ingest_data import clean_html_content, fetch_url, process_local_pdfs


def test_clean_html_content():
    sample_html = """
    <!DOCTYPE html>
    <html>
    <head><title>About Gnani AI | Frontier Voice AI</title></head>
    <body>
        <nav><a href="/home">Home</a><a href="/about">About</a></nav>
        <header>Header content</header>
        <script>console.log("secret script");</script>
        <style>.red { color: red; }</style>
        <main>
            <h1>Gnani.ai Voice Solutions</h1>
            <p>Gnani.ai builds conversational AI engines and voice biometrics for enterprise applications.</p>
        </main>
        <footer>Copyright 2026 Gnani.ai</footer>
    </body>
    </html>
    """
    title, text = clean_html_content(sample_html)
    assert "About Gnani AI" in title
    assert "Gnani.ai builds conversational AI engines" in text
    assert "secret script" not in text
    assert "Copyright 2026" not in text


def test_processed_documents_exist_and_valid():
    doc_files = list(PROCESSED_DOCS_DIR.glob("*.json"))
    assert len(doc_files) >= 10, f"Expected at least 10 processed docs, got {len(doc_files)}"

    for df in doc_files[:5]:
        with open(df, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert "id" in data
        assert "title" in data
        assert "source_url" in data
        assert "source_type" in data
        assert "content" in data
        assert len(data["content"]) > 50


def test_source_inventory_integrity():
    assert INVENTORY_FILE.exists(), "source_inventory.json does not exist"
    with open(INVENTORY_FILE, "r", encoding="utf-8") as f:
        inv = json.load(f)

    assert inv["company"] == "Gnani.ai"
    assert inv["total_documents"] > 0
    assert inv["total_characters"] > 10000
    assert len(inv["sources"]) > 0


def test_fetch_url_handles_broken_or_invalid_urls():
    session = requests.Session()
    # Invalid domain
    res = fetch_url("https://this-domain-definitely-does-not-exist-gnani-xyz.com", session)
    assert res is None

    # 404 URL
    res_404 = fetch_url("https://www.gnani.ai/this-page-does-not-exist-404-test", session)
    assert res_404 is None
