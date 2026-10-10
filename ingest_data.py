"""
Data ingestion pipeline for Gnani.ai Company Knowledge Assistant.
Crawls official web pages from gnani.ai, processes local PDFs/documents,
cleans text, extracts metadata, and saves processed records under data/processed/.
"""

import json
import logging
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader

from config import (
    COMPANY_NAME,
    DATA_DIR,
    INVENTORY_FILE,
    OFFICIAL_BASE_URL,
    OFFICIAL_SEED_URLS,
    PROCESSED_DOCS_DIR,
    RAW_DOCS_DIR,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("GnaniIngest")

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}


def clean_html_content(html: str) -> tuple[str, str]:
    """
    Parses HTML, decomposes non-content elements, extracts title and clean text.
    Returns (title, cleaned_text).
    """
    soup = BeautifulSoup(html, "html.parser")

    # Extract title
    title = "Gnani.ai Official Document"
    if soup.title and soup.title.string:
        title = soup.title.string.strip()
    elif soup.find("h1"):
        title = soup.find("h1").get_text(strip=True)

    # Clean title
    title = re.sub(r"\s+", " ", title).strip()

    # Remove irrelevant or boilerplate tags
    for tag in soup(
        [
            "script",
            "style",
            "noscript",
            "nav",
            "footer",
            "svg",
            "iframe",
            "form",
            "button",
            "header",
        ]
    ):
        tag.decompose()

    # Prefer main, article, or primary content containers if available
    main_content = soup.find("main") or soup.find("article") or soup.find("body") or soup

    raw_text = main_content.get_text(separator=" \n ")

    # Normalize whitespace while preserving paragraphs
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in raw_text.splitlines()]
    clean_lines = [l for l in lines if len(l) > 15 or (len(l) > 3 and l.isupper())]
    normalized_text = "\n\n".join(clean_lines)

    return title, normalized_text


def fetch_url(url: str, session: requests.Session) -> Optional[tuple[str, str]]:
    """
    Fetches a web page with error handling.
    Returns (title, text) or None if inaccessible.
    """
    try:
        logger.info(f"Fetching: {url}")
        resp = session.get(url, headers=HEADERS, timeout=12)
        if resp.status_code == 200:
            return clean_html_content(resp.text)
        else:
            logger.warning(f"Failed to fetch {url}: HTTP {resp.status_code}")
            return None
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching {url}: {e}")
        return None


def discover_sitemap_urls(session: requests.Session, max_urls: int = 40) -> List[str]:
    """
    Discovers high-value English URLs from sitemap.xml.
    """
    sitemap_url = f"{OFFICIAL_BASE_URL}/sitemap.xml"
    discovered = []
    try:
        resp = session.get(sitemap_url, headers=HEADERS, timeout=10)
        if resp.status_code == 200:
            soup = BeautifulSoup(resp.text, "xml")
            for loc in soup.find_all("loc"):
                url = loc.text.strip()
                # Keep English pages only, avoid international /ja-jp
                if "/ja-jp" not in url and url.startswith("https://www.gnani.ai"):
                    # Focus on product, solution, career, about, research pages
                    if any(
                        sub in url
                        for sub in [
                            "about",
                            "career",
                            "technology",
                            "agent",
                            "product",
                            "solution",
                            "inya",
                            "armour",
                            "aura",
                            "assist",
                            "insurance",
                            "banking",
                            "automotive",
                            "customer",
                            "research",
                        ]
                    ):
                        discovered.append(url)
    except Exception as e:
        logger.warning(f"Could not parse sitemap: {e}")

    # Combine with seed URLs and deduplicate
    combined = list(dict.fromkeys(OFFICIAL_SEED_URLS + discovered))
    return combined[:max_urls]


def process_local_pdfs() -> List[Dict]:
    """
    Reads all PDF files placed in data/raw_docs/ and extracts text.
    """
    pdf_docs = []
    pdf_files = list(RAW_DOCS_DIR.glob("*.pdf"))
    for pdf_path in pdf_files:
        try:
            logger.info(f"Processing local PDF: {pdf_path.name}")
            reader = PdfReader(str(pdf_path))
            full_text = []
            for i, page in enumerate(reader.pages):
                page_text = page.extract_text()
                if page_text:
                    full_text.append(page_text)

            content = "\n\n".join(full_text).strip()
            if content:
                doc = {
                    "id": f"pdf_{pdf_path.stem}",
                    "title": pdf_path.stem.replace("_", " ").title(),
                    "source_url": str(pdf_path),
                    "source_type": "official_pdf",
                    "retrieval_date": datetime.now(timezone.utc).isoformat(),
                    "content": content,
                    "char_count": len(content),
                }
                pdf_docs.append(doc)
        except Exception as e:
            logger.error(f"Error reading PDF {pdf_path}: {e}")
    return pdf_docs


def run_ingestion() -> Dict:
    """
    Executes the end-to-end ingestion pipeline:
    1. Crawls official Gnani.ai pages
    2. Ingests local PDFs
    3. Writes structured JSON files under data/processed/
    4. Generates data/source_inventory.json
    """
    session = requests.Session()
    urls_to_crawl = discover_sitemap_urls(session)
    logger.info(f"Discovered {len(urls_to_crawl)} official pages to process.")

    all_documents = []
    inventory_items = []

    for url in urls_to_crawl:
        time.sleep(0.2)  # Respectful polite delay
        res = fetch_url(url, session)
        if not res:
            inventory_items.append(
                {
                    "url": url,
                    "status": "failed_or_skipped",
                    "title": "",
                    "retrieval_date": datetime.now(timezone.utc).isoformat(),
                    "char_count": 0,
                }
            )
            continue

        title, content = res
        if len(content) < 100:
            logger.warning(f"Skipping {url}: insufficient content ({len(content)} chars)")
            continue

        url_path = urlparse(url).path.strip("/").replace("/", "_") or "homepage"
        doc_id = f"web_{url_path}"

        doc_data = {
            "id": doc_id,
            "title": title,
            "source_url": url,
            "source_type": "official_website",
            "retrieval_date": datetime.now(timezone.utc).isoformat(),
            "content": content,
            "char_count": len(content),
        }

        # Save individual processed document
        doc_file = PROCESSED_DOCS_DIR / f"{doc_id}.json"
        with open(doc_file, "w", encoding="utf-8") as f:
            json.dump(doc_data, f, ensure_ascii=False, indent=2)

        all_documents.append(doc_data)
        inventory_items.append(
            {
                "url": url,
                "status": "success",
                "title": title,
                "retrieval_date": doc_data["retrieval_date"],
                "char_count": len(content),
            }
        )

    # Ingest local PDFs
    local_docs = process_local_pdfs()
    for ldoc in local_docs:
        doc_file = PROCESSED_DOCS_DIR / f"{ldoc['id']}.json"
        with open(doc_file, "w", encoding="utf-8") as f:
            json.dump(ldoc, f, ensure_ascii=False, indent=2)
        all_documents.append(ldoc)
        inventory_items.append(
            {
                "url": ldoc["source_url"],
                "status": "success",
                "title": ldoc["title"],
                "retrieval_date": ldoc["retrieval_date"],
                "char_count": ldoc["char_count"],
            }
        )

    total_chars = sum(d["char_count"] for d in all_documents)
    inventory = {
        "company": COMPANY_NAME,
        "official_website": OFFICIAL_BASE_URL,
        "last_updated": datetime.now(timezone.utc).isoformat(),
        "total_documents": len(all_documents),
        "total_characters": total_chars,
        "sources": inventory_items,
    }

    with open(INVENTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(inventory, f, ensure_ascii=False, indent=2)

    logger.info(
        f"Ingestion complete! Processed {len(all_documents)} official documents "
        f"({total_chars:,} total characters). Inventory saved to {INVENTORY_FILE.name}."
    )
    return inventory


if __name__ == "__main__":
    run_ingestion()
