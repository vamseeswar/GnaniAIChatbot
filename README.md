---
title: Gnani.ai Company Knowledge Assistant
emoji: 🎙️
colorFrom: indigo
colorTo: blue
sdk: streamlit
app_file: app.py
pinned: false
---

# 🎙️ Gnani.ai Company Knowledge Assistant

A secure, grounded, enterprise Retrieval-Augmented Generation (RAG) conversational assistant built exclusively for **[Gnani.ai](https://www.gnani.ai/)**, the frontier Voice AI & Conversational AI company.

Built with **Python**, **Streamlit**, **LangChain**, **Groq**, **Hugging Face Sentence-Transformers**, and **FAISS**.

---

## 🌟 Key Highlights & Architecture

- **Strict Company Scope**: Answers questions exclusively about Gnani.ai's official products, technologies, industry use cases, leadership, and careers.
- **Hallucination Prevention**: If evidence is missing, the assistant reliably outputs:
  > *"I’m sorry, I don’t have enough verified information in the Gnani.ai knowledge base to answer that question. Please refer to the official Gnani.ai website."*
- **Clickable Citations**: Every generated response includes direct clickable links to the official Gnani.ai source pages.
- **Security & Injection Defense**: Multi-layered guardrails detect and neutralize role-override attempts, prompt injections, and credential disclosure requests.
- **Free & Open-Source Stack**:
  - **LLM**: Groq API (`llama-3.3-70b-versatile` / `llama-3.1-8b-instant`) — Free tier
  - **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` — 100% local, fast, zero API cost
  - **Vector Store**: FAISS CPU — High-performance vector similarity search
  - **UI**: Streamlit with custom CSS enterprise dark theme

---

## 📊 Ingested Knowledge Base Inventory

The automated crawler (`ingest_data.py`) collected and indexed **39 official pages** containing **263,154 verified characters** and **454 contextual chunks**:

| Category | Official Pages Ingested |
| :--- | :--- |
| **Core Platforms** | Inya Platform, Armour365 Voice Biometrics, Aura365 Analytics, Assist365 Agent Assist, Automate365 Workflow AI |
| **Industry Solutions** | Agentic AI for Banking, Insurance, Automotive, BPO Contact Centers, Healthcare, Retail, Telecom, Real Estate, Hospitality, EdTech, HR |
| **Enterprise Features** | Customer Experience AI, Customer Support AI, Marketing Automation, Analytics & QA AI, Internal Operations Automation |
| **Company & Careers** | About Us, Technology Overview, Official Careers & Openings, Customer Stories, Testimonials, Privacy Policy |

Full inventory is cataloged in [`data/source_inventory.json`](data/source_inventory.json).

---

## 🚀 Quickstart & Local Setup (Windows PowerShell)

### 1. Clone or Open the Workspace
```powershell
cd c:\Chatbot_GnaniAI
```

### 2. Create and Activate Virtual Environment
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Copy the `.env.example` file to `.env`:
```powershell
Copy-Item .env.example .env
```
Edit `.env` and set your free Groq API key:
```env
GROQ_API_KEY=gsk_your_groq_api_key_here
GROQ_MODEL=llama-3.3-70b-versatile
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
TOP_K=4
```
*(Get your free API key at [https://console.groq.com/](https://console.groq.com/))*

### 5. Ingest Knowledge & Build Index (Optional if pre-built)
```powershell
python ingest_data.py
python rag_pipeline.py
```

### 6. Run the Application
```powershell
streamlit run app.py
```
Open your browser at **`http://localhost:8501`**.

---

## 🧪 Automated Test Suite

Run all automated unit tests covering relevant queries, off-topic rejections, prompt injection guards, and retrieval threshold:

```powershell
python -m pytest -v tests/
```

### Test Suite Results:
- `test_relevant_gnani_company_queries` — **PASSED**
- `test_unrelated_questions_are_rejected` — **PASSED**
- `test_security_and_prompt_injection_defense` — **PASSED**
- `test_unsupported_facts_trigger_refusal` — **PASSED**
- `test_missing_api_key_reliability` — **PASSED**
- `test_clean_html_content` — **PASSED**
- `test_processed_documents_exist_and_valid` — **PASSED**
- `test_source_inventory_integrity` — **PASSED**
- `test_fetch_url_handles_broken_or_invalid_urls` — **PASSED**
- `test_embedding_model_dimensions` — **PASSED**
- `test_vector_store_loads_and_retrieves` — **PASSED**
- `test_retrieval_product_specific_queries` — **PASSED**
- `test_distance_threshold_filtering` — **PASSED**

**Status**: 13/13 Passed (100% Pass Rate).

---

## 🌐 Cloud Deployment Guide (Hugging Face Spaces)

This repository includes native Hugging Face Spaces Streamlit configuration in `README.md` frontmatter.

### Step-by-Step Deployment:
1. Log in to [Hugging Face](https://huggingface.co/) and click **New Space**.
2. Select:
   - **Space SDK**: `Streamlit`
   - **Space Hardware**: `Free CPU (2 vCPU, 16GB RAM)`
   - **Visibility**: `Public`
3. In **Settings -> Variables and Secrets**:
   - Add Secret: `GROQ_API_KEY` = your Groq API Key
4. Push this repository to your Space:
   ```bash
   git remote add space https://huggingface.co/spaces/YOUR_USERNAME/gnani-ai-chatbot
   git push space main
   ```
5. Hugging Face Spaces will build and provide your permanent public live URL:
   `https://huggingface.co/spaces/YOUR_USERNAME/gnani-ai-chatbot`

---

## 📁 Project Structure

```
Chatbot_GnaniAI/
├── app.py                     # Streamlit application with custom enterprise styling
├── rag_pipeline.py            # Local SentenceTransformers + FAISS + Groq RAG chain
├── ingest_data.py             # Official site crawler + PDF ingestion + inventory generator
├── config.py                  # Central configuration, models, paths, and guardrails
├── requirements.txt           # Production dependency specifications
├── .env.example               # Environment template
├── .gitignore                 # Secret and cache exclusion
├── packages.txt               # OS package definitions for cloud deployment
├── README.md                  # Documentation and Hugging Face Spaces spec
├── data/
│   ├── processed/             # 39 cleaned official JSON documents
│   ├── raw_docs/              # Directory for custom manual PDF additions
│   └── source_inventory.json  # Complete catalog of ingested company documents
├── vector_store/              # Persisted FAISS CPU vector index files
└── tests/
    ├── conftest.py            # Test path configuration
    ├── test_ingestion.py      # Crawler and HTML parser unit tests
    ├── test_retrieval.py      # Vector similarity and chunking tests
    └── test_company_scope.py  # Guardrail and security unit tests
```

---

## ⚖️ License & Disclaimer

This assistant is designed strictly for educational, informational, and demonstration purposes regarding **Gnani.ai**. Content is grounded strictly in public official materials from [gnani.ai](https://www.gnani.ai/).
