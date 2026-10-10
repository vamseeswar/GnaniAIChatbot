"""
Gnani.ai Company Knowledge Assistant - Streamlit Web Application
A production-grade, grounded conversational AI assistant exclusively for Gnani.ai
using RAG, LangChain, Groq, FAISS, and SentenceTransformers.
"""

import os
import sys
import time
from pathlib import Path

import streamlit as st

# Configure Streamlit page settings
st.set_page_config(
    page_title="Gnani.ai Company Knowledge Assistant",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling for Enterprise Aesthetics
CUSTOM_CSS = """
<style>
    /* Global Styling */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    /* Main Container Padding */
    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 3.5rem;
        max-width: 950px;
    }

    /* Header Banner */
    .gnani-header {
        background: linear-gradient(135deg, #0b0f19 0%, #171e33 50%, #1e1b4b 100%);
        border: 1px solid rgba(99, 102, 241, 0.25);
        border-radius: 16px;
        padding: 24px 28px;
        margin-bottom: 24px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3), 0 8px 10px -6px rgba(0, 0, 0, 0.3);
    }
    
    .gnani-header h1 {
        color: #ffffff;
        font-size: 26px;
        font-weight: 700;
        margin: 0 0 8px 0;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    
    .gnani-badge {
        background: rgba(99, 102, 241, 0.2);
        color: #818cf8;
        border: 1px solid rgba(99, 102, 241, 0.4);
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }
    
    .gnani-header p {
        color: #94a3b8;
        font-size: 14px;
        margin: 0;
        line-height: 1.5;
    }

    /* Source Citation Cards */
    .source-container {
        margin-top: 14px;
        padding-top: 12px;
        border-top: 1px dashed rgba(255, 255, 255, 0.12);
    }

    .source-label {
        font-size: 11px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        color: #818cf8;
        margin-bottom: 8px;
        display: flex;
        align-items: center;
        gap: 6px;
    }

    .source-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(99, 102, 241, 0.3);
        border-radius: 8px;
        padding: 6px 12px;
        margin: 4px 6px 4px 0;
        font-size: 12px;
        color: #e2e8f0;
        text-decoration: none;
        transition: all 0.2s ease;
    }

    .source-pill:hover {
        background: rgba(99, 102, 241, 0.25);
        border-color: #818cf8;
        color: #ffffff;
        text-decoration: none;
        transform: translateY(-1px);
    }

    /* Sidebar Enhancements */
    .sidebar-section {
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 14px;
        margin-bottom: 16px;
    }

    .sidebar-title {
        font-size: 12px;
        font-weight: 700;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 8px;
    }

    .status-badge {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background-color: #10b981;
        margin-right: 6px;
    }

    /* Footer */
    .gnani-footer {
        text-align: center;
        color: #64748b;
        font-size: 12px;
        margin-top: 40px;
        padding-top: 16px;
        border-top: 1px solid rgba(255, 255, 255, 0.08);
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# Lazy import of RAG pipeline
from config import (
    COMPANY_NAME,
    DEFAULT_GROQ_MODEL,
    FAST_GROQ_MODEL,
    GROQ_API_KEY,
    GROQ_MODEL,
    INVENTORY_FILE,
    OFFICIAL_BASE_URL,
    VECTOR_STORE_DIR,
)
from rag_pipeline import GnaniRAGPipeline


@st.cache_resource(show_spinner="Initializing Gnani Knowledge Engine...")
def get_rag_pipeline(api_key: str, model_name: str) -> GnaniRAGPipeline:
    return GnaniRAGPipeline(groq_api_key=api_key, groq_model=model_name)


# Load Inventory Stats
inventory_data = {}
if INVENTORY_FILE.exists():
    try:
        import json
        with open(INVENTORY_FILE, "r", encoding="utf-8") as f:
            inventory_data = json.load(f)
    except Exception:
        pass

total_docs = inventory_data.get("total_documents", 39)
total_chars = inventory_data.get("total_characters", 263154)
last_updated = inventory_data.get("last_updated", "Recent")
if last_updated != "Recent":
    last_updated = last_updated.split("T")[0]

# --- SIDEBAR CONFIGURATION ---
with st.sidebar:
    st.markdown(
        f"""
        <div style="display:flex; align-items:center; gap:10px; margin-bottom:12px;">
            <div style="font-size:28px;">🎙️</div>
            <div>
                <div style="font-weight:700; font-size:16px; color:#ffffff;">Gnani.ai Assistant</div>
                <div style="font-size:11px; color:#818cf8;">Official Knowledge Base</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # API Key Configuration
    st.markdown('<div class="sidebar-title">Groq Inference API</div>', unsafe_allow_html=True)
    env_key = GROQ_API_KEY
    user_api_key = st.text_input(
        "Groq API Key (Free)",
        type="password",
        value=env_key,
        help="Get your free key at https://console.groq.com/",
        placeholder="gsk_...",
    )

    selected_model = st.selectbox(
        "Groq Model",
        options=[DEFAULT_GROQ_MODEL, FAST_GROQ_MODEL],
        index=0,
        help="Choose high-quality 70B versatile model or ultra-fast 8B instant model.",
    )

    if user_api_key:
        st.markdown(
            '<div style="font-size:12px; color:#10b981; margin-top:-6px; margin-bottom:12px;">'
            '● Groq LLM Inference Active</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div style="font-size:12px; color:#f59e0b; margin-top:-6px; margin-bottom:12px;">'
            '▲ Free Groq Key Recommended for full synthesis</div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # Knowledge Base Status
    st.markdown('<div class="sidebar-title">Knowledge Base Status</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="sidebar-section">
            <div style="font-size:13px; color:#f1f5f9; margin-bottom:4px;">
                <span class="status-badge"></span><strong>39 Official Pages Ingested</strong>
            </div>
            <div style="font-size:12px; color:#94a3b8;">• Chunks Indexed: <strong>454</strong></div>
            <div style="font-size:12px; color:#94a3b8;">• Total Characters: <strong>{total_chars:,}</strong></div>
            <div style="font-size:12px; color:#94a3b8;">• Embeddings: <strong>all-MiniLM-L6-v2</strong></div>
            <div style="font-size:12px; color:#94a3b8;">• Index Engine: <strong>FAISS CPU</strong></div>
            <div style="font-size:12px; color:#94a3b8;">• Updated: <strong>{last_updated}</strong></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Scope & Supported Topics
    st.markdown('<div class="sidebar-title">Verified Topics In Scope</div>', unsafe_allow_html=True)
    st.markdown(
        """
        - 🗣️ **Inya Platform & Voice Agents**
        - 🛡️ **Armour365 Voice Biometrics**
        - 📊 **Aura365 Speech Analytics**
        - 🤖 **Assist365 Agent Assist AI**
        - ⚡ **Automate365 Enterprise AI**
        - 🏦 **Industry Use Cases** (Banking, Insurance, Automotive, BPO)
        - 🏢 **Company Overview & Leadership**
        - 💼 **Official Careers & Openings**
        """
    )

    st.markdown("---")

    # Clear Chat Action
    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# --- MAIN INTERFACE ---
st.markdown(
    """
    <div class="gnani-header">
        <h1>
            <span>Gnani.ai Knowledge Assistant</span>
            <span class="gnani-badge">Official RAG Engine</span>
        </h1>
        <p>
            An enterprise AI assistant strictly grounded in official Gnani.ai documentation, 
            voice AI product architectures, industry use cases, and company updates.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Suggested Questions Grid
SUGGESTED_QUESTIONS = [
    "What is Gnani.ai?",
    "What products and services does Gnani.ai offer?",
    "What industries does Gnani.ai serve?",
    "What AI solutions are documented on its official website?",
    "What use cases are described by Gnani.ai?",
    "Where can I find official career information?",
]

# Session State for Conversation
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Hello! I am the **Gnani.ai Company Knowledge Assistant**.\n\n"
                "I am here to answer questions exclusively about Gnani.ai's voice AI platforms "
                "(Inya, Armour365, Assist365, Aura365, Automate365), enterprise solutions, "
                "industry use cases, technologies, and company information. "
                "All answers are grounded in official documentation with direct source citations.\n\n"
                "How can I assist you with Gnani.ai today?"
            ),
            "sources": [{"title": "Gnani.ai Official Website", "url": OFFICIAL_BASE_URL}],
        }
    ]

# Display Suggested Question Chips
st.markdown(
    '<div style="font-size:12px; font-weight:600; color:#94a3b8; margin-bottom:6px;">💡 Suggested Questions:</div>',
    unsafe_allow_html=True,
)
cols = st.columns(3)
clicked_question = None

for i, q in enumerate(SUGGESTED_QUESTIONS):
    col = cols[i % 3]
    if col.button(q, key=f"sugg_{i}", use_container_width=True):
        clicked_question = q

# Display Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="👤" if msg["role"] == "user" else "🎙️"):
        st.markdown(msg["content"])
        if msg.get("sources"):
            st.markdown(
                '<div class="source-container"><div class="source-label">🔗 Official Sources & Citations:</div>',
                unsafe_allow_html=True,
            )
            sources_html = ""
            for s in msg["sources"]:
                title = s.get("title", "Official Page")
                url = s.get("url", OFFICIAL_BASE_URL)
                sources_html += f'<a class="source-pill" href="{url}" target="_blank">📄 {title} ↗</a> '
            st.markdown(sources_html + "</div>", unsafe_allow_html=True)

# User Input Handling
prompt = st.chat_input("Ask any question about Gnani.ai (e.g. What is Armour365?)...")
user_query = clicked_question or prompt

if user_query:
    # Append User Message
    st.session_state.messages.append({"role": "user", "content": user_query, "sources": []})
    with st.chat_message("user", avatar="👤"):
        st.markdown(user_query)

    # Process Assistant Response
    with st.chat_message("assistant", avatar="🎙️"):
        with st.spinner("Searching verified Gnani.ai knowledge base..."):
            pipeline = get_rag_pipeline(user_api_key, selected_model)
            response = pipeline.answer_query(user_query)

            answer_text = response.get("answer", "")
            sources = response.get("sources", [])

            st.markdown(answer_text)

            if sources:
                st.markdown(
                    '<div class="source-container"><div class="source-label">🔗 Official Sources & Citations:</div>',
                    unsafe_allow_html=True,
                )
                sources_html = ""
                for s in sources:
                    title = s.get("title", "Official Page")
                    url = s.get("url", OFFICIAL_BASE_URL)
                    sources_html += f'<a class="source-pill" href="{url}" target="_blank">📄 {title} ↗</a> '
                st.markdown(sources_html + "</div>", unsafe_allow_html=True)

            # Save in history
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer_text,
                    "sources": sources,
                }
            )

# Footer
st.markdown(
    f"""
    <div class="gnani-footer">
        Answers are generated strictly from public official Gnani.ai documentation. 
        Visit the <a href="{OFFICIAL_BASE_URL}" target="_blank" style="color:#818cf8;">Official Gnani.ai Website</a> for live enterprise consultations.
    </div>
    """,
    unsafe_allow_html=True,
)
