"""
app.py - Streamlit UI for the RAG Study Assistant
"""

import os
import streamlit as st
from ingest import load_and_split_pdf
from vector_store import create_vector_store
from main import answer_query

# ─── Page Configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Study Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    .main-header {
        background: linear-gradient(135deg, #0f0f23 0%, #1a1a3e 50%, #0d1117 100%);
        padding: 2.5rem 2rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        border: 1px solid #2d2d5e;
        text-align: center;
    }

    .main-header h1 {
        font-family: 'Space Mono', monospace;
        font-size: 2.2rem;
        font-weight: 700;
        color: #e2e8f0;
        letter-spacing: -0.02em;
        margin: 0;
    }

    .main-header p {
        color: #7c86a2;
        font-size: 1rem;
        margin-top: 0.5rem;
        font-weight: 300;
    }

    .answer-box {
        background: #0d1117;
        border: 1px solid #30363d;
        border-left: 4px solid #58a6ff;
        border-radius: 8px;
        padding: 1.5rem;
        margin-top: 1rem;
    }

    .answer-box h4 {
        color: #58a6ff;
        font-family: 'Space Mono', monospace;
        font-size: 0.75rem;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        margin-bottom: 0.75rem;
    }

    .answer-text {
        color: #e2e8f0;
        font-size: 1rem;
        line-height: 1.7;
    }

    .not-found-box {
        background: #1a0a0a;
        border: 1px solid #4d1919;
        border-left: 4px solid #f85149;
        border-radius: 8px;
        padding: 1.5rem;
        margin-top: 1rem;
    }

    .not-found-box h4 {
        color: #f85149;
        font-family: 'Space Mono', monospace;
        font-size: 0.75rem;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        margin-bottom: 0.5rem;
    }

    .source-badge {
        display: inline-block;
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 20px;
        padding: 0.2rem 0.75rem;
        font-size: 0.75rem;
        color: #8b949e;
        margin-right: 0.4rem;
        font-family: 'Space Mono', monospace;
    }

    .status-success {
        background: #0a1a0a;
        border: 1px solid #1e4d1e;
        border-radius: 8px;
        padding: 1rem 1.25rem;
        color: #3fb950;
        font-size: 0.9rem;
    }

    .stButton > button {
        background: linear-gradient(135deg, #238636, #2ea043);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1.5rem;
        font-family: 'DM Sans', sans-serif;
        font-weight: 500;
        font-size: 0.95rem;
        cursor: pointer;
        transition: opacity 0.2s;
        width: 100%;
    }

    .stButton > button:hover {
        opacity: 0.85;
    }

    .sidebar-section {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1rem;
    }

    hr {
        border-color: #21262d;
    }
</style>
""", unsafe_allow_html=True)

# ─── Data Directory Setup ──────────────────────────────────────────────────────
DATA_DIR = "data"
PDF_PATH = os.path.join(DATA_DIR, "temp.pdf")
os.makedirs(DATA_DIR, exist_ok=True)

# ─── Session State ─────────────────────────────────────────────────────────────
if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ─── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Configuration")

    api_key = st.text_input(
        "Gemini API Key",
        type="password",
        placeholder="AIza...",
        help="Get your key from https://aistudio.google.com"
    )

    st.markdown("---")
    st.markdown("### 📄 Upload Document")

    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
        help="Upload the PDF document you want to query."
    )

    if uploaded_file is not None:
        with open(PDF_PATH, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.success(f"✅ **{uploaded_file.name}** uploaded")

        if st.button("🔄 Process Document"):
            if not os.path.exists(PDF_PATH):
                st.error("PDF file not found. Please re-upload.")
            else:
                with st.spinner("Processing PDF... This may take a moment."):
                    try:
                        chunks = load_and_split_pdf(PDF_PATH)
                        create_vector_store(chunks)
                        st.session_state.pdf_processed = True
                        st.session_state.chat_history = []
                        st.success(f"✅ Processed **{len(chunks)}** chunks")
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")

    st.markdown("---")

    if st.session_state.pdf_processed:
        st.markdown(
            '<div style="color:#3fb950;font-size:0.85rem;">● Document indexed and ready</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<div style="color:#8b949e;font-size:0.85rem;">○ No document indexed yet</div>',
            unsafe_allow_html=True
        )

    if st.session_state.chat_history:
        if st.button("🗑️ Clear History"):
            st.session_state.chat_history = []
            st.rerun()

    st.markdown("---")
    st.markdown(
        '<div style="color:#484f58;font-size:0.75rem;">RAG Study Assistant · Powered by Gemini + FAISS</div>',
        unsafe_allow_html=True
    )

# ─── Main Area ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>📚 RAG Study Assistant</h1>
    <p>Ask questions · Get answers strictly from your document</p>
</div>
""", unsafe_allow_html=True)

# ─── Guard: API Key ────────────────────────────────────────────────────────────
if not api_key:
    st.info("🔑 Enter your **Gemini API Key** in the sidebar to get started.")
    st.stop()

# ─── Guard: Document ───────────────────────────────────────────────────────────
if not st.session_state.pdf_processed:
    st.info("📄 Upload and **Process a PDF document** from the sidebar first.")
    st.stop()

# ─── Chat History Display ──────────────────────────────────────────────────────
for item in st.session_state.chat_history:
    with st.chat_message("user"):
        st.write(item["question"])
    with st.chat_message("assistant"):
        answer = item["answer"]
        sources = item.get("sources", [])

        if answer.strip().lower() == "not found":
            st.markdown(
                f'''<div class="not-found-box">
                    <h4>⚠ Not Found in Document</h4>
                    <div class="answer-text">The answer to this question was not found in the uploaded document.</div>
                </div>''',
                unsafe_allow_html=True
            )
        else:
            source_badges = "".join(
                f'<span class="source-badge">📄 {s}</span>' for s in sources
            )
            st.markdown(
                f'''<div class="answer-box">
                    <h4>✦ Answer</h4>
                    <div class="answer-text">{answer}</div>
                </div>''',
                unsafe_allow_html=True
            )
            if sources:
                st.markdown(
                    f'<div style="margin-top:0.75rem;">{source_badges}</div>',
                    unsafe_allow_html=True
                )

# ─── Query Input ───────────────────────────────────────────────────────────────
question = st.chat_input("Ask a question about your document...")

if question:
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching document and generating answer..."):
            try:
                result = answer_query(question, api_key)
                answer = result["answer"]
                sources = result.get("sources", [])

                if answer.strip().lower() == "not found":
                    st.markdown(
                        f'''<div class="not-found-box">
                            <h4>⚠ Not Found in Document</h4>
                            <div class="answer-text">The answer to this question was not found in the uploaded document.</div>
                        </div>''',
                        unsafe_allow_html=True
                    )
                else:
                    source_badges = "".join(
                        f'<span class="source-badge">📄 {s}</span>' for s in sources
                    )
                    st.markdown(
                        f'''<div class="answer-box">
                            <h4>✦ Answer</h4>
                            <div class="answer-text">{answer}</div>
                        </div>''',
                        unsafe_allow_html=True
                    )
                    if sources:
                        st.markdown(
                            f'<div style="margin-top:0.75rem;">{source_badges}</div>',
                            unsafe_allow_html=True
                        )

                with st.expander("🔍 View Retrieved Context"):
                    st.code(result.get("context_used", ""), language="text")

            except FileNotFoundError as e:
                st.error(f"❌ {str(e)}")
                answer, sources = "Error", []
            except Exception as e:
                st.error(f"❌ Unexpected error: {str(e)}")
                answer, sources = "Error", []

    st.session_state.chat_history.append({
        "question": question,
        "answer": answer,
        "sources": sources
    })
