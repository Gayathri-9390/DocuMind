"""
app.py
Streamlit UI for DocuMind - an AI Document Q&A Assistant (RAG system).

Run with: streamlit run app.py
"""

import os
import tempfile
import streamlit as st
from dotenv import load_dotenv
from rag_pipeline import load_and_chunk_pdfs, build_vectorstore, get_qa_chain, ask_question

load_dotenv()  # reads variables from a local .env file, if present

st.set_page_config(page_title="DocuMind - AI Document Q&A", page_icon="📄", layout="wide")

st.title("📄 DocuMind — AI Document Q&A Assistant")
st.caption("Upload PDFs, ask questions, get answers grounded in your documents (RAG-powered).")

# ---------- Sidebar: setup ----------
with st.sidebar:
    st.header("⚙️ Setup")
    default_key = os.getenv("GROQ_API_KEY", "")
    if default_key:
        # Key is set via environment — don't expose the field publicly
        groq_api_key = default_key
        st.success("API key loaded ✅", icon="🔑")
    else:
        groq_api_key = st.text_input(
            "Groq API Key",
            type="password",
            help="Get a free key at https://console.groq.com/keys",
        )
    st.divider()
    uploaded_files = st.file_uploader(
        "Upload PDF document(s)",
        type=["pdf"],
        accept_multiple_files=True,
        key="pdf_uploader",
    )
    process_btn = st.button("🔄 Process Documents", use_container_width=True)

# ---------- Session state ----------
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ---------- Process documents ----------
if process_btn:
    if not groq_api_key:
        st.sidebar.error("Please enter your Groq API key.")
    elif not uploaded_files:
        st.sidebar.error("Please upload at least one PDF.")
    else:
        with st.spinner("Reading, chunking, and embedding your documents..."):
            temp_paths = []
            temp_dir = tempfile.mkdtemp()
            for f in uploaded_files:
                # Keep the original filename so source citations are readable
                safe_path = os.path.join(temp_dir, f.name)
                with open(safe_path, "wb") as out:
                    out.write(f.read())
                temp_paths.append(safe_path)

            chunks = load_and_chunk_pdfs(temp_paths)
            vectorstore = build_vectorstore(chunks)
            st.session_state.qa_chain = get_qa_chain(vectorstore, groq_api_key)

            for p in temp_paths:
                os.unlink(p)
            os.rmdir(temp_dir)

        st.sidebar.success(f"Processed {len(uploaded_files)} document(s) into {len(chunks)} chunks ✅")

# ---------- Chat interface ----------
st.divider()

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📚 Sources used"):
                for i, src in enumerate(msg["sources"], 1):
                    st.markdown(f"**Source {i}** — `{src.metadata.get('source_file', 'unknown')}` (page {src.metadata.get('page', '?')})")
                    st.text(src.page_content[:300] + "...")

question = st.chat_input("Ask a question about your uploaded document(s)...")

if question:
    if st.session_state.qa_chain is None:
        st.error("Please upload and process a document first (use the sidebar).")
    else:
        st.session_state.chat_history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer, sources = ask_question(st.session_state.qa_chain, question)
                st.markdown(answer)
                if sources:
                    with st.expander("📚 Sources used"):
                        for i, src in enumerate(sources, 1):
                            st.markdown(f"**Source {i}** — `{src.metadata.get('source_file', 'unknown')}` (page {src.metadata.get('page', '?')})")
                            st.text(src.page_content[:300] + "...")

        st.session_state.chat_history.append({"role": "assistant", "content": answer, "sources": sources})