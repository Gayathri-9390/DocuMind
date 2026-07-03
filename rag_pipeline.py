"""
rag_pipeline.py
Core RAG logic: load PDFs -> chunk -> embed -> store in ChromaDB -> retrieve -> answer via Groq LLM.
Supports conversation memory via ConversationalRetrievalChain.
"""

import os
import uuid
import chromadb
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate

# ---------- CONFIG ----------
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "llama-3.1-8b-instant"

# Prompt for combining retrieved context + conversation history into an answer
QA_PROMPT = PromptTemplate(
    template="""You are a helpful assistant answering questions using ONLY the context below.
Always extract and state SPECIFIC facts, numbers, or figures from the context when asked
(e.g. percentages, metrics, names, dates). If the answer is not in the context, say
"I couldn't find that in the uploaded document(s)."
Be concise and accurate.

Context:
{context}

Question: {question}

Answer:""",
    input_variables=["context", "question"],
)


def load_and_chunk_pdfs(pdf_paths: list[str]):
    """Load one or more PDFs and split into overlapping text chunks."""
    all_docs = []
    for path in pdf_paths:
        loader = PyPDFLoader(path)
        docs = loader.load()
        for d in docs:
            d.metadata["source_file"] = os.path.basename(path)
        all_docs.extend(docs)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(all_docs)
    return chunks


def build_vectorstore(chunks):
    """Embed chunks into a fresh in-memory ChromaDB collection."""
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    collection_name = f"documind_{uuid.uuid4().hex[:8]}"
    chroma_client = chromadb.EphemeralClient()
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=collection_name,
        client=chroma_client,
    )
    return vectorstore


def get_qa_chain(vectorstore, groq_api_key: str, k: int = 8):
    """Build a ConversationalRetrievalChain with sliding-window memory (last 5 turns)."""
    llm = ChatGroq(
        groq_api_key=groq_api_key,
        model_name=LLM_MODEL,
        temperature=0.1,
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    # Keep last 5 question-answer pairs in memory — balances context vs token cost
    memory = ConversationBufferWindowMemory(
        k=5,
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",
    )

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": QA_PROMPT},
    )
    return qa_chain


def ask_question(qa_chain, question: str):
    """Run a question through the chain and return answer + deduplicated source chunks."""
    result = qa_chain.invoke({"question": question})
    answer = result["answer"]
    raw_sources = result.get("source_documents", [])

    # Deduplicate identical chunks
    seen = set()
    sources = []
    for src in raw_sources:
        key = (src.metadata.get("source_file"), src.metadata.get("page"), src.page_content[:50])
        if key not in seen:
            seen.add(key)
            sources.append(src)

    return answer, sources