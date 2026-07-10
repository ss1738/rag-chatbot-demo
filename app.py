"""
RAG Chatbot, answer questions from your own documents, with source citations.

A clean demo of a production RAG pipeline: upload docs (PDF/TXT/MD) or use the sample set,
ask questions, and get answers grounded in the content with the exact sources shown.

Run:  streamlit run app.py
"""
import os
from pathlib import Path

import streamlit as st

from rag import RAGIndex, read_document, answer

st.set_page_config(page_title="RAG Chatbot", page_icon="🤖", layout="wide")

SAMPLE_DIR = Path(__file__).parent / "sample_docs"


@st.cache_resource(show_spinner=False)
def build_index(doc_key: tuple, docs: tuple) -> RAGIndex:
    """Cached index build. doc_key changes when the document set changes."""
    idx = RAGIndex()
    idx.add_documents(list(docs))
    return idx


def load_sample_docs() -> list[tuple[str, str]]:
    docs = []
    for p in sorted(SAMPLE_DIR.glob("*")):
        if p.is_file():
            docs.append((p.read_text(encoding="utf-8", errors="ignore"), p.name))
    return docs


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.title("🤖 RAG Chatbot")
    st.caption("Answers grounded in your documents, with the sources cited.")

    st.subheader("1. Model")
    provider = st.selectbox("LLM provider", ["Groq", "OpenAI", "Retrieval only (no key)"])
    key_env = {"Groq": "GROQ_API_KEY", "OpenAI": "OPENAI_API_KEY"}.get(provider)
    api_key = ""
    if key_env:
        api_key = st.text_input(f"{provider} API key", type="password",
                                value=os.environ.get(key_env, "")
                                or st.secrets.get(key_env, "") if hasattr(st, "secrets") else "")
        st.caption("Groq offers a free key at console.groq.com, fastest way to try it.")

    st.subheader("2. Knowledge base")
    uploaded = st.file_uploader("Upload PDF / TXT / MD", type=["pdf", "txt", "md"],
                                accept_multiple_files=True)
    use_samples = st.checkbox("Use the sample documents", value=not uploaded)
    top_k = st.slider("Passages to retrieve", 2, 8, 4)

# ---------------------------------------------------------------- build the KB
docs: list[tuple[str, str]] = []
if uploaded:
    for f in uploaded:
        docs.append((read_document(f.name, f.getvalue()), f.name))
if use_samples or not docs:
    docs += load_sample_docs()

doc_key = tuple(sorted(name for _, name in docs))
index = build_index(doc_key, tuple(docs)) if docs else None
n_chunks = len(index.passages) if index else 0

# ---------------------------------------------------------------- main
st.markdown("### Chat with your documents")
st.caption(f"Indexed **{len(docs)} document(s)** into **{n_chunks} searchable passages**. "
           "Ask a question and every answer is grounded in the sources shown.")

if "history" not in st.session_state:
    st.session_state.history = []

for role, content, sources in st.session_state.history:
    with st.chat_message(role):
        st.markdown(content)
        if sources:
            with st.expander(f"Sources ({len(sources)})"):
                for i, (p, score) in enumerate(sources):
                    st.markdown(f"**[{i + 1}] {p.source}** · match {score:.2f}")
                    st.caption(p.text[:500] + ("..." if len(p.text) > 500 else ""))

prompt = st.chat_input("Ask something about the documents...")
if prompt:
    st.session_state.history.append(("user", prompt, None))
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        if not index:
            st.markdown("Add a document or enable the sample docs in the sidebar first.")
        else:
            with st.spinner("Retrieving + answering..."):
                retrieved = index.retrieve(prompt, k=top_k)
                key = api_key if provider in ("Groq", "OpenAI") else ""
                prov = provider if provider in ("Groq", "OpenAI") else "none"
                reply = answer(prompt, retrieved, prov, key)
            st.markdown(reply)
            with st.expander(f"Sources ({len(retrieved)})"):
                for i, (p, score) in enumerate(retrieved):
                    st.markdown(f"**[{i + 1}] {p.source}** · match {score:.2f}")
                    st.caption(p.text[:500] + ("..." if len(p.text) > 500 else ""))
            st.session_state.history.append(("assistant", reply, retrieved))
