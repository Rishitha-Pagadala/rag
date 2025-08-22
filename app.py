# app.py
import streamlit as st
from rag import HyDERetriever
import os

st.set_page_config(page_title="HyDE + RAG demo", layout="centered")

st.title("HyDE + RAG demo")

# get HF token from Streamlit secrets
try:
    HF_TOKEN = st.secrets["HUGGINGFACE_TOKEN"]
except Exception:
    st.error("HUGGINGFACE_TOKEN not found in Streamlit secrets. Add it in your app settings.")
    st.stop()

# load docs from docs/ directory
@st.cache_resource
def load_docs():
    docs_path = os.path.join(os.path.dirname(__file__), "docs")
    docs = []
    if not os.path.exists(docs_path):
        return docs
    for fn in os.listdir(docs_path):
        if fn.endswith(".txt"):
            with open(os.path.join(docs_path, fn), "r", encoding="utf-8") as f:
                docs.append(f.read())
    return docs

docs = load_docs()
if not docs:
    st.error("No docs found in docs/. Add some .txt files to the docs/ folder in the repo.")
    st.stop()

# Build retriever once (cached)
@st.cache_resource
def make_retriever(token, docs):
    return HyDERetriever(hf_token=token, docs=docs)

retriever = make_retriever(HF_TOKEN, docs)

query = st.text_input("Ask a question about the docs", "")

if st.button("Search"):
    if not query.strip():
        st.warning("Please type a question.")
    else:
        with st.spinner("Running HyDE + retrieval..."):
            results, hyde = retriever.retrieve(query, top_k=3)
            st.subheader("HyDE (hypothetical answer)")
            st.write(hyde)

            st.subheader("Retrieved documents (score = similarity)")
            retrieved_texts = []
            for doc, score in results:
                st.write(f"**score**: {score:.4f}")
                st.code(doc[:1000] + ("..." if len(doc) > 1000 else ""))
                retrieved_texts.append(doc)

            st.subheader("Final answer (synthesized)")
            final = retriever.final_answer(query, retrieved_texts)
            st.write(final)
