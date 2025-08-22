# rag.py
import os
from typing import List, Tuple
import numpy as np
from huggingface_hub import InferenceClient
from sentence_transformers import SentenceTransformer
import faiss

# Default model IDs
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GENERATION_MODEL = "facebook/opt-1.3b" 


class HyDERetriever:
    def __init__(self, hf_token: str, docs: List[str]):
        # local embedding model
        self.embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

        # HuggingFace Inference Client for generation
        self.gen_api = InferenceClient(model="facebook/opt-1.3b", token=hf_token)

        # store documents
        self.docs = docs
        self.doc_embeddings = self._embed_texts(docs)

        # build FAISS index
        self.index, self.id_to_doc = self._build_faiss(self.doc_embeddings, docs)

    def _embed_texts(self, texts: List[str]) -> np.ndarray:
        """Embed multiple texts into vectors."""
        embeddings = self.embedder.encode(
            texts, convert_to_numpy=True, normalize_embeddings=True
        )
        return np.array(embeddings, dtype=np.float32)

    def _build_faiss(self, embeddings: np.ndarray, docs: List[str]):
        """Build a FAISS index for similarity search."""
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)  # cosine similarity (after normalization)
        index.add(embeddings)
        return index, docs

    def _embed_single(self, text: str) -> np.ndarray:
        """Embed a single text into vector."""
        vec = self.embedder.encode(
            [text], convert_to_numpy=True, normalize_embeddings=True
        )
        return np.array(vec, dtype=np.float32)

    def _generate_hyde(self, query: str, max_length: int = 128) -> str:
        """Generate hypothetical answer using generation model."""
        prompt = (
            f"Draft a concise helpful answer (few sentences) "
            f"to the user's question:\n\nQuestion: {query}\n\nAnswer:"
        )

        text = self.gen_api.text_generation(
            prompt,
            max_new_tokens=max_length,
            do_sample=False,
        )
        return text.strip()

    def retrieve(self, query: str, top_k: int = 3) -> Tuple[List[Tuple[str, float]], str]:
        """Retrieve relevant docs based on HyDE-generated query."""
        # 1) HyDE: generate hypothetical answer
        hyde_text = self._generate_hyde(query)

        # 2) Embed the HyDE text
        hyde_vec = self._embed_single(hyde_text)

        # 3) Search in FAISS
        D, I = self.index.search(hyde_vec, top_k)
        results = []
        for score, idx in zip(D[0], I[0]):
            if idx < 0:
                continue
            results.append((self.id_to_doc[idx], float(score)))
        return results, hyde_text

    def final_answer(self, query: str, retrieved: List[str], max_length: int = 200) -> str:
        """Generate final answer using retrieved docs as context."""
        context = "\n\n---\n\n".join(retrieved)
        prompt = (
            f"Use the following context to answer the question as helpfully and concisely as possible.\n\n"
            f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
        )

        text = self.gen_api.text_generation(
            prompt,
            max_new_tokens=max_length,
            do_sample=False,
        )
        return text.strip()
