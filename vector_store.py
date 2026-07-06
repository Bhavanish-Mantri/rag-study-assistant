"""
vector_store.py - Embedding generation and FAISS vector store management
"""

import os
import shutil
import stat
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

FAISS_INDEX_PATH = "faiss_index"


_embeddings_instance = None


def get_embeddings() -> HuggingFaceEmbeddings:
    """
    Initialize and return HuggingFace sentence-transformers embeddings.
    """
    global _embeddings_instance
    if _embeddings_instance is None:
        _embeddings_instance = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
    return _embeddings_instance



def create_vector_store(chunks: list) -> FAISS:
    """
    Create a FAISS vector store from document chunks.
    Deletes any existing index before creating a new one.

    Args:
        chunks: List of Document objects.

    Returns:
        FAISS vector store instance.
    """
    def handle_remove_readonly(func, path, exc):
        os.chmod(path, stat.S_IWRITE)
        func(path)

    # Delete old index if it exists
    if os.path.exists(FAISS_INDEX_PATH):
        shutil.rmtree(FAISS_INDEX_PATH, onerror=handle_remove_readonly)

    embeddings = get_embeddings()
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local(FAISS_INDEX_PATH)
    return vector_store


def load_vector_store() -> FAISS:
    """
    Load FAISS index from disk.

    Returns:
        FAISS vector store instance.

    Raises:
        FileNotFoundError: If the FAISS index does not exist.
    """
    if not os.path.exists(FAISS_INDEX_PATH):
        raise FileNotFoundError(
            "FAISS index not found. Please upload and process a PDF first."
        )

    embeddings = get_embeddings()
    vector_store = FAISS.load_local(
        FAISS_INDEX_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )
    return vector_store