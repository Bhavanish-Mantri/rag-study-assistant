"""
retriever.py - Retrieve top-k relevant chunks from FAISS vector store
"""

from langchain_community.vectorstores import FAISS
from vector_store import load_vector_store


def get_relevant_chunks(query: str, k: int = 3) -> list:
    """
    Retrieve the top-k most relevant document chunks for a given query.

    Args:
        query: User's question string.
        k: Number of top chunks to retrieve (default: 3).

    Returns:
        List of Document objects most relevant to the query.

    Raises:
        FileNotFoundError: If FAISS index has not been created yet.
    """
    vector_store: FAISS = load_vector_store()
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )
    docs = retriever.invoke(query)
    return docs


def format_context(docs: list) -> str:
    """
    Format retrieved documents into a single context string.

    Args:
        docs: List of Document objects.

    Returns:
        Concatenated context string.
    """
    context_parts = []
    for i, doc in enumerate(docs, 1):
        page = doc.metadata.get("page", "N/A")
        context_parts.append(
            f"[Chunk {i} | Page {page}]:\n{doc.page_content.strip()}"
        )
    return "\n\n".join(context_parts)
