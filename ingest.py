"""
ingest.py - PDF loading and text chunking
"""

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_and_split_pdf(pdf_path: str) -> list:
    """
    Load a PDF file and split it into chunks.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        List of Document chunks.
    """
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", " ", ""]
    )

    chunks = splitter.split_documents(documents)
    return chunks
