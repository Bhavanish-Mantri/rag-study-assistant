"""
main.py - Core RAG pipeline using Gemini API (google-genai SDK)
"""

from google import genai
from google.genai import types
from retriever import get_relevant_chunks, format_context

# ─── Configuration ────────────────────────────────────────────────────────────
GEMINI_MODEL = "gemini-2.5-flash"

STRICT_PROMPT_TEMPLATE = """You are a strict document question-answering assistant.

RULES (MANDATORY - DO NOT BREAK):
1. Answer ONLY using the context provided below.
2. Do NOT use any external knowledge, training data, or assumptions.
3. If the answer cannot be found in the context, respond with exactly: Not found
4. Do NOT make up information. Do NOT infer beyond what is explicitly stated.
5. Keep your answer concise and directly based on the context.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER (strictly from context only):"""


def answer_query(question: str, api_key: str) -> dict:
    """
    Full RAG pipeline: retrieve relevant chunks, build prompt, call Gemini.

    Args:
        question: User's natural language question.
        api_key: Google Gemini API key.

    Returns:
        Dict with keys: 'answer', 'sources', 'context_used'
    """
    # Step 1: Configure Gemini client
    client = genai.Client(api_key=api_key)

    # Step 2: Retrieve top-k relevant chunks
    docs = get_relevant_chunks(question, k=3)

    if not docs:
        return {
            "answer": "Not found",
            "sources": [],
            "context_used": ""
        }

    # Step 3: Format context from chunks
    context = format_context(docs)

    # Step 4: Build strict prompt
    prompt = STRICT_PROMPT_TEMPLATE.format(
        context=context,
        question=question
    )

    # Step 5: Call Gemini model
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.0,
            max_output_tokens=512,
        )
    )

    answer = response.text.strip()

    # Step 6: Collect source metadata
    sources = []
    for doc in docs:
        page = doc.metadata.get("page", "N/A")
        sources.append(f"Page {page}")

    return {
        "answer": answer,
        "sources": list(dict.fromkeys(sources)),
        "context_used": context
    }