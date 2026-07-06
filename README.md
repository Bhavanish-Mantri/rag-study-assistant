# 📚 RAG Study Assistant

A production-ready Retrieval Augmented Generation (RAG) system that lets you chat with any PDF — strictly answering from document content only, with zero hallucination.

---

## 🧩 Problem Statement

Large Language Models tend to hallucinate when asked domain-specific questions. This project solves that by grounding every answer strictly in the content of a user-uploaded PDF. If the answer isn't in the document, the system returns **"Not found"** — never guesses.

---

## 🏗️ Architecture

```
User uploads PDF
      │
      ▼
[ingest.py] ──► PyPDFLoader → RecursiveCharacterTextSplitter
      │              chunk_size=500, overlap=50
      ▼
[vector_store.py] ──► HuggingFaceEmbeddings (all-MiniLM-L6-v2)
      │                    └──► FAISS Index (saved to disk)
      ▼
User asks question
      │
      ▼
[retriever.py] ──► Load FAISS → Top-3 similar chunks
      │
      ▼
[main.py] ──► Strict prompt → Gemini 1.5 Flash → Answer
      │
      ▼
[app.py] ──► Streamlit UI (chat interface + source display)
```

---

## 📁 Project Structure

```
rag-study-assistant/
│
├── ingest.py          # PDF loading and chunking
├── vector_store.py    # Embeddings + FAISS index management
├── retriever.py       # Semantic search over FAISS index
├── main.py            # Gemini RAG pipeline (strict prompt)
├── app.py             # Streamlit chat UI
├── requirements.txt   # Python dependencies
├── README.md          # This file
└── data/
    └── temp.pdf       # Uploaded PDF (auto-created)
```

---

## ⚙️ Tech Stack

| Layer        | Technology                          |
|--------------|-------------------------------------|
| UI           | Streamlit                           |
| PDF Parsing  | LangChain + PyPDFLoader             |
| Chunking     | RecursiveCharacterTextSplitter      |
| Embeddings   | SentenceTransformers (MiniLM-L6-v2) |
| Vector Store | FAISS (faiss-cpu)                   |
| LLM          | Google Gemini 1.5 Flash             |
| Orchestration| LangChain + langchain-community     |

---

## 🚀 Steps to Run

### 1. Clone / Download the project

```bash
cd rag-study-assistant
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Get a Gemini API Key

- Visit: https://aistudio.google.com/app/apikey
- Create a free API key

### 5. Run the app

```bash
streamlit run app.py
```

### 6. Use the app

1. Paste your **Gemini API Key** in the sidebar
2. Upload a **PDF file**
3. Click **"Process Document"** — wait for indexing to complete
4. Type your question in the chat input
5. Get answers sourced strictly from your document

---

## 💬 Sample Input / Output

**Document:** A research paper on "Transformer Architecture in NLP"

**Question:**
```
What is the attention mechanism used in transformers?
```

**Answer:**
```
The attention mechanism in transformers computes a weighted sum of values,
where the weights are determined by the compatibility of a query with
corresponding keys using scaled dot-product attention.

📄 Page 3  📄 Page 5
```

---

**Question (not in document):**
```
What is the GDP of France?
```

**Answer:**
```
Not found
```

---

## 🔒 Anti-Hallucination Design

The system enforces strict grounding through:

1. **Temperature = 0.0** — deterministic, no creative generation
2. **Explicit prompt rules** — model is instructed to respond ONLY from context
3. **"Not found" fallback** — if context doesn't contain the answer, the model is instructed to say exactly "Not found"
4. **No general knowledge leakage** — context window contains only the retrieved chunks

---

## 🛠️ Configuration

| Parameter      | Default           | Location          |
|----------------|-------------------|-------------------|
| Chunk Size     | 500 chars         | `ingest.py`       |
| Chunk Overlap  | 50 chars          | `ingest.py`       |
| Top-K Chunks   | 3                 | `retriever.py`    |
| Embedding Model| all-MiniLM-L6-v2  | `vector_store.py` |
| Gemini Model   | gemini-1.5-flash  | `main.py`         |
| Max Tokens     | 512               | `main.py`         |
| Temperature    | 0.0               | `main.py`         |

---

## 📊 Automated Ragas Evaluation

The project includes an automated evaluation suite using the **Ragas** framework to measure system performance on key metrics:

* **Faithfulness**: Verifies if the answer is derived strictly from the text (no hallucination).
* **Answer Relevancy**: Verifies if the answer directly addresses the question.
* **Context Precision**: Verifies if the retriever places the most relevant chunks at the top.
* **Context Recall**: Verifies if all necessary information to answer the question is successfully retrieved.

### Running the Evaluation Suite

1. **Generate the Test Document**:
   Run the helper script to create a sample academic PDF in the `data/` folder:
   ```bash
   python generate_sample_pdf.py
   ```

2. **Run the Evaluation Script**:
   - **Demo / Mock Mode** (no API key required):
     ```bash
     python evaluate_ragas.py --demo
     ```
   - **Live Mode** (uses Gemini LLM as the evaluator/judge):
     ```bash
     # Windows CMD
     set GEMINI_API_KEY=your_gemini_api_key_here
     python evaluate_ragas.py
     
     # Windows PowerShell
     $env:GEMINI_API_KEY="your_gemini_api_key_here"
     python evaluate_ragas.py
     ```

3. **Check the Report**:
   The script will print a summary table to the console and save a detailed markdown report to `evaluation_report.md`.

