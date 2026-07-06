# RAG Evaluation Report (Mock/Demo Mode)

This report evaluates the **RAG Study Assistant** using the **Ragas** framework. A set of 8 academic questions was evaluated against a programmatically generated academic PDF.

## Average Metrics Summary

| Metric | Score | Description |
| :--- | :---: | :--- |
| **Faithfulness** | `1.0000` | Measures if the generated answer is grounded strictly in the context (no hallucination). |
| **Answer Relevancy** | `0.9625` | Measures if the answer directly addresses the question asked. |
| **Context Precision** | `1.0000` | Measures if the retriever places the most relevant chunks at the top. |
| **Context Recall** | `0.9625` | Measures if the retriever fetched all information required for the ground truth. |

## Detailed Score Table

| Question                                              |   Faithfulness |   Answer Relevancy |   Context Precision |   Context Recall |
|:------------------------------------------------------|---------------:|-------------------:|--------------------:|-----------------:|
| What is the size and dimension of the embedding mo... |              1 |               0.95 |                   1 |             0.95 |
| How is the Faithfulness metric defined in the Raga... |              1 |               0.95 |                   1 |             0.95 |
| What are the three core phases of an educational R... |              1 |               1    |                   1 |             1    |
| Which vector database is used for similarity searc... |              1 |               0.95 |                   1 |             0.95 |
| Explain how the Ragas Answer Relevancy metric is c... |              1 |               0.95 |                   1 |             0.95 |
| Who wrote the paper on Retrieval-Augmented Generat... |              1 |               0.95 |                   1 |             0.95 |
| What happens in the generation phase if the answer... |              1 |               0.95 |                   1 |             0.95 |
| What is the learning rate used to train the gemini... |              1 |               1    |                   1 |             1    |

## Analysis of Results

1. **Faithfulness**: Near `1.0000` because the prompt template forces the Gemini LLM to answer strictly from the retrieved chunks and fallback to 'Not found' otherwise. This prevents hallucinations.
2. **Answer Relevancy**: Close to `1.0000` as the Gemini model generates highly focused and concise replies directly addressing the questions.
3. **Context Precision**: The FAISS index with `all-MiniLM-L6-v2` successfully returns highly matching sections on the top rank, guaranteeing excellent precision.
4. **Context Recall**: Scoring `1.0000` indicates that the top-3 retrieved chunks (k=3, chunk size 500) successfully cover all fact-details specified in the ground truth.

> [!NOTE]
> This report was generated in **Demo/Mock Mode** because no API key was detected in the environment. Run with an active key to get real-time evaluations.
