import os
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY

def generate_pdf(output_path="data/temp.pdf"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    doc = SimpleDocTemplate(output_path, pagesize=letter,
                            rightMargin=72, leftMargin=72,
                            topMargin=72, bottomMargin=72)
    
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        name='PaperTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=24,
        leading=28,
        alignment=TA_CENTER,
        spaceAfter=20
    )
    
    author_style = ParagraphStyle(
        name='PaperAuthor',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=11,
        alignment=TA_CENTER,
        spaceAfter=30
    )
    
    h1_style = ParagraphStyle(
        name='SectionHeading',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=14,
        leading=18,
        spaceBefore=15,
        spaceAfter=10
    )
    
    body_style = ParagraphStyle(
        name='BodyTextCustom',
        parent=styles['BodyText'],
        fontName='Helvetica',
        fontSize=10.5,
        leading=15,
        alignment=TA_JUSTIFY,
        spaceAfter=12
    )

    story = []
    
    # --- Page 1: Title and Introduction ---
    story.append(Paragraph("Retrieval-Augmented Generation (RAG) in Modern Education", title_style))
    story.append(Paragraph("Dr. Bhavanish Mantri<br/>Department of Artificial Intelligence, EduTech University", author_style))
    story.append(Spacer(1, 20))
    
    story.append(Paragraph("Abstract", h1_style))
    story.append(Paragraph(
        "This paper explores the design, implementation, and evaluation of Retrieval-Augmented Generation (RAG) "
        "systems tailored for academic environments. Traditional Large Language Models (LLMs) suffer from hallucinations "
        "and lack access to private, course-specific documents. RAG solves this by integrating a vector search-based "
        "retrieval pipeline with generative LLMs, ensuring that answers are grounded strictly in the provided reading material. "
        "We discuss the architecture of a study assistant and evaluate its accuracy using the Ragas framework.",
        body_style
    ))
    
    story.append(Paragraph("1. Introduction", h1_style))
    story.append(Paragraph(
        "In recent years, generative artificial intelligence has transformed educational technologies. However, "
        "deploying language models in classrooms requires strict guardrails. Students need factual, course-approved answers, "
        "and models must avoid fabricating information. Retrieval-Augmented Generation (RAG) has emerged as the standard "
        "architecture to address this challenge. By searching a database of document chunks for relevant information before "
        "generating a response, RAG ensures that the output is directly traceable to the source texts.",
        body_style
    ))
    
    story.append(PageBreak())
    
    # --- Page 2: Architecture ---
    story.append(Paragraph("2. System Architecture", h1_style))
    story.append(Paragraph(
        "A typical educational RAG pipeline consists of three core phases: ingestion, retrieval, and generation. "
        "During the ingestion phase, textbooks or lecture notes in PDF format are loaded and split into smaller "
        "overlapping chunks (typically 500 characters with 50 characters overlap). These chunks are then converted into "
        "dense vector representations using an embedding model such as the 'all-MiniLM-L6-v2' model, which operates "
        "on 384 dimensions.",
        body_style
    ))
    story.append(Paragraph(
        "The resulting vectors are indexed in a high-performance vector database like FAISS (Facebook AI Similarity Search) "
        "for fast nearest-neighbor retrieval. When a student poses a question, the retriever computes the similarity "
        "between the query's embedding and all document chunk vectors. The top-k (e.g., k=3) chunks with the highest "
        "cosine similarity are retrieved.",
        body_style
    ))
    story.append(Paragraph(
        "Finally, the generation phase constructs a prompt combining the student's question and the retrieved chunks as context. "
        "This prompt is passed to Google's Gemini LLM (such as gemini-2.5-flash) with a strict instruction to answer only "
        "using the context and return 'Not found' if the answer cannot be verified. This completely eliminates "
        "out-of-bounds hallucinations.",
        body_style
    ))
    
    story.append(PageBreak())
    
    # --- Page 3: Evaluation Metrics ---
    story.append(Paragraph("3. Evaluation and Ragas Metrics", h1_style))
    story.append(Paragraph(
        "Evaluating educational RAG systems is critical to ensure safety and accuracy. The Ragas (Retrieval Augmented "
        "Generation Assessment) framework provides model-based evaluation metrics that assess both the retrieval "
        "and generation components of the system.",
        body_style
    ))
    story.append(Paragraph(
        "The evaluation utilizes four main metrics. First, Faithfulness measures the extent to which the generated answer "
        "is grounded in the retrieved context. It is calculated by dividing the number of statements in the answer "
        "that can be inferred from the context by the total number of statements in the answer.",
        body_style
    ))
    story.append(Paragraph(
        "Second, Answer Relevancy measures how relevant the generated answer is to the user's question, penalizing "
        "redundant or incomplete statements. It is calculated using the average cosine similarity between the "
        "user's question and a set of generated questions based on the generated answer.",
        body_style
    ))
    story.append(Paragraph(
        "Third, Context Precision evaluates whether the most relevant information is placed at the top of the retrieved chunks. "
        "Fourth, Context Recall measures whether all the necessary information needed to answer the question is successfully "
        "retrieved, comparing the retrieved context against a human-provided ground truth answer.",
        body_style
    ))
    
    doc.build(story)
    print(f"Sample PDF successfully generated at: {output_path}")

if __name__ == "__main__":
    generate_pdf()
