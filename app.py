import streamlit as st
import requests
import PyPDF2
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

stored_chunks = []
stored_embeddings = []


def extract_text_from_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text


def chunk_text(text, chunk_size=800):
    sentences = text.split(".")
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) < chunk_size:
            current_chunk += sentence + "."
        else:
            chunks.append(current_chunk)
            current_chunk = sentence + "."

    if current_chunk:
        chunks.append(current_chunk)

    return chunks



def generate_answer(context, question):

    prompt = f"""
You are an AI tutor helping students revise concepts from their study material.

Use ONLY the information provided in the context.

Context:
{context}

Question:
{question}

Create a clear study flashcard.

Rules:
- If the answer is not present in the context, say: "Answer not found in the uploaded material."
- Do not leave any section empty.

Return STRICTLY in this format:

Definition:
Explanation:
Key Points:
- point
- point
- point
"""

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "mistral",
            "prompt": prompt,
            "stream": False
        }
    )

    result = response.json()
    return result["response"]

st.title("📚 Student Flashcards Generator")

uploaded_file = st.file_uploader("Upload your PDF", type=["pdf"])

if uploaded_file:
    text = extract_text_from_pdf(uploaded_file)

    stored_chunks = chunk_text(text)
    stored_embeddings = embedding_model.encode(stored_chunks)

    st.success("PDF uploaded successfully! Ask a question below.")

question = st.text_input("Enter your question")

if st.button("Generate Flashcard"):

    if not stored_chunks:
        st.warning("Please upload a PDF first.")
    else:

        question_embedding = embedding_model.encode([question])

        similarities = cosine_similarity(
            question_embedding, stored_embeddings
        )[0]

        best_index = np.argmax(similarities)

        best_context = stored_chunks[best_index]

        answer = generate_answer(best_context, question)

        try:
            definition = answer.split("Explanation:")[0].replace("Definition:", "").strip()
            explanation = answer.split("Explanation:")[1].split("Key Points:")[0].strip()
            keypoints = answer.split("Key Points:")[1].strip().split("\n")
        except:
            definition = answer
            explanation = ""
            keypoints = []

        st.subheader("Definition")
        st.write(definition)

        st.subheader("Explanation")
        st.write(explanation)

        st.subheader("Key Points")

        for point in keypoints:
            if point.strip():
                st.write(point)