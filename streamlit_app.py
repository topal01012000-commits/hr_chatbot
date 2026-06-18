import streamlit as st
from docx import Document
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai
import os
from dotenv import load_dotenv

# -------------------
# Load API Key
# -------------------
load_dotenv("key.env")

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model_llm = genai.GenerativeModel("gemini-2.5-flash")

# -------------------
# Load HR Manual
# -------------------
@st.cache_data
def load_data():
    doc = Document("hr_manual.docx")
    text = "\n".join([p.text for p in doc.paragraphs])

    chunk_size = 2000
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = embed_model.encode(chunks)

    return chunks, embeddings, embed_model

chunks, chunk_embeddings, embed_model = load_data()

# -------------------
# UI
# -------------------
st.title("🏢 HR Policy Chatbot")
st.write("Ask any question related to HR policies")

question = st.text_input("Enter your question:")

if question:

    q_embedding = embed_model.encode([question])

    scores = cosine_similarity(q_embedding, chunk_embeddings)[0]

    top_indices = scores.argsort()[-3:][::-1]
    top_chunks = [chunks[i] for i in top_indices]

    context = "\n\n---\n\n".join(top_chunks)

    prompt = f"""
You are an HR assistant.

RULES:
- Answer ONLY using HR manual content.
- If not found, say "Not mentioned in HR policy".

HR CONTENT:
{context}

QUESTION:
{question}
"""

    response = model_llm.generate_content(prompt)

    st.subheader("Answer")
    st.write(response.text)