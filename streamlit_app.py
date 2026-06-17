import streamlit as st
from docx import Document
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai

# ----------------------------
# CONFIG (STREAMLIT CLOUD SAFE)
# ----------------------------
genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
model_llm = genai.GenerativeModel("gemini-2.5-flash")

# ----------------------------
# LOAD & CACHE DATA
# ----------------------------
@st.cache_resource
def load_models_and_data():
    # Load HR document
    doc = Document("hr_manual.docx")
    text = "\n".join([p.text for p in doc.paragraphs])

    # Chunking
    chunk_size = 2000
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

    # Embeddings model
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")

    # Precompute embeddings (VERY IMPORTANT FOR SPEED)
    embeddings = embed_model.encode(chunks)

    return chunks, embeddings, embed_model

chunks, chunk_embeddings, embed_model = load_models_and_data()

# ----------------------------
# UI
# ----------------------------
st.set_page_config(page_title="HR Chatbot", layout="centered")

st.title("🏢 HR Policy Chatbot")
st.write("Ask any HR-related question from your company handbook.")

# ----------------------------
# INPUT
# ----------------------------
question = st.text_input("Enter your question:")

# ----------------------------
# PROCESS
# ----------------------------
if question:

    with st.spinner("Searching HR manual..."):

        # Embed question
        q_embedding = embed_model.encode([question])

        # Similarity search
        scores = cosine_similarity(q_embedding, chunk_embeddings)[0]

        # Top 3 chunks
        top_indices = scores.argsort()[-3:][::-1]
        top_chunks = [chunks[i] for i in top_indices]

        context = "\n\n---\n\n".join(top_chunks)

        # Prompt
        prompt = f"""
You are an HR assistant for a company.

RULES:
- Answer ONLY using HR manual context.
- If not present, say "Not mentioned in HR policy".
- Be clear and professional.

HR CONTEXT:
{context}

QUESTION:
{question}
"""

        # Gemini response
        response = model_llm.generate_content(prompt)

    st.subheader("Answer")
    st.write(response.text)

    # Optional: show sources (debug mode)
    with st.expander("Show retrieved HR sections"):
        for i, chunk in enumerate(top_chunks, 1):
            st.markdown(f"**Section {i}:**")
            st.write(chunk)