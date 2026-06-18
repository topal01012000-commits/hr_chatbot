import streamlit as st
from docx import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai

# ---------------- UI CONFIG ----------------
st.set_page_config(
    page_title="HR Policy Assistant",
    page_icon="🏢",
    layout="centered"
)

st.title("🏢 HR Policy Assistant")
st.caption("Ask anything about company HR policies")

# ---------------- API KEY ----------------
genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
model = genai.GenerativeModel("gemini-1.5-flash")

# ---------------- LOAD DOC ----------------
@st.cache_data
def load_text():
    doc = Document("hr_manual.docx")
    text = "\n".join([p.text for p in doc.paragraphs])
    return text

text = load_text()

# ---------------- CHUNKING ----------------
chunk_size = 1500
chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

vectorizer = TfidfVectorizer()
chunk_vectors = vectorizer.fit_transform(chunks)

# ---------------- CHAT HISTORY ----------------
if "chat" not in st.session_state:
    st.session_state.chat = []

# ---------------- INPUT BOX ----------------
user_input = st.text_input("💬 Ask your HR question:")

if user_input:

    question_vector = vectorizer.transform([user_input])
    similarities = cosine_similarity(question_vector, chunk_vectors)[0]

    best_chunk = chunks[similarities.argmax()]

    prompt = f"""
    You are an HR assistant.
    Answer ONLY using the context below.

    Context:
    {best_chunk}

    Question:
    {user_input}
    """

    response = model.generate_content(prompt)

    answer = response.text

    # save chat
    st.session_state.chat.append(("You", user_input))
    st.session_state.chat.append(("AI", answer))

# ---------------- CHAT DISPLAY ----------------
for role, msg in st.session_state.chat:
    if role == "You":
        st.markdown(f"**🧑 You:** {msg}")
    else:
        st.markdown(f"**🤖 HR Bot:** {msg}")
        st.divider()

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.header("📌 Instructions")
    st.write("""
    - Ask questions about HR policies  
    - Example: leave policy, dress code  
    - AI will search HR manual and answer  
    """)

    if st.button("🧹 Clear Chat"):
        st.session_state.chat = []