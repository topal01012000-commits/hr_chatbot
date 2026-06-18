import streamlit as st
from docx import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from google import genai

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="HR Policy Assistant",
    page_icon="🏢",
    layout="centered"
)

st.title("🏢 HR Policy Assistant")
st.write("Ask anything about HR policies and get instant answers.")

# ---------------- API KEY (Streamlit Secrets) ----------------
client = genai.Client(api_key=st.secrets["GOOGLE_API_KEY"])

# ---------------- LOAD HR DOCUMENT ----------------
@st.cache_data
def load_doc():
    doc = Document("hr_manual.docx")
    text = "\n".join([p.text for p in doc.paragraphs if p.text.strip() != ""])
    return text

text = load_doc()

# ---------------- CHUNKING ----------------
chunk_size = 1500
chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

# ---------------- TF-IDF VECTOR SEARCH ----------------
vectorizer = TfidfVectorizer()
chunk_vectors = vectorizer.fit_transform(chunks)

# ---------------- CHAT HISTORY ----------------
if "chat" not in st.session_state:
    st.session_state.chat = []

# ---------------- INPUT ----------------
user_question = st.text_input("💬 Ask your HR question:")

if user_question:

    question_vec = vectorizer.transform([user_question])
    scores = cosine_similarity(question_vec, chunk_vectors)[0]
    best_chunk = chunks[scores.argmax()]

    prompt = f"""
You are a professional HR assistant.

Rules:
- Answer ONLY from the given HR policy context
- If answer is not found, say "Not mentioned in HR manual"
- Keep answer simple and professional

Context:
{best_chunk}

Question:
{user_question}
"""

    # ---------------- GEMINI RESPONSE ----------------
    response = client.models.generate_content(
        model="gemini-1.5-flash-latest"
        contents=prompt
    )

    answer = response.text

    # save chat
    st.session_state.chat.append(("You", user_question))
    st.session_state.chat.append(("HR Bot", answer))

# ---------------- CHAT UI ----------------
st.divider()

for role, msg in st.session_state.chat:
    if role == "You":
        st.markdown(f"🧑 **You:** {msg}")
    else:
        st.markdown(f"🤖 **HR Bot:** {msg}")
        st.markdown("---")

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.header("📌 HR Assistant Guide")

    st.info("""
    ✔ Ask HR policy questions  
    ✔ Example: leave policy, dress code  
    ✔ Answers come from HR manual  
    """)

    if st.button("🧹 Clear Chat"):
        st.session_state.chat = []