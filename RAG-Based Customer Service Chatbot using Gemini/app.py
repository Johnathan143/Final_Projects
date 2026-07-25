import os
import streamlit as st

from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import pickle
from rag_pipeline import ask_chatbot

# PAGE CONFIG
st.set_page_config(
    page_title="AI Customer Support Chatbot",
    page_icon="🤖",
    layout="wide"
)
st.markdown("""
<style>
/* Hide Streamlit UI */
#MainMenu {visibility:hidden;}
footer {visibility:hidden;}
header {visibility:hidden;}
/* Background */
.stApp{
    background:#202123;
}
/* Main container */
.block-container{
    padding-top:2rem;
    max-width:1100px;
}
/* Sidebar */
[data-testid="stSidebar"]{
    background:#171717;
}
/* Chat bubbles */
[data-testid="stChatMessage"]{
    border-radius:18px;
    padding:18px;
    margin-bottom:15px;
    border:none;
}
/* Assistant */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]){
    background:#2A2B32;
}
/* User */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]){
    background:#343541;
}
/* Input */
[data-testid="stChatInput"]{
    border-radius:25px;
}
/* Buttons */
.stButton>button{
    width:100%;
    border-radius:16px;
    background:#2A2B32;
    color:white;
    border:1px solid #444;
    padding:15px;
    transition:0.3s;
}
.stButton>button:hover{
    background:#3B3C43;
    border-color:#10A37F;

}
</style>
""", unsafe_allow_html=True)


st.markdown("""
<h1 style="text-align:center;">
🤖 AI Customer Support Assistant
</h1>

<p style="text-align:center;color:#B0B0B0;">
Powered by RAG • NLP • FAISS • Gemini
</p>
""", unsafe_allow_html=True)

# CHAT HISTORY
if "messages" not in st.session_state:
    st.session_state.messages = []

    st.markdown(
        "<h2 style='text-align:center;'>👋 Welcome!</h2>",
        unsafe_allow_html=True)
    st.markdown(
        "<p style='text-align:center;'>How can I help you today?</p>",
        unsafe_allow_html=True
    )
    st.write("I'm your AI Customer Support Assistant.")
    st.write("You can ask me about:")

col1,col2=st.columns(2)

with col1:

    st.button("📦 Track my Order")

    st.button("💰 Refund Policy")

    st.button("💳 Payment Methods")

with col2:

    st.button("🚚 Shipping")

    st.button("👤 Account")

    st.button("🛠 Technical Support")




# LOAD API KEY
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if GOOGLE_API_KEY is None:
    st.error("Google API Key not found.")
    st.stop()

# LOAD NLP PIPELINE
@st.cache_resource
def load_nlp_pipeline():
    with open("data/nlp_pipeline.pkl", "rb") as f:
        return pickle.load(f)
nlp = load_nlp_pipeline()

def detect_intent(question):
    question = question.lower()
    scores = {}
    for category, words in nlp["category_keywords"].items():
        score = 0
        for word in words:
            if word.lower() in question:
                score += 1
        scores[category] = score
    best_category = max(scores, key=scores.get)
    if scores[best_category] == 0:
        return None
    return best_category

def expand_query(question, intent):
    if intent is None:
        return question
    keywords = nlp["category_keywords"][intent]
    expanded = question + " " + " ".join(keywords)
    return expanded

# CACHE EMBEDDING MODEL
@st.cache_resource
def load_embedding():
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-base-en-v1.5",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
embedding_model = load_embedding()

# CACHE FAISS
@st.cache_resource
def load_vectorstore():
    return FAISS.load_local(
        "vectorstore/faiss_index",
        embedding_model,
        allow_dangerous_deserialization=True
    )


db = load_vectorstore()
retriever = retriever = db.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 5,
        "fetch_k": 20
        }
        )


# GEMINI
@st.cache_resource
def load_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-3.5-flash-lite",
        temperature=0.2
    )

llm = load_llm()


# PROMPT
prompt = ChatPromptTemplate.from_template("""
You are an AI customer support assistant.

Answer the user's question using the provided context.

If multiple context documents are relevant, combine the information into a clear answer.

If the answer is partially available, answer with the available information.

Only say:

"I couldn't find that information in the knowledge base."

if none of the retrieved context is relevant.

Context:
{context}

Question:
{question}

Answer:
""")
parser = StrOutputParser()


# RAG FUNCTION
def ask_chatbot(question):

    # NLP
    intent = detect_intent(question)
    expanded_question = expand_query(question, intent)
    # RAG
    docs = retriever.invoke(expanded_question)
    context = "\n\n".join(
        doc.page_content for doc in docs
    )
    chain = prompt | llm | parser
    response = chain.invoke(
        {
            "context": context,
            "question": question
        }
    )
    return response, intent


# CHAT HISTORY
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# USER INPUT
user_question = st.chat_input("Ask anything...")


if user_question:

    st.session_state.messages.append(
        {
            "role":"user",
            "content":user_question
        }
    )
    with st.chat_message("user"):
        st.markdown(user_question)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                answer, intent = ask_chatbot(user_question)
            except Exception as e:
                answer = f"Error:\n\n{e}"
            st.markdown(answer)
    st.session_state.messages.append(
        {
            "role":"assistant",
            "content":answer
        }
    )


# SIDEBAR
with st.sidebar:
    st.title("🤖 Customer Support")
    st.success("Online")
    st.divider()
    st.subheader("Technology")
    st.markdown("""
- 🧠 NLP
- 📚 FAISS
- 🤖 Gemini
- 🔗 LangChain
""")

    st.divider()
    st.subheader("Knowledge Base")
    st.metric("FAQs","500")
    st.metric("Documents","1 PDF")
    st.divider()
    st.subheader("Session")
    st.metric(
        "Messages",
        len(st.session_state.messages)
    )
    if st.button("🗑 Clear Chat"):
        st.session_state.messages=[]
        st.rerun()