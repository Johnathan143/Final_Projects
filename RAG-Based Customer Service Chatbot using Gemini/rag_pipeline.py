import os
import pickle

from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


# LOAD API KEY

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if GOOGLE_API_KEY is None:
    raise ValueError("Google API Key not found.")


# NLP PIPELINE

with open(BASE_DIR / "data" / "nlp_pipeline.pkl", "rb") as f:
    nlp = pickle.load(f)


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


# EMBEDDING MODEL

embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-base-en-v1.5",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

# VECTOR DATABASE
VECTORSTORE_PATH = BASE_DIR / "vectorstore" / "faiss_index"

db = FAISS.load_local(
    folder_path=str(VECTORSTORE_PATH),
    embeddings=embedding_model,
    allow_dangerous_deserialization=True
)

retriever = db.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 5,
        "fetch_k": 20,
    },
)


# GEMINI
llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    temperature=0.2,
)


# PROMPT
prompt = ChatPromptTemplate.from_template(
"""
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
"""
)

parser = StrOutputParser()


# MAIN RAG FUNCTION
def ask_chatbot(question):

    intent = detect_intent(question)

    expanded_question = expand_query(question, intent)

    docs = retriever.invoke(expanded_question)

    context = "\n\n".join(
        doc.page_content for doc in docs
    )

    chain = prompt | llm | parser

    response = chain.invoke(
        {
            "context": context,
            "question": question,
        }
    )

    return response, intent, docs