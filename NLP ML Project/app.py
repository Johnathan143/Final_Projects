import streamlit as st
import joblib
import re
import string

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Load models
model = joblib.load("logistic_regression.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")

# NLP objects
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# Text cleaning
def clean_text(text):

    text = str(text)
    text = text.lower()

    text = re.sub(r'http\\S+|www\\S+|https\\S+', '', text)
    text = re.sub(r'@\\w+', '', text)
    text = re.sub(r'#', '', text)
    text = re.sub(r'\\d+', '', text)

    text = text.translate(
        str.maketrans('', '', string.punctuation)
    )

    text = re.sub(r'\\s+', ' ', text).strip()

    tokens = text.split()

    tokens = [
        lemmatizer.lemmatize(word)
        for word in tokens
        if word not in stop_words
    ]

    return " ".join(tokens)

# UI
st.set_page_config(
    page_title="Actionable Message Classifier",
    page_icon="📩"
)

st.title("📩 Social Media Actionable Message Classifier")

st.write(
    "Predict whether a social media message requires action or response."
)

user_input = st.text_area(
    "Enter Message",
    height=150
)

if st.button("Predict"):

    cleaned = clean_text(user_input)

    vector = tfidf.transform([cleaned])

    prediction = model.predict(vector)[0]

    if prediction == 1:
        st.success("✅ Actionable Message")
    else:
        st.info("ℹ️ Non-Actionable Message")