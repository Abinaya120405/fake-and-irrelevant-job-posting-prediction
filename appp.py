import streamlit as st
import pickle
import numpy as np
import re
import os

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="Fake Job Detector", layout="wide")

# -------------------------------
# STYLING (like your ECG app)
# -------------------------------
bg = "#0b1f3a"
card = "#112d4e"
accent = "#3a86ff"
text = "white"

st.markdown(f"""
<style>
body {{
    background-color: {bg};
    color: {text};
}}

.main-title {{
    text-align: center;
    font-size: 32px;
    font-weight: bold;
    color: {accent};
    margin-bottom: 20px;
}}

.card {{
    background-color: {card};
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.3);
}}

.result-box {{
    padding: 15px;
    border-radius: 10px;
    font-size: 22px;
    font-weight: bold;
    text-align: center;
}}

.real {{
    background-color: #2ecc71;
    color: white;
}}

.fake {{
    background-color: #e74c3c;
    color: white;
}}

.stButton>button {{
    width: 100%;
    height: 45px;
    border-radius: 10px;
    background-color: {accent};
    color: white;
    font-size: 16px;
}}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# TITLE
# -------------------------------
st.markdown('<div class="main-title">🕵️ Fake Job Posting Detection System</div>', unsafe_allow_html=True)

# -------------------------------
# LOAD MODEL SAFELY
# -------------------------------
@st.cache_resource
def load_model():
    try:
        if not os.path.exists("model.pkl") or not os.path.exists("vectorizer.pkl"):
            return None, None

        model = pickle.load(open("model.pkl", "rb"))
        vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
        return model, vectorizer

    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

model, vectorizer = load_model()

# -------------------------------
# TEXT CLEANING
# -------------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z ]', '', text)
    return text

# -------------------------------
# LAYOUT
# -------------------------------
col1, col2 = st.columns([1, 2])

# -------------------------------
# INPUT SECTION
# -------------------------------
with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.subheader("Enter Job Description")

    user_input = st.text_area("Paste job description here...", height=250)

    detect = st.button("🔍 Detect")

    st.markdown('</div>', unsafe_allow_html=True)

# -------------------------------
# OUTPUT SECTION
# -------------------------------
with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)

    if detect:

        if user_input.strip() == "":
            st.warning("Please enter some text")

        elif model is None or vectorizer is None:
            st.warning("⚠️ Model not loaded. Showing demo result.")

            if len(user_input) > 150:
                result = "REAL JOB"
                css_class = "result-box real"
            else:
                result = "FAKE JOB"
                css_class = "result-box fake"

            st.markdown(f'<div class="{css_class}">{result}</div>', unsafe_allow_html=True)

        else:
            cleaned = clean_text(user_input)
            vector = vectorizer.transform([cleaned])

            pred = model.predict(vector)[0]

            if pred == 1:
                result = "FAKE JOB"
                css_class = "result-box fake"
            else:
                result = "REAL JOB"
                css_class = "result-box real"

            st.markdown(f'<div class="{css_class}">{result}</div>', unsafe_allow_html=True)

            st.write("Prediction Value:", pred)

            # Explanation
            if pred == 1:
                st.info("This job posting contains patterns commonly found in fraudulent listings.")
            else:
                st.info("This job posting appears legitimate based on model analysis.")

    else:
        st.info("Enter a job description and click Detect.")

    st.markdown('</div>', unsafe_allow_html=True)