import streamlit as st
import pickle
import re
import os

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="TrueHire Portal", layout="wide")

# -------------------------------
# SIMPLE USER LOGIN SYSTEM
# -------------------------------
users = {
    "admin": "1234",
    "user": "1234"
}

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

def login():
    st.title("🔐 Login to TrueHire")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username in users and users[username] == password:
            st.session_state.logged_in = True
            st.session_state.user = username
            st.success("Login successful")
            st.rerun()
        else:
            st.error("Invalid credentials")

if not st.session_state.logged_in:
    login()
    st.stop()

# -------------------------------
# LOAD MODEL
# -------------------------------
@st.cache_resource
def load_model():
    if not os.path.exists("model.pkl") or not os.path.exists("vectorizer.pkl"):
        return None, None

    model = pickle.load(open("model.pkl", "rb"))
    vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
    return model, vectorizer

model, vectorizer = load_model()

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z ]', '', text)
    return text

# -------------------------------
# SAMPLE JOB DATABASE
# -------------------------------
jobs = [
    {"title": "Software Engineer", "desc": "Develop web applications using Python and React."},
    {"title": "Data Entry Job", "desc": "Work from home. Earn money easily. Registration fee required."},
    {"title": "AI Engineer", "desc": "Work on ML models and AI systems."},
    {"title": "Online Job", "desc": "No experience needed. Earn ₹5000 daily. Pay registration fee."}
]

# -------------------------------
# UI
# -------------------------------
st.sidebar.title("TrueHire Portal")
page = st.sidebar.radio("Navigate", ["Home", "Search Jobs", "Post Job", "Logout"])

# -------------------------------
# HOME PAGE
# -------------------------------
if page == "Home":
    st.title("🏢 Welcome to TrueHire")
    st.write(f"Logged in as: {st.session_state.user}")

    st.info("This platform detects fake job postings using Machine Learning.")

# -------------------------------
# SEARCH JOBS
# -------------------------------
elif page == "Search Jobs":

    st.title("🔍 Available Jobs")

    for i, job in enumerate(jobs):

        desc_clean = clean_text(job["desc"])

        if model and vectorizer:
            vector = vectorizer.transform([desc_clean])
            pred = model.predict(vector)[0]
        else:
            pred = 1 if "fee" in desc_clean else 0  # fallback

        tag = "🟢 REAL" if pred == 0 else "🔴 FAKE"

        with st.container():
            st.subheader(job["title"])
            st.write(job["desc"])
            st.write("Status:", tag)

            if pred == 0:
                if st.button(f"Apply Job {i}"):
                    st.success("Applied Successfully!")
            else:
                st.warning("Cannot apply to suspicious job")

            st.markdown("---")

# -------------------------------
# POST JOB
# -------------------------------
elif page == "Post Job":

    st.title("📤 Post a Job")

    title = st.text_input("Job Title")
    desc = st.text_area("Job Description")

    if st.button("Submit Job"):

        if title and desc:
            desc_clean = clean_text(desc)

            if model and vectorizer:
                vector = vectorizer.transform([desc_clean])
                pred = model.predict(vector)[0]
            else:
                pred = 1 if "fee" in desc_clean else 0

            if pred == 1:
                st.error("🚨 This job looks FAKE. Cannot post.")
            else:
                jobs.append({"title": title, "desc": desc})
                st.success("✅ Job posted successfully!")
        else:
            st.warning("Fill all fields")

# -------------------------------
# LOGOUT
# -------------------------------
elif page == "Logout":
    st.session_state.logged_in = False
    st.rerun()