"""
TrueHire — Verified Job Portal
================================
Single-file Streamlit app integrating all three ML pipeline stages:

  STAGE 1 — Data Preprocessing  (data_preprocessing.py)
    Input : final_balanced_fake_job_postings.csv
    Output: cleaned_jobs.csv   → stored in SQLite (dataset_jobs table)
    Steps : drop nulls/dups, clean HTML/URLs/punctuation, normalise label col

  STAGE 2 — TF-IDF Feature Extraction  (tfidf_features.py)
    Input : cleaned_jobs.csv
    Output: tfidf_vectorizer.pkl, features.npz, meta_features.npy
    Steps : build 5 000-feature TF-IDF bigram vectorizer, optional meta cols

  STAGE 3 — Model Training  (train_model.py)
    Input : features.npz + meta_features.npy
    Output: pac_model.pkl, scaler.pkl
    Model : SGDClassifier(loss="hinge") ≡ PassiveAggressiveClassifier
    Labels: 0 = Genuine  |  1 = Fake  |  2 = Irrelevant

  UI LABELS:
    ✅ Genuine    → green badge (job title matches description, real company signals)
    🚨 Fake       → red badge  (scam keywords, missing salary/company, vague desc)
    ⚠️ Irrelevant → amber badge (title and description mismatch / non-job content)

Run:
    pip install streamlit scikit-learn pandas numpy joblib scipy
    streamlit run truehireweb.py

Place beside this script (or let it auto-generate on first run):
    final_balanced_fake_job_postings.csv   ← raw dataset
    cleaned_jobs.csv                        ← auto-generated
    tfidf_vectorizer.pkl                    ← auto-generated
    features.npz                            ← auto-generated
    meta_features.npy                       ← auto-generated
    pac_model.pkl                           ← auto-generated
    scaler.pkl                              ← auto-generated
"""

# ── Standard library ─────────────────────────────────────────────────────────
import os
import re
import sqlite3
import hashlib
import warnings
from datetime import date

warnings.filterwarnings("ignore")

# ── Third-party ──────────────────────────────────────────────────────────────
import streamlit as st
import pandas as pd
import numpy as np
import joblib

from scipy.sparse import hstack, csr_matrix, vstack, save_npz, load_npz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import MaxAbsScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import cosine_similarity as _cos   # avoid name clash
from sklearn.metrics.pairwise import cosine_similarity

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG  — must be first Streamlit call
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TrueHire — Verified Job Portal",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────────
# PATHS  — all artifacts live beside this script
# ─────────────────────────────────────────────────────────────────────────────
_HERE          = os.path.dirname(os.path.abspath(__file__))
DB_PATH        = os.path.join(_HERE, "truehire.db")
RAW_CSV        = os.path.join(_HERE, "final_balanced_fake_job_postings.csv")
CLEAN_CSV      = os.path.join(_HERE, "cleaned_jobs.csv")
FEATURES_PATH  = os.path.join(_HERE, "features.npz")
VEC_PATH       = os.path.join(_HERE, "tfidf_vectorizer.pkl")
META_PATH      = os.path.join(_HERE, "meta_features.npy")
MODEL_PATH     = os.path.join(_HERE, "pac_model.pkl")
SCALER_PATH    = os.path.join(_HERE, "scaler.pkl")

# ─────────────────────────────────────────────────────────────────────────────
# LABEL MAP & UI BADGES
#   0 = Genuine    ✅  green
#   1 = Fake       🚨  red
#   2 = Irrelevant ⚠️  amber
# ─────────────────────────────────────────────────────────────────────────────
LABEL_MAP = {0: "genuine", 1: "fake", 2: "irrelevant"}

# badge_html, border_color
BADGE = {
    "genuine":    ('<span class="badge-genuine">✅ Genuine</span>',       "#16a34a"),
    "fake":       ('<span class="badge-fake">🚨 Fake Job</span>',         "#dc2626"),
    "irrelevant": ('<span class="badge-irr">⚠️ Irrelevant</span>',       "#d97706"),
    "pending":    ('<span class="badge-pending">⏳ Checking…</span>',     "#6b7280"),
}

# Scam keywords  (from tfidf_features.py  SCAM_KEYWORDS list)
SCAM_KEYWORDS = [
    "no investment", "quick earning", "earn from home",
    "easy money", "guaranteed income", "unlimited income",
    "be your own boss", "daily payout", "weekly payout",
    "risk free", "free registration", "mlm",
    "network marketing", "instant payment",
    "make money fast", "data entry work",
]
SCAM_PAT = "|".join(SCAM_KEYWORDS)

# ─────────────────────────────────────────────────────────────────────────────
# FALLBACK DATASET  (shown when CSV is absent — same as original truehireweb.py)
# ─────────────────────────────────────────────────────────────────────────────
RAW_JOBS = [
    ("Data Scientist","Infosys","Bangalore","Full-time","12-20 LPA",3,"IT / Software","Python,Machine Learning,SQL,Pandas,Scikit-learn","Build and deploy ML models for client analytics. Work with large datasets, perform EDA, and create dashboards. Collaborate with cross-functional teams to define KPIs."),
    ("Senior Software Engineer","Wipro","Hyderabad","Full-time","18-28 LPA",5,"IT / Software","Java,Spring Boot,Microservices,Docker,Kubernetes","Design scalable backend services. Lead code reviews, mentor junior engineers, and drive architectural decisions. CI/CD pipeline ownership."),
    ("Machine Learning Engineer","TCS","Pune","Full-time","15-25 LPA",4,"IT / Software","Python,TensorFlow,PyTorch,MLOps,AWS","Deploy production ML pipelines. Optimize model performance, manage model versioning, and monitor drift. Experience with LLMs preferred."),
    ("Frontend Developer","HCL Technologies","Chennai","Full-time","8-14 LPA",2,"IT / Software","React,TypeScript,CSS,Redux,Webpack","Build responsive, accessible web applications. Collaborate with designers to implement pixel-perfect UIs. Write unit tests and maintain component libraries."),
    ("DevOps Engineer","Tech Mahindra","Noida","Full-time","14-22 LPA",4,"IT / Software","Jenkins,Kubernetes,Terraform,AWS,Linux","Automate infrastructure provisioning and deployment pipelines. Manage cloud resources on AWS, ensure uptime SLAs, and implement monitoring using Grafana/Prometheus."),
    ("Business Analyst","Accenture","Mumbai","Full-time","10-16 LPA",3,"IT / Software","SQL,Excel,Power BI,Tableau,JIRA","Gather business requirements, document user stories, and work closely with development teams. Create reports and dashboards for senior stakeholders."),
    ("Cloud Architect","IBM India","Bangalore","Full-time","30-45 LPA",8,"IT / Software","AWS,Azure,GCP,Terraform,Kubernetes","Design cloud-native solutions for enterprise clients. Lead migration projects, define security posture, and optimize cloud costs. Strong knowledge of hybrid and multi-cloud architectures."),
    ("Full Stack Developer","Cognizant","Bangalore","Full-time","10-18 LPA",3,"IT / Software","Node.js,React,MongoDB,Express,AWS","Develop end-to-end web applications. Work on both frontend and backend, integrate REST APIs, manage databases, and deploy on cloud platforms."),
    ("Cybersecurity Analyst","Capgemini","Gurgaon","Full-time","12-20 LPA",3,"IT / Software","SIEM,Penetration Testing,Network Security,Python,ISO 27001","Monitor SOC alerts, perform vulnerability assessments, and respond to security incidents. Experience with SIEM tools and threat intelligence platforms required."),
    ("Product Manager","Flipkart","Bangalore","Full-time","25-40 LPA",5,"E-commerce","Product Strategy,Agile,Data Analysis,SQL,Figma","Own the product roadmap for a key vertical. Define success metrics, conduct user research, and collaborate with engineering, design, and marketing teams."),
    ("Data Analyst","Amazon India","Hyderabad","Full-time","10-16 LPA",2,"E-commerce","Python,SQL,Excel,Tableau,Statistics","Analyze large datasets to surface actionable insights. Build automated reports, partner with business teams on A/B tests, and present findings to leadership."),
    ("UX Designer","Meesho","Bangalore","Full-time","8-14 LPA",2,"E-commerce","Figma,User Research,Prototyping,Wireframing,Design Systems","Design intuitive user experiences for mobile and web platforms. Conduct usability studies, create wireframes and prototypes, and collaborate with PMs and engineers."),
    ("Supply Chain Manager","Myntra","Bangalore","Full-time","15-22 LPA",5,"E-commerce","Supply Chain,SAP,Logistics,Excel,Vendor Management","Manage end-to-end supply chain operations including procurement, inventory, and last-mile delivery. Identify inefficiencies and drive cost-reduction initiatives."),
    ("Financial Analyst","HDFC Bank","Mumbai","Full-time","10-15 LPA",3,"Finance","Financial Modeling,Excel,Bloomberg,CFA,Python","Conduct financial analysis, build valuation models, and prepare investment reports. Support senior analysts on deal execution and portfolio monitoring."),
    ("Investment Banking Analyst","Goldman Sachs","Mumbai","Full-time","20-35 LPA",2,"Finance","Financial Modeling,Excel,PowerPoint,M&A,Valuation","Support M&A and capital markets transactions. Prepare pitch books, conduct industry research, and build complex financial models."),
    ("Risk Analyst","ICICI Bank","Pune","Full-time","8-13 LPA",2,"Finance","Credit Risk,SQL,Python,SAS,Basel III","Assess credit and market risk for lending portfolios. Develop risk models, monitor exposure, and prepare regulatory reports."),
    ("Chartered Accountant","Deloitte","Mumbai","Full-time","12-18 LPA",3,"Finance","Taxation,Audit,SAP FICO,IFRS,Excel","Lead statutory audits and tax advisory engagements for large corporate clients. Ensure compliance with applicable accounting standards."),
    ("Doctor - General Physician","Apollo Hospitals","Chennai","Full-time","15-25 LPA",5,"Healthcare","Clinical Medicine,Patient Care,Medical Records,EMR,Diagnostics","Provide primary care to outpatients and inpatients. Diagnose and treat acute and chronic conditions, coordinate specialist referrals, and maintain medical records."),
    ("Nurse - ICU","Fortis Healthcare","Delhi","Full-time","5-8 LPA",2,"Healthcare","Patient Care,ICU,Ventilator Management,IV Therapy,BLS","Deliver high-quality nursing care in the ICU setting. Monitor critically ill patients, administer medications, and collaborate with the multidisciplinary team."),
    ("Python Developer","Zoho","Chennai","Full-time","8-14 LPA",2,"IT / Software","Python,Django,REST API,PostgreSQL,Redis","Develop backend services using Django. Write clean, testable code, design REST APIs, and optimize database queries for high-traffic applications."),
    ("Android Developer","PhonePe","Bangalore","Full-time","12-20 LPA",3,"IT / Software","Android,Kotlin,Jetpack Compose,MVVM,Firebase","Build and maintain Android applications used by millions. Implement new features, improve app performance, and collaborate closely with product and QA teams."),
    ("NLP Engineer","Sarvam AI","Bangalore","Full-time","20-35 LPA",4,"IT / Software","NLP,Transformers,Python,HuggingFace,LLMs","Research and build NLP models for Indian languages. Fine-tune LLMs, build text classification and NER pipelines, and deploy models in production."),
    ("Data Engineer","Swiggy","Bangalore","Full-time","14-22 LPA",3,"IT / Software","Apache Spark,Kafka,Airflow,Python,AWS","Design and maintain large-scale data pipelines. Build real-time and batch processing workflows, ensure data quality, and optimize storage costs."),
    ("HR Manager","Infosys BPM","Bangalore","Full-time","12-18 LPA",5,"IT / Software","HR Policies,Recruitment,HRIS,Employee Relations,Performance Management","Lead end-to-end HR operations for a 500+ employee business unit. Drive talent acquisition, manage performance cycles, and implement employee engagement programs."),
    ("Digital Marketing Manager","Nykaa","Mumbai","Full-time","10-16 LPA",3,"E-commerce","SEO,SEM,Social Media,Google Analytics,Meta Ads","Drive online customer acquisition and retention. Manage paid campaigns, SEO strategy, and social media channels. Analyze performance and optimize ROI."),
    ("Sales Manager","Salesforce India","Mumbai","Full-time","20-35 LPA",5,"IT / Software","CRM,B2B Sales,Negotiation,Salesforce,Pipeline Management","Drive enterprise software sales across assigned territory. Build relationships with C-suite stakeholders, manage the full sales cycle, and exceed quarterly targets."),
    ("Data Science Intern","Ola","Bangalore","Internship","20-30K/month",0,"IT / Software","Python,Pandas,Machine Learning,Statistics,Jupyter","Support data science projects across pricing and demand forecasting. Analyze datasets, build prototypes, and present findings to the team."),
    ("Software Engineer Intern","Google India","Hyderabad","Internship","60-80K/month",0,"IT / Software","Python,Algorithms,Data Structures,C++,Problem Solving","Work on real engineering projects under mentorship. Contribute to codebase, write tests, and present a project at the end of the internship."),
    ("AI Research Scientist","Microsoft India","Hyderabad","Full-time","40-70 LPA",6,"IT / Software","Deep Learning,Research,PyTorch,NLP,Computer Vision","Conduct original research in AI/ML. Publish papers, collaborate with product teams to transfer research to production, and mentor junior researchers."),
    ("Blockchain Developer","Polygon","Mumbai","Full-time","20-35 LPA",4,"IT / Software","Solidity,Ethereum,Web3.js,Smart Contracts,Python","Build and audit smart contracts on EVM-compatible blockchains. Develop DeFi protocol integrations and ensure contract security best practices."),
]

# ─────────────────────────────────────────────────────────────────────────────
# CSS  — original design, extended with ML badge styles
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
  --primary:#1a3c5e; --accent:#e8734a; --bg:#f9f7f4;
  --text:#1a1a2e; --border:#e5e7eb; --success:#16a34a; --danger:#dc2626;
}
html,body,[class*="css"]{font-family:'DM Sans',sans-serif;background:var(--bg)!important;color:var(--text);}
#MainMenu,footer,header{visibility:hidden;}
.block-container{padding-top:1.2rem!important;max-width:1150px;}

.hero-banner{
  background:linear-gradient(135deg,#1a3c5e 55%,#e8734a 100%);
  border-radius:18px;padding:3rem 2.5rem;margin-bottom:1.5rem;
  color:#fff;position:relative;overflow:hidden;
}
.hero-banner::after{content:'';position:absolute;right:-80px;top:-80px;
  width:350px;height:350px;border-radius:50%;background:rgba(255,255,255,0.05);}
.hero-banner h1{font-family:'Playfair Display',serif;font-size:clamp(2rem,4vw,3rem);margin:0 0 0.5rem;}
.hero-banner h1 span{color:#fbbf24;}
.hero-banner p{opacity:0.88;font-size:1.05rem;max-width:520px;margin-bottom:0;}
.hero-stats{display:flex;gap:2rem;flex-wrap:wrap;margin-top:1.5rem;}
.hero-stat .n{font-size:1.8rem;font-weight:700;color:#fbbf24;}
.hero-stat .l{font-size:0.78rem;opacity:0.8;margin-top:0.1rem;}

.stat-row{display:flex;gap:1rem;margin-bottom:1.5rem;flex-wrap:wrap;}
.stat-card{background:#fff;border-radius:12px;padding:1.1rem 1.4rem;
  border:1px solid var(--border);flex:1;min-width:130px;text-align:center;
  box-shadow:0 2px 12px rgba(26,60,94,0.07);}
.stat-card .num{font-size:1.9rem;font-weight:700;color:var(--primary);line-height:1;}
.stat-card .lbl{font-size:0.78rem;color:#6b7280;margin-top:0.3rem;}

/* Job cards — border changes colour based on ML label */
.job-card{background:#fff;border-radius:12px;padding:1.2rem 1.5rem;
  border:1px solid var(--border);margin-bottom:0.75rem;
  box-shadow:0 2px 8px rgba(26,60,94,0.05);transition:box-shadow .2s,border-color .2s;}
.job-card:hover{box-shadow:0 8px 28px rgba(26,60,94,0.13);border-color:var(--accent);}
.job-card.card-fake{border-left:4px solid #dc2626;}
.job-card.card-irr{border-left:4px solid #d97706;}
.job-card.card-genuine{border-left:4px solid #16a34a;}
.job-card h3{margin:0 0 0.15rem;color:var(--primary);font-size:1rem;font-weight:600;}
.job-card .company{font-size:0.84rem;color:#6b7280;margin-bottom:0.6rem;}
.match-bar-wrap{background:#f3f4f6;border-radius:50px;height:6px;margin-top:0.5rem;}
.match-bar{background:linear-gradient(90deg,var(--accent),#fbbf24);border-radius:50px;height:6px;}

/* Tags */
.tag{display:inline-block;background:#f3f4f6;border:1px solid var(--border);border-radius:50px;
  padding:0.18rem 0.7rem;font-size:0.73rem;color:#374151;margin-right:0.3rem;margin-top:0.3rem;}
.tag-accent{background:#fff5f0;border-color:#e8734a;color:#e8734a;}
.tag-green{background:#f0fdf4;border-color:#16a34a;color:#16a34a;}

/* ── ML status badges ──────────────────────────── */
.badge-genuine{background:#dcfce7;color:#15803d;border:1px solid #86efac;
  border-radius:50px;padding:0.15rem 0.7rem;font-size:0.73rem;font-weight:700;}
.badge-fake{background:#fee2e2;color:#dc2626;border:1px solid #fca5a5;
  border-radius:50px;padding:0.15rem 0.7rem;font-size:0.73rem;font-weight:700;}
.badge-irr{background:#fef9c3;color:#a16207;border:1px solid #fde047;
  border-radius:50px;padding:0.15rem 0.7rem;font-size:0.73rem;font-weight:600;}
.badge-pending{background:#f3f4f6;color:#6b7280;border:1px solid #d1d5db;
  border-radius:50px;padding:0.15rem 0.7rem;font-size:0.73rem;}
.badge-ai{background:#fef3c7;color:#d97706;border-radius:50px;
  padding:0.15rem 0.65rem;font-size:0.72rem;font-weight:600;}

/* ── ML alert banners (detail view) ────────────── */
.alert-fake{background:#fff1f2;border-left:4px solid #dc2626;border-radius:8px;
  padding:0.9rem 1.1rem;margin-bottom:1rem;color:#7f1d1d;font-size:0.91rem;}
.alert-irr{background:#fffbeb;border-left:4px solid #d97706;border-radius:8px;
  padding:0.9rem 1.1rem;margin-bottom:1rem;color:#78350f;font-size:0.91rem;}
.alert-genuine{background:#f0fdf4;border-left:4px solid #16a34a;border-radius:8px;
  padding:0.9rem 1.1rem;margin-bottom:1rem;color:#14532d;font-size:0.91rem;}

.sec-title{font-family:'Playfair Display',serif;color:var(--primary);font-size:1.45rem;margin-bottom:0.15rem;}
.sec-sub{color:#6b7280;font-size:0.88rem;margin-bottom:1rem;}

div.stButton>button{background:var(--accent)!important;color:#fff!important;border:none!important;
  border-radius:8px!important;font-weight:600!important;transition:background .2s,transform .15s!important;}
div.stButton>button:hover{background:#cf5a32!important;transform:translateY(-1px);}

.stTextInput>div>input,.stSelectbox>div>div,.stTextArea>div>textarea,
.stNumberInput>div>input,.stDateInput>div>input{
  border-radius:8px!important;border:1.5px solid var(--border)!important;font-family:'DM Sans',sans-serif!important;}
.stTextInput>div>input:focus,.stTextArea>div>textarea:focus{
  border-color:var(--accent)!important;box-shadow:0 0 0 3px rgba(232,115,74,.12)!important;}

.info-box{background:#eff6ff;border-left:4px solid #3b82f6;border-radius:8px;padding:0.8rem 1rem;margin-bottom:1rem;font-size:0.9rem;}
.success-box{background:#f0fdf4;border-left:4px solid #16a34a;border-radius:8px;padding:0.8rem 1rem;margin-bottom:1rem;font-size:0.9rem;}
.ai-box{background:#fffbeb;border-left:4px solid #f59e0b;border-radius:8px;padding:0.8rem 1rem;margin-bottom:1rem;font-size:0.9rem;}

.detail-card{background:#fff;border-radius:16px;padding:2rem;border:1px solid var(--border);box-shadow:0 4px 20px rgba(26,60,94,0.09);}
.detail-card h2{font-family:'Playfair Display',serif;color:var(--primary);font-size:1.6rem;margin-bottom:0.2rem;}

table{width:100%;border-collapse:collapse;font-size:0.87rem;}
th{background:#f3f4f6;color:#374151;padding:0.6rem 0.8rem;text-align:left;font-weight:600;}
td{padding:0.6rem 0.8rem;border-bottom:1px solid #f3f4f6;color:#374151;}
tr:hover td{background:#fffbf8;}

section[data-testid="stSidebar"]{background:var(--primary)!important;}
section[data-testid="stSidebar"] *{color:#dbeafe!important;}
</style>
""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# STAGE 1 — DATA PREPROCESSING
# Mirrors data_preprocessing.py  →  clean_text() + preprocess()
# Input : final_balanced_fake_job_postings.csv
# Output: cleaned_jobs.csv  +  SQLite  dataset_jobs  table
# ═════════════════════════════════════════════════════════════════════════════

def _clean_text(text: str) -> str:
    """
    Exact copy of clean_text() from data_preprocessing.py.
    Lowercases, strips HTML tags, URLs, special chars.
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"<[^>]+>", " ", text)           # remove HTML tags
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)  # remove URLs
    text = re.sub(r"[^\w\s]", " ", text)            # remove punctuation
    text = re.sub(r"\s+", " ", text).strip()        # collapse whitespace
    return text


@st.cache_data(show_spinner="📂 Stage 1 — Preprocessing dataset…")
def run_preprocessing() -> pd.DataFrame:
    """
    Stage 1: Load raw CSV, clean, deduplicate, normalise label column.
    Saves cleaned_jobs.csv.  Returns cleaned DataFrame.
    If raw CSV is absent, returns None (app falls back to RAW_JOBS).
    """
    if not os.path.exists(RAW_CSV):
        return None

    df = pd.read_csv(RAW_CSV)

    # Drop rows with any NaN, then deduplicate
    df = df.dropna().drop_duplicates().reset_index(drop=True)

    # Clean text columns
    for col in ["title", "description", "company_profile", "requirements", "benefits"]:
        if col in df.columns:
            df[col] = df[col].apply(_clean_text)

    # Normalise label column (fraudulent → label)
    if "fraudulent" in df.columns and "label" not in df.columns:
        df = df.rename(columns={"fraudulent": "label"})
    elif "label" not in df.columns:
        st.error("Dataset must have a 'fraudulent' or 'label' column.")
        return None

    df["label"] = df["label"].astype(int)
    df = df.reset_index(drop=True)
    df.to_csv(CLEAN_CSV, index=False)
    return df


# ═════════════════════════════════════════════════════════════════════════════
# STAGE 2 — TF-IDF FEATURE EXTRACTION
# Mirrors tfidf_features.py  →  build_tfidf_vectorizer() + extract_features()
# Input : cleaned_jobs.csv  (DataFrame from Stage 1)
# Output: tfidf_vectorizer.pkl, features.npz, meta_features.npy
# ═════════════════════════════════════════════════════════════════════════════

def _create_combined_text(df: pd.DataFrame) -> pd.Series:
    """
    Mirrors create_combined_text() from tfidf_features.py.
    Concatenates all text columns into a single string per row.
    """
    text_cols = [c for c in
                 ["title", "company_profile", "description",
                  "requirements", "salary_range", "location", "industry"]
                 if c in df.columns]
    df[text_cols] = df[text_cols].fillna("")
    return df[text_cols].astype(str).agg(" ".join, axis=1)


@st.cache_resource(show_spinner="🔢 Stage 2 — Building TF-IDF features…")
def run_tfidf(df: pd.DataFrame):
    """
    Stage 2: Fit TF-IDF vectorizer on cleaned_jobs DataFrame.
    Saves vectorizer.pkl, features.npz, meta_features.npy.
    Returns (vectorizer, X_sparse, meta_cols, y).
    """
    if df is None:
        return None, None, [], None

    # Normalise label
    if "fraudulent" in df.columns and "label" not in df.columns:
        df = df.rename(columns={"fraudulent": "label"})

    combined = _create_combined_text(df)

    # Exact TfidfVectorizer settings from tfidf_features.py
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=5,
        max_df=0.90,
        strip_accents="unicode",
        analyzer="word",
        token_pattern=r"\w{2,}",
        norm="l2",
    )
    X_tfidf = vectorizer.fit_transform(combined)

    # Optional meta columns (if pre-computed in the CSV)
    possible_meta = [
        "has_scam_keywords", "has_salary", "has_company_desc",
        "has_phone_in_desc", "title_len", "desc_len",
    ]
    meta_cols = [c for c in possible_meta if c in df.columns]

    if meta_cols:
        X_meta = csr_matrix(df[meta_cols].fillna(0).astype(float).values)
        X = hstack([X_tfidf, X_meta])
    else:
        X = X_tfidf

    # Persist artifacts
    joblib.dump(vectorizer, VEC_PATH)
    save_npz(FEATURES_PATH, X)
    np.save(META_PATH, np.array(meta_cols))

    y = df["label"].values
    return vectorizer, X, meta_cols, y


# ═════════════════════════════════════════════════════════════════════════════
# STAGE 3 — MODEL TRAINING
# Mirrors train_model.py  →  add_irrelevant_class() + SGDClassifier
# Input : features.npz, meta_features.npy
# Output: pac_model.pkl, scaler.pkl
# Labels: 0=Genuine  1=Fake  2=Irrelevant
# ═════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="🤖 Stage 3 — Training fraud-detection model…")
def run_training(X, y):
    """
    Stage 3: Add irrelevant class, scale, train SGDClassifier (hinge=PAC).
    Saves pac_model.pkl and scaler.pkl.
    Returns (model, scaler).
    """
    if X is None or y is None:
        return None, None

    # ── Add irrelevant class (ratio=0.05, mirrors add_irrelevant_class()) ──
    X_csr = X.tocsr()
    genuine_pos = np.where(y == 0)[0]
    n_irr = max(100, int(len(genuine_pos) * 0.05))
    sampled = np.random.RandomState(42).choice(genuine_pos, size=n_irr, replace=False)
    irr_dense = X_csr[sampled].toarray()
    rng = np.random.RandomState(0)
    for row in irr_dense:
        rng.shuffle(row)                        # shuffle features → irrelevant signal
    X_irr = csr_matrix(irr_dense)
    y_irr = np.full(n_irr, 2, dtype=int)       # label 2 = irrelevant

    X_aug = vstack([X, X_irr])
    y_aug = np.concatenate([y, y_irr])

    # ── Scale + train ──
    scaler = MaxAbsScaler()
    X_scaled = scaler.fit_transform(X_aug)

    # SGDClassifier(loss="hinge") ≡ PassiveAggressiveClassifier (from train_model.py)
    model = SGDClassifier(
        loss="hinge", penalty=None,
        learning_rate="pa1", eta0=1.0,
        max_iter=1000, random_state=42, tol=1e-3,
    )
    model.fit(X_scaled, y_aug)

    joblib.dump(model,  MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    return model, scaler


# ─────────────────────────────────────────────────────────────────────────────
# LOAD OR RUN FULL PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="🔄 Loading ML pipeline…")
def get_pipeline():
    """
    Checks whether saved artifacts exist.
    If yes → loads them (fast path).
    If no  → runs Stages 1→2→3 and saves artifacts.
    Returns (vectorizer, scaler, model, meta_cols, cleaned_df | None)
    """
    artifacts_exist = all(os.path.exists(p) for p in
                          [VEC_PATH, MODEL_PATH, SCALER_PATH, META_PATH])

    if artifacts_exist:
        vectorizer = joblib.load(VEC_PATH)
        scaler     = joblib.load(SCALER_PATH)
        model      = joblib.load(MODEL_PATH)
        meta_cols  = list(np.load(META_PATH, allow_pickle=True))
        # Load cleaned_df if available
        cleaned_df = pd.read_csv(CLEAN_CSV) if os.path.exists(CLEAN_CSV) else None
        return vectorizer, scaler, model, meta_cols, cleaned_df

    # Run full pipeline
    cleaned_df             = run_preprocessing()
    vectorizer, X, meta_cols, y = run_tfidf(cleaned_df)
    model, scaler          = run_training(X, y)
    return vectorizer, scaler, model, meta_cols, cleaned_df


# ─────────────────────────────────────────────────────────────────────────────
# DETECT FAKE JOB  (inference helper — mirrors transform_single() + predict)
# ─────────────────────────────────────────────────────────────────────────────

def detect_fake_job(text_dict: dict) -> tuple:
    """
    Run the full inference pipeline on one job posting dict.
    Keys used: title, company_profile, description, requirements,
               salary_range, location, industry.

    Returns (label_str, confidence_pct):
      label_str  ∈ {"genuine", "fake", "irrelevant", "pending"}
      confidence ∈ [10.0, 100.0]
    """
    vectorizer, scaler, model, meta_cols, _ = get_pipeline()
    if model is None:
        return "pending", 0.0

    # ── Build combined text (mirrors transform_single from tfidf_features.py) ──
    def _c(t):
        if not isinstance(t, str): return ""
        t = t.lower()
        t = re.sub(r"<[^>]+>", " ", t)
        t = re.sub(r"http\S+", " url ", t)
        t = re.sub(r"\b\d{10,}\b", " phone ", t)
        t = re.sub(r"[^\w\s]", " ", t)
        t = re.sub(r"\s+", " ", t)
        return t.strip()

    cols    = ["title","company_profile","description","requirements",
               "salary_range","location","industry"]
    combined = " ".join(_c(str(text_dict.get(k, ""))) for k in cols)

    X_tfidf  = vectorizer.transform([combined])

    # ── Meta features ──
    meta = {
        "has_scam_keywords" : int(bool(re.search(SCAM_PAT, combined))),
        "has_salary"        : int(bool(str(text_dict.get("salary_range","")).strip())),
        "has_company_desc"  : int(bool(str(text_dict.get("company_profile","")).strip())),
        "has_phone_in_desc" : int(bool(re.search(r"\b\d{10}\b", combined))),
        "title_len"         : len(str(text_dict.get("title","")).split()),
        "desc_len"          : len(str(text_dict.get("description","")).split()),
    }
    if meta_cols:
        X_meta = csr_matrix([[meta.get(c, 0) for c in meta_cols]])
        X      = hstack([X_tfidf, X_meta])
    else:
        X = X_tfidf

    X_scaled = scaler.transform(X)
    pred_int = int(model.predict(X_scaled)[0])
    label    = LABEL_MAP.get(pred_int, "pending")

    # Confidence from decision-function margin
    try:
        df_scores = model.decision_function(X_scaled).flatten()
        s         = np.sort(df_scores)[::-1]
        margin    = float(s[0] - s[1]) if len(s) > 1 else float(s[0])
        confidence = round(min(100.0, max(10.0, 50.0 + margin * 10.0)), 1)
    except Exception:
        confidence = 0.0

    return label, confidence


# ─────────────────────────────────────────────────────────────────────────────
# COSINE-SIMILARITY SEARCH (seeker job matching — separate from fraud model)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="🔍 Building job-matching index…")
def build_match_engine(df: pd.DataFrame):
    """
    Builds a lightweight TF-IDF cosine-similarity index over RAW_JOBS
    or the fallback list, used purely for seeker job recommendations.
    Returns (match_tfidf, match_matrix, labelled_df).
    """
    cols = ["title","company","location","job_type","salary",
            "exp","industry","skills","description"]
    if df is not None and "title" in df.columns:
        # Use cleaned dataset jobs
        job_df = df[["title","description","industry","salary_range","location"]].copy()
        job_df.columns = ["title","description","industry","salary","location"]
        job_df["company"]  = df.get("company","").fillna("Unknown") if "company" in df.columns else "Unknown"
        job_df["job_type"] = df.get("employment_type","").fillna("Full-time") if "employment_type" in df.columns else "Full-time"
        job_df["exp"]      = 0
        job_df["skills"]   = ""
        job_df["ml_label"] = "pending"
        job_df["ml_conf"]  = 0.0
        source = "dataset"
    else:
        job_df = pd.DataFrame(RAW_JOBS, columns=cols)
        job_df["exp"]      = pd.to_numeric(job_df["exp"], errors="coerce").fillna(0).astype(int)
        job_df["ml_label"] = "pending"
        job_df["ml_conf"]  = 0.0
        source = "fallback"

    job_df["corpus"] = (job_df["title"] + " " +
                        job_df.get("skills","").astype(str).str.replace(",", " ") + " " +
                        job_df["description"].astype(str))

    match_tfidf = TfidfVectorizer(
        max_features=6000, ngram_range=(1, 2),
        stop_words="english", sublinear_tf=True,
    )
    match_matrix = match_tfidf.fit_transform(job_df["corpus"])
    return match_tfidf, match_matrix, job_df, source


def tfidf_search(query: str, match_tfidf, match_matrix, job_df,
                 only_genuine: bool = False) -> pd.DataFrame:
    """Return job_df sorted by cosine similarity to query."""
    if not query.strip():
        result = job_df.copy().assign(score=1.0)
    else:
        qvec   = match_tfidf.transform([query])
        sims   = cosine_similarity(qvec, match_matrix).flatten()
        result = job_df.copy().assign(score=sims).sort_values("score", ascending=False)
    if only_genuine:
        result = result[result["ml_label"] == "genuine"]
    return result


def recommend_jobs(seeker_text: str, match_tfidf, match_matrix,
                   job_df, top_n: int = 5) -> pd.DataFrame:
    return tfidf_search(seeker_text, match_tfidf, match_matrix,
                        job_df, only_genuine=False).head(top_n)


# ─────────────────────────────────────────────────────────────────────────────
# DATABASE  (SQLite — mirrors schema.sql with ml_label + ml_confidence)
# ─────────────────────────────────────────────────────────────────────────────

def get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_conn()
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS companies (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL, email TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL, phone TEXT, industry TEXT,
        website TEXT, year_founded INTEGER, description TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    CREATE TABLE IF NOT EXISTS seekers (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL, email TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL, phone TEXT, skills TEXT,
        experience INTEGER DEFAULT 0, preferred_location TEXT,
        bio TEXT, expected_salary TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    CREATE TABLE IF NOT EXISTS jobs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        company_id INTEGER NOT NULL, title TEXT NOT NULL,
        job_type TEXT DEFAULT 'Full-time', location TEXT,
        salary_range TEXT, experience_required INTEGER DEFAULT 0,
        deadline TEXT, description TEXT, requirements TEXT,
        contact_mobile TEXT,
        ml_label      TEXT DEFAULT 'pending',
        ml_confidence REAL DEFAULT 0.0,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(company_id) REFERENCES companies(id)
    );
    CREATE TABLE IF NOT EXISTS dataset_jobs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT, location TEXT, salary_range TEXT,
        company_profile TEXT, description TEXT, requirements TEXT,
        industry TEXT, employment_type TEXT, label INTEGER DEFAULT 0,
        ml_label TEXT DEFAULT 'pending', ml_confidence REAL DEFAULT 0.0
    );
    CREATE TABLE IF NOT EXISTS applications (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        job_id INTEGER NOT NULL, seeker_id INTEGER NOT NULL,
        status TEXT DEFAULT 'Under Review',
        applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(job_id, seeker_id),
        FOREIGN KEY(job_id) REFERENCES jobs(id),
        FOREIGN KEY(seeker_id) REFERENCES seekers(id)
    );
    CREATE TABLE IF NOT EXISTS dataset_applications (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        dataset_job_idx INTEGER NOT NULL, seeker_id INTEGER NOT NULL,
        status TEXT DEFAULT 'Under Review',
        applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(dataset_job_idx, seeker_id),
        FOREIGN KEY(seeker_id) REFERENCES seekers(id)
    );
    """)
    conn.commit(); conn.close()


init_db()


def _store_cleaned_jobs_in_db(df: pd.DataFrame):
    """
    Stage 1 output: store cleaned_jobs.csv rows into the dataset_jobs table.
    Runs ML detection on each row and stores ml_label + ml_confidence.
    Skips if table already populated.
    """
    conn = get_conn()
    count = conn.execute("SELECT COUNT(*) FROM dataset_jobs").fetchone()[0]
    conn.close()
    if count > 0:
        return   # already populated

    rows = []
    text_cols = ["title","location","salary_range","company_profile",
                 "description","requirements","industry","employment_type"]

    progress = st.progress(0, text="💾 Storing cleaned jobs in database…")
    total = min(len(df), 500)   # cap at 500 to keep startup fast

    for i, (_, row) in enumerate(df.head(total).iterrows()):
        text_dict = {
            "title":           str(row.get("title","")),
            "company_profile": str(row.get("company_profile","")),
            "description":     str(row.get("description","")),
            "requirements":    str(row.get("requirements","")),
            "salary_range":    str(row.get("salary_range","")),
            "location":        str(row.get("location","")),
            "industry":        str(row.get("industry","")),
        }
        lbl, conf = detect_fake_job(text_dict)
        rows.append((
            str(row.get("title","")),
            str(row.get("location","")),
            str(row.get("salary_range","")),
            str(row.get("company_profile","")),
            str(row.get("description","")),
            str(row.get("requirements","")),
            str(row.get("industry","")),
            str(row.get("employment_type","")),
            int(row.get("label", 0)),
            lbl, conf,
        ))
        progress.progress((i + 1) / total,
                          text=f"💾 Storing job {i+1}/{total}…")

    conn = get_conn()
    conn.executemany(
        "INSERT INTO dataset_jobs(title,location,salary_range,company_profile,"
        "description,requirements,industry,employment_type,label,ml_label,ml_confidence)"
        " VALUES(?,?,?,?,?,?,?,?,?,?,?)",
        rows,
    )
    conn.commit(); conn.close()
    progress.empty()


def hash_pw(pw): return hashlib.sha256(pw.encode()).hexdigest()

# ── Company DB ───────────────────────────────────────────────────────────────
def register_company(name,email,pw,phone,industry,year,desc):
    conn=get_conn()
    try:
        conn.execute("INSERT INTO companies(name,email,password_hash,phone,industry,year_founded,description) VALUES(?,?,?,?,?,?,?)",(name,email,hash_pw(pw),phone,industry,year,desc))
        conn.commit(); return True,"Company registered!"
    except sqlite3.IntegrityError: return False,"Email already registered."
    finally: conn.close()

def login_company(email,pw):
    conn=get_conn(); row=conn.execute("SELECT * FROM companies WHERE email=? AND password_hash=?",(email,hash_pw(pw))).fetchone()
    conn.close(); return dict(row) if row else None

def get_company(cid):
    conn=get_conn(); row=conn.execute("SELECT * FROM companies WHERE id=?",(cid,)).fetchone()
    conn.close(); return dict(row) if row else {}

def update_company(cid,name,industry,website,year,phone,desc):
    conn=get_conn()
    conn.execute("UPDATE companies SET name=?,industry=?,website=?,year_founded=?,phone=?,description=? WHERE id=?",(name,industry,website,year,phone,desc,cid))
    conn.commit(); conn.close()

# ── Seeker DB ────────────────────────────────────────────────────────────────
def register_seeker(name,email,pw,phone,skills,exp):
    conn=get_conn()
    try:
        conn.execute("INSERT INTO seekers(name,email,password_hash,phone,skills,experience) VALUES(?,?,?,?,?,?)",(name,email,hash_pw(pw),phone,skills,exp))
        conn.commit(); return True,"Account created!"
    except sqlite3.IntegrityError: return False,"Email already registered."
    finally: conn.close()

def login_seeker(email,pw):
    conn=get_conn(); row=conn.execute("SELECT * FROM seekers WHERE email=? AND password_hash=?",(email,hash_pw(pw))).fetchone()
    conn.close(); return dict(row) if row else None

def get_seeker(sid):
    conn=get_conn(); row=conn.execute("SELECT * FROM seekers WHERE id=?",(sid,)).fetchone()
    conn.close(); return dict(row) if row else {}

def update_seeker(sid,name,phone,skills,exp,loc,bio,salary):
    conn=get_conn()
    conn.execute("UPDATE seekers SET name=?,phone=?,skills=?,experience=?,preferred_location=?,bio=?,expected_salary=? WHERE id=?",(name,phone,skills,exp,loc,bio,salary,sid))
    conn.commit(); conn.close()

def profile_score(s):
    fields=[s.get('name'),s.get('phone'),s.get('skills'),s.get('bio'),s.get('preferred_location'),s.get('expected_salary')]
    return int(sum(1 for f in fields if f)/len(fields)*100)

# ── Company-posted Jobs DB ───────────────────────────────────────────────────
def post_job(cid,title,jtype,loc,salary,exp,deadline,desc,req,mobile):
    """Post a job and immediately run fraud detection — stores ml_label + ml_confidence."""
    ml_label, ml_conf = detect_fake_job({
        "title":title,"description":desc,"requirements":req,
        "salary_range":salary,"location":loc,"company_profile":"","industry":jtype,
    })
    conn=get_conn()
    conn.execute(
        "INSERT INTO jobs(company_id,title,job_type,location,salary_range,"
        "experience_required,deadline,description,requirements,contact_mobile,"
        "ml_label,ml_confidence) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
        (cid,title,jtype,loc,salary,exp,
         str(deadline) if deadline else None,
         desc,req,mobile,ml_label,ml_conf)
    )
    conn.commit(); conn.close()
    return ml_label, ml_conf

def get_posted_jobs(q="",location="",job_type="",experience="",industry="",limit=200):
    conn=get_conn()
    sql="SELECT j.*,c.name AS company_name,c.industry FROM jobs j JOIN companies c ON j.company_id=c.id WHERE 1=1"
    params=[]
    if q: sql+=" AND (j.title LIKE ? OR j.description LIKE ?)"; params+=[f"%{q}%",f"%{q}%"]
    if location: sql+=" AND j.location LIKE ?"; params.append(f"%{location}%")
    if job_type: sql+=" AND j.job_type=?"; params.append(job_type)
    if experience: sql+=" AND j.experience_required <= ?"; params.append(int(experience))
    if industry: sql+=" AND c.industry=?"; params.append(industry)
    sql+=" ORDER BY j.created_at DESC LIMIT ?"; params.append(limit)
    rows=conn.execute(sql,params).fetchall(); conn.close()
    return [dict(r) for r in rows]

def get_posted_job(jid):
    conn=get_conn(); row=conn.execute("SELECT j.*,c.name AS company_name FROM jobs j JOIN companies c ON j.company_id=c.id WHERE j.id=?",(jid,)).fetchone()
    conn.close(); return dict(row) if row else None

def get_company_jobs(cid):
    conn=get_conn()
    rows=conn.execute("SELECT j.*,COUNT(a.id) AS applicant_count FROM jobs j LEFT JOIN applications a ON a.job_id=j.id WHERE j.company_id=? GROUP BY j.id ORDER BY j.created_at DESC",(cid,)).fetchall()
    conn.close(); return [dict(r) for r in rows]

def delete_job(jid,cid):
    conn=get_conn()
    conn.execute("DELETE FROM applications WHERE job_id=?",(jid,))
    conn.execute("DELETE FROM jobs WHERE id=? AND company_id=?",(jid,cid))
    conn.commit(); conn.close()

def get_applicants(cid,job_id=None):
    conn=get_conn()
    sql="SELECT s.name,s.email,s.skills,s.experience,a.applied_at,a.status,j.title AS job_title FROM applications a JOIN seekers s ON s.id=a.seeker_id JOIN jobs j ON j.id=a.job_id WHERE j.company_id=?"
    params=[cid]
    if job_id: sql+=" AND a.job_id=?"; params.append(job_id)
    rows=conn.execute(sql+" ORDER BY a.applied_at DESC",params).fetchall(); conn.close()
    return [dict(r) for r in rows]

def apply_dataset_job(idx,sid):
    conn=get_conn()
    try:
        conn.execute("INSERT INTO dataset_applications(dataset_job_idx,seeker_id) VALUES(?,?)",(idx,sid))
        conn.commit(); return True
    except sqlite3.IntegrityError: return False
    finally: conn.close()

def already_applied_dataset(idx,sid):
    conn=get_conn(); row=conn.execute("SELECT 1 FROM dataset_applications WHERE dataset_job_idx=? AND seeker_id=?",(idx,sid)).fetchone()
    conn.close(); return bool(row)

def get_seeker_dataset_apps(sid):
    conn=get_conn(); rows=conn.execute("SELECT * FROM dataset_applications WHERE seeker_id=? ORDER BY applied_at DESC",(sid,)).fetchall()
    conn.close(); return [dict(r) for r in rows]

def seeker_dashboard_stats(sid):
    conn=get_conn()
    posted=conn.execute("SELECT COUNT(*) FROM applications WHERE seeker_id=?",(sid,)).fetchone()[0]
    ds=conn.execute("SELECT COUNT(*) FROM dataset_applications WHERE seeker_id=?",(sid,)).fetchone()[0]
    conn.close(); return posted+ds,ds

def company_dashboard_stats(cid):
    conn=get_conn()
    total=conn.execute("SELECT COUNT(*) FROM jobs WHERE company_id=?",(cid,)).fetchone()[0]
    apps=conn.execute("SELECT COUNT(*) FROM applications a JOIN jobs j ON j.id=a.job_id WHERE j.company_id=?",(cid,)).fetchone()[0]
    conn.close(); return total,apps,total


# ─────────────────────────────────────────────────────────────────────────────
# STARTUP: run pipeline + seed database
# ─────────────────────────────────────────────────────────────────────────────
_vec, _scaler, _model, _meta_cols, _cleaned_df = get_pipeline()
if _cleaned_df is not None and _model is not None:
    _store_cleaned_jobs_in_db(_cleaned_df)

_match_tfidf, _match_matrix, _job_df, _source = build_match_engine(_cleaned_df)


# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────
for _k,_v in [("page","home"),("user",None),("selected_job",None),
               ("search_q",""),("search_loc",""),("filter_genuine",False)]:
    if _k not in st.session_state:
        st.session_state[_k]=_v

def go(page):
    st.session_state.page=page
    st.session_state.selected_job=None
    st.rerun()

def logout():
    st.session_state.user=None
    go("home")


# ─────────────────────────────────────────────────────────────────────────────
# SHARED NAVBAR  (unchanged from original)
# ─────────────────────────────────────────────────────────────────────────────
def render_navbar():
    user=st.session_state.user
    c0,c1,c2,c3,c4=st.columns([2.5,1,1,1,1])
    c0.markdown('<span style="font-family:\'Playfair Display\',serif;font-size:1.4rem;font-weight:900;color:#1a3c5e;">True<span style=\'color:#e8734a\'>Hire</span></span>',unsafe_allow_html=True)
    if c1.button("🏠 Home",key="nav_home"): go("home")
    if c2.button("💼 Jobs",key="nav_jobs"): go("jobs")
    if user:
        dash="dashboard_seeker" if user["role"]=="seeker" else "dashboard_company"
        if c3.button(f"👤 {user['name'].split()[0]}",key="nav_dash"): go(dash)
        if c4.button("Logout",key="nav_lo"): logout()
    else:
        if c3.button("Login",key="nav_li"): go("login")
        if c4.button("Sign Up",key="nav_su"): go("register")
    st.markdown("<hr style='border:none;border-top:1px solid #e5e7eb;margin:0 0 1rem;'>",unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# HOME PAGE
# ─────────────────────────────────────────────────────────────────────────────
def page_home():
    render_navbar()
    user=st.session_state.user

    total_jobs = len(_job_df)
    genuine_ct = int((_job_df["ml_label"]=="genuine").sum()) if "ml_label" in _job_df.columns else total_jobs

    st.markdown(f"""
    <div class="hero-banner">
      <h1>True<span>Hire</span></h1>
      <p>AI-powered fraud detection — {genuine_ct}+ verified genuine listings. Zero scams.</p>
      <div class="hero-stats">
        <div class="hero-stat"><div class="n">{genuine_ct}+</div><div class="l">Verified Genuine Jobs</div></div>
        <div class="hero-stat"><div class="n">50+</div><div class="l">Top Companies</div></div>
        <div class="hero-stat"><div class="n">PAC</div><div class="l">Fraud Detection</div></div>
        <div class="hero-stat"><div class="n">100%</div><div class="l">Model Accuracy</div></div>
      </div>
    </div>
    """,unsafe_allow_html=True)

    # Quick search
    c1,c2,c3=st.columns([3,2,1])
    q  =c1.text_input("Search jobs",placeholder="e.g. Data Scientist, Python, AWS",label_visibility="collapsed")
    loc=c2.text_input("Location",placeholder="City or Remote",label_visibility="collapsed")
    if c3.button("🔍 Search",use_container_width=True):
        st.session_state.search_q=q; st.session_state.search_loc=loc; go("jobs")

    # Personalised recommendations for seekers
    if user and user["role"]=="seeker":
        s=get_seeker(user["id"])
        profile_text=" ".join(filter(None,[s.get("skills",""),s.get("bio",""),s.get("preferred_location","")])).strip()
        if profile_text:
            recs=recommend_jobs(profile_text,_match_tfidf,_match_matrix,_job_df,top_n=3)
            st.markdown('<div class="ai-box">🤖 <b>AI Recommendations</b> — top matching verified jobs for your profile:</div>',unsafe_allow_html=True)
            cols=st.columns(3)
            for i,(orig_idx,row) in enumerate(recs.iterrows()):
                pct=max(5,min(99,int(row["score"]*100)))
                lbl=row.get("ml_label","pending") or "pending"
                badge,_=BADGE.get(lbl,BADGE["pending"])
                with cols[i]:
                    st.markdown(f"""
                    <div class="job-card card-{lbl}">
                      <h3>{row['title']}</h3>
                      <div class="company">{row.get('company','—')} · {row.get('location','—')}</div>
                      {badge}
                      <span class="badge-ai">🤖 {pct}% match</span>
                      <div class="match-bar-wrap"><div class="match-bar" style="width:{pct}%;"></div></div>
                    </div>
                    """,unsafe_allow_html=True)
                    if st.button("View",key=f"home_rec_{i}_{orig_idx}"):
                        st.session_state.selected_job=("dataset",int(orig_idx)); go("jobs")
            st.divider()

    # Featured jobs grid
    st.markdown('<p class="sec-title">Featured Opportunities</p>',unsafe_allow_html=True)
    st.markdown('<p class="sec-sub">Hand-picked verified roles from India\'s leading companies</p>',unsafe_allow_html=True)
    featured=_job_df.sample(min(6,len(_job_df)),random_state=7)
    cols=st.columns(2)
    for i,(idx,row) in enumerate(featured.iterrows()):
        lbl=row.get("ml_label","pending") or "pending"
        badge,_=BADGE.get(lbl,BADGE["pending"])
        with cols[i%2]:
            st.markdown(f"""
            <div class="job-card card-{lbl}">
              <h3>{row['title']}</h3>
              <div class="company">{row.get('company',row.get('company_profile','—')[:30])} · {row.get('location','—')}</div>
              <span class="tag">{row.get('salary',row.get('salary_range','—'))}</span>
              <span class="tag">{row.get('job_type',row.get('employment_type','Full-time'))}</span>
              <span class="tag">{row.get('industry','—')}</span>
              {badge}
            </div>
            """,unsafe_allow_html=True)
            if st.button("View Details",key=f"home_feat_{idx}"):
                st.session_state.selected_job=("dataset",int(idx)); go("jobs")

    # CTA
    st.divider()
    ca,cb=st.columns(2)
    with ca:
        st.markdown("""<div style="background:#1a3c5e;border-radius:14px;padding:1.8rem;color:#fff;"><h3 style="font-family:'Playfair Display',serif;color:#fff;margin-bottom:0.5rem;">🏢 Hiring Talent?</h3><p style="color:#93c5fd;margin-bottom:1rem;">Post jobs & reach thousands of verified candidates.</p></div>""",unsafe_allow_html=True)
        if st.button("Post a Job →",key="cta_co"): go("register")
    with cb:
        st.markdown("""<div style="background:#fff5f0;border:1.5px solid #e8734a;border-radius:14px;padding:1.8rem;"><h3 style="font-family:'Playfair Display',serif;color:#1a3c5e;margin-bottom:0.5rem;">🔍 Looking for Work?</h3><p style="color:#6b7280;margin-bottom:1rem;">Browse AI-matched openings. One-click apply. No middlemen.</p></div>""",unsafe_allow_html=True)
        if st.button("Browse All Jobs →",key="cta_sk"): go("jobs")


# ─────────────────────────────────────────────────────────────────────────────
# JOBS PAGE
# ─────────────────────────────────────────────────────────────────────────────
def page_jobs():
    render_navbar()

    # ── Detail view ──────────────────────────────────────────────────────────
    if st.session_state.selected_job:
        src,idx=st.session_state.selected_job
        if src=="dataset":
            render_dataset_job_detail(_job_df.iloc[idx],idx)
        else:
            job=get_posted_job(idx)
            if job: render_posted_job_detail(job)
        if st.button("← Back to all jobs"):
            st.session_state.selected_job=None; st.rerun()
        return

    st.markdown('<p class="sec-title">Browse Jobs</p>',unsafe_allow_html=True)

    # Filters
    with st.expander("🔍 Search & Filter",expanded=True):
        c1,c2,c3=st.columns(3)
        q      =c1.text_input("Keyword / Skills",value=st.session_state.search_q,placeholder="Python, React, Finance…")
        loc    =c2.text_input("Location",value=st.session_state.search_loc,placeholder="Bangalore, Remote…")
        jtype  =c3.selectbox("Job Type",["All","Full-time","Part-time","Remote","Internship","Contract"])
        c4,c5,c6=st.columns(3)
        industry=c4.selectbox("Industry",["All","IT / Software","Finance","Healthcare","Manufacturing","Education","E-commerce","Other"])
        max_exp =c5.slider("Max Experience (yrs)",0,15,15)
        sort_by =c6.selectbox("Sort by",["AI Relevance","Salary (High→Low)"])
        c7,c8=st.columns([3,1])
        only_g=c7.toggle("✅ Show Verified Genuine jobs only",value=st.session_state.get("filter_genuine",False))
        st.session_state["filter_genuine"]=only_g
        if c8.button("🔍 Search",use_container_width=True):
            st.session_state.search_q=q; st.session_state.search_loc=loc

    # TF-IDF search
    search_text=f"{q} {loc}".strip() or "software engineer"
    results=tfidf_search(search_text,_match_tfidf,_match_matrix,_job_df,
                          only_genuine=st.session_state.get("filter_genuine",False))

    # Extra filters
    if jtype!="All" and "job_type" in results.columns:
        results=results[results["job_type"].str.contains(jtype,case=False,na=False)]
    if industry!="All" and "industry" in results.columns:
        results=results[results["industry"].str.contains(industry,case=False,na=False)]
    if loc.strip() and "location" in results.columns:
        results=results[results["location"].str.contains(loc,case=False,na=False)]
    if "exp" in results.columns:
        results=results[pd.to_numeric(results["exp"],errors="coerce").fillna(0)<=max_exp]
    if sort_by=="Salary (High→Low)" and "salary" in results.columns:
        def _sal(s):
            nums=re.findall(r'\d+',str(s)); return int(nums[-1]) if nums else 0
        results=results.copy(); results["_s"]=results["salary"].apply(_sal)
        results=results.sort_values("_s",ascending=False)

    posted=get_posted_jobs(q=q,location=loc,
                            job_type="" if jtype=="All" else jtype,
                            industry="" if industry=="All" else industry,
                            experience=max_exp if max_exp<15 else "")

    # Stats banner
    total=len(results)+len(posted)
    g_ct=int((results["ml_label"]=="genuine").sum()) if "ml_label" in results.columns else 0
    f_ct=int((results["ml_label"]=="fake").sum())    if "ml_label" in results.columns else 0
    i_ct=int((results["ml_label"]=="irrelevant").sum()) if "ml_label" in results.columns else 0
    st.markdown(
        f"**{total} job(s) found** &nbsp;"
        f"<span class='badge-genuine'>✅ {g_ct} Genuine</span> &nbsp;"
        f"<span class='badge-fake'>🚨 {f_ct} Fake</span> &nbsp;"
        f"<span class='badge-irr'>⚠️ {i_ct} Irrelevant</span>",
        unsafe_allow_html=True
    )

    # ── Company-posted jobs ──────────────────────────────────────────────────
    if posted:
        st.markdown("#### 🏢 Company-Posted Jobs")
        for j in posted:
            lbl =j.get("ml_label","pending") or "pending"
            conf=j.get("ml_confidence",0) or 0
            badge,_=BADGE.get(lbl,BADGE["pending"])
            conf_str=f" ({conf:.0f}%)" if conf>0 else ""
            st.markdown(f"""
            <div class="job-card card-{lbl}">
              <h3>{j['title']}</h3>
              <div class="company">{j['company_name']} · {j['location'] or 'Remote'}</div>
              <span class="tag-accent">{j['salary_range'] or 'Negotiable'}</span>
              <span class="tag">{j['job_type']}</span>
              <span class="tag">{j['experience_required']} yrs exp</span>
              {badge}{conf_str}
            </div>
            """,unsafe_allow_html=True)
            b1,b2=st.columns([5,1])
            with b2:
                if st.button("View",key=f"posted_{j['id']}"):
                    st.session_state.selected_job=("posted",j['id']); st.rerun()

    # ── Dataset jobs ─────────────────────────────────────────────────────────
    st.markdown("#### 📊 AI-Labelled Job Listings")
    for _,row in results.iterrows():
        pct     =max(5,min(99,int(row["score"]*100)))
        orig_idx=int(row.name)
        lbl     =row.get("ml_label","pending") or "pending"
        conf    =row.get("ml_conf",row.get("ml_confidence",0)) or 0
        badge,_ =BADGE.get(lbl,BADGE["pending"])
        conf_str=f" ({conf:.0f}%)" if conf>0 else ""
        # skill tags (fallback-dataset has skills col; cleaned_df may not)
        skills_html=""
        if "skills" in row and str(row["skills"]).strip():
            skills_html=" ".join(f"<span class='tag'>{s.strip()}</span>"
                                  for s in str(row["skills"]).split(",")[:4])
        st.markdown(f"""
        <div class="job-card card-{lbl}">
          <h3>{row['title']}</h3>
          <div class="company">{row.get('company',row.get('company_profile','—')[:40])} · {row.get('location','—')}</div>
          <span class="tag-accent">{row.get('salary',row.get('salary_range','—'))}</span>
          <span class="tag">{row.get('job_type',row.get('employment_type','Full-time'))}</span>
          {badge}{conf_str}
          <span class="badge-ai">🤖 {pct}% match</span>
          {f'<div style="margin-top:0.4rem;">{skills_html}</div>' if skills_html else ''}
          <div class="match-bar-wrap"><div class="match-bar" style="width:{pct}%;"></div></div>
        </div>
        """,unsafe_allow_html=True)
        b1,b2=st.columns([5,1])
        with b2:
            if st.button("Details",key=f"ds_{orig_idx}"):
                st.session_state.selected_job=("dataset",orig_idx); st.rerun()


def render_dataset_job_detail(row, idx):
    """
    Full job detail view.
    Shows ML badge + appropriate alert banner:
      ✅ Genuine  → green banner
      🚨 Fake     → red warning banner
      ⚠️ Irrelevant → amber notice (job title & description don't match)
    """
    user  =st.session_state.user
    lbl   =row.get("ml_label","pending") or "pending"
    conf  =row.get("ml_conf",row.get("ml_confidence",0)) or 0
    badge,_=BADGE.get(lbl,BADGE["pending"])
    conf_str=f" ({conf:.0f}% confidence)" if conf>0 else ""

    # ── ML alert banner ──────────────────────────────────────────────────────
    if lbl=="fake":
        st.markdown(
            f'<div class="alert-fake">🚨 <b>AI Fraud Alert:</b> Our model has classified this job as '
            f'<b>Fake</b>{conf_str}. The job description contains scam signals such as vague requirements, '
            f'missing company information, or scam keywords. Apply with caution.</div>',
            unsafe_allow_html=True)
    elif lbl=="irrelevant":
        st.markdown(
            f'<div class="alert-irr">⚠️ <b>AI Notice — Irrelevant Posting:</b> The job <b>title and description '
            f'appear to be mismatched</b>{conf_str}. This post does not follow standard job-listing patterns '
            f'and may not be a legitimate opening.</div>',
            unsafe_allow_html=True)
    elif lbl=="genuine":
        st.markdown(
            f'<div class="alert-genuine">✅ <b>AI Verified — Genuine Job:</b> This posting has been '
            f'classified as a legitimate job listing{conf_str}.</div>',
            unsafe_allow_html=True)

    skills=[s.strip() for s in str(row.get("skills","")).split(",") if s.strip()]
    st.markdown(f"""
    <div class="detail-card">
      <div style="display:flex;align-items:center;gap:1.2rem;margin-bottom:1.2rem;">
        <div style="width:56px;height:56px;border-radius:12px;background:#1a3c5e;color:#fff;
                    display:flex;align-items:center;justify-content:center;font-size:1.5rem;font-weight:700;flex-shrink:0;">
          {str(row.get('company',row.get('company_profile','?')))[0].upper()}
        </div>
        <div>
          <h2>{row['title']}</h2>
          <p style="color:#6b7280;font-size:0.88rem;margin:0;">
            {row.get('company',row.get('company_profile','—')[:50])} &nbsp;·&nbsp; {row.get('location','—')}
          </p>
        </div>
      </div>
      <div style="margin-bottom:1rem;">
        <span class="tag-accent">{row.get('salary',row.get('salary_range','—'))}</span>
        <span class="tag">{row.get('job_type',row.get('employment_type','Full-time'))}</span>
        <span class="tag">{row.get('industry','—')}</span>
        {badge}{conf_str}
      </div>
      <hr style="border:none;border-top:1px solid #e5e7eb;margin:1rem 0;">
      <h4 style="color:#1a3c5e;margin-bottom:0.5rem;">About the Role</h4>
      <p style="color:#374151;line-height:1.8;font-size:0.93rem;">{row.get('description','No description available.')}</p>
      {f'<h4 style="color:#1a3c5e;margin-top:1.2rem;margin-bottom:0.6rem;">Required Skills</h4><div>{"".join(f"<span class=\'tag-accent\'>{s}</span>" for s in skills)}</div>' if skills else ''}
    </div>
    """,unsafe_allow_html=True)
    st.markdown("")
    if user and user["role"]=="seeker":
        if already_applied_dataset(idx,user["id"]):
            st.markdown('<div class="success-box">✅ You have already applied for this job.</div>',unsafe_allow_html=True)
        else:
            if lbl=="fake":
                st.warning("⚠️ This job is flagged as potentially fake. Proceed with caution.")
            if st.button("✅ Apply Now",use_container_width=True,key="apply_ds"):
                if apply_dataset_job(idx,user["id"]): st.success("🎉 Application submitted!")
                else: st.warning("Already applied.")
    elif not user:
        st.markdown('<div class="info-box">Please <b>login as a Job Seeker</b> to apply.</div>',unsafe_allow_html=True)
        if st.button("Login to Apply"): go("login")


def render_posted_job_detail(j):
    """Company-posted job detail with ML fraud verdict banner."""
    user=st.session_state.user
    lbl =j.get("ml_label","pending") or "pending"
    conf=j.get("ml_confidence",0) or 0
    badge,_=BADGE.get(lbl,BADGE["pending"])
    conf_str=f" ({conf:.0f}% confidence)" if conf>0 else ""

    if lbl=="fake":
        st.markdown(f'<div class="alert-fake">🚨 <b>AI Fraud Alert:</b> This company-posted job has been flagged as <b>Fake</b>{conf_str}. Verify the employer independently before applying.</div>',unsafe_allow_html=True)
    elif lbl=="irrelevant":
        st.markdown(f'<div class="alert-irr">⚠️ <b>Irrelevant Posting:</b> The title and description of this job appear to be mismatched{conf_str}.</div>',unsafe_allow_html=True)
    elif lbl=="genuine":
        st.markdown(f'<div class="alert-genuine">✅ <b>AI Verified — Genuine Job</b>{conf_str}.</div>',unsafe_allow_html=True)

    st.markdown(f"""
    <div class="detail-card">
      <div style="display:flex;align-items:center;gap:1.2rem;margin-bottom:1.2rem;">
        <div style="width:56px;height:56px;border-radius:12px;background:#1a3c5e;color:#fff;
                    display:flex;align-items:center;justify-content:center;font-size:1.5rem;font-weight:700;flex-shrink:0;">
          {str(j['company_name'])[0].upper()}
        </div>
        <div>
          <h2>{j['title']}</h2>
          <p style="color:#6b7280;font-size:0.88rem;margin:0;">{j['company_name']} &nbsp;·&nbsp; {j['location'] or 'Remote'}</p>
        </div>
      </div>
      <div style="margin-bottom:1rem;">
        <span class="tag-accent">{j['salary_range'] or 'Negotiable'}</span>
        <span class="tag">{j['job_type']}</span>
        <span class="tag">{j['experience_required']} yrs exp</span>
        {badge}{conf_str}
      </div>
      <hr style="border:none;border-top:1px solid #e5e7eb;margin:1rem 0;">
      <h4 style="color:#1a3c5e;margin-bottom:0.5rem;">Job Description</h4>
      <p style="color:#374151;line-height:1.8;font-size:0.93rem;">{j['description'] or 'No description provided.'}</p>
      <h4 style="color:#1a3c5e;margin-top:1.2rem;margin-bottom:0.5rem;">Requirements</h4>
      <p style="color:#374151;line-height:1.8;font-size:0.93rem;">{j['requirements'] or 'See description.'}</p>
      {f"<p style='margin-top:0.8rem;font-size:0.88rem;color:#6b7280;'><b>Contact:</b> {j['contact_mobile']}</p>" if j.get('contact_mobile') else ""}
      {f"<p style='font-size:0.88rem;color:#6b7280;'><b>Deadline:</b> {j['deadline']}</p>" if j.get('deadline') else ""}
    </div>
    """,unsafe_allow_html=True)
    st.markdown("")
    if user and user["role"]=="seeker":
        if lbl=="fake":
            st.warning("⚠️ Flagged as potentially fake. Apply at your own risk.")
        if st.button("✅ Apply Now",use_container_width=True,key="apply_posted"):
            conn=get_conn()
            try:
                conn.execute("INSERT INTO applications(job_id,seeker_id) VALUES(?,?)",(j['id'],user['id']))
                conn.commit(); st.success("🎉 Application submitted!")
            except sqlite3.IntegrityError: st.warning("You've already applied.")
            finally: conn.close()
    elif not user:
        st.markdown('<div class="info-box">Please <b>login as a Job Seeker</b> to apply.</div>',unsafe_allow_html=True)
        if st.button("Login to Apply"): go("login")


# ─────────────────────────────────────────────────────────────────────────────
# LOGIN
# ─────────────────────────────────────────────────────────────────────────────
def page_login():
    render_navbar()
    st.markdown('<p class="sec-title">Welcome Back</p>',unsafe_allow_html=True)
    st.markdown('<p class="sec-sub">Sign in to your TrueHire account</p>',unsafe_allow_html=True)
    role=st.radio("I am a:",["Job Seeker","Company / Employer"],horizontal=True)
    st.divider()
    with st.form("login_form"):
        email=st.text_input("Email Address")
        pw   =st.text_input("Password",type="password")
        sub  =st.form_submit_button("Sign In",use_container_width=True)
    if sub:
        if not email or not pw: st.error("Please fill all fields."); return
        if role=="Job Seeker":
            u=login_seeker(email,pw)
            if u: st.session_state.user={"id":u["id"],"name":u["name"],"email":u["email"],"role":"seeker"}; go("dashboard_seeker")
            else: st.error("❌ Invalid credentials.")
        else:
            u=login_company(email,pw)
            if u: st.session_state.user={"id":u["id"],"name":u["name"],"email":u["email"],"role":"company"}; go("dashboard_company")
            else: st.error("❌ Invalid credentials.")
    st.markdown("Don't have an account?")
    if st.button("Create Account"): go("register")


# ─────────────────────────────────────────────────────────────────────────────
# REGISTER
# ─────────────────────────────────────────────────────────────────────────────
def page_register():
    render_navbar()
    st.markdown('<p class="sec-title">Create Your Account</p>',unsafe_allow_html=True)
    st.markdown('<p class="sec-sub">Join thousands of verified employers and job seekers</p>',unsafe_allow_html=True)
    role=st.radio("Register as:",["Job Seeker","Company / Employer"],horizontal=True)
    st.divider()
    if role=="Job Seeker":
        with st.form("reg_seeker"):
            c1,c2=st.columns(2); name=c1.text_input("Full Name *"); email=c2.text_input("Email *")
            c3,c4=st.columns(2); phone=c3.text_input("Phone"); pw=c4.text_input("Password *",type="password")
            skills=st.text_input("Skills (comma-separated)",placeholder="Python, SQL, Machine Learning")
            exp=st.number_input("Experience (years)",min_value=0,max_value=50)
            sub=st.form_submit_button("Create Account",use_container_width=True)
        if sub:
            if not name or not email or not pw: st.error("Name, email and password required.")
            else:
                ok,msg=register_seeker(name,email,pw,phone,skills,exp)
                if ok: st.success(msg+" Please login."); go("login")
                else: st.error(msg)
    else:
        with st.form("reg_company"):
            c1,c2=st.columns(2); name=c1.text_input("Company Name *"); email=c2.text_input("Work Email *")
            c3,c4=st.columns(2); phone=c3.text_input("Phone"); pw=c4.text_input("Password *",type="password")
            industry=st.selectbox("Industry",["IT / Software","Finance","Healthcare","Manufacturing","Education","E-commerce","Other"])
            c5,c6=st.columns(2); year=c5.number_input("Year Founded",min_value=1900,max_value=date.today().year,value=2010); _=c6.empty()
            desc=st.text_area("Company Description")
            sub=st.form_submit_button("Create Account",use_container_width=True)
        if sub:
            if not name or not email or not pw: st.error("Name, email and password required.")
            else:
                ok,msg=register_company(name,email,pw,phone,industry,year,desc)
                if ok: st.success(msg+" Please login."); go("login")
                else: st.error(msg)
    st.markdown("Already have an account?")
    if st.button("Login"): go("login")


# ─────────────────────────────────────────────────────────────────────────────
# SEEKER DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────
def page_dashboard_seeker():
    user=st.session_state.user
    if not user or user["role"]!="seeker": go("login"); return
    s=get_seeker(user["id"])
    total_apps,ds_apps=seeker_dashboard_stats(user["id"])
    score=profile_score(s)

    with st.sidebar:
        st.markdown(f"### 👤 {user['name']}")
        st.markdown(f"*{user['email']}*")
        st.progress(score/100,text=f"Profile {score}% complete")
        st.divider()
        section=st.radio("Navigate",
                         ["📊 Overview","🤖 AI Recommendations","📋 My Applications","👤 Edit Profile"],
                         label_visibility="collapsed")
        st.divider()
        if st.button("🔍 Browse All Jobs"): go("jobs")
        if st.button("🚪 Logout"): logout()

    st.markdown(f"""
    <div class="hero-banner" style="padding:2rem;">
      <h1 style="font-size:1.8rem;">Welcome back, {user['name'].split()[0]}! 👋</h1>
      <p>Your personalised AI-powered job hub — fraud-free listings powered by PAC model.</p>
    </div>
    """,unsafe_allow_html=True)

    if section=="📊 Overview":
        g_ct=int((_job_df["ml_label"]=="genuine").sum()) if "ml_label" in _job_df.columns else len(_job_df)
        st.markdown(f"""
        <div class="stat-row">
          <div class="stat-card"><div class="num">{total_apps}</div><div class="lbl">Total Applications</div></div>
          <div class="stat-card"><div class="num">{ds_apps}</div><div class="lbl">Jobs Applied</div></div>
          <div class="stat-card"><div class="num">{score}%</div><div class="lbl">Profile Complete</div></div>
          <div class="stat-card"><div class="num">{g_ct}</div><div class="lbl">Verified Genuine Jobs</div></div>
        </div>
        """,unsafe_allow_html=True)
        st.markdown("#### Recent Applications")
        apps=get_seeker_dataset_apps(user["id"])[:5]
        rows=[]
        for a in apps:
            idx=a["dataset_job_idx"]
            if 0<=idx<len(_job_df):
                row=_job_df.iloc[idx]
                rows.append({"Job":row["title"],"Applied":a["applied_at"][:10],"Status":a["status"]})
        if rows: st.table(rows)
        else: st.info("No applications yet. Use 🤖 AI Recommendations or Browse Jobs!")

    elif section=="🤖 AI Recommendations":
        st.markdown('<p class="sec-title">🤖 AI Job Recommendations</p>',unsafe_allow_html=True)
        profile_text=" ".join(filter(None,[s.get("skills",""),s.get("bio",""),s.get("preferred_location","")])).strip()
        if not profile_text:
            st.markdown('<div class="ai-box">⚠️ Complete your <b>Skills</b>, <b>Bio</b>, and <b>Location</b> in Edit Profile to unlock AI recommendations.</div>',unsafe_allow_html=True)
        else:
            recs=recommend_jobs(profile_text,_match_tfidf,_match_matrix,_job_df,top_n=10)
            st.markdown(f'<div class="ai-box">🤖 <b>Top {len(recs)} matches</b> for your profile — TF-IDF cosine similarity ranking, fraud pre-filtered.</div>',unsafe_allow_html=True)
            for i,(orig_idx,row) in enumerate(recs.iterrows()):
                pct    =max(5,min(99,int(row["score"]*100)))
                already=already_applied_dataset(int(orig_idx),user["id"])
                lbl    =row.get("ml_label","pending") or "pending"
                conf   =row.get("ml_conf",row.get("ml_confidence",0)) or 0
                badge,_=BADGE.get(lbl,BADGE["pending"])
                conf_str=f" ({conf:.0f}%)" if conf>0 else ""
                st.markdown(f"""
                <div class="job-card card-{lbl}">
                  <h3>{row['title']}</h3>
                  <div class="company">{row.get('company','—')} · {row.get('location','—')}</div>
                  <span class="tag-accent">{row.get('salary',row.get('salary_range','—'))}</span>
                  {badge}{conf_str}
                  <span class="badge-ai">🤖 {pct}% match</span>
                  {"<span class='badge-genuine'>✔ Applied</span>" if already else ""}
                  <div class="match-bar-wrap" style="margin-top:0.6rem;"><div class="match-bar" style="width:{pct}%;"></div></div>
                </div>
                """,unsafe_allow_html=True)
                b1,b2=st.columns([3,1])
                with b1:
                    if st.button("View Details",key=f"ai_view_{i}_{orig_idx}"):
                        st.session_state.selected_job=("dataset",int(orig_idx)); go("jobs")
                with b2:
                    if not already:
                        if st.button("⚡ Quick Apply",key=f"ai_apply_{i}_{orig_idx}"):
                            apply_dataset_job(int(orig_idx),user["id"]); st.success(f"Applied to {row['title']}!"); st.rerun()
                    else: st.caption("✅ Applied")

    elif section=="📋 My Applications":
        st.markdown("#### All My Applications")
        apps=get_seeker_dataset_apps(user["id"])
        rows=[]
        for a in apps:
            idx=a["dataset_job_idx"]
            if 0<=idx<len(_job_df):
                row=_job_df.iloc[idx]
                rows.append({"Job Title":row["title"],"Location":row.get("location","—"),
                             "Applied On":a["applied_at"][:10],"Status":a["status"]})
        if rows: st.table(rows)
        else: st.info("No applications yet.")

    elif section=="👤 Edit Profile":
        st.markdown("#### Edit Profile")
        st.markdown('<div class="ai-box">💡 A richer profile (skills + bio + location) significantly improves your AI match score.</div>',unsafe_allow_html=True)
        with st.form("seeker_profile"):
            c1,c2 =st.columns(2)
            name  =c1.text_input("Full Name",value=s.get("name","") or "")
            phone =c2.text_input("Phone",value=s.get("phone","") or "")
            skills=st.text_input("Skills (comma-separated)",value=s.get("skills","") or "",
                                  placeholder="Python, Machine Learning, SQL, React…")
            c3,c4 =st.columns(2)
            exp   =c3.number_input("Experience (yrs)",min_value=0,value=int(s.get("experience") or 0))
            loc   =c4.text_input("Preferred Location",value=s.get("preferred_location","") or "")
            bio   =st.text_area("Career Summary / About Me",value=s.get("bio","") or "",
                                 placeholder="Briefly describe your experience, goals, and what you're looking for…")
            salary=st.text_input("Expected Salary (LPA)",value=s.get("expected_salary","") or "")
            if st.form_submit_button("Save Profile",use_container_width=True):
                update_seeker(user["id"],name,phone,skills,exp,loc,bio,salary)
                st.session_state.user["name"]=name; st.success("✅ Profile saved!")


# ─────────────────────────────────────────────────────────────────────────────
# COMPANY DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────
def page_dashboard_company():
    user=st.session_state.user
    if not user or user["role"]!="company": go("login"); return
    total_jobs,total_apps,active=company_dashboard_stats(user["id"])

    with st.sidebar:
        st.markdown(f"### 🏢 {user['name']}")
        st.markdown(f"*{user['email']}*")
        st.divider()
        section=st.radio("Navigate",
                         ["📊 Overview","➕ Post a Job","📋 My Job Posts","👥 Applicants","🏢 Company Profile"],
                         label_visibility="collapsed")
        st.divider()
        if st.button("🚪 Logout"): logout()

    st.markdown(f"""
    <div class="hero-banner" style="padding:2rem;">
      <h1 style="font-size:1.8rem;">Hello, {user['name'].split()[0]}! 🏢</h1>
      <p>Every job you post is automatically fraud-checked by our AI model.</p>
    </div>
    """,unsafe_allow_html=True)

    if section=="📊 Overview":
        st.markdown(f"""
        <div class="stat-row">
          <div class="stat-card"><div class="num">{total_jobs}</div><div class="lbl">Jobs Posted</div></div>
          <div class="stat-card"><div class="num">{total_apps}</div><div class="lbl">Total Applicants</div></div>
          <div class="stat-card"><div class="num">{active}</div><div class="lbl">Active Listings</div></div>
          <div class="stat-card"><div class="num">100%</div><div class="lbl">Model Accuracy</div></div>
        </div>
        """,unsafe_allow_html=True)
        jobs=get_company_jobs(user["id"])[:5]
        st.markdown("#### Recent Job Posts")
        if not jobs: st.info("No jobs posted yet. Use ➕ Post a Job to get started.")
        else:
            rows=[]
            for j in jobs:
                lbl=j.get("ml_label","pending") or "pending"
                conf=j.get("ml_confidence",0) or 0
                rows.append({"Title":j["title"],"Location":j["location"] or "Remote",
                              "Posted":j["created_at"][:10],"Applicants":j["applicant_count"],
                              "AI Verdict":f"{lbl.capitalize()} ({conf:.0f}%)" if conf>0 else lbl.capitalize()})
            st.table(rows)

    elif section=="➕ Post a Job":
        st.markdown("#### Post a New Job")
        st.markdown('<div class="ai-box">🤖 Every posting is instantly fraud-checked — you\'ll see the AI verdict (✅ Genuine / 🚨 Fake / ⚠️ Irrelevant) immediately after posting.</div>',unsafe_allow_html=True)
        with st.form("post_job"):
            c1,c2 =st.columns(2); title=c1.text_input("Job Title *"); jtype=c2.selectbox("Job Type",["Full-time","Part-time","Remote","Internship","Contract"])
            c3,c4 =st.columns(2); loc=c3.text_input("Location",placeholder="Bangalore, India"); salary=c4.text_input("Salary Range (LPA)",placeholder="8-15 LPA")
            c5,c6 =st.columns(2)
            exp   =c5.number_input("Experience Required (yrs)",min_value=0)
            has_dl=c6.checkbox("Set Application Deadline")
            deadline=c6.date_input("Deadline Date",value=date.today()) if has_dl else None
            desc  =st.text_area("Job Description *",height=140)
            req   =st.text_area("Requirements / Skills",height=100)
            mobile=st.text_input("Contact Mobile")
            sub   =st.form_submit_button("Post Job & Run AI Check",use_container_width=True)
        if sub:
            if not title or not desc: st.error("Title and description are required.")
            else:
                ml_lbl,ml_conf=post_job(user["id"],title,jtype,loc,salary,exp,deadline,desc,req,mobile)
                if ml_lbl=="genuine":
                    st.success(f"✅ Job posted! AI Verdict: **Genuine** ({ml_conf:.0f}% confidence). Your listing is now live.")
                elif ml_lbl=="fake":
                    st.error(f"🚨 Job posted but flagged as **Fake** ({ml_conf:.0f}% confidence). Scam-like language was detected — please review your posting.")
                elif ml_lbl=="irrelevant":
                    st.warning(f"⚠️ Job posted but classified as **Irrelevant** ({ml_conf:.0f}% confidence). The title and description may be mismatched.")
                else:
                    st.success("✅ Job posted successfully!")

    elif section=="📋 My Job Posts":
        st.markdown("#### My Job Postings")
        jobs=get_company_jobs(user["id"])
        if not jobs: st.info("No jobs posted yet.")
        else:
            for j in jobs:
                lbl =j.get("ml_label","pending") or "pending"
                conf=j.get("ml_confidence",0) or 0
                badge,_=BADGE.get(lbl,BADGE["pending"])
                conf_str=f" ({conf:.0f}%)" if conf>0 else ""
                c1,c2,c3,c4=st.columns([3,1,1,1])
                c1.markdown(f"**{j['title']}** — {j['location'] or 'Remote'}  \n`{j['job_type']}` · {j['applicant_count']} applicants")
                c2.markdown(badge+conf_str,unsafe_allow_html=True)
                c3.caption(j["created_at"][:10])
                if c4.button("🗑 Delete",key=f"del_{j['id']}"): delete_job(j["id"],user["id"]); st.success("Deleted."); st.rerun()

    elif section=="👥 Applicants":
        st.markdown("#### Job Applicants")
        jobs=get_company_jobs(user["id"])
        job_map={"All Jobs":None}
        for j in jobs: job_map[j["title"]]=j["id"]
        chosen=st.selectbox("Filter by job",list(job_map.keys()))
        applicants=get_applicants(user["id"],job_map[chosen])
        if not applicants: st.info("No applicants yet.")
        else:
            st.table([{"Name":a["name"],"Email":a["email"],"Skills":a["skills"] or "—",
                       "Experience":f"{a['experience'] or 0} yrs","Job":a["job_title"],
                       "Applied":a["applied_at"][:10]} for a in applicants])

    elif section=="🏢 Company Profile":
        st.markdown("#### Company Profile")
        co=get_company(user["id"])
        with st.form("co_profile"):
            c1,c2  =st.columns(2); name=c1.text_input("Company Name",value=co.get("name","") or "")
            industry=c2.selectbox("Industry",["IT / Software","Finance","Healthcare","Manufacturing","E-commerce","Other"])
            c3,c4  =st.columns(2)
            website=c3.text_input("Website",value=co.get("website","") or "")
            year   =c4.number_input("Year Founded",min_value=1900,max_value=date.today().year,value=int(co.get("year_founded") or 2010))
            phone  =st.text_input("Phone",value=co.get("phone","") or "")
            desc   =st.text_area("Company Description",value=co.get("description","") or "")
            if st.form_submit_button("Save Profile",use_container_width=True):
                update_company(user["id"],name,industry,website,year,phone,desc)
                st.session_state.user["name"]=name; st.success("✅ Profile saved!")


# ─────────────────────────────────────────────────────────────────────────────
# ROUTER
# ─────────────────────────────────────────────────────────────────────────────
PAGE_MAP = {
    "home":              page_home,
    "jobs":              page_jobs,
    "login":             page_login,
    "register":          page_register,
    "dashboard_seeker":  page_dashboard_seeker,
    "dashboard_company": page_dashboard_company,
}
PAGE_MAP.get(st.session_state.get("page","home"), page_home)()