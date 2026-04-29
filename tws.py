"""
TrueHire — Verified Job Portal
Single-file Streamlit app: frontend + SQLite backend + TF-IDF search + PAC model
Integrated with:
  - cleaned_jobs.csv  (real dataset)
  - data_preprocessing.py  logic (text cleaning)
  - tfidf_features.py  logic (feature extraction)
  - train_model.py / pac_model.pkl  (fake vs real vs irrelevant classification)

Run:
  pip install streamlit scikit-learn pandas numpy scipy joblib
  streamlit run truehireweb.py
"""

import os, re, warnings, hashlib
import sqlite3
from datetime import date

import numpy as np
import pandas as pd
import joblib
import streamlit as st
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TrueHire — Verified Job Portal",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="collapsed",
)

DB_PATH = "truehire.db"

# ─────────────────────────────────────────────────────────────────────────────
# PATHS  —  resolve relative to this script so it works from any CWD
# ─────────────────────────────────────────────────────────────────────────────
_BASE = os.path.dirname(os.path.abspath(__file__))
CLEANED_CSV_PATH   = os.path.join(_BASE, "cleaned_jobs.csv")
PAC_MODEL_PATH     = os.path.join(_BASE, "pac_model.pkl")
VECTORIZER_PATH    = os.path.join(_BASE, "tfidf_vectorizer.pkl")
META_FEATURES_PATH = os.path.join(_BASE, "meta_features.npy")

# ─────────────────────────────────────────────────────────────────────────────
# SCAM KEYWORDS  (from tfidf_features.py)
# ─────────────────────────────────────────────────────────────────────────────
SCAM_KEYWORDS = [
    "no investment", "quick earning", "earn from home",
    "easy money", "guaranteed income", "unlimited income",
    "be your own boss", "daily payout", "weekly payout",
    "risk free", "free registration", "mlm",
    "network marketing", "instant payment",
    "make money fast", "data entry work",
    "earn money fast", "work from home earn",
]
SCAM_PATTERN = "|".join(re.escape(k) for k in SCAM_KEYWORDS)

# ─────────────────────────────────────────────────────────────────────────────
# IRRELEVANT JOBS  (added as requested — label = "irrelevant")
# ─────────────────────────────────────────────────────────────────────────────
IRRELEVANT_JOBS = [
    {
        "title": "Astrological Chart Reader",
        "location": "Remote, Worldwide",
        "company_profile": "We offer spiritual guidance and cosmic advice.",
        "description": "Read star charts and provide daily horoscope advice via WhatsApp. No qualifications needed — just belief in the cosmos.",
        "requirements": "Know your sun sign. Must own a crystal ball (optional).",
        "salary_range": "Earn unlimited* (*no guarantee)",
        "industry": "Spirituality / Wellness",
        "employment_type": "Freelance",
        "label": "irrelevant",
    },
    {
        "title": "Professional Netflix Watcher",
        "location": "Your Couch, Anywhere",
        "company_profile": "StreamRate Inc. — we rate content so you don't have to.",
        "description": "Watch Netflix shows 8 hours/day and fill out a 5-minute survey. Must have fast WiFi and love of binge-watching.",
        "requirements": "Netflix account. Working eyes. Stable internet.",
        "salary_range": "₹500/day paid in OTT credits",
        "industry": "Entertainment",
        "employment_type": "Part-time",
        "label": "irrelevant",
    },
    {
        "title": "Zombie Apocalypse Survival Consultant",
        "location": "Undisclosed Bunker Location",
        "company_profile": "ZombiePrep LLC — preparing humanity for the inevitable.",
        "description": "Train civilians in zombie evasion tactics. Must be able to run backwards while waving a torch. Crossbow skills a plus.",
        "requirements": "Survival instinct. No fear of the undead.",
        "salary_range": "Payment in canned goods and ammunition",
        "industry": "Preparedness / Defense",
        "employment_type": "Contract",
        "label": "irrelevant",
    },
    {
        "title": "Moon Dust Collector",
        "location": "Moon (Occasional Earth travel)",
        "company_profile": "LunarMineCo — we make space work for you.",
        "description": "Collect regolith samples on the lunar surface. Must be comfortable with zero gravity and 2-week communication blackouts.",
        "requirements": "Space suit provided. Must hold breath for 8 seconds.",
        "salary_range": "Pay negotiable upon safe return",
        "industry": "Space / Mining",
        "employment_type": "Full-time",
        "label": "irrelevant",
    },
    {
        "title": "Chief Snack Officer (CSO)",
        "location": "Any Office Pantry, India",
        "company_profile": "SnackHub — democratizing snack access across corporates.",
        "description": "Responsible for curating the weekly office snack list, taste-testing new chips, and maintaining biscuit inventory. Reports to CEO (Chief Eating Officer).",
        "requirements": "Strong opinions about Parle-G. Dislike of raisins preferred.",
        "salary_range": "₹0 + unlimited snacks",
        "industry": "Food & Beverages",
        "employment_type": "Volunteer",
        "label": "irrelevant",
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# CSS
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
.hero-banner::after{
  content:'';position:absolute;right:-80px;top:-80px;
  width:350px;height:350px;border-radius:50%;background:rgba(255,255,255,0.05);
}
.hero-banner h1{font-family:'Playfair Display',serif;font-size:clamp(2rem,4vw,3rem);margin:0 0 0.5rem;}
.hero-banner h1 span{color:#fbbf24;}
.hero-banner p{opacity:0.88;font-size:1.05rem;max-width:520px;margin-bottom:0;}
.hero-stats{display:flex;gap:2rem;flex-wrap:wrap;margin-top:1.5rem;}
.hero-stat .n{font-size:1.8rem;font-weight:700;color:#fbbf24;}
.hero-stat .l{font-size:0.78rem;opacity:0.8;margin-top:0.1rem;}

.stat-row{display:flex;gap:1rem;margin-bottom:1.5rem;flex-wrap:wrap;}
.stat-card{
  background:#fff;border-radius:12px;padding:1.1rem 1.4rem;
  border:1px solid var(--border);flex:1;min-width:130px;text-align:center;
  box-shadow:0 2px 12px rgba(26,60,94,0.07);
}
.stat-card .num{font-size:1.9rem;font-weight:700;color:var(--primary);line-height:1;}
.stat-card .lbl{font-size:0.78rem;color:#6b7280;margin-top:0.3rem;}

.job-card{
  background:#fff;border-radius:12px;padding:1.2rem 1.5rem;
  border:1px solid var(--border);margin-bottom:0.75rem;
  box-shadow:0 2px 8px rgba(26,60,94,0.05);
  transition:box-shadow .2s,border-color .2s;
}
.job-card:hover{box-shadow:0 8px 28px rgba(26,60,94,0.13);border-color:var(--accent);}
.job-card h3{margin:0 0 0.15rem;color:var(--primary);font-size:1rem;font-weight:600;}
.job-card .company{font-size:0.84rem;color:#6b7280;margin-bottom:0.6rem;}
.match-bar-wrap{background:#f3f4f6;border-radius:50px;height:6px;margin-top:0.5rem;}
.match-bar{background:linear-gradient(90deg,var(--accent),#fbbf24);border-radius:50px;height:6px;}

.tag{display:inline-block;background:#f3f4f6;border:1px solid var(--border);border-radius:50px;padding:0.18rem 0.7rem;font-size:0.73rem;color:#374151;margin-right:0.3rem;margin-top:0.3rem;}
.tag-accent{background:#fff5f0;border-color:#e8734a;color:#e8734a;}
.tag-green{background:#f0fdf4;border-color:#16a34a;color:#16a34a;}
.badge-verified{background:#e8f5e9;color:#16a34a;border-radius:50px;padding:0.15rem 0.65rem;font-size:0.72rem;font-weight:600;}
.badge-fake{background:#fff0f0;color:#dc2626;border:1px solid #fca5a5;border-radius:50px;padding:0.15rem 0.65rem;font-size:0.72rem;font-weight:700;}
.badge-irrelevant{background:#fefce8;color:#a16207;border:1px solid #fde68a;border-radius:50px;padding:0.15rem 0.65rem;font-size:0.72rem;font-weight:600;}
.badge-ai{background:#fef3c7;color:#d97706;border-radius:50px;padding:0.15rem 0.65rem;font-size:0.72rem;font-weight:600;}

.sec-title{font-family:'Playfair Display',serif;color:var(--primary);font-size:1.45rem;margin-bottom:0.15rem;}
.sec-sub{color:#6b7280;font-size:0.88rem;margin-bottom:1rem;}

div.stButton>button{background:var(--accent)!important;color:#fff!important;border:none!important;border-radius:8px!important;font-weight:600!important;transition:background .2s,transform .15s!important;}
div.stButton>button:hover{background:#cf5a32!important;transform:translateY(-1px);}

.stTextInput>div>input,.stSelectbox>div>div,.stTextArea>div>textarea,.stNumberInput>div>input,.stDateInput>div>input{border-radius:8px!important;border:1.5px solid var(--border)!important;font-family:'DM Sans',sans-serif!important;}
.stTextInput>div>input:focus,.stTextArea>div>textarea:focus{border-color:var(--accent)!important;box-shadow:0 0 0 3px rgba(232,115,74,.12)!important;}

.info-box{background:#eff6ff;border-left:4px solid #3b82f6;border-radius:8px;padding:0.8rem 1rem;margin-bottom:1rem;font-size:0.9rem;}
.success-box{background:#f0fdf4;border-left:4px solid #16a34a;border-radius:8px;padding:0.8rem 1rem;margin-bottom:1rem;font-size:0.9rem;}
.ai-box{background:#fffbeb;border-left:4px solid #f59e0b;border-radius:8px;padding:0.8rem 1rem;margin-bottom:1rem;font-size:0.9rem;}
.warn-box{background:#fff0f0;border-left:4px solid #dc2626;border-radius:8px;padding:0.8rem 1rem;margin-bottom:1rem;font-size:0.9rem;}

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


# ─────────────────────────────────────────────────────────────────────────────
# DATA PREPROCESSING (from data_preprocessing.py)
# ─────────────────────────────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    """Clean a single text field — mirrors data_preprocessing.py:clean_text()."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE EXTRACTION (from tfidf_features.py)
# ─────────────────────────────────────────────────────────────────────────────
def create_combined_text_series(df: pd.DataFrame) -> pd.Series:
    """Combine text columns — mirrors tfidf_features.py:create_combined_text()."""
    cols = ["title", "company_profile", "description",
            "requirements", "salary_range", "location", "industry"]
    cols = [c for c in cols if c in df.columns]
    return df[cols].fillna("").astype(str).agg(" ".join, axis=1)


def transform_for_classification(text_dict: dict, vectorizer, meta_cols: list):
    """
    Build feature vector for a single job — mirrors tfidf_features.py:transform_single().
    Returns sparse matrix ready for pac_model.predict().
    """
    def _clean(t):
        if not isinstance(t, str): return ""
        t = t.lower()
        t = re.sub(r"<[^>]+>", " ", t)
        t = re.sub(r"http\S+", " url ", t)
        t = re.sub(r"\b\d{10,}\b", " phone ", t)
        t = re.sub(r"[^\w\s]", " ", t)
        return re.sub(r"\s+", " ", t).strip()

    combined = " ".join([
        _clean(text_dict.get("title", "")),
        _clean(text_dict.get("company_profile", "")),
        _clean(text_dict.get("description", "")),
        _clean(text_dict.get("requirements", "")),
        _clean(text_dict.get("salary_range", "")),
        _clean(text_dict.get("location", "")),
        _clean(text_dict.get("industry", "")),
    ])

    X_tfidf = vectorizer.transform([combined])

    meta = {
        "has_scam_keywords": int(bool(re.search(SCAM_PATTERN, combined))),
        "has_salary":        int(bool(text_dict.get("salary_range", "").strip())),
        "has_company_desc":  int(bool(text_dict.get("company_profile", "").strip())),
        "has_phone_in_desc": int(bool(re.search(r"\b\d{10}\b", combined))),
        "title_len":         len(text_dict.get("title", "").split()),
        "desc_len":          len(text_dict.get("description", "").split()),
    }

    if meta_cols:
        X_meta = csr_matrix([[meta.get(c, 0) for c in meta_cols]])
        return hstack([X_tfidf, X_meta])
    return X_tfidf


# ─────────────────────────────────────────────────────────────────────────────
# LABEL HELPERS
# ─────────────────────────────────────────────────────────────────────────────
# PAC model classes: 0 = Real, 1 = Fake, 2 = Irrelevant
CLASS_MAP = {0: "Real", 1: "Fake", 2: "Irrelevant"}

def label_badge(predicted_class: int) -> str:
    if predicted_class == 0:
        return "<span class='badge-verified'>✔ Real Job</span>"
    elif predicted_class == 1:
        return "<span class='badge-fake'>⚠ Fake Job</span>"
    else:
        return "<span class='badge-irrelevant'>❓ Irrelevant</span>"

def dataset_label_badge(label_val) -> str:
    """Badge from the CSV fraudulent column: 0=Real, 1=Fake."""
    try:
        v = int(label_val)
    except Exception:
        v = -1
    if v == 0:
        return "<span class='badge-verified'>✔ Real (dataset)</span>"
    elif v == 1:
        return "<span class='badge-fake'>⚠ Fake (dataset)</span>"
    return ""


# ─────────────────────────────────────────────────────────────────────────────
# ML ENGINE  —  loads PAC model + TF-IDF vectorizer + cleaned_jobs.csv
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="🤖 Loading AI classification engine…")
def build_ml_engine():
    """
    1. Load cleaned_jobs.csv  (data_preprocessing output).
    2. Load tfidf_vectorizer.pkl  (tfidf_features output).
    3. Load pac_model.pkl  (train_model output).
    4. Build a secondary TF-IDF matrix for cosine-similarity job search.
    5. Run pac_model on every CSV row to pre-compute predicted labels.
    Returns: df, search_tfidf, search_matrix, pac_model, pac_vec, meta_cols
    """
    # ── Load dataset ──────────────────────────────────────────────────────────
    df = pd.read_csv(cleaned_jobs.csv)

    # Normalise label column
    if "fraudulent" in df.columns and "label" not in df.columns:
        df = df.rename(columns={"fraudulent": "label"})
    if "label" not in df.columns:
        df["label"] = 0

    df["label"] = pd.to_numeric(df["label"], errors="coerce").fillna(0).astype(int)
    df = df.reset_index(drop=True)

    # Fill key text columns
    for col in ["title", "company_profile", "description",
                 "requirements", "salary_range", "location",
                 "industry", "employment_type"]:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("")

    # ── Load PAC model + vectorizer ───────────────────────────────────────────
    pac_model = joblib.load(pac_model.pkl)
    pac_vec   = joblib.load(tfidf_vectorizer.pkl)
    meta_cols = list(np.load(meta_features.pkl, allow_pickle=True))

    # ── Pre-classify every row using the PAC model ────────────────────────────
    combined_texts = create_combined_text_series(df)
    X_all = pac_vec.transform(combined_texts)
    df["pac_prediction"] = pac_model.predict(X_all)

    # ── Build a separate TF-IDF matrix for cosine-similarity search ───────────
    search_corpus = (
        df["title"] + " " +
        df["industry"].str.replace(",", " ") + " " +
        df["description"]
    )
    search_tfidf = TfidfVectorizer(
        max_features=6000,
        ngram_range=(1, 2),
        stop_words="english",
        sublinear_tf=True,
    )
    search_matrix = search_tfidf.fit_transform(search_corpus)

    return df, search_tfidf, search_matrix, pac_model, pac_vec, meta_cols


def classify_job(text_dict: dict, pac_model, pac_vec, meta_cols) -> int:
    """Classify a single job posting. Returns 0=Real, 1=Fake, 2=Irrelevant."""
    X = transform_for_classification(text_dict, pac_vec, meta_cols)
    return int(pac_model.predict(X)[0])


def tfidf_search(query: str, df, search_tfidf, search_matrix):
    """Return df sorted by cosine similarity to query."""
    if not query.strip():
        return df.copy().assign(score=1.0)
    qvec = search_tfidf.transform([query])
    sims = cosine_similarity(qvec, search_matrix).flatten()
    return df.copy().assign(score=sims).sort_values("score", ascending=False)


# ─────────────────────────────────────────────────────────────────────────────
# DATABASE
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
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(company_id) REFERENCES companies(id)
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
        dataset_job_idx INTEGER NOT NULL,
        seeker_id INTEGER NOT NULL,
        status TEXT DEFAULT 'Under Review',
        applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(dataset_job_idx, seeker_id),
        FOREIGN KEY(seeker_id) REFERENCES seekers(id)
    );
    """)
    conn.commit(); conn.close()


init_db()


def hash_pw(pw): return hashlib.sha256(pw.encode()).hexdigest()


# ── Company ──
def register_company(name, email, pw, phone, industry, year, desc):
    conn = get_conn()
    try:
        conn.execute("INSERT INTO companies(name,email,password_hash,phone,industry,year_founded,description) VALUES(?,?,?,?,?,?,?)",
                     (name, email, hash_pw(pw), phone, industry, year, desc))
        conn.commit(); return True, "Company registered!"
    except sqlite3.IntegrityError: return False, "Email already registered."
    finally: conn.close()


def login_company(email, pw):
    conn = get_conn()
    row = conn.execute("SELECT * FROM companies WHERE email=? AND password_hash=?",
                       (email, hash_pw(pw))).fetchone()
    conn.close(); return dict(row) if row else None


def get_company(cid):
    conn = get_conn()
    row = conn.execute("SELECT * FROM companies WHERE id=?", (cid,)).fetchone()
    conn.close(); return dict(row) if row else {}


def update_company(cid, name, industry, website, year, phone, desc):
    conn = get_conn()
    conn.execute("UPDATE companies SET name=?,industry=?,website=?,year_founded=?,phone=?,description=? WHERE id=?",
                 (name, industry, website, year, phone, desc, cid))
    conn.commit(); conn.close()


# ── Seeker ──
def register_seeker(name, email, pw, phone, skills, exp):
    conn = get_conn()
    try:
        conn.execute("INSERT INTO seekers(name,email,password_hash,phone,skills,experience) VALUES(?,?,?,?,?,?)",
                     (name, email, hash_pw(pw), phone, skills, exp))
        conn.commit(); return True, "Account created!"
    except sqlite3.IntegrityError: return False, "Email already registered."
    finally: conn.close()


def login_seeker(email, pw):
    conn = get_conn()
    row = conn.execute("SELECT * FROM seekers WHERE email=? AND password_hash=?",
                       (email, hash_pw(pw))).fetchone()
    conn.close(); return dict(row) if row else None


def get_seeker(sid):
    conn = get_conn()
    row = conn.execute("SELECT * FROM seekers WHERE id=?", (sid,)).fetchone()
    conn.close(); return dict(row) if row else {}


def update_seeker(sid, name, phone, skills, exp, loc, bio, salary):
    conn = get_conn()
    conn.execute("UPDATE seekers SET name=?,phone=?,skills=?,experience=?,preferred_location=?,bio=?,expected_salary=? WHERE id=?",
                 (name, phone, skills, exp, loc, bio, salary, sid))
    conn.commit(); conn.close()


def profile_score(s):
    fields = [s.get("name"), s.get("phone"), s.get("skills"),
               s.get("bio"), s.get("preferred_location"), s.get("expected_salary")]
    return int(sum(1 for f in fields if f) / len(fields) * 100)


# ── Posted Jobs ──
def post_job(cid, title, jtype, loc, salary, exp, deadline, desc, req, mobile):
    conn = get_conn()
    conn.execute("INSERT INTO jobs(company_id,title,job_type,location,salary_range,experience_required,deadline,description,requirements,contact_mobile) VALUES(?,?,?,?,?,?,?,?,?,?)",
                 (cid, title, jtype, loc, salary, exp, str(deadline) if deadline else None, desc, req, mobile))
    conn.commit(); conn.close()


def get_posted_jobs(q="", location="", job_type="", industry="", experience="", limit=200):
    conn = get_conn()
    sql = "SELECT j.*,c.name AS company_name,c.industry FROM jobs j JOIN companies c ON j.company_id=c.id WHERE 1=1"
    params = []
    if q: sql += " AND (j.title LIKE ? OR j.description LIKE ?)"; params += [f"%{q}%", f"%{q}%"]
    if location: sql += " AND j.location LIKE ?"; params.append(f"%{location}%")
    if job_type: sql += " AND j.job_type=?"; params.append(job_type)
    if experience: sql += " AND j.experience_required <= ?"; params.append(int(experience))
    if industry: sql += " AND c.industry=?"; params.append(industry)
    sql += " ORDER BY j.created_at DESC LIMIT ?"; params.append(limit)
    rows = conn.execute(sql, params).fetchall(); conn.close()
    return [dict(r) for r in rows]


def get_posted_job(jid):
    conn = get_conn()
    row = conn.execute("SELECT j.*,c.name AS company_name FROM jobs j JOIN companies c ON j.company_id=c.id WHERE j.id=?", (jid,)).fetchone()
    conn.close(); return dict(row) if row else None


def get_company_jobs(cid):
    conn = get_conn()
    rows = conn.execute("SELECT j.*,COUNT(a.id) AS applicant_count FROM jobs j LEFT JOIN applications a ON a.job_id=j.id WHERE j.company_id=? GROUP BY j.id ORDER BY j.created_at DESC", (cid,)).fetchall()
    conn.close(); return [dict(r) for r in rows]


def delete_job(jid, cid):
    conn = get_conn()
    conn.execute("DELETE FROM applications WHERE job_id=?", (jid,))
    conn.execute("DELETE FROM jobs WHERE id=? AND company_id=?", (jid, cid))
    conn.commit(); conn.close()


def get_applicants(cid, job_id=None):
    conn = get_conn()
    sql = "SELECT s.name,s.email,s.skills,s.experience,a.applied_at,a.status,j.title AS job_title FROM applications a JOIN seekers s ON s.id=a.seeker_id JOIN jobs j ON j.id=a.job_id WHERE j.company_id=?"
    params = [cid]
    if job_id: sql += " AND a.job_id=?"; params.append(job_id)
    rows = conn.execute(sql + " ORDER BY a.applied_at DESC", params).fetchall()
    conn.close(); return [dict(r) for r in rows]


# ── Dataset Applications ──
def apply_dataset_job(idx, sid):
    conn = get_conn()
    try:
        conn.execute("INSERT INTO dataset_applications(dataset_job_idx,seeker_id) VALUES(?,?)", (idx, sid))
        conn.commit(); return True
    except sqlite3.IntegrityError: return False
    finally: conn.close()


def already_applied_dataset(idx, sid):
    conn = get_conn()
    row = conn.execute("SELECT 1 FROM dataset_applications WHERE dataset_job_idx=? AND seeker_id=?", (idx, sid)).fetchone()
    conn.close(); return bool(row)


def get_seeker_dataset_apps(sid):
    conn = get_conn()
    rows = conn.execute("SELECT * FROM dataset_applications WHERE seeker_id=? ORDER BY applied_at DESC", (sid,)).fetchall()
    conn.close(); return [dict(r) for r in rows]


def seeker_dashboard_stats(sid):
    conn = get_conn()
    posted = conn.execute("SELECT COUNT(*) FROM applications WHERE seeker_id=?", (sid,)).fetchone()[0]
    ds     = conn.execute("SELECT COUNT(*) FROM dataset_applications WHERE seeker_id=?", (sid,)).fetchone()[0]
    conn.close(); return posted + ds, ds


def company_dashboard_stats(cid):
    conn = get_conn()
    total = conn.execute("SELECT COUNT(*) FROM jobs WHERE company_id=?", (cid,)).fetchone()[0]
    apps  = conn.execute("SELECT COUNT(*) FROM applications a JOIN jobs j ON j.id=a.job_id WHERE j.company_id=?", (cid,)).fetchone()[0]
    conn.close(); return total, apps, total


# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────
for _k, _v in [("page", "home"), ("user", None), ("selected_job", None),
                ("search_q", ""), ("search_loc", "")]:
    if _k not in st.session_state:
        st.session_state[_k] = _v


def go(page):
    st.session_state.page = page
    st.session_state.selected_job = None
    st.rerun()


def logout():
    st.session_state.user = None
    go("home")


# ─────────────────────────────────────────────────────────────────────────────
# NAVBAR
# ─────────────────────────────────────────────────────────────────────────────
def render_navbar():
    user = st.session_state.user
    c0, c1, c2, c3, c4 = st.columns([2.5, 1, 1, 1, 1])
    c0.markdown('<span style="font-family:\'Playfair Display\',serif;font-size:1.4rem;font-weight:900;color:#1a3c5e;">True<span style=\'color:#e8734a\'>Hire</span></span>', unsafe_allow_html=True)
    if c1.button("🏠 Home", key="nav_home"): go("home")
    if c2.button("💼 Jobs", key="nav_jobs"): go("jobs")
    if user:
        dash = "dashboard_seeker" if user["role"] == "seeker" else "dashboard_company"
        if c3.button(f"👤 {user['name'].split()[0]}", key="nav_dash"): go(dash)
        if c4.button("Logout", key="nav_lo"): logout()
    else:
        if c3.button("Login",   key="nav_li"): go("login")
        if c4.button("Sign Up", key="nav_su"): go("register")
    st.markdown("<hr style='border:none;border-top:1px solid #e5e7eb;margin:0 0 1rem;'>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# HOME PAGE
# ─────────────────────────────────────────────────────────────────────────────
def page_home():
    render_navbar()
    df, search_tfidf, search_matrix, pac_model, pac_vec, meta_cols = build_ml_engine()
    user = st.session_state.user

    real_count = int((df["pac_prediction"] == 0).sum())
    fake_count = int((df["pac_prediction"] == 1).sum())

    st.markdown(f"""
    <div class="hero-banner">
      <h1>True<span>Hire</span></h1>
      <p>AI-powered job matching. PAC model detects fake listings. Zero scams.</p>
      <div class="hero-stats">
        <div class="hero-stat"><div class="n">{len(df):,}</div><div class="l">Total Listings</div></div>
        <div class="hero-stat"><div class="n">{real_count:,}</div><div class="l">✔ Real Jobs</div></div>
        <div class="hero-stat"><div class="n">{fake_count:,}</div><div class="l">⚠ Fake Detected</div></div>
        <div class="hero-stat"><div class="n">PAC</div><div class="l">AI Classifier</div></div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Quick search
    c1, c2, c3 = st.columns([3, 2, 1])
    q   = c1.text_input("Search jobs", placeholder="e.g. Data Scientist, Python, AWS", label_visibility="collapsed")
    loc = c2.text_input("Location",    placeholder="City or Remote",                   label_visibility="collapsed")
    if c3.button("🔍 Search", use_container_width=True):
        st.session_state.search_q   = q
        st.session_state.search_loc = loc
        go("jobs")

    # Featured jobs (mix of real and fake for demonstration)
    st.markdown('<p class="sec-title">Featured Listings</p>', unsafe_allow_html=True)
    st.markdown('<p class="sec-sub">Classified by PAC model — fraudulent value shown for each listing (0 = Real, 1 = Fake)</p>', unsafe_allow_html=True)

    featured = df.sample(min(6, len(df)), random_state=7)
    cols = st.columns(2)
    for i, (idx, row) in enumerate(featured.iterrows()):
        pac_pred   = int(row["pac_prediction"])
        csv_label  = int(row["label"])
        badge      = label_badge(pac_pred)
        ds_badge   = dataset_label_badge(csv_label)
        with cols[i % 2]:
            st.markdown(f"""
            <div class="job-card">
              <h3>{row['title']}</h3>
              <div class="company">{row.get('location','') or 'Location N/A'}</div>
              <span class="tag">{row.get('employment_type','') or 'N/A'}</span>
              <span class="tag">{row.get('industry','') or '—'}</span>
              {badge}
              {ds_badge}
              <div style="font-size:0.72rem;color:#6b7280;margin-top:0.4rem;">
                Fraudulent value: <b>{csv_label}</b> &nbsp;|&nbsp; PAC prediction: <b>{CLASS_MAP[pac_pred]}</b>
              </div>
            </div>
            """, unsafe_allow_html=True)
            if st.button("View Details", key=f"home_feat_{idx}"):
                st.session_state.selected_job = ("dataset", int(idx))
                go("jobs")

    # Irrelevant jobs section
    st.divider()
    st.markdown('<p class="sec-title">🎭 Irrelevant Listings</p>', unsafe_allow_html=True)
    st.markdown('<p class="sec-sub">These jobs have been flagged as irrelevant — neither fake nor real job opportunities</p>', unsafe_allow_html=True)

    irr_cols = st.columns(2)
    for i, job in enumerate(IRRELEVANT_JOBS):
        with irr_cols[i % 2]:
            st.markdown(f"""
            <div class="job-card" style="border-color:#fde68a;">
              <h3>{job['title']}</h3>
              <div class="company">{job['location']}</div>
              <span class="tag">{job.get('employment_type','N/A')}</span>
              <span class="tag">{job.get('industry','—')}</span>
              <span class="badge-irrelevant">❓ Irrelevant</span>
              <div style="font-size:0.72rem;color:#6b7280;margin-top:0.4rem;">
                {job['description'][:90]}…
              </div>
            </div>
            """, unsafe_allow_html=True)

    # CTA
    st.divider()
    ca, cb = st.columns(2)
    with ca:
        st.markdown("""<div style="background:#1a3c5e;border-radius:14px;padding:1.8rem;color:#fff;"><h3 style="font-family:'Playfair Display',serif;color:#fff;margin-bottom:0.5rem;">🏢 Hiring Talent?</h3><p style="color:#93c5fd;margin-bottom:1rem;">Post jobs & reach thousands of verified candidates for free.</p></div>""", unsafe_allow_html=True)
        if st.button("Post a Job →", key="cta_co"): go("register")
    with cb:
        st.markdown("""<div style="background:#fff5f0;border:1.5px solid #e8734a;border-radius:14px;padding:1.8rem;"><h3 style="font-family:'Playfair Display',serif;color:#1a3c5e;margin-bottom:0.5rem;">🔍 Looking for Work?</h3><p style="color:#6b7280;margin-bottom:1rem;">Browse AI-classified listings. Real jobs highlighted, fake ones flagged.</p></div>""", unsafe_allow_html=True)
        if st.button("Browse All Jobs →", key="cta_sk"): go("jobs")


# ─────────────────────────────────────────────────────────────────────────────
# JOBS PAGE
# ─────────────────────────────────────────────────────────────────────────────
def page_jobs():
    render_navbar()
    df, search_tfidf, search_matrix, pac_model, pac_vec, meta_cols = build_ml_engine()

    # ── Job detail view ──
    if st.session_state.selected_job:
        src, idx = st.session_state.selected_job
        if src == "dataset":
            render_dataset_job_detail(df, idx, pac_model, pac_vec, meta_cols)
        elif src == "irrelevant":
            render_irrelevant_job_detail(IRRELEVANT_JOBS[idx])
        else:
            job = get_posted_job(idx)
            if job: render_posted_job_detail(job, pac_model, pac_vec, meta_cols)
        if st.button("← Back to all jobs"):
            st.session_state.selected_job = None
            st.rerun()
        return

    st.markdown('<p class="sec-title">Browse Jobs</p>', unsafe_allow_html=True)

    # Filters
    with st.expander("🔍 Search & Filter", expanded=True):
        c1, c2, c3 = st.columns(3)
        q        = c1.text_input("Keyword / Skills", value=st.session_state.search_q,  placeholder="Python, Finance…")
        loc      = c2.text_input("Location",          value=st.session_state.search_loc, placeholder="New York, Remote…")
        show_fake = c3.selectbox("Show listings", ["All (Real + Fake)", "Real Only (label=0)", "Fake Only (label=1)", "Irrelevant Only"])
        if st.button("🔍 Search Jobs", use_container_width=True):
            st.session_state.search_q   = q
            st.session_state.search_loc = loc

    # TF-IDF search on dataset
    search_text = f"{q} {loc}".strip() or "engineer developer analyst"
    results = tfidf_search(search_text, df, search_tfidf, search_matrix)

    # Apply filters
    if loc.strip():
        results = results[results["location"].str.contains(loc, case=False, na=False)]
    if show_fake == "Real Only (label=0)":
        results = results[results["label"] == 0]
    elif show_fake == "Fake Only (label=1)":
        results = results[results["label"] == 1]
    elif show_fake == "Irrelevant Only":
        results = results.iloc[0:0]  # empty — irrelevant shown below

    # Company-posted jobs
    posted = get_posted_jobs(q=q, location=loc)

    total = len(results) + len(posted)
    st.markdown(f"**{total} listing(s) found** &nbsp; <span class='badge-ai'>🤖 Ranked by TF-IDF · Classified by PAC</span>", unsafe_allow_html=True)

    # Company-posted
    if posted:
        st.markdown("#### 🏢 Company-Posted Jobs")
        for j in posted:
            job_dict = {
                "title": j["title"], "description": j.get("description", ""),
                "company_profile": "", "requirements": j.get("requirements", ""),
                "salary_range": j.get("salary_range", ""),
                "location": j.get("location", ""), "industry": "",
            }
            pac_pred = classify_job(job_dict, pac_model, pac_vec, meta_cols)
            badge = label_badge(pac_pred)
            st.markdown(f"""
            <div class="job-card">
              <h3>{j['title']}</h3>
              <div class="company">{j['company_name']} · {j['location'] or 'Remote'}</div>
              <span class="tag-accent">{j['salary_range'] or 'Negotiable'}</span>
              <span class="tag">{j['job_type']}</span>
              {badge}
              <div style="font-size:0.72rem;color:#6b7280;margin-top:0.3rem;">PAC model: <b>{CLASS_MAP[pac_pred]}</b></div>
            </div>
            """, unsafe_allow_html=True)
            b1, b2 = st.columns([5, 1])
            with b2:
                if st.button("Apply", key=f"posted_{j['id']}"):
                    st.session_state.selected_job = ("posted", j["id"]); st.rerun()

    # Dataset listings
    if show_fake != "Irrelevant Only":
        st.markdown("#### 📊 Cleaned Dataset Listings (cleaned_jobs.csv)")
        st.markdown("""
        <div class="ai-box" style="font-size:0.82rem;">
          🤖 <b>Classification key:</b> &nbsp;
          <span class="badge-verified">✔ Real Job</span> = PAC predicts 0 (genuine) &nbsp;|&nbsp;
          <span class="badge-fake">⚠ Fake Job</span> = PAC predicts 1 (fraudulent) &nbsp;|&nbsp;
          Dataset <b>fraudulent</b> column shown as "label"
        </div>
        """, unsafe_allow_html=True)

        for _, row in results.head(80).iterrows():
            pct      = max(5, min(99, int(row["score"] * 100)))
            orig_idx = int(row.name)
            pac_pred = int(row["pac_prediction"])
            csv_lbl  = int(row["label"])
            badge    = label_badge(pac_pred)
            ds_badge = dataset_label_badge(csv_lbl)

            st.markdown(f"""
            <div class="job-card">
              <h3>{row['title']}</h3>
              <div class="company">{row['location'] or 'N/A'} &nbsp;·&nbsp; {row.get('industry','')}</div>
              <span class="tag">{row.get('employment_type','') or 'N/A'}</span>
              <span class="tag">{row.get('required_experience','') or ''}</span>
              {badge} {ds_badge}
              <span class="badge-ai">🤖 {pct}% match</span>
              <div style="font-size:0.72rem;color:#6b7280;margin-top:0.35rem;">
                Fraudulent value (CSV): <b>{csv_lbl}</b> &nbsp;|&nbsp; PAC prediction: <b>{CLASS_MAP[pac_pred]}</b>
              </div>
              <div class="match-bar-wrap"><div class="match-bar" style="width:{pct}%;"></div></div>
            </div>
            """, unsafe_allow_html=True)
            b1, b2 = st.columns([5, 1])
            with b2:
                if st.button("Details", key=f"ds_{orig_idx}"):
                    st.session_state.selected_job = ("dataset", orig_idx); st.rerun()

    # Irrelevant section
    if show_fake in ("All (Real + Fake)", "Irrelevant Only"):
        st.markdown("#### 🎭 Irrelevant Listings")
        for i, job in enumerate(IRRELEVANT_JOBS):
            st.markdown(f"""
            <div class="job-card" style="border-color:#fde68a;">
              <h3>{job['title']}</h3>
              <div class="company">{job['location']}</div>
              <span class="tag">{job.get('employment_type','N/A')}</span>
              <span class="tag">{job.get('industry','—')}</span>
              <span class="badge-irrelevant">❓ Irrelevant</span>
              <div style="font-size:0.72rem;color:#6b7280;margin-top:0.35rem;">
                {job['description'][:110]}…
              </div>
            </div>
            """, unsafe_allow_html=True)
            if st.button("View Details", key=f"irr_{i}"):
                st.session_state.selected_job = ("irrelevant", i); st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# JOB DETAIL VIEWS
# ─────────────────────────────────────────────────────────────────────────────
def render_dataset_job_detail(df, idx, pac_model, pac_vec, meta_cols):
    row = df.iloc[idx]
    user     = st.session_state.user
    pac_pred = int(row["pac_prediction"])
    csv_lbl  = int(row["label"])
    badge    = label_badge(pac_pred)

    warning_html = ""
    if pac_pred == 1:
        warning_html = "<div class='warn-box'>⚠️ <b>Warning:</b> Our PAC classifier has flagged this listing as likely <b>FAKE</b>. Proceed with caution.</div>"
    elif pac_pred == 2:
        warning_html = "<div class='warn-box' style='border-color:#a16207;background:#fefce8;'>❓ This listing has been classified as <b>Irrelevant</b>.</div>"

    st.markdown(f"""
    <div class="detail-card">
      <div style="display:flex;align-items:center;gap:1.2rem;margin-bottom:1.2rem;">
        <div style="width:56px;height:56px;border-radius:12px;background:#1a3c5e;color:#fff;
                    display:flex;align-items:center;justify-content:center;font-size:1.5rem;font-weight:700;flex-shrink:0;">
          {str(row['title'])[0].upper()}
        </div>
        <div>
          <h2>{row['title']}</h2>
          <p style="color:#6b7280;font-size:0.88rem;margin:0;">{row.get('location','') or 'Location N/A'} &nbsp;·&nbsp; {row.get('industry','')}</p>
        </div>
      </div>
      <div style="margin-bottom:1rem;">
        <span class="tag">{row.get('employment_type','') or 'N/A'}</span>
        <span class="tag">{row.get('required_experience','') or ''}</span>
        <span class="tag">{row.get('required_education','') or ''}</span>
        {badge}
        {dataset_label_badge(csv_lbl)}
      </div>
      <div style="background:#f9f7f4;border-radius:8px;padding:0.6rem 1rem;margin-bottom:1rem;font-size:0.82rem;color:#374151;">
        📊 <b>Fraudulent value (CSV label):</b> <code>{csv_lbl}</code>
        &nbsp; → &nbsp; {('🟢 Real Job' if csv_lbl==0 else '🔴 Fake Job')}
        &nbsp;&nbsp;&nbsp;
        🤖 <b>PAC prediction:</b> <code>{pac_pred}</code>
        &nbsp; → &nbsp; <b>{CLASS_MAP[pac_pred]}</b>
      </div>
      <hr style="border:none;border-top:1px solid #e5e7eb;margin:1rem 0;">
      <h4 style="color:#1a3c5e;margin-bottom:0.5rem;">Company Profile</h4>
      <p style="color:#374151;line-height:1.8;font-size:0.93rem;">{row.get('company_profile','N/A') or 'N/A'}</p>
      <h4 style="color:#1a3c5e;margin-top:1.2rem;margin-bottom:0.5rem;">Job Description</h4>
      <p style="color:#374151;line-height:1.8;font-size:0.93rem;">{row.get('description','N/A') or 'N/A'}</p>
      <h4 style="color:#1a3c5e;margin-top:1.2rem;margin-bottom:0.5rem;">Requirements</h4>
      <p style="color:#374151;line-height:1.8;font-size:0.93rem;">{row.get('requirements','N/A') or 'N/A'}</p>
    </div>
    """, unsafe_allow_html=True)

    if warning_html:
        st.markdown(warning_html, unsafe_allow_html=True)

    if user and user["role"] == "seeker" and pac_pred != 1:
        if already_applied_dataset(idx, user["id"]):
            st.markdown('<div class="success-box">✅ You have already applied for this job.</div>', unsafe_allow_html=True)
        else:
            if st.button("✅ Apply Now", use_container_width=True, key="apply_ds"):
                if apply_dataset_job(idx, user["id"]): st.success("🎉 Application submitted!")
                else: st.warning("Already applied.")
    elif user and pac_pred == 1:
        st.error("⚠️ Applications blocked — this job has been flagged as FAKE by the PAC classifier.")
    elif not user:
        st.markdown('<div class="info-box">Please <b>login as a Job Seeker</b> to apply.</div>', unsafe_allow_html=True)
        if st.button("Login to Apply"): go("login")


def render_irrelevant_job_detail(job: dict):
    st.markdown(f"""
    <div class="detail-card" style="border-color:#fde68a;">
      <div style="display:flex;align-items:center;gap:1.2rem;margin-bottom:1.2rem;">
        <div style="width:56px;height:56px;border-radius:12px;background:#a16207;color:#fff;
                    display:flex;align-items:center;justify-content:center;font-size:1.5rem;font-weight:700;flex-shrink:0;">
          ❓
        </div>
        <div>
          <h2>{job['title']}</h2>
          <p style="color:#6b7280;font-size:0.88rem;margin:0;">{job['location']}</p>
        </div>
      </div>
      <div style="margin-bottom:1rem;">
        <span class="tag">{job.get('employment_type','N/A')}</span>
        <span class="tag">{job.get('industry','—')}</span>
        <span class="badge-irrelevant">❓ Irrelevant</span>
      </div>
      <div style="background:#fefce8;border-radius:8px;padding:0.6rem 1rem;margin-bottom:1rem;font-size:0.82rem;color:#374151;">
        This listing has been classified as <b>Irrelevant</b> — it does not represent a genuine job opportunity.
      </div>
      <hr style="border:none;border-top:1px solid #e5e7eb;margin:1rem 0;">
      <h4 style="color:#1a3c5e;margin-bottom:0.5rem;">Description</h4>
      <p style="color:#374151;line-height:1.8;font-size:0.93rem;">{job['description']}</p>
      <h4 style="color:#1a3c5e;margin-top:1.2rem;margin-bottom:0.5rem;">Requirements</h4>
      <p style="color:#374151;line-height:1.8;font-size:0.93rem;">{job['requirements']}</p>
      <p style="color:#6b7280;font-size:0.85rem;margin-top:1rem;">💰 Salary: {job['salary_range']}</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<div class="warn-box" style="border-color:#a16207;background:#fefce8;">❓ Applications are <b>not available</b> for irrelevant listings.</div>', unsafe_allow_html=True)


def render_posted_job_detail(j, pac_model, pac_vec, meta_cols):
    user = st.session_state.user
    job_dict = {
        "title": j["title"], "description": j.get("description", ""),
        "company_profile": "", "requirements": j.get("requirements", ""),
        "salary_range": j.get("salary_range", ""),
        "location": j.get("location", ""), "industry": "",
    }
    pac_pred = classify_job(job_dict, pac_model, pac_vec, meta_cols)
    badge = label_badge(pac_pred)

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
        {badge}
      </div>
      <div style="background:#f9f7f4;border-radius:8px;padding:0.6rem 1rem;margin-bottom:1rem;font-size:0.82rem;">
        🤖 <b>PAC Model Classification:</b> <code>{pac_pred}</code> → <b>{CLASS_MAP[pac_pred]}</b>
      </div>
      <hr style="border:none;border-top:1px solid #e5e7eb;margin:1rem 0;">
      <h4 style="color:#1a3c5e;margin-bottom:0.5rem;">Job Description</h4>
      <p style="color:#374151;line-height:1.8;font-size:0.93rem;">{j['description'] or 'No description provided.'}</p>
      <h4 style="color:#1a3c5e;margin-top:1.2rem;margin-bottom:0.5rem;">Requirements</h4>
      <p style="color:#374151;line-height:1.8;font-size:0.93rem;">{j['requirements'] or 'See description.'}</p>
      {f"<p style='margin-top:0.8rem;font-size:0.88rem;color:#6b7280;'><b>Contact:</b> {j['contact_mobile']}</p>" if j.get('contact_mobile') else ""}
      {f"<p style='font-size:0.88rem;color:#6b7280;'><b>Deadline:</b> {j['deadline']}</p>" if j.get('deadline') else ""}
    </div>
    """, unsafe_allow_html=True)

    if pac_pred == 1:
        st.error("⚠️ This company-posted job has been flagged as FAKE by the PAC classifier. Applications blocked.")
        return

    if user and user["role"] == "seeker":
        if st.button("✅ Apply Now", use_container_width=True, key="apply_posted"):
            conn = get_conn()
            try:
                conn.execute("INSERT INTO applications(job_id,seeker_id) VALUES(?,?)", (j["id"], user["id"]))
                conn.commit(); st.success("🎉 Application submitted!")
            except sqlite3.IntegrityError: st.warning("You've already applied.")
            finally: conn.close()
    elif not user:
        st.markdown('<div class="info-box">Please <b>login as a Job Seeker</b> to apply.</div>', unsafe_allow_html=True)
        if st.button("Login to Apply"): go("login")


# ─────────────────────────────────────────────────────────────────────────────
# LOGIN
# ─────────────────────────────────────────────────────────────────────────────
def page_login():
    render_navbar()
    st.markdown('<p class="sec-title">Welcome Back</p>', unsafe_allow_html=True)
    st.markdown('<p class="sec-sub">Sign in to your TrueHire account</p>', unsafe_allow_html=True)
    role = st.radio("I am a:", ["Job Seeker", "Company / Employer"], horizontal=True)
    st.divider()
    with st.form("login_form"):
        email = st.text_input("Email Address")
        pw    = st.text_input("Password", type="password")
        sub   = st.form_submit_button("Sign In", use_container_width=True)
    if sub:
        if not email or not pw: st.error("Please fill all fields."); return
        if role == "Job Seeker":
            u = login_seeker(email, pw)
            if u: st.session_state.user = {"id": u["id"], "name": u["name"], "email": u["email"], "role": "seeker"}; go("dashboard_seeker")
            else: st.error("❌ Invalid credentials.")
        else:
            u = login_company(email, pw)
            if u: st.session_state.user = {"id": u["id"], "name": u["name"], "email": u["email"], "role": "company"}; go("dashboard_company")
            else: st.error("❌ Invalid credentials.")
    st.markdown("Don't have an account?")
    if st.button("Create Account"): go("register")


# ─────────────────────────────────────────────────────────────────────────────
# REGISTER
# ─────────────────────────────────────────────────────────────────────────────
def page_register():
    render_navbar()
    st.markdown('<p class="sec-title">Create Your Account</p>', unsafe_allow_html=True)
    st.markdown('<p class="sec-sub">Join thousands of verified employers and job seekers</p>', unsafe_allow_html=True)
    role = st.radio("Register as:", ["Job Seeker", "Company / Employer"], horizontal=True)
    st.divider()
    if role == "Job Seeker":
        with st.form("reg_seeker"):
            c1, c2 = st.columns(2); name=c1.text_input("Full Name *"); email=c2.text_input("Email *")
            c3, c4 = st.columns(2); phone=c3.text_input("Phone"); pw=c4.text_input("Password *", type="password")
            skills = st.text_input("Skills (comma-separated)", placeholder="Python, SQL, Machine Learning")
            exp    = st.number_input("Experience (years)", min_value=0, max_value=50)
            sub    = st.form_submit_button("Create Account", use_container_width=True)
        if sub:
            if not name or not email or not pw: st.error("Name, email and password required.")
            else:
                ok, msg = register_seeker(name, email, pw, phone, skills, exp)
                if ok: st.success(msg + " Please login."); go("login")
                else:  st.error(msg)
    else:
        with st.form("reg_company"):
            c1, c2 = st.columns(2); name=c1.text_input("Company Name *"); email=c2.text_input("Work Email *")
            c3, c4 = st.columns(2); phone=c3.text_input("Phone"); pw=c4.text_input("Password *", type="password")
            industry = st.selectbox("Industry", ["IT / Software", "Finance", "Healthcare", "Manufacturing", "Education", "E-commerce", "Other"])
            c5, c6   = st.columns(2); year=c5.number_input("Year Founded", min_value=1900, max_value=date.today().year, value=2010); _=c6.empty()
            desc     = st.text_area("Company Description")
            sub      = st.form_submit_button("Create Account", use_container_width=True)
        if sub:
            if not name or not email or not pw: st.error("Name, email and password required.")
            else:
                ok, msg = register_company(name, email, pw, phone, industry, year, desc)
                if ok: st.success(msg + " Please login."); go("login")
                else:  st.error(msg)
    st.markdown("Already have an account?")
    if st.button("Login"): go("login")


# ─────────────────────────────────────────────────────────────────────────────
# SEEKER DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────
def page_dashboard_seeker():
    user = st.session_state.user
    if not user or user["role"] != "seeker": go("login"); return
    df, _, _, _, _, _ = build_ml_engine()
    s = get_seeker(user["id"])
    total_apps, ds_apps = seeker_dashboard_stats(user["id"])
    score = profile_score(s)

    with st.sidebar:
        st.markdown(f"### 👤 {user['name']}")
        st.markdown(f"*{user['email']}*")
        st.progress(score / 100, text=f"Profile {score}% complete")
        st.divider()
        section = st.radio("Navigate",
                           ["📊 Overview", "📋 My Applications", "👤 Edit Profile"],
                           label_visibility="collapsed")
        st.divider()
        if st.button("🔍 Browse All Jobs"): go("jobs")
        if st.button("🚪 Logout"):          logout()

    st.markdown(f"""
    <div class="hero-banner" style="padding:2rem;">
      <h1 style="font-size:1.8rem;">Welcome back, {user['name'].split()[0]}! 👋</h1>
      <p>Your AI-powered job hub — real jobs highlighted, fakes blocked.</p>
    </div>
    """, unsafe_allow_html=True)

    if section == "📊 Overview":
        real_count = int((df["pac_prediction"] == 0).sum())
        fake_count = int((df["pac_prediction"] == 1).sum())
        st.markdown(f"""
        <div class="stat-row">
          <div class="stat-card"><div class="num">{total_apps}</div><div class="lbl">My Applications</div></div>
          <div class="stat-card"><div class="num">{score}%</div><div class="lbl">Profile Complete</div></div>
          <div class="stat-card"><div class="num">{real_count:,}</div><div class="lbl">✔ Real Jobs</div></div>
          <div class="stat-card"><div class="num">{fake_count:,}</div><div class="lbl">⚠ Fake Blocked</div></div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("#### Recent Applications")
        apps = get_seeker_dataset_apps(user["id"])[:5]
        rows = []
        for a in apps:
            idx = a["dataset_job_idx"]
            if 0 <= idx < len(df):
                row = df.iloc[idx]
                rows.append({"Job": row["title"], "Location": row.get("location",""), "Applied": a["applied_at"][:10], "Status": a["status"]})
        if rows: st.table(rows)
        else: st.info("No applications yet. Browse Jobs to get started!")

    elif section == "📋 My Applications":
        st.markdown("#### All My Applications")
        apps = get_seeker_dataset_apps(user["id"])
        rows = []
        for a in apps:
            idx = a["dataset_job_idx"]
            if 0 <= idx < len(df):
                row = df.iloc[idx]
                rows.append({"Job Title": row["title"], "Location": row.get("location",""),
                              "Applied On": a["applied_at"][:10], "Status": a["status"],
                              "PAC Label": CLASS_MAP.get(int(row["pac_prediction"]), "?")})
        if rows: st.table(rows)
        else: st.info("No applications yet.")

    elif section == "👤 Edit Profile":
        st.markdown("#### Edit Profile")
        with st.form("seeker_profile"):
            c1, c2 = st.columns(2)
            name   = c1.text_input("Full Name",   value=s.get("name", "") or "")
            phone  = c2.text_input("Phone",        value=s.get("phone", "") or "")
            skills = st.text_input("Skills (comma-separated)", value=s.get("skills", "") or "")
            c3, c4 = st.columns(2)
            exp    = c3.number_input("Experience (yrs)", min_value=0, value=int(s.get("experience") or 0))
            loc    = c4.text_input("Preferred Location", value=s.get("preferred_location", "") or "")
            bio    = st.text_area("Career Summary / About Me", value=s.get("bio", "") or "")
            salary = st.text_input("Expected Salary", value=s.get("expected_salary", "") or "")
            if st.form_submit_button("Save Profile", use_container_width=True):
                update_seeker(user["id"], name, phone, skills, exp, loc, bio, salary)
                st.session_state.user["name"] = name; st.success("✅ Profile saved!")


# ─────────────────────────────────────────────────────────────────────────────
# COMPANY DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────
def page_dashboard_company():
    user = st.session_state.user
    if not user or user["role"] != "company": go("login"); return

    with st.sidebar:
        st.markdown(f"### 🏢 {user['name']}")
        st.markdown(f"*{user['email']}*")
        st.divider()
        section = st.radio("Navigate",
                           ["📊 Overview", "➕ Post a Job", "📋 My Job Posts", "👥 Applicants", "🏢 Company Profile"],
                           label_visibility="collapsed")
        st.divider()
        if st.button("🚪 Logout"): logout()

    st.markdown(f"""
    <div class="hero-banner" style="padding:2rem;">
      <h1 style="font-size:1.8rem;">Employer Dashboard 🏢</h1>
      <p>Manage your job postings and review applicants — {user['name']}</p>
    </div>
    """, unsafe_allow_html=True)

    total_jobs, total_apps, active = company_dashboard_stats(user["id"])

    if section == "📊 Overview":
        st.markdown(f"""
        <div class="stat-row">
          <div class="stat-card"><div class="num">{total_jobs}</div><div class="lbl">Jobs Posted</div></div>
          <div class="stat-card"><div class="num">{total_apps}</div><div class="lbl">Total Applicants</div></div>
          <div class="stat-card"><div class="num">{active}</div><div class="lbl">Active Listings</div></div>
        </div>
        """, unsafe_allow_html=True)
        jobs = get_company_jobs(user["id"])[:5]
        st.markdown("#### Recent Job Posts")
        if not jobs: st.info("No jobs posted yet. Use ➕ Post a Job to get started.")
        else: st.table([{"Title": j["title"], "Location": j["location"] or "Remote", "Posted": j["created_at"][:10], "Applicants": j["applicant_count"]} for j in jobs])

    elif section == "➕ Post a Job":
        st.markdown("#### Post a New Job")
        st.markdown('<div class="ai-box">🤖 After posting, the PAC model will automatically classify your job as Real, Fake, or Irrelevant.</div>', unsafe_allow_html=True)
        with st.form("post_job"):
            c1, c2 = st.columns(2); title=c1.text_input("Job Title *"); jtype=c2.selectbox("Job Type", ["Full-time", "Part-time", "Remote", "Internship", "Contract"])
            c3, c4 = st.columns(2); loc=c3.text_input("Location", placeholder="City, Country"); salary=c4.text_input("Salary Range", placeholder="$50k-$80k")
            c5, c6 = st.columns(2)
            exp     = c5.number_input("Experience Required (yrs)", min_value=0)
            has_dl  = c6.checkbox("Set Application Deadline")
            deadline = c6.date_input("Deadline Date", value=date.today()) if has_dl else None
            desc    = st.text_area("Job Description *", height=140)
            req     = st.text_area("Requirements / Skills", height=100)
            mobile  = st.text_input("Contact Mobile")
            sub     = st.form_submit_button("Post Job", use_container_width=True)
        if sub:
            if not title or not desc: st.error("Title and description are required.")
            else: post_job(user["id"], title, jtype, loc, salary, exp, deadline, desc, req, mobile); st.success("✅ Job posted successfully!")

    elif section == "📋 My Job Posts":
        st.markdown("#### My Job Postings")
        jobs = get_company_jobs(user["id"])
        if not jobs: st.info("No jobs posted yet.")
        else:
            for j in jobs:
                c1, c2, c3 = st.columns([4, 1, 1])
                c1.markdown(f"**{j['title']}** — {j['location'] or 'Remote'}  \n`{j['job_type']}` · {j['applicant_count']} applicants")
                c2.caption(j["created_at"][:10])
                if c3.button("🗑 Delete", key=f"del_{j['id']}"): delete_job(j["id"], user["id"]); st.success("Deleted."); st.rerun()

    elif section == "👥 Applicants":
        st.markdown("#### Job Applicants")
        jobs    = get_company_jobs(user["id"])
        job_map = {"All Jobs": None}
        for j in jobs: job_map[j["title"]] = j["id"]
        chosen     = st.selectbox("Filter by job", list(job_map.keys()))
        applicants = get_applicants(user["id"], job_map[chosen])
        if not applicants: st.info("No applicants yet.")
        else:
            st.table([{"Name": a["name"], "Email": a["email"], "Skills": a["skills"] or "—",
                        "Experience": f"{a['experience'] or 0} yrs", "Job": a["job_title"], "Applied": a["applied_at"][:10]} for a in applicants])

    elif section == "🏢 Company Profile":
        st.markdown("#### Company Profile")
        co = get_company(user["id"])
        with st.form("co_profile"):
            c1, c2   = st.columns(2); name=c1.text_input("Company Name", value=co.get("name", "") or "")
            industry = c2.selectbox("Industry", ["IT / Software", "Finance", "Healthcare", "Manufacturing", "E-commerce", "Other"])
            c3, c4   = st.columns(2); website=c3.text_input("Website", value=co.get("website", "") or ""); year=c4.number_input("Year Founded", min_value=1900, max_value=date.today().year, value=int(co.get("year_founded") or 2010))
            phone    = st.text_input("Phone", value=co.get("phone", "") or "")
            desc     = st.text_area("Company Description", value=co.get("description", "") or "")
            if st.form_submit_button("Save Profile", use_container_width=True):
                update_company(user["id"], name, industry, website, year, phone, desc)
                st.session_state.user["name"] = name; st.success("✅ Profile saved!")


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
PAGE_MAP.get(st.session_state.get("page", "home"), page_home)()
