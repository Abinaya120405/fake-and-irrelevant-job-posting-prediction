"""
TrueHire — Fake & Real Job Classification Portal
=================================================
Deployment-ready single-file Streamlit app.

✅ Loads cleaned_jobs.csv (7,590 rows, label: 0=Real 1=Fake)
✅ Trains SGDClassifier (PAC-equivalent) + TF-IDF at startup — NO pkl files needed
✅ Classifies every job as Real / Fake / Irrelevant using fraudulent value
✅ Live classification of company-posted jobs and user-submitted jobs
✅ Irrelevant jobs section
✅ Full auth: job seeker + employer dashboards
✅ SQLite backend for users, posted jobs, applications

Run locally:
    pip install streamlit scikit-learn pandas numpy scipy
    streamlit run truehireweb.py

Deploy on Streamlit Cloud:
    Push truehireweb.py + cleaned_jobs.csv + requirements.txt to GitHub
    requirements.txt: streamlit, scikit-learn, pandas, numpy, scipy
"""

import os, re, warnings, hashlib, sqlite3
from datetime import date

import numpy as np
import pandas as pd
import streamlit as st
from scipy.sparse import csr_matrix, hstack, vstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import MaxAbsScaler
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings("ignore")

# ── Page config (must be FIRST Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="TrueHire — Job Verification Portal",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Paths ─────────────────────────────────────────────────────────────────────
_BASE            = os.path.dirname(os.path.abspath(__file__))
CLEANED_CSV_PATH = os.path.join(_BASE, "cleaned_jobs.csv")
DB_PATH          = os.path.join(_BASE, "truehire.db")

# ── Label map ─────────────────────────────────────────────────────────────────
# 0 = Real  |  1 = Fake  |  2 = Irrelevant
CLASS_LABEL = {0: "Real", 1: "Fake", 2: "Irrelevant"}

# ── Scam keywords (from tfidf_features.py) ───────────────────────────────────
SCAM_KW = [
    "no investment", "quick earning", "earn from home", "easy money",
    "guaranteed income", "unlimited income", "be your own boss",
    "daily payout", "weekly payout", "risk free", "free registration",
    "mlm", "network marketing", "instant payment",
    "make money fast", "data entry work",
]
SCAM_PATTERN = "|".join(re.escape(k) for k in SCAM_KW)

# ── Irrelevant joke jobs ──────────────────────────────────────────────────────
IRRELEVANT_JOBS = [
    {
        "title": "Astrological Chart Reader",
        "location": "Remote / Cosmos",
        "employment_type": "Freelance",
        "industry": "Spirituality",
        "salary_range": "Unlimited* (*unverified)",
        "description": "Provide daily horoscope advice via WhatsApp. Read star charts for clients. No qualifications needed — just belief in the cosmos.",
        "requirements": "Must know your own sun sign. Crystal ball optional.",
        "company_profile": "CosmoGuide Inc. — spiritual guidance since the dawn of time.",
    },
    {
        "title": "Professional Netflix Watcher",
        "location": "Your Couch",
        "employment_type": "Part-time",
        "industry": "Entertainment",
        "salary_range": "Paid in OTT credits",
        "description": "Watch Netflix 8 hrs/day and submit a 5-minute survey. Must have fast Wi-Fi and unlimited snacks.",
        "requirements": "Active Netflix account. Working eyes. High binge tolerance.",
        "company_profile": "StreamRate Inc. — we rate content so you don't have to.",
    },
    {
        "title": "Zombie Apocalypse Survival Consultant",
        "location": "Undisclosed Bunker",
        "employment_type": "Contract",
        "industry": "Preparedness / Defense",
        "salary_range": "Canned goods + ammunition",
        "description": "Train civilians in zombie evasion and barricade construction. Crossbow proficiency a strong plus.",
        "requirements": "Survival instinct. Zero fear of the undead. Own torch preferred.",
        "company_profile": "ZombiePrep LLC — since before it was too late.",
    },
    {
        "title": "Moon Dust Collector",
        "location": "Moon (Earth travel occasional)",
        "employment_type": "Full-time",
        "industry": "Space / Mining",
        "salary_range": "Negotiable upon safe return",
        "description": "Collect regolith samples on the lunar surface. Must tolerate 2-week blackouts and zero-gravity lunch.",
        "requirements": "Space suit provided. Must hold breath for 8 seconds minimum.",
        "company_profile": "LunarMineCo — making space work for humanity.",
    },
    {
        "title": "Chief Snack Officer (CSO)",
        "location": "Office Pantry, Anywhere",
        "employment_type": "Volunteer",
        "industry": "Food & Beverages",
        "salary_range": "Unlimited snacks (no cash)",
        "description": "Curate weekly snack lists, taste-test new chips, maintain biscuit inventory. Reports to CEO (Chief Eating Officer).",
        "requirements": "Strong opinions on Parle-G. Dislike of raisins strongly preferred.",
        "company_profile": "SnackHub — democratising pantry access since 2019.",
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# CSS — refined dark-navy + amber design
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:ital,wght@0,600;0,900;1,400&family=DM+Sans:wght@300;400;500;600;700&display=swap');

:root {
  --navy:#0f2744; --navy2:#1a3c5e; --amber:#e8a020; --accent:#e8734a;
  --real:#16a34a; --fake:#dc2626; --irr:#a16207;
  --bg:#f5f3ef; --card:#ffffff; --border:#e2ddd6; --text:#1c1a17;
  --muted:#6b6560;
}
*{box-sizing:border-box;}
html,body,[class*="css"]{font-family:'DM Sans',sans-serif;background:var(--bg)!important;color:var(--text);}
#MainMenu,footer,header{visibility:hidden;}
.block-container{padding-top:1rem!important;max-width:1180px!important;}

/* ── Hero ── */
.hero{background:var(--navy);border-radius:20px;padding:3rem 2.8rem;margin-bottom:1.5rem;
      color:#fff;position:relative;overflow:hidden;}
.hero::before{content:'';position:absolute;top:-120px;right:-120px;width:500px;height:500px;
              border-radius:50%;background:radial-gradient(circle,rgba(232,160,32,.18),transparent 70%);}
.hero h1{font-family:'Fraunces',serif;font-size:clamp(2.2rem,4.5vw,3.4rem);font-weight:900;
          line-height:1.05;margin:0 0 .6rem;}
.hero h1 em{font-style:italic;color:var(--amber);}
.hero p{opacity:.82;font-size:1.05rem;max-width:540px;line-height:1.7;margin:0;}
.hero-pills{display:flex;gap:.8rem;flex-wrap:wrap;margin-top:1.6rem;}
.hero-pill{background:rgba(255,255,255,.1);border:1px solid rgba(255,255,255,.18);
            border-radius:50px;padding:.35rem 1rem;font-size:.8rem;color:#e2ddd6;}
.hero-pill strong{color:var(--amber);}

/* ── Stat cards ── */
.stat-row{display:flex;gap:1rem;flex-wrap:wrap;margin-bottom:1.5rem;}
.stat-card{background:var(--card);border-radius:14px;padding:1.2rem 1.5rem;flex:1;min-width:140px;
            border:1px solid var(--border);box-shadow:0 2px 12px rgba(15,39,68,.06);text-align:center;}
.stat-card .n{font-family:'Fraunces',serif;font-size:2rem;font-weight:900;color:var(--navy2);line-height:1;}
.stat-card .l{font-size:.75rem;color:var(--muted);margin-top:.25rem;letter-spacing:.03em;text-transform:uppercase;}

/* ── Job card ── */
.jcard{background:var(--card);border-radius:14px;padding:1.3rem 1.6rem;border:1px solid var(--border);
        margin-bottom:.85rem;box-shadow:0 2px 8px rgba(15,39,68,.05);
        transition:box-shadow .18s,border-color .18s,transform .18s;}
.jcard:hover{box-shadow:0 10px 32px rgba(15,39,68,.12);border-color:var(--accent);transform:translateY(-2px);}
.jcard h3{margin:0 0 .2rem;color:var(--navy2);font-size:1.02rem;font-weight:700;}
.jcard .meta{font-size:.82rem;color:var(--muted);margin-bottom:.55rem;}
.bar-wrap{background:#ede9e3;border-radius:50px;height:5px;margin-top:.6rem;}
.bar{background:linear-gradient(90deg,var(--accent),var(--amber));border-radius:50px;height:5px;}

/* ── Badges ── */
.tag{display:inline-block;background:#f0ede8;border:1px solid var(--border);border-radius:50px;
      padding:.16rem .65rem;font-size:.71rem;color:#4b4740;margin:.15rem .2rem 0 0;}
.tag-salary{background:#fff8ed;border-color:var(--amber);color:#92600a;}
.b-real{background:#dcfce7;color:var(--real);border:1px solid #86efac;border-radius:50px;
         padding:.14rem .65rem;font-size:.71rem;font-weight:700;}
.b-fake{background:#fee2e2;color:var(--fake);border:1px solid #fca5a5;border-radius:50px;
         padding:.14rem .65rem;font-size:.71rem;font-weight:700;}
.b-irr{background:#fef9c3;color:var(--irr);border:1px solid #fde047;border-radius:50px;
        padding:.14rem .65rem;font-size:.71rem;font-weight:700;}
.b-ai{background:#fef3c7;color:#b45309;border-radius:50px;padding:.14rem .65rem;font-size:.71rem;font-weight:600;}

/* ── Sections ── */
.sec-title{font-family:'Fraunces',serif;color:var(--navy2);font-size:1.5rem;font-weight:900;margin-bottom:.1rem;}
.sec-sub{color:var(--muted);font-size:.87rem;margin-bottom:1.1rem;}

/* ── Info boxes ── */
.box-info{background:#eff6ff;border-left:4px solid #3b82f6;border-radius:10px;padding:.85rem 1.1rem;
           margin-bottom:1rem;font-size:.89rem;}
.box-ok{background:#f0fdf4;border-left:4px solid var(--real);border-radius:10px;padding:.85rem 1.1rem;
         margin-bottom:1rem;font-size:.89rem;}
.box-warn{background:#fef2f2;border-left:4px solid var(--fake);border-radius:10px;padding:.85rem 1.1rem;
           margin-bottom:1rem;font-size:.89rem;}
.box-ai{background:#fffbeb;border-left:4px solid var(--amber);border-radius:10px;padding:.85rem 1.1rem;
         margin-bottom:1rem;font-size:.89rem;}
.box-irr{background:#fefce8;border-left:4px solid var(--irr);border-radius:10px;padding:.85rem 1.1rem;
          margin-bottom:1rem;font-size:.89rem;}

/* ── Detail card ── */
.detail-card{background:var(--card);border-radius:18px;padding:2.2rem;border:1px solid var(--border);
              box-shadow:0 6px 24px rgba(15,39,68,.09);}
.detail-card h2{font-family:'Fraunces',serif;color:var(--navy2);font-size:1.65rem;font-weight:900;margin-bottom:.2rem;}

/* ── Fraud pill ── */
.fpill{display:inline-flex;align-items:center;gap:.4rem;border-radius:8px;
        padding:.4rem .9rem;font-size:.8rem;font-weight:700;margin:.4rem 0;}
.fpill-real{background:#dcfce7;color:var(--real);border:1.5px solid #86efac;}
.fpill-fake{background:#fee2e2;color:var(--fake);border:1.5px solid #fca5a5;}
.fpill-irr{background:#fef9c3;color:var(--irr);border:1.5px solid #fde047;}

/* ── Buttons ── */
div.stButton>button{background:var(--navy2)!important;color:#fff!important;border:none!important;
    border-radius:10px!important;font-weight:600!important;font-family:'DM Sans',sans-serif!important;
    transition:background .18s,transform .15s!important;}
div.stButton>button:hover{background:var(--accent)!important;transform:translateY(-1px);}

/* ── Inputs ── */
.stTextInput>div>input,.stTextArea>div>textarea,.stNumberInput>div>input{
    border-radius:10px!important;border:1.5px solid var(--border)!important;}
.stTextInput>div>input:focus,.stTextArea>div>textarea:focus{
    border-color:var(--accent)!important;box-shadow:0 0 0 3px rgba(232,115,74,.15)!important;}

/* ── Table ── */
table{width:100%;border-collapse:collapse;font-size:.86rem;}
th{background:#f5f3ef;color:#374151;padding:.6rem .9rem;text-align:left;font-weight:600;border-bottom:2px solid var(--border);}
td{padding:.6rem .9rem;border-bottom:1px solid var(--border);color:#374151;}
tr:hover td{background:#faf8f4;}

/* ── Sidebar ── */
section[data-testid="stSidebar"]{background:var(--navy)!important;}
section[data-testid="stSidebar"] *{color:#d1e3f8!important;}
section[data-testid="stSidebar"] .stRadio label{color:#d1e3f8!important;}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# ML ENGINE
# Trains SGDClassifier (PAC-mode: loss='hinge', penalty=None, learning_rate='pa1')
# from cleaned_jobs.csv at startup — no .pkl files required.
# Mirrors train_model.py logic exactly.
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="🛡️ Training PAC classifier from cleaned_jobs.csv…")
def build_ml_engine():
    # ── 1. Load & normalise CSV (data_preprocessing.py) ────────────────────
    df = pd.read_csv(CLEANED_CSV_PATH)
    if "fraudulent" in df.columns and "label" not in df.columns:
        df = df.rename(columns={"fraudulent": "label"})
    df["label"] = pd.to_numeric(df["label"], errors="coerce").fillna(0).astype(int)
    df = df.reset_index(drop=True)

    TEXT_COLS = ["title","company_profile","description","requirements",
                 "salary_range","location","industry"]
    for col in TEXT_COLS + ["employment_type","required_experience","required_education"]:
        if col not in df.columns: df[col] = ""
        df[col] = df[col].fillna("").astype(str)

    # ── 2. Combined text (tfidf_features.py:create_combined_text) ──────────
    combined = df[TEXT_COLS].agg(" ".join, axis=1)

    # ── 3. TF-IDF vectoriser (tfidf_features.py:build_tfidf_vectorizer) ────
    pac_vec = TfidfVectorizer(
        max_features=5000, ngram_range=(1, 2), sublinear_tf=True,
        min_df=5, max_df=0.90, strip_accents="unicode",
        analyzer="word", token_pattern=r"\w{2,}", norm="l2",
    )
    X_tfidf = pac_vec.fit_transform(combined)

    # ── 4. Meta features (tfidf_features.py:extract_features) ──────────────
    meta_df = pd.DataFrame({
        "has_scam_keywords": combined.str.contains(SCAM_PATTERN, regex=True).astype(int),
        "has_salary":        (df["salary_range"].str.strip() != "").astype(int),
        "has_company_desc":  (df["company_profile"].str.strip() != "").astype(int),
        "has_phone_in_desc": combined.str.contains(r"\b\d{10}\b", regex=True).astype(int),
        "title_len":         df["title"].str.split().str.len().fillna(0).astype(int),
        "desc_len":          df["description"].str.split().str.len().fillna(0).astype(int),
    })
    META_COLS = list(meta_df.columns)
    X_meta = csr_matrix(meta_df.values.astype(float))
    X_full = hstack([X_tfidf, X_meta]).tocsr()

    # ── 5. Augment with irrelevant class (train_model.py:add_irrelevant_class)
    rng      = np.random.RandomState(42)
    real_idx = np.where(df["label"].values == 0)[0]
    n_irr    = max(100, int(len(real_idx) * 0.05))
    sampled  = rng.choice(real_idx, size=n_irr, replace=False)
    irr_arr  = X_full[sampled].toarray()
    for r in irr_arr: rng.shuffle(r)
    X_aug = vstack([X_full, csr_matrix(irr_arr)])
    y_aug = np.concatenate([df["label"].values, np.full(n_irr, 2, dtype=int)])

    # ── 6. Scale + SGD/PAC (train_model.py) ────────────────────────────────
    scaler = MaxAbsScaler()
    X_scaled = scaler.fit_transform(X_aug)
    # SGDClassifier(loss='hinge',penalty=None,learning_rate='pa1') == PAC
    model = SGDClassifier(
        loss="hinge", penalty=None, learning_rate="pa1", eta0=1.0,
        max_iter=1000, random_state=42, tol=1e-3,
    )
    model.fit(X_scaled, y_aug)

    # ── 7. Pre-classify all CSV rows ────────────────────────────────────────
    df["pac_pred"] = model.predict(scaler.transform(X_full)).astype(int)

    # ── 8. TF-IDF for cosine-similarity search ──────────────────────────────
    search_vec = TfidfVectorizer(
        max_features=6000, ngram_range=(1, 2),
        stop_words="english", sublinear_tf=True,
    )
    search_mat = search_vec.fit_transform(
        df["title"] + " " + df["industry"] + " " + df["description"]
    )

    return df, pac_vec, META_COLS, scaler, model, search_vec, search_mat


def classify_text_dict(td: dict, pac_vec, META_COLS, scaler, model) -> int:
    """Classify a raw job dict → 0 Real | 1 Fake | 2 Irrelevant."""
    def _clean(t):
        if not isinstance(t, str): return ""
        t = t.lower()
        t = re.sub(r"<[^>]+>", " ", t)
        t = re.sub(r"https?://\S+", " ", t)
        t = re.sub(r"[^\w\s]", " ", t)
        return re.sub(r"\s+", " ", t).strip()

    combined = " ".join(_clean(str(td.get(k, "")))
                        for k in ["title","company_profile","description",
                                   "requirements","salary_range","location","industry"])
    X_tfidf = pac_vec.transform([combined])
    meta = {
        "has_scam_keywords": int(bool(re.search(SCAM_PATTERN, combined))),
        "has_salary":        int(bool(str(td.get("salary_range","")).strip())),
        "has_company_desc":  int(bool(str(td.get("company_profile","")).strip())),
        "has_phone_in_desc": int(bool(re.search(r"\b\d{10}\b", combined))),
        "title_len":         len(str(td.get("title","")).split()),
        "desc_len":          len(str(td.get("description","")).split()),
    }
    X_meta = csr_matrix([[meta.get(c, 0) for c in META_COLS]])
    X = hstack([X_tfidf, X_meta]) if META_COLS else X_tfidf
    return int(model.predict(scaler.transform(X))[0])


def do_search(query: str, df, search_vec, search_mat) -> pd.DataFrame:
    if not query.strip():
        return df.copy().assign(score=1.0)
    sims = cosine_similarity(search_vec.transform([query]), search_mat).flatten()
    return df.copy().assign(score=sims).sort_values("score", ascending=False)


# ─────────────────────────────────────────────────────────────────────────────
# BADGE HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def pac_badge(cls: int) -> str:
    if cls == 0: return "<span class='b-real'>✔ Real</span>"
    if cls == 1: return "<span class='b-fake'>✘ Fake</span>"
    return "<span class='b-irr'>? Irrelevant</span>"

def fraud_pill(val: int) -> str:
    if val == 0:
        return "<span class='fpill fpill-real'>🟢  &nbsp;→&nbsp; Real Job</span>"
    if val == 1:
        return "<span class='fpill fpill-fake'>🔴 &nbsp;→&nbsp; Fake Job</span>"
    return "<span class='fpill fpill-irr'>🟡</span>"


# ─────────────────────────────────────────────────────────────────────────────
# DATABASE
# ─────────────────────────────────────────────────────────────────────────────
def _conn():
    c = sqlite3.connect(DB_PATH, check_same_thread=False)
    c.row_factory = sqlite3.Row
    return c

def init_db():
    c = _conn()
    c.executescript("""
    CREATE TABLE IF NOT EXISTS companies(
        id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL,
        email TEXT UNIQUE NOT NULL, password_hash TEXT NOT NULL,
        phone TEXT, industry TEXT, website TEXT, year_founded INTEGER,
        description TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP);
    CREATE TABLE IF NOT EXISTS seekers(
        id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL,
        email TEXT UNIQUE NOT NULL, password_hash TEXT NOT NULL,
        phone TEXT, skills TEXT, experience INTEGER DEFAULT 0,
        preferred_location TEXT, bio TEXT, expected_salary TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP);
    CREATE TABLE IF NOT EXISTS jobs(
        id INTEGER PRIMARY KEY AUTOINCREMENT, company_id INTEGER NOT NULL,
        title TEXT NOT NULL, job_type TEXT DEFAULT 'Full-time',
        location TEXT, salary_range TEXT, experience_required INTEGER DEFAULT 0,
        deadline TEXT, description TEXT, requirements TEXT,
        contact_mobile TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(company_id) REFERENCES companies(id));
    CREATE TABLE IF NOT EXISTS applications(
        id INTEGER PRIMARY KEY AUTOINCREMENT, job_id INTEGER NOT NULL,
        seeker_id INTEGER NOT NULL, status TEXT DEFAULT 'Under Review',
        applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(job_id, seeker_id),
        FOREIGN KEY(job_id) REFERENCES jobs(id),
        FOREIGN KEY(seeker_id) REFERENCES seekers(id));
    CREATE TABLE IF NOT EXISTS ds_applications(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ds_idx INTEGER NOT NULL, seeker_id INTEGER NOT NULL,
        status TEXT DEFAULT 'Under Review',
        applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(ds_idx, seeker_id),
        FOREIGN KEY(seeker_id) REFERENCES seekers(id));
    """)
    c.commit(); c.close()

init_db()
_hp = lambda pw: hashlib.sha256(pw.encode()).hexdigest()

# ── Company ──
def co_register(name, email, pw, phone, industry, year, desc):
    c = _conn()
    try:
        c.execute("INSERT INTO companies(name,email,password_hash,phone,industry,year_founded,description) VALUES(?,?,?,?,?,?,?)",
                  (name, email, _hp(pw), phone, industry, year, desc))
        c.commit(); return True, "Registered!"
    except sqlite3.IntegrityError: return False, "Email already registered."
    finally: c.close()

def co_login(email, pw):
    c = _conn(); r = c.execute("SELECT * FROM companies WHERE email=? AND password_hash=?", (email, _hp(pw))).fetchone()
    c.close(); return dict(r) if r else None

def co_get(cid):
    c = _conn(); r = c.execute("SELECT * FROM companies WHERE id=?", (cid,)).fetchone()
    c.close(); return dict(r) if r else {}

def co_update(cid, name, industry, website, year, phone, desc):
    c = _conn()
    c.execute("UPDATE companies SET name=?,industry=?,website=?,year_founded=?,phone=?,description=? WHERE id=?",
              (name, industry, website, year, phone, desc, cid))
    c.commit(); c.close()

# ── Seeker ──
def sk_register(name, email, pw, phone, skills, exp):
    c = _conn()
    try:
        c.execute("INSERT INTO seekers(name,email,password_hash,phone,skills,experience) VALUES(?,?,?,?,?,?)",
                  (name, email, _hp(pw), phone, skills, exp))
        c.commit(); return True, "Account created!"
    except sqlite3.IntegrityError: return False, "Email already registered."
    finally: c.close()

def sk_login(email, pw):
    c = _conn(); r = c.execute("SELECT * FROM seekers WHERE email=? AND password_hash=?", (email, _hp(pw))).fetchone()
    c.close(); return dict(r) if r else None

def sk_get(sid):
    c = _conn(); r = c.execute("SELECT * FROM seekers WHERE id=?", (sid,)).fetchone()
    c.close(); return dict(r) if r else {}

def sk_update(sid, name, phone, skills, exp, loc, bio, salary):
    c = _conn()
    c.execute("UPDATE seekers SET name=?,phone=?,skills=?,experience=?,preferred_location=?,bio=?,expected_salary=? WHERE id=?",
              (name, phone, skills, exp, loc, bio, salary, sid))
    c.commit(); c.close()

def profile_score(s):
    fields = ["name","phone","skills","bio","preferred_location","expected_salary"]
    filled = sum(1 for f in fields if s.get(f))
    return int(filled / len(fields) * 100)

# ── Posted Jobs ──
def job_post(cid, title, jtype, loc, salary, exp, deadline, desc, req, mobile):
    c = _conn()
    c.execute("INSERT INTO jobs(company_id,title,job_type,location,salary_range,experience_required,deadline,description,requirements,contact_mobile) VALUES(?,?,?,?,?,?,?,?,?,?)",
              (cid, title, jtype, loc, salary, exp, str(deadline) if deadline else None, desc, req, mobile))
    c.commit(); c.close()

def jobs_get(q="", location="", job_type="", limit=200):
    c = _conn()
    sql = "SELECT j.*,co.name AS company_name FROM jobs j JOIN companies co ON j.company_id=co.id WHERE 1=1"
    p = []
    if q: sql += " AND (j.title LIKE ? OR j.description LIKE ?)"; p += [f"%{q}%", f"%{q}%"]
    if location: sql += " AND j.location LIKE ?"; p.append(f"%{location}%")
    if job_type: sql += " AND j.job_type=?"; p.append(job_type)
    sql += " ORDER BY j.created_at DESC LIMIT ?"; p.append(limit)
    rows = c.execute(sql, p).fetchall(); c.close()
    return [dict(r) for r in rows]

def job_get(jid):
    c = _conn(); r = c.execute("SELECT j.*,co.name AS company_name FROM jobs j JOIN companies co ON j.company_id=co.id WHERE j.id=?", (jid,)).fetchone()
    c.close(); return dict(r) if r else None

def co_jobs(cid):
    c = _conn(); rows = c.execute("SELECT j.*,COUNT(a.id) AS appl FROM jobs j LEFT JOIN applications a ON a.job_id=j.id WHERE j.company_id=? GROUP BY j.id ORDER BY j.created_at DESC", (cid,)).fetchall()
    c.close(); return [dict(r) for r in rows]

def job_delete(jid, cid):
    c = _conn(); c.execute("DELETE FROM applications WHERE job_id=?", (jid,)); c.execute("DELETE FROM jobs WHERE id=? AND company_id=?", (jid, cid)); c.commit(); c.close()

def co_applicants(cid, jid=None):
    c = _conn()
    sql = "SELECT s.name,s.email,s.skills,s.experience,a.applied_at,a.status,j.title AS job_title FROM applications a JOIN seekers s ON s.id=a.seeker_id JOIN jobs j ON j.id=a.job_id WHERE j.company_id=?"
    p = [cid]
    if jid: sql += " AND a.job_id=?"; p.append(jid)
    rows = c.execute(sql + " ORDER BY a.applied_at DESC", p).fetchall(); c.close()
    return [dict(r) for r in rows]

def co_stats(cid):
    c = _conn(); t = c.execute("SELECT COUNT(*) FROM jobs WHERE company_id=?", (cid,)).fetchone()[0]; a = c.execute("SELECT COUNT(*) FROM applications a JOIN jobs j ON j.id=a.job_id WHERE j.company_id=?", (cid,)).fetchone()[0]; c.close(); return t, a

# ── Dataset applications ──
def ds_apply(idx, sid):
    c = _conn()
    try: c.execute("INSERT INTO ds_applications(ds_idx,seeker_id) VALUES(?,?)", (idx, sid)); c.commit(); return True
    except sqlite3.IntegrityError: return False
    finally: c.close()

def ds_applied(idx, sid):
    c = _conn(); r = c.execute("SELECT 1 FROM ds_applications WHERE ds_idx=? AND seeker_id=?", (idx, sid)).fetchone(); c.close(); return bool(r)

def ds_my_apps(sid):
    c = _conn(); rows = c.execute("SELECT * FROM ds_applications WHERE seeker_id=? ORDER BY applied_at DESC", (sid,)).fetchall(); c.close(); return [dict(r) for r in rows]

def sk_stats(sid):
    c = _conn(); a = c.execute("SELECT COUNT(*) FROM applications WHERE seeker_id=?", (sid,)).fetchone()[0]; b = c.execute("SELECT COUNT(*) FROM ds_applications WHERE seeker_id=?", (sid,)).fetchone()[0]; c.close(); return a + b, b


# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────
for k, v in [("page","home"),("user",None),("sel_job",None),("sq",""),("sl","")]:
    if k not in st.session_state: st.session_state[k] = v

def go(page, job=None):
    st.session_state.page = page
    st.session_state.sel_job = job
    st.rerun()

def logout():
    st.session_state.user = None; go("home")


# ─────────────────────────────────────────────────────────────────────────────
# NAVBAR
# ─────────────────────────────────────────────────────────────────────────────
def navbar():
    u = st.session_state.user
    c0,c1,c2,c3,c4 = st.columns([2.4,1,1,1.2,1.2])
    c0.markdown(
        '<span style="font-family:\'Fraunces\',serif;font-size:1.5rem;font-weight:900;color:#0f2744;">'
        '🛡️ True<em style="color:#e8a020;font-style:italic;">Hire</em></span>',
        unsafe_allow_html=True)
    if c1.button("Home",  key="n_home"): go("home")
    if c2.button("Jobs",  key="n_jobs"): go("jobs")
    if u:
        dash = "dash_sk" if u["role"] == "seeker" else "dash_co"
        if c3.button(f"👤 {u['name'].split()[0]}", key="n_dash"): go(dash)
        if c4.button("Logout", key="n_lo"): logout()
    else:
        if c3.button("Login",   key="n_li"): go("login")
        if c4.button("Sign Up", key="n_su"): go("register")
    st.markdown("<hr style='border:none;border-top:1px solid #e2ddd6;margin:.2rem 0 1rem;'>",
                unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# HOME PAGE
# ─────────────────────────────────────────────────────────────────────────────
def page_home():
    navbar()
    df, pac_vec, META_COLS, scaler, model, search_vec, search_mat = build_ml_engine()

    real_n = int((df["pac_pred"] == 0).sum())
    fake_n = int((df["pac_pred"] == 1).sum())

    st.markdown(f"""
    <div class="hero">
      <h1>Detect <em>Fake Jobs</em><br>Before They Fool You</h1>
      <p>AI-powered PAC classifier analyses every listing in real time using the
         <b>fraudulent</b> label from our dataset.<br>
         0 = Real &nbsp;|&nbsp; 1 = Fake &nbsp;|&nbsp; 2 = Irrelevant</p>
      <div class="hero-pills">
        <div class="hero-pill"><strong>{len(df):,}</strong> listings</div>
        <div class="hero-pill"><strong style="color:#4ade80">{real_n:,}</strong> real jobs</div>
        <div class="hero-pill"><strong style="color:#f87171">{fake_n:,}</strong> fake detected</div>
        <div class="hero-pill"><strong>SGD/PAC</strong> model</div>
        <div class="hero-pill"><strong>TF-IDF</strong> search</div>
      </div>
    </div>""", unsafe_allow_html=True)

    # Quick search
    c1, c2, c3 = st.columns([3, 2, 1])
    q   = c1.text_input("Search",   placeholder="Data Scientist, Python, Finance…", label_visibility="collapsed")
    loc = c2.text_input("Location", placeholder="City or Remote",                    label_visibility="collapsed")
    if c3.button("🔍 Search", use_container_width=True):
        st.session_state.sq = q; st.session_state.sl = loc; go("jobs")

    # ── Live classify input ──────────────────────────────────────────────────
    st.markdown('<p class="sec-title" style="margin-top:1.6rem;">🔬 Live Job Classifier</p>', unsafe_allow_html=True)
    st.markdown('<p class="sec-sub">Paste any job description below — PAC model will classify it instantly</p>', unsafe_allow_html=True)
    with st.expander("▶ Try the classifier", expanded=False):
        lt = st.text_area("Job text (title + description)", height=110,
                           placeholder="e.g. Earn money fast! No investment needed. Work from home…")
        ls = st.text_input("Salary range (optional)", placeholder="e.g. ₹5000/week")
        if st.button("Classify Now", key="live_cls"):
            if lt.strip():
                cls = classify_text_dict(
                    {"title": lt[:120], "description": lt, "salary_range": ls},
                    pac_vec, META_COLS, scaler, model
                )
                color = {"Real":"#dcfce7","Fake":"#fee2e2","Irrelevant":"#fef9c3"}[CLASS_LABEL[cls]]
                icon  = {"Real":"✅","Fake":"⚠️","Irrelevant":"❓"}[CLASS_LABEL[cls]]
                st.markdown(
                    f'<div style="background:{color};border-radius:12px;padding:1rem 1.4rem;'
                    f'font-size:1rem;font-weight:700;">'
                    f'{icon} PAC Model Prediction: <b>{CLASS_LABEL[cls]}</b> '
                    f'(fraudulent value = {cls})</div>',
                    unsafe_allow_html=True)
            else:
                st.warning("Please enter some job text first.")

    # ── Featured listings ────────────────────────────────────────────────────
    st.markdown('<p class="sec-title" style="margin-top:1.8rem;">Featured Listings</p>', unsafe_allow_html=True)
    st.markdown('<p class="sec-sub">Randomly sampled — fraudulent value (0/1) and PAC prediction shown for each</p>', unsafe_allow_html=True)

    featured = df.sample(min(6, len(df)), random_state=7)
    cols = st.columns(2)
    for i, (idx, row) in enumerate(featured.iterrows()):
        cls     = int(row["pac_pred"])
        csv_lbl = int(row["label"])
        with cols[i % 2]:
            st.markdown(f"""
            <div class="jcard">
              <h3>{row['title']}</h3>
              <div class="meta">{row.get('location','') or 'N/A'} &nbsp;·&nbsp; {row.get('industry','') or '—'}</div>
              <span class="tag">{row.get('employment_type','') or 'N/A'}</span>
              {pac_badge(cls)}
              <div style="margin-top:.5rem;">{fraud_pill(csv_lbl)}</div>
              <div style="font-size:.7rem;color:#9ca3af;margin-top:.3rem;">
                Dataset fraudulent value: <b>{csv_lbl}</b> &nbsp;|&nbsp; PAC: <b>{CLASS_LABEL[cls]}</b>
              </div>
            </div>""", unsafe_allow_html=True)
            if st.button("View Details", key=f"home_f_{idx}"):
                go("jobs", job=("dataset", int(idx)))

    # ── Irrelevant jobs ──────────────────────────────────────────────────────
    st.divider()
    st.markdown('<p class="sec-title">🎭 Irrelevant Listings</p>', unsafe_allow_html=True)
    st.markdown('<p class="sec-sub">These don\'t fit Real or Fake — they\'re just irrelevant to any genuine hiring</p>', unsafe_allow_html=True)
    ic = st.columns(2)
    for i, job in enumerate(IRRELEVANT_JOBS):
        with ic[i % 2]:
            st.markdown(f"""
            <div class="jcard" style="border-color:#fde047;">
              <h3>{job['title']}</h3>
              <div class="meta">{job['location']} &nbsp;·&nbsp; {job['industry']}</div>
              <span class="tag">{job['employment_type']}</span>
              <span class="b-irr">? Irrelevant</span>
              <div style="font-size:.7rem;color:#9ca3af;margin-top:.35rem;">{job['description'][:90]}…</div>
            </div>""", unsafe_allow_html=True)
            if st.button("View", key=f"irr_h_{i}"):
                go("jobs", job=("irrelevant", i))

    # CTA
    st.divider()
    ca, cb = st.columns(2)
    with ca:
        st.markdown("""<div style="background:var(--navy);border-radius:16px;padding:1.8rem;color:#fff;">
          <h3 style="font-family:'Fraunces',serif;color:#fff;margin-bottom:.4rem;">🏢 Hiring?</h3>
          <p style="color:#93c5fd;margin-bottom:1rem;">Post jobs and reach thousands of verified candidates.</p>
          </div>""", unsafe_allow_html=True)
        if st.button("Post a Job →", key="cta_co"): go("register")
    with cb:
        st.markdown("""<div style="background:#fff8ed;border:1.5px solid #e8a020;border-radius:16px;padding:1.8rem;">
          <h3 style="font-family:'Fraunces',serif;color:var(--navy2);margin-bottom:.4rem;">🔍 Job Hunting?</h3>
          <p style="color:#6b6560;margin-bottom:1rem;">Real jobs highlighted, fakes blocked automatically.</p>
          </div>""", unsafe_allow_html=True)
        if st.button("Browse All Jobs →", key="cta_sk"): go("jobs")


# ─────────────────────────────────────────────────────────────────────────────
# JOBS PAGE
# ─────────────────────────────────────────────────────────────────────────────
def page_jobs():
    navbar()
    df, pac_vec, META_COLS, scaler, model, search_vec, search_mat = build_ml_engine()

    # ── Detail view ─────────────────────────────────────────────────────────
    if st.session_state.sel_job:
        src, idx = st.session_state.sel_job
        if src == "dataset":      detail_dataset(df, idx, pac_vec, META_COLS, scaler, model)
        elif src == "irrelevant": detail_irrelevant(IRRELEVANT_JOBS[idx])
        elif src == "posted":
            j = job_get(idx)
            if j: detail_posted(j, pac_vec, META_COLS, scaler, model)
        if st.button("← Back to listings"):
            st.session_state.sel_job = None; st.rerun()
        return

    st.markdown('<p class="sec-title">Browse & Filter Jobs</p>', unsafe_allow_html=True)

    with st.expander("🔍 Search & Filter", expanded=True):
        c1, c2, c3 = st.columns(3)
        q    = c1.text_input("Keyword",  value=st.session_state.sq, placeholder="Python, Finance…")
        loc  = c2.text_input("Location", value=st.session_state.sl, placeholder="New York, Remote…")
        show = c3.selectbox("Filter by label", ["All", "Real Only (label=0)", "Fake Only (label=1)", "Irrelevant Only"])
        if st.button("🔍 Apply Search", use_container_width=True):
            st.session_state.sq = q; st.session_state.sl = loc

    # TF-IDF ranked search
    q_text  = f"{q} {loc}".strip() or "engineer developer analyst"
    results = do_search(q_text, df, search_vec, search_mat)
    if loc.strip():
        results = results[results["location"].str.contains(loc, case=False, na=False)]
    if show == "Real Only (label=0)":     results = results[results["label"] == 0]
    elif show == "Fake Only (label=1)":   results = results[results["label"] == 1]
    elif show == "Irrelevant Only":        results = results.iloc[0:0]

    posted = jobs_get(q=q, location=loc)

    total = len(results) + len(posted) + (len(IRRELEVANT_JOBS) if "Irrelevant" in show else 0)
    st.markdown(
        f"**{total} listing(s)** &nbsp;"
        f"<span class='b-ai'>🤖 TF-IDF ranked · PAC classified</span> &nbsp;"
        f"<span style='font-size:.8rem;color:#6b6560;'>fraudulent: 0=Real 1=Fake 2=Irrelevant</span>",
        unsafe_allow_html=True)

    st.markdown("""<div class="box-ai" style="font-size:.81rem;margin-bottom:.8rem;">
      <b>Classification key:</b> &nbsp;
      <span class="b-real">✔ Real</span> PAC=0 &nbsp;|&nbsp;
      <span class="b-fake">✘ Fake</span> PAC=1 &nbsp;|&nbsp;
      <span class="b-irr">? Irrelevant</span> PAC=2 &nbsp;|&nbsp;
      <b>fraudulent value</b> = raw CSV label (0 or 1)
    </div>""", unsafe_allow_html=True)

    # Company-posted jobs
    if posted and show not in ("Irrelevant Only",):
        st.markdown("#### 🏢 Company-Posted Jobs")
        for j in posted:
            cls = classify_text_dict(
                {"title": j["title"], "description": j.get("description",""),
                 "company_profile": "", "requirements": j.get("requirements",""),
                 "salary_range": j.get("salary_range",""), "location": j.get("location",""), "industry": ""},
                pac_vec, META_COLS, scaler, model)
            st.markdown(f"""
            <div class="jcard">
              <h3>{j['title']}</h3>
              <div class="meta">{j['company_name']} &nbsp;·&nbsp; {j['location'] or 'Remote'}</div>
              <span class="tag-salary">{j['salary_range'] or 'Negotiable'}</span>
              <span class="tag">{j['job_type']}</span>
              {pac_badge(cls)}
              <div style="font-size:.7rem;color:#9ca3af;margin-top:.3rem;">PAC: <b>{CLASS_LABEL[cls]}</b></div>
            </div>""", unsafe_allow_html=True)
            _, b = st.columns([5, 1])
            with b:
                if st.button("Apply", key=f"p_{j['id']}"):
                    go("jobs", job=("posted", j["id"]))

    # Dataset listings
    if show != "Irrelevant Only":
        st.markdown("#### 📊 Dataset Listings (cleaned_jobs.csv)")
        for _, row in results.head(80).iterrows():
            pct     = max(5, min(99, int(row["score"] * 100)))
            oi      = int(row.name)
            cls     = int(row["pac_pred"])
            csv_lbl = int(row["label"])
            st.markdown(f"""
            <div class="jcard">
              <h3>{row['title']}</h3>
              <div class="meta">{row.get('location','') or 'N/A'} &nbsp;·&nbsp; {row.get('industry','') or '—'}</div>
              <span class="tag">{row.get('employment_type','') or 'N/A'}</span>
              {pac_badge(cls)} {fraud_pill(csv_lbl)}
              <span class="b-ai">🤖 {pct}% match</span>
              <div style="font-size:.7rem;color:#9ca3af;margin-top:.3rem;">
                CSV fraudulent value: <b>{csv_lbl}</b> &nbsp;|&nbsp; PAC prediction: <b>{CLASS_LABEL[cls]}</b>
              </div>
              <div class="bar-wrap"><div class="bar" style="width:{pct}%;"></div></div>
            </div>""", unsafe_allow_html=True)
            _, b = st.columns([5, 1])
            with b:
                if st.button("Details", key=f"ds_{oi}"):
                    go("jobs", job=("dataset", oi))

    # Irrelevant listings
    if show in ("All", "Irrelevant Only"):
        st.markdown("#### 🎭 Irrelevant Listings")
        for i, job in enumerate(IRRELEVANT_JOBS):
            st.markdown(f"""
            <div class="jcard" style="border-color:#fde047;">
              <h3>{job['title']}</h3>
              <div class="meta">{job['location']} &nbsp;·&nbsp; {job['industry']}</div>
              <span class="tag">{job['employment_type']}</span>
              <span class="b-irr">? Irrelevant</span>
              <div style="font-size:.7rem;color:#9ca3af;margin-top:.3rem;">{job['description'][:100]}…</div>
            </div>""", unsafe_allow_html=True)
            if st.button("View Details", key=f"irr_{i}"):
                go("jobs", job=("irrelevant", i))


# ─────────────────────────────────────────────────────────────────────────────
# DETAIL PAGES
# ─────────────────────────────────────────────────────────────────────────────
def detail_dataset(df, idx, pac_vec, META_COLS, scaler, model):
    row     = df.iloc[idx]
    user    = st.session_state.user
    cls     = int(row["pac_pred"])
    csv_lbl = int(row["label"])

    st.markdown(f"""
    <div class="detail-card">
      <div style="display:flex;align-items:center;gap:1.3rem;margin-bottom:1.3rem;">
        <div style="width:58px;height:58px;border-radius:14px;background:#0f2744;color:#fff;
                    display:flex;align-items:center;justify-content:center;font-family:'Fraunces',serif;
                    font-size:1.5rem;font-weight:900;flex-shrink:0;">
          {str(row['title'])[0].upper()}
        </div>
        <div>
          <h2>{row['title']}</h2>
          <p style="color:#6b6560;font-size:.88rem;margin:0;">
            {row.get('location','') or 'N/A'} &nbsp;·&nbsp; {row.get('industry','') or '—'}
          </p>
        </div>
      </div>
      <div style="margin-bottom:.9rem;">
        <span class="tag">{row.get('employment_type','') or 'N/A'}</span>
        <span class="tag">{row.get('required_experience','') or ''}</span>
        <span class="tag">{row.get('required_education','') or ''}</span>
        &nbsp;{pac_badge(cls)}
      </div>
      <div style="background:#f5f3ef;border-radius:10px;padding:.75rem 1.1rem;margin-bottom:1rem;">
        {fraud_pill(csv_lbl)}
        <div style="font-size:.78rem;color:#6b6560;margin-top:.35rem;">
          CSV <code>fraudulent</code> column = <b>{csv_lbl}</b>
          &nbsp;→&nbsp; {'🟢 Real Job' if csv_lbl==0 else '🔴 Fake Job'}
          &nbsp;&nbsp;&nbsp;
          PAC model output = <b>{cls}</b> → <b>{CLASS_LABEL[cls]}</b>
        </div>
      </div>
      <hr style="border:none;border-top:1px solid #e2ddd6;margin:1rem 0;">
      <h4 style="color:#0f2744;margin-bottom:.4rem;">Company Profile</h4>
      <p style="color:#374151;line-height:1.8;font-size:.92rem;">{row.get('company_profile','') or '—'}</p>
      <h4 style="color:#0f2744;margin-top:1.2rem;margin-bottom:.4rem;">Job Description</h4>
      <p style="color:#374151;line-height:1.8;font-size:.92rem;">{row.get('description','') or '—'}</p>
      <h4 style="color:#0f2744;margin-top:1.2rem;margin-bottom:.4rem;">Requirements</h4>
      <p style="color:#374151;line-height:1.8;font-size:.92rem;">{row.get('requirements','') or '—'}</p>
    </div>""", unsafe_allow_html=True)

    if cls == 1:
        st.markdown('<div class="box-warn">⚠️ <b>Blocked:</b> PAC model classified this as FAKE. Applications are disabled for your safety.</div>', unsafe_allow_html=True)
    elif cls == 2:
        st.markdown('<div class="box-irr">❓ Classified as Irrelevant — applications unavailable.</div>', unsafe_allow_html=True)
    elif user and user["role"] == "seeker":
        if ds_applied(idx, user["id"]):
            st.markdown('<div class="box-ok">✅ You have already applied for this job.</div>', unsafe_allow_html=True)
        else:
            if st.button("✅ Apply Now", use_container_width=True, key="apply_ds"):
                if ds_apply(idx, user["id"]): st.success("🎉 Application submitted!")
                else: st.warning("Already applied.")
    else:
        st.markdown('<div class="box-info">Please <b>login as a Job Seeker</b> to apply.</div>', unsafe_allow_html=True)
        if st.button("Login to Apply"): go("login")


def detail_irrelevant(job):
    st.markdown(f"""
    <div class="detail-card" style="border-color:#fde047;">
      <div style="display:flex;align-items:center;gap:1.3rem;margin-bottom:1.3rem;">
        <div style="width:58px;height:58px;border-radius:14px;background:#a16207;color:#fff;
                    display:flex;align-items:center;justify-content:center;font-size:1.7rem;flex-shrink:0;">❓</div>
        <div>
          <h2>{job['title']}</h2>
          <p style="color:#6b6560;font-size:.88rem;margin:0;">{job['location']} &nbsp;·&nbsp; {job['industry']}</p>
        </div>
      </div>
      <div style="margin-bottom:.9rem;">
        <span class="tag">{job['employment_type']}</span>
        <span class="b-irr">? Irrelevant</span>
      </div>
      <div class="box-irr">This listing is classified as <b>Irrelevant</b> (fraudulent value = 2)
        — not a genuine job opportunity.</div>
      <h4 style="color:#0f2744;margin-bottom:.4rem;">Description</h4>
      <p style="color:#374151;line-height:1.8;font-size:.92rem;">{job['description']}</p>
      <h4 style="color:#0f2744;margin-top:1.2rem;margin-bottom:.4rem;">Requirements</h4>
      <p style="color:#374151;line-height:1.8;font-size:.92rem;">{job['requirements']}</p>
      <p style="color:#9ca3af;font-size:.83rem;margin-top:1rem;">💰 {job['salary_range']}</p>
    </div>""", unsafe_allow_html=True)
    st.markdown('<div class="box-irr">❓ Applications are not available for irrelevant listings.</div>', unsafe_allow_html=True)


def detail_posted(j, pac_vec, META_COLS, scaler, model):
    user = st.session_state.user
    cls  = classify_text_dict(
        {"title": j["title"], "description": j.get("description",""),
         "company_profile": "", "requirements": j.get("requirements",""),
         "salary_range": j.get("salary_range",""), "location": j.get("location",""), "industry": ""},
        pac_vec, META_COLS, scaler, model)

    st.markdown(f"""
    <div class="detail-card">
      <div style="display:flex;align-items:center;gap:1.3rem;margin-bottom:1.3rem;">
        <div style="width:58px;height:58px;border-radius:14px;background:#0f2744;color:#fff;
                    display:flex;align-items:center;justify-content:center;font-family:'Fraunces',serif;
                    font-size:1.5rem;font-weight:900;flex-shrink:0;">
          {str(j['company_name'])[0].upper()}
        </div>
        <div>
          <h2>{j['title']}</h2>
          <p style="color:#6b6560;font-size:.88rem;margin:0;">{j['company_name']} &nbsp;·&nbsp; {j['location'] or 'Remote'}</p>
        </div>
      </div>
      <div style="margin-bottom:.9rem;">
        <span class="tag-salary">{j['salary_range'] or 'Negotiable'}</span>
        <span class="tag">{j['job_type']}</span>
        <span class="tag">{j['experience_required']} yrs exp</span>
        &nbsp;{pac_badge(cls)}
      </div>
      <div style="background:#f5f3ef;border-radius:10px;padding:.75rem 1.1rem;margin-bottom:1rem;font-size:.78rem;color:#6b6560;">
        🤖 PAC model prediction: <b>{cls}</b> → <b>{CLASS_LABEL[cls]}</b>
        &nbsp;(live classification of this posted job)
      </div>
      <hr style="border:none;border-top:1px solid #e2ddd6;margin:1rem 0;">
      <h4 style="color:#0f2744;margin-bottom:.4rem;">Description</h4>
      <p style="color:#374151;line-height:1.8;font-size:.92rem;">{j['description'] or '—'}</p>
      <h4 style="color:#0f2744;margin-top:1.2rem;margin-bottom:.4rem;">Requirements</h4>
      <p style="color:#374151;line-height:1.8;font-size:.92rem;">{j['requirements'] or '—'}</p>
      {f"<p style='font-size:.85rem;color:#6b6560;margin-top:.8rem;'>📞 {j['contact_mobile']}</p>" if j.get('contact_mobile') else ""}
    </div>""", unsafe_allow_html=True)

    if cls == 1:
        st.markdown('<div class="box-warn">⚠️ Blocked: PAC model flagged this as FAKE. Applications disabled.</div>', unsafe_allow_html=True)
        return

    if user and user["role"] == "seeker":
        if st.button("✅ Apply Now", use_container_width=True, key="apply_posted"):
            c = _conn()
            try: c.execute("INSERT INTO applications(job_id,seeker_id) VALUES(?,?)", (j["id"], user["id"])); c.commit(); st.success("🎉 Application submitted!")
            except sqlite3.IntegrityError: st.warning("You've already applied.")
            finally: c.close()
    else:
        st.markdown('<div class="box-info">Please <b>login as a Job Seeker</b> to apply.</div>', unsafe_allow_html=True)
        if st.button("Login to Apply"): go("login")


# ─────────────────────────────────────────────────────────────────────────────
# AUTH PAGES
# ─────────────────────────────────────────────────────────────────────────────
def page_login():
    navbar()
    st.markdown('<p class="sec-title">Sign In</p>', unsafe_allow_html=True)
    role = st.radio("I am a:", ["Job Seeker", "Company / Employer"], horizontal=True)
    st.divider()
    with st.form("lf"):
        email = st.text_input("Email"); pw = st.text_input("Password", type="password")
        sub = st.form_submit_button("Sign In", use_container_width=True)
    if sub:
        if not email or not pw: st.error("Fill all fields."); return
        u = sk_login(email, pw) if role == "Job Seeker" else co_login(email, pw)
        rk = "seeker" if role == "Job Seeker" else "company"
        if u:
            st.session_state.user = {"id": u["id"], "name": u["name"], "email": u["email"], "role": rk}
            go("dash_sk" if rk == "seeker" else "dash_co")
        else: st.error("Invalid credentials.")
    if st.button("Create account"): go("register")


def page_register():
    navbar()
    st.markdown('<p class="sec-title">Create Account</p>', unsafe_allow_html=True)
    role = st.radio("Register as:", ["Job Seeker", "Company / Employer"], horizontal=True)
    st.divider()
    if role == "Job Seeker":
        with st.form("rs"):
            c1,c2 = st.columns(2); name=c1.text_input("Full Name *"); email=c2.text_input("Email *")
            c3,c4 = st.columns(2); phone=c3.text_input("Phone"); pw=c4.text_input("Password *", type="password")
            skills = st.text_input("Skills (comma-separated)", placeholder="Python, SQL, ML")
            exp = st.number_input("Experience (years)", min_value=0, max_value=50)
            sub = st.form_submit_button("Create Account", use_container_width=True)
        if sub:
            if not name or not email or not pw: st.error("Name, email and password required.")
            else:
                ok, msg = sk_register(name, email, pw, phone, skills, exp)
                if ok: st.success(msg); go("login")
                else: st.error(msg)
    else:
        with st.form("rc"):
            c1,c2 = st.columns(2); name=c1.text_input("Company Name *"); email=c2.text_input("Work Email *")
            c3,c4 = st.columns(2); phone=c3.text_input("Phone"); pw=c4.text_input("Password *", type="password")
            industry = st.selectbox("Industry", ["IT / Software","Finance","Healthcare","Manufacturing","Education","E-commerce","Other"])
            year = st.number_input("Year Founded", min_value=1900, max_value=date.today().year, value=2010)
            desc = st.text_area("Company Description")
            sub = st.form_submit_button("Create Account", use_container_width=True)
        if sub:
            if not name or not email or not pw: st.error("Name, email and password required.")
            else:
                ok, msg = co_register(name, email, pw, phone, industry, year, desc)
                if ok: st.success(msg); go("login")
                else: st.error(msg)
    if st.button("Already have an account? Login"): go("login")


# ─────────────────────────────────────────────────────────────────────────────
# SEEKER DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────
def page_dash_seeker():
    u = st.session_state.user
    if not u or u["role"] != "seeker": go("login"); return
    df, *_ = build_ml_engine()
    s = sk_get(u["id"]); total, ds = sk_stats(u["id"]); sc = profile_score(s)

    with st.sidebar:
        st.markdown(f"### 👤 {u['name']}\n*{u['email']}*")
        st.progress(sc / 100, text=f"Profile {sc}% complete")
        st.divider()
        section = st.radio("Navigate",
            ["📊 Overview","📋 My Applications","👤 Edit Profile"], label_visibility="collapsed")
        st.divider()
        if st.button("💼 Browse Jobs"): go("jobs")
        if st.button("🚪 Logout"): logout()

    real_n = int((df["pac_pred"] == 0).sum()); fake_n = int((df["pac_pred"] == 1).sum())
    st.markdown(f"""
    <div class="hero" style="padding:2rem;">
      <h1 style="font-size:1.8rem;">Welcome back, <em>{u['name'].split()[0]}</em>! 👋</h1>
      <p>Your AI-protected job hub — fakes blocked automatically.</p>
    </div>""", unsafe_allow_html=True)

    if section == "📊 Overview":
        st.markdown(f"""
        <div class="stat-row">
          <div class="stat-card"><div class="n">{total}</div><div class="l">Applications</div></div>
          <div class="stat-card"><div class="n">{sc}%</div><div class="l">Profile Score</div></div>
          <div class="stat-card"><div class="n">{real_n:,}</div><div class="l">Real Jobs</div></div>
          <div class="stat-card"><div class="n">{fake_n:,}</div><div class="l">Fake Blocked</div></div>
        </div>""", unsafe_allow_html=True)
        apps = ds_my_apps(u["id"])[:5]; rows = []
        for a in apps:
            i = a["ds_idx"]
            if 0 <= i < len(df):
                r = df.iloc[i]
                rows.append({"Job": r["title"], "Location": r.get("location",""), "Applied": a["applied_at"][:10], "Status": a["status"]})
        st.markdown("#### Recent Applications")
        if rows: st.table(rows)
        else: st.info("No applications yet. Browse Jobs to get started!")

    elif section == "📋 My Applications":
        apps = ds_my_apps(u["id"]); rows = []
        for a in apps:
            i = a["ds_idx"]
            if 0 <= i < len(df):
                r = df.iloc[i]
                rows.append({"Job": r["title"], "Location": r.get("location",""),
                              "Applied": a["applied_at"][:10], "Status": a["status"],
                              "PAC": CLASS_LABEL.get(int(r["pac_pred"]), "?")})
        st.markdown("#### All Applications")
        if rows: st.table(rows)
        else: st.info("No applications yet.")

    elif section == "👤 Edit Profile":
        with st.form("sp"):
            c1,c2 = st.columns(2)
            name   = c1.text_input("Full Name",   value=s.get("name","") or "")
            phone  = c2.text_input("Phone",        value=s.get("phone","") or "")
            skills = st.text_input("Skills",       value=s.get("skills","") or "")
            c3,c4  = st.columns(2)
            exp    = c3.number_input("Experience (yrs)", min_value=0, value=int(s.get("experience") or 0))
            loc    = c4.text_input("Preferred Location", value=s.get("preferred_location","") or "")
            bio    = st.text_area("About Me", value=s.get("bio","") or "")
            salary = st.text_input("Expected Salary", value=s.get("expected_salary","") or "")
            if st.form_submit_button("Save Profile", use_container_width=True):
                sk_update(u["id"], name, phone, skills, exp, loc, bio, salary)
                st.session_state.user["name"] = name; st.success("✅ Saved!")


# ─────────────────────────────────────────────────────────────────────────────
# COMPANY DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────
def page_dash_company():
    u = st.session_state.user
    if not u or u["role"] != "company": go("login"); return

    with st.sidebar:
        st.markdown(f"### 🏢 {u['name']}\n*{u['email']}*"); st.divider()
        section = st.radio("Navigate",
            ["📊 Overview","➕ Post a Job","📋 My Postings","👥 Applicants","🏢 Profile"],
            label_visibility="collapsed")
        st.divider()
        if st.button("🚪 Logout"): logout()

    t, a = co_stats(u["id"])
    st.markdown(f"""
    <div class="hero" style="padding:2rem;">
      <h1 style="font-size:1.8rem;">Employer Dashboard 🏢</h1>
      <p>Manage your job postings — {u['name']}</p>
    </div>""", unsafe_allow_html=True)

    if section == "📊 Overview":
        st.markdown(f"""
        <div class="stat-row">
          <div class="stat-card"><div class="n">{t}</div><div class="l">Jobs Posted</div></div>
          <div class="stat-card"><div class="n">{a}</div><div class="l">Total Applicants</div></div>
        </div>""", unsafe_allow_html=True)
        jobs = co_jobs(u["id"])[:5]
        if not jobs: st.info("No jobs posted yet. Use ➕ Post a Job.")
        else: st.table([{"Title": j["title"], "Location": j["location"] or "Remote", "Posted": j["created_at"][:10], "Applicants": j["appl"]} for j in jobs])

    elif section == "➕ Post a Job":
        st.markdown("#### Post a New Job")
        st.markdown('<div class="box-ai">🤖 PAC model will automatically classify your posted job as Real, Fake, or Irrelevant.</div>', unsafe_allow_html=True)
        with st.form("pj"):
            c1,c2 = st.columns(2); title=c1.text_input("Job Title *"); jtype=c2.selectbox("Type",["Full-time","Part-time","Remote","Internship","Contract"])
            c3,c4 = st.columns(2); loc=c3.text_input("Location"); salary=c4.text_input("Salary Range")
            exp = st.number_input("Experience Required (yrs)", min_value=0)
            desc = st.text_area("Job Description *", height=130)
            req  = st.text_area("Requirements", height=90)
            mob  = st.text_input("Contact Mobile")
            sub  = st.form_submit_button("Post Job", use_container_width=True)
        if sub:
            if not title or not desc: st.error("Title and description required.")
            else: job_post(u["id"],title,jtype,loc,salary,exp,None,desc,req,mob); st.success("✅ Job posted!")

    elif section == "📋 My Postings":
        jobs = co_jobs(u["id"])
        if not jobs: st.info("No jobs yet.")
        else:
            for j in jobs:
                c1,c2,c3 = st.columns([4,1,1])
                c1.markdown(f"**{j['title']}** — {j['location'] or 'Remote'}\n`{j['job_type']}` · {j['appl']} applicants")
                c2.caption(j["created_at"][:10])
                if c3.button("🗑", key=f"del_{j['id']}"): job_delete(j["id"], u["id"]); st.success("Deleted."); st.rerun()

    elif section == "👥 Applicants":
        jobs = co_jobs(u["id"]); jmap = {"All Jobs": None}
        for j in jobs: jmap[j["title"]] = j["id"]
        chosen = st.selectbox("Filter", list(jmap.keys()))
        appl = co_applicants(u["id"], jmap[chosen])
        if not appl: st.info("No applicants yet.")
        else: st.table([{"Name": a["name"],"Email": a["email"],"Skills": a["skills"] or "—","Exp": f"{a['experience'] or 0} yrs","Job": a["job_title"],"Applied": a["applied_at"][:10]} for a in appl])

    elif section == "🏢 Profile":
        co = co_get(u["id"])
        with st.form("cp"):
            c1,c2 = st.columns(2); name=c1.text_input("Company Name", value=co.get("name","") or "")
            industry = c2.selectbox("Industry", ["IT / Software","Finance","Healthcare","Manufacturing","E-commerce","Other"])
            c3,c4 = st.columns(2); website=c3.text_input("Website", value=co.get("website","") or ""); year=c4.number_input("Year Founded", min_value=1900, max_value=date.today().year, value=int(co.get("year_founded") or 2010))
            phone = st.text_input("Phone", value=co.get("phone","") or "")
            desc  = st.text_area("Description", value=co.get("description","") or "")
            if st.form_submit_button("Save", use_container_width=True):
                co_update(u["id"],name,industry,website,year,phone,desc)
                st.session_state.user["name"] = name; st.success("✅ Saved!")


# ─────────────────────────────────────────────────────────────────────────────
# ROUTER
# ─────────────────────────────────────────────────────────────────────────────
def _route():
    page = st.session_state.get("page", "home")
    # Handle sel_job being set by go() before page change
    if st.session_state.get("sel_job") and page != "jobs":
        st.session_state.page = "jobs"
        page = "jobs"
    {
        "home":     page_home,
        "jobs":     page_jobs,
        "login":    page_login,
        "register": page_register,
        "dash_sk":  page_dash_seeker,
        "dash_co":  page_dash_company,
    }.get(page, page_home)()

_route()
