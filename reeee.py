"""
TrueHire — Fake & Real Job Classification Portal
Uses your pac_model.pkl (PassiveAggressiveClassifier) + tfidf_vectorizer.pkl directly.

Required files (same folder):
  cleaned_jobs.csv | pac_model.pkl | tfidf_vectorizer.pkl

requirements.txt:
  streamlit>=1.35.0
  scikit-learn>=1.3.0
  pandas>=2.0.0
  numpy>=1.24.0
  scipy>=1.11.0
  joblib>=1.3.0
"""

import os, re, warnings, hashlib, sqlite3
from datetime import date

import numpy as np
import pandas as pd
import joblib
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="TrueHire — Job Verification Portal",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

_BASE    = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(_BASE, "cleaned_jobs for project.csv")
MOD_PATH = os.path.join(_BASE, "pac_model.pkl")
VEC_PATH = os.path.join(_BASE, "tfidf_vectorizer.pkl")
DB_PATH  = os.path.join(_BASE, "truehire.db")

CLASS_LABEL = {0: "Real", 1: "Fake", 2: "Irrelevant"}
TEXT_COLS   = ["title","company_profile","description","requirements","salary_range","location","industry"]

IRRELEVANT_JOBS = [
    {"title":"Astrological Chart Reader","location":"Remote / Cosmos","employment_type":"Freelance",
     "industry":"Spirituality","salary_range":"Unlimited* (unverified)",
     "description":"Read star charts and provide daily horoscope advice via WhatsApp. No qualifications needed.",
     "requirements":"Know your own sun sign. Crystal ball optional.","company_profile":"CosmoGuide Inc."},
    {"title":"Professional Netflix Watcher","location":"Your Couch","employment_type":"Part-time",
     "industry":"Entertainment","salary_range":"Paid in OTT credits",
     "description":"Watch Netflix 8 hrs/day and submit a 5-minute survey. Fast Wi-Fi and unlimited snacks required.",
     "requirements":"Active Netflix account. Working eyes. High binge tolerance.","company_profile":"StreamRate Inc."},
    {"title":"Zombie Apocalypse Survival Consultant","location":"Undisclosed Bunker","employment_type":"Contract",
     "industry":"Preparedness / Defense","salary_range":"Canned goods + ammunition",
     "description":"Train civilians in zombie evasion and barricade construction. Crossbow skills a plus.",
     "requirements":"Survival instinct. Zero fear of the undead.","company_profile":"ZombiePrep LLC."},
    {"title":"Moon Dust Collector","location":"Moon (Earth travel occasional)","employment_type":"Full-time",
     "industry":"Space / Mining","salary_range":"Negotiable upon safe return",
     "description":"Collect regolith samples on the lunar surface. Tolerate 2-week communication blackouts.",
     "requirements":"Space suit provided. Must hold breath 8 seconds minimum.","company_profile":"LunarMineCo."},
    {"title":"Chief Snack Officer (CSO)","location":"Office Pantry, Anywhere","employment_type":"Volunteer",
     "industry":"Food & Beverages","salary_range":"Unlimited snacks (no cash)",
     "description":"Curate weekly snack lists, taste-test chips, maintain biscuit inventory. Reports to CEO.",
     "requirements":"Strong opinions on Parle-G. Dislike of raisins preferred.","company_profile":"SnackHub."},
]

# ── Email validator ───────────────────────────────────────────────────────────
def is_valid_email(email: str) -> bool:
    return bool(re.match(r'^[^@\s]+@[^@\s]+\.[^@\s]+$', email.strip()))

def is_valid_phone(phone: str) -> bool:
    digits = re.sub(r'\D', '', phone)
    return len(digits) >= 10

# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:ital,wght@0,700;0,900;1,500&family=DM+Sans:wght@300;400;500;600;700&display=swap');
:root{--navy:#0f2744;--navy2:#1a3c5e;--amber:#e8a020;--accent:#e8734a;
  --real:#16a34a;--fake:#dc2626;--irr:#a16207;
  --bg:#f5f3ef;--card:#fff;--border:#e2ddd6;--text:#1c1a17;--muted:#6b6560;}
*{box-sizing:border-box;}
html,body,[class*="css"]{font-family:'DM Sans',sans-serif;background:var(--bg)!important;color:var(--text);}
#MainMenu,footer,header{visibility:hidden;}
.block-container{padding-top:.8rem!important;max-width:1180px!important;}

.hero{background:linear-gradient(135deg,#0f2744 60%,#1a3c5e 100%);border-radius:20px;
  padding:3rem 3rem;margin-bottom:1.5rem;color:#fff;position:relative;overflow:hidden;}
.hero::before{content:'';position:absolute;top:-100px;right:-80px;width:420px;height:420px;
  border-radius:50%;background:radial-gradient(circle,rgba(232,160,32,.2),transparent 70%);}
.hero::after{content:'';position:absolute;bottom:-60px;left:-60px;width:280px;height:280px;
  border-radius:50%;background:radial-gradient(circle,rgba(232,115,74,.12),transparent 70%);}
.hero h1{font-family:'Fraunces',serif;font-size:clamp(2rem,4vw,3rem);font-weight:900;
  line-height:1.08;margin:0 0 .6rem;position:relative;z-index:1;}
.hero h1 em{font-style:italic;color:var(--amber);}
.hero p{opacity:.88;font-size:1rem;max-width:560px;line-height:1.75;margin:0;position:relative;z-index:1;}
.hero-pills{display:flex;gap:.7rem;flex-wrap:wrap;margin-top:1.4rem;position:relative;z-index:1;}
.hero-pill{background:rgba(255,255,255,.12);border:1px solid rgba(255,255,255,.22);
  border-radius:50px;padding:.3rem .95rem;font-size:.78rem;color:#dbeafe;}
.hero-pill strong{color:var(--amber);}

.stat-row{display:flex;gap:1rem;flex-wrap:wrap;margin-bottom:1.4rem;}
.stat-card{background:var(--card);border-radius:14px;padding:1.2rem 1.5rem;flex:1;min-width:130px;
  border:1px solid var(--border);box-shadow:0 2px 12px rgba(15,39,68,.07);text-align:center;}
.stat-card .n{font-family:'Fraunces',serif;font-size:2rem;font-weight:900;color:var(--navy2);line-height:1;}
.stat-card .l{font-size:.73rem;color:var(--muted);margin-top:.25rem;letter-spacing:.04em;text-transform:uppercase;}

.jcard{background:var(--card);border-radius:14px;padding:1.25rem 1.5rem;border:1px solid var(--border);
  margin-bottom:.8rem;box-shadow:0 2px 8px rgba(15,39,68,.05);
  transition:box-shadow .2s,border-color .2s,transform .2s;}
.jcard:hover{box-shadow:0 10px 32px rgba(15,39,68,.13);border-color:var(--accent);transform:translateY(-2px);}
.jcard h3{margin:0 0 .2rem;color:var(--navy2);font-size:1rem;font-weight:700;}
.jcard .meta{font-size:.81rem;color:var(--muted);margin-bottom:.5rem;}
.bar-wrap{background:#ede9e3;border-radius:50px;height:5px;margin-top:.55rem;}
.bar{background:linear-gradient(90deg,var(--accent),var(--amber));border-radius:50px;height:5px;}

.tag{display:inline-block;background:#f0ede8;border:1px solid var(--border);border-radius:50px;
  padding:.15rem .62rem;font-size:.7rem;color:#4b4740;margin:.12rem .18rem 0 0;}
.tag-sal{background:#fff8ed;border-color:var(--amber);color:#92600a;}
.b-real{background:#dcfce7;color:var(--real);border:1px solid #86efac;border-radius:50px;
  padding:.13rem .62rem;font-size:.7rem;font-weight:700;}
.b-fake{background:#fee2e2;color:var(--fake);border:1px solid #fca5a5;border-radius:50px;
  padding:.13rem .62rem;font-size:.7rem;font-weight:700;}
.b-irr{background:#fef9c3;color:var(--irr);border:1px solid #fde047;border-radius:50px;
  padding:.13rem .62rem;font-size:.7rem;font-weight:700;}
.b-ai{background:#fef3c7;color:#b45309;border-radius:50px;padding:.13rem .62rem;font-size:.7rem;font-weight:600;}

.fpill{display:inline-flex;align-items:center;gap:.4rem;border-radius:8px;
  padding:.35rem .85rem;font-size:.78rem;font-weight:700;margin:.35rem 0;}
.fp-real{background:#dcfce7;color:var(--real);border:1.5px solid #86efac;}
.fp-fake{background:#fee2e2;color:var(--fake);border:1.5px solid #fca5a5;}
.fp-irr{background:#fef9c3;color:var(--irr);border:1.5px solid #fde047;}

.sec-title{font-family:'Fraunces',serif;color:var(--navy2);font-size:1.45rem;font-weight:900;margin-bottom:.1rem;}
.sec-sub{color:var(--muted);font-size:.86rem;margin-bottom:1rem;}

.box-info{background:#eff6ff;border-left:4px solid #3b82f6;border-radius:10px;
  padding:.8rem 1rem;margin-bottom:1rem;font-size:.88rem;}
.box-ok{background:#f0fdf4;border-left:4px solid var(--real);border-radius:10px;
  padding:.8rem 1rem;margin-bottom:1rem;font-size:.88rem;}
.box-warn{background:#fef2f2;border-left:4px solid var(--fake);border-radius:10px;
  padding:.8rem 1rem;margin-bottom:1rem;font-size:.88rem;}
.box-ai{background:#fffbeb;border-left:4px solid var(--amber);border-radius:10px;
  padding:.8rem 1rem;margin-bottom:1rem;font-size:.88rem;}
.box-irr{background:#fefce8;border-left:4px solid var(--irr);border-radius:10px;
  padding:.8rem 1rem;margin-bottom:1rem;font-size:.88rem;}

.auth-card{background:var(--card);border-radius:18px;padding:2.2rem 2.5rem;
  border:1px solid var(--border);box-shadow:0 8px 32px rgba(15,39,68,.1);max-width:520px;margin:0 auto;}
.auth-card h2{font-family:'Fraunces',serif;color:var(--navy2);font-size:1.6rem;
  font-weight:900;margin-bottom:.2rem;text-align:center;}
.auth-card .sub{color:var(--muted);font-size:.86rem;text-align:center;margin-bottom:1.5rem;}
.field-label{font-size:.82rem;font-weight:600;color:var(--navy2);margin-bottom:.25rem;display:block;}
.field-hint{font-size:.72rem;color:var(--muted);margin-top:.2rem;}
.divider-text{display:flex;align-items:center;gap:.8rem;margin:1rem 0;color:var(--muted);font-size:.8rem;}
.divider-text::before,.divider-text::after{content:'';flex:1;border-top:1px solid var(--border);}

.detail-card{background:var(--card);border-radius:18px;padding:2.2rem;border:1px solid var(--border);
  box-shadow:0 6px 24px rgba(15,39,68,.09);}
.detail-card h2{font-family:'Fraunces',serif;color:var(--navy2);font-size:1.6rem;font-weight:900;margin-bottom:.2rem;}

div.stButton>button{background:var(--navy2)!important;color:#fff!important;border:none!important;
  border-radius:10px!important;font-weight:600!important;font-family:'DM Sans',sans-serif!important;
  padding:.5rem 1.2rem!important;transition:all .18s!important;}
div.stButton>button:hover{background:var(--accent)!important;transform:translateY(-1px);box-shadow:0 4px 12px rgba(232,115,74,.3)!important;}
div.stButton>button:active{transform:translateY(0)!important;}
div[data-testid="stFormSubmitButton"]>button{background:var(--navy2)!important;color:#fff!important;
  border:none!important;border-radius:10px!important;font-weight:600!important;width:100%!important;
  padding:.65rem!important;font-size:1rem!important;}
div[data-testid="stFormSubmitButton"]>button:hover{background:var(--accent)!important;}

.stTextInput>div>input,.stTextArea>div>textarea,.stNumberInput>div>input,.stSelectbox>div>div{
  border-radius:10px!important;border:1.5px solid var(--border)!important;
  font-family:'DM Sans',sans-serif!important;}
.stTextInput>div>input:focus,.stTextArea>div>textarea:focus{
  border-color:var(--accent)!important;box-shadow:0 0 0 3px rgba(232,115,74,.15)!important;}

table{width:100%;border-collapse:collapse;font-size:.85rem;}
th{background:#f5f3ef;color:var(--navy2);padding:.65rem .9rem;text-align:left;font-weight:700;
  border-bottom:2px solid var(--border);}
td{padding:.65rem .9rem;border-bottom:1px solid var(--border);color:#374151;}
tr:hover td{background:#faf8f4;}

section[data-testid="stSidebar"]{background:var(--navy)!important;}
section[data-testid="stSidebar"] *{color:#d1e3f8!important;}
section[data-testid="stSidebar"] .stProgress>div>div{background:var(--amber)!important;}

.nav-logo{font-family:'Fraunces',serif;font-size:1.4rem;font-weight:900;color:var(--navy);}
.nav-logo em{font-style:italic;color:var(--amber);}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# ML ENGINE — loads pac_model.pkl + tfidf_vectorizer.pkl directly
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="🛡️ Loading PAC model and dataset…")
def build_ml_engine():
    df = pd.read_csv(CSV_PATH)
    if "fraudulent" in df.columns and "label" not in df.columns:
        df = df.rename(columns={"fraudulent": "label"})
    df["label"] = pd.to_numeric(df["label"], errors="coerce").fillna(0).astype(int)
    df = df.reset_index(drop=True)
    for col in TEXT_COLS + ["employment_type","required_experience","required_education"]:
        if col not in df.columns: df[col] = ""
        df[col] = df[col].fillna("").astype(str)

    # Load your existing pac_model.pkl and tfidf_vectorizer.pkl
    pac_vec = joblib.load(VEC_PATH)
    model   = joblib.load(MOD_PATH)

    # Pre-classify all CSV rows with your PAC model
    combined = df[TEXT_COLS].agg(" ".join, axis=1)
    X = pac_vec.transform(combined)
    df["pac_pred"] = model.predict(X).astype(int)

    # Separate TF-IDF for cosine search
    search_vec = TfidfVectorizer(max_features=6000, ngram_range=(1,2),
                                  stop_words="english", sublinear_tf=True)
    search_mat = search_vec.fit_transform(
        df["title"] + " " + df["industry"] + " " + df["description"])
    return df, pac_vec, model, search_vec, search_mat


def pac_classify(td: dict, pac_vec, model) -> int:
    """Classify one job dict using your pac_model.pkl."""
    def _c(t):
        if not isinstance(t, str): return ""
        t = t.lower(); t = re.sub(r"<[^>]+>"," ",t)
        t = re.sub(r"https?://\S+"," ",t); t = re.sub(r"[^\w\s]"," ",t)
        return re.sub(r"\s+"," ",t).strip()
    text = " ".join(_c(str(td.get(k,""))) for k in TEXT_COLS)
    return int(model.predict(pac_vec.transform([text]))[0])


def do_search(q, df, sv, sm):
    if not q.strip(): return df.copy().assign(score=1.0)
    sims = cosine_similarity(sv.transform([q]), sm).flatten()
    return df.copy().assign(score=sims).sort_values("score", ascending=False)


# ─────────────────────────────────────────────────────────────────────────────
# BADGE HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def pbadge(c):
    if c==0: return "<span class='b-real'>✔ Real</span>"
    if c==1: return "<span class='b-fake'>✘ Fake</span>"
    return "<span class='b-irr'>? Irrelevant</span>"

def fpill(v):
    if v==0: return "<span class='fpill fp-real'>🟢 fraudulent=0 → Real</span>"
    if v==1: return "<span class='fpill fp-fake'>🔴 fraudulent=1 → Fake</span>"
    return "<span class='fpill fp-irr'>🟡 Irrelevant</span>"


# ─────────────────────────────────────────────────────────────────────────────
# DATABASE
# ─────────────────────────────────────────────────────────────────────────────
def _conn():
    c = sqlite3.connect(DB_PATH, check_same_thread=False)
    c.row_factory = sqlite3.Row; return c

def init_db():
    c = _conn(); c.executescript("""
    CREATE TABLE IF NOT EXISTS companies(
        id INTEGER PRIMARY KEY AUTOINCREMENT,name TEXT NOT NULL,
        email TEXT UNIQUE NOT NULL,password_hash TEXT NOT NULL,
        phone TEXT,industry TEXT,website TEXT,year_founded INTEGER,
        description TEXT,created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP);
    CREATE TABLE IF NOT EXISTS seekers(
        id INTEGER PRIMARY KEY AUTOINCREMENT,name TEXT NOT NULL,
        email TEXT UNIQUE NOT NULL,password_hash TEXT NOT NULL,
        phone TEXT,skills TEXT,experience INTEGER DEFAULT 0,
        preferred_location TEXT,bio TEXT,expected_salary TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP);
    CREATE TABLE IF NOT EXISTS jobs(
        id INTEGER PRIMARY KEY AUTOINCREMENT,company_id INTEGER NOT NULL,
        title TEXT NOT NULL,job_type TEXT DEFAULT 'Full-time',
        location TEXT,salary_range TEXT,experience_required INTEGER DEFAULT 0,
        description TEXT,requirements TEXT,contact_mobile TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(company_id) REFERENCES companies(id));
    CREATE TABLE IF NOT EXISTS applications(
        id INTEGER PRIMARY KEY AUTOINCREMENT,job_id INTEGER NOT NULL,
        seeker_id INTEGER NOT NULL,status TEXT DEFAULT 'Under Review',
        applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(job_id,seeker_id),
        FOREIGN KEY(job_id) REFERENCES jobs(id),
        FOREIGN KEY(seeker_id) REFERENCES seekers(id));
    CREATE TABLE IF NOT EXISTS ds_applications(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ds_idx INTEGER NOT NULL,seeker_id INTEGER NOT NULL,
        status TEXT DEFAULT 'Under Review',
        applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(ds_idx,seeker_id),
        FOREIGN KEY(seeker_id) REFERENCES seekers(id));
    """); c.commit(); c.close()

init_db()
_hp = lambda pw: hashlib.sha256(pw.encode()).hexdigest()

def co_register(name,email,pw,phone,industry,year,desc):
    c=_conn()
    try: c.execute("INSERT INTO companies(name,email,password_hash,phone,industry,year_founded,description) VALUES(?,?,?,?,?,?,?)",(name,email,_hp(pw),phone,industry,year,desc)); c.commit(); return True,"Registered!"
    except sqlite3.IntegrityError: return False,"Email already registered."
    finally: c.close()

def co_login(email,pw):
    c=_conn(); r=c.execute("SELECT * FROM companies WHERE email=? AND password_hash=?",(email,_hp(pw))).fetchone(); c.close(); return dict(r) if r else None

def co_get(cid):
    c=_conn(); r=c.execute("SELECT * FROM companies WHERE id=?",(cid,)).fetchone(); c.close(); return dict(r) if r else {}

def co_update(cid,name,industry,website,year,phone,desc):
    c=_conn(); c.execute("UPDATE companies SET name=?,industry=?,website=?,year_founded=?,phone=?,description=? WHERE id=?",(name,industry,website,year,phone,desc,cid)); c.commit(); c.close()

def sk_register(name,email,pw,phone,skills,exp):
    c=_conn()
    try: c.execute("INSERT INTO seekers(name,email,password_hash,phone,skills,experience) VALUES(?,?,?,?,?,?)",(name,email,_hp(pw),phone,skills,exp)); c.commit(); return True,"Account created!"
    except sqlite3.IntegrityError: return False,"Email already registered."
    finally: c.close()

def sk_login(email,pw):
    c=_conn(); r=c.execute("SELECT * FROM seekers WHERE email=? AND password_hash=?",(email,_hp(pw))).fetchone(); c.close(); return dict(r) if r else None

def sk_get(sid):
    c=_conn(); r=c.execute("SELECT * FROM seekers WHERE id=?",(sid,)).fetchone(); c.close(); return dict(r) if r else {}

def sk_update(sid,name,phone,skills,exp,loc,bio,salary):
    c=_conn(); c.execute("UPDATE seekers SET name=?,phone=?,skills=?,experience=?,preferred_location=?,bio=?,expected_salary=? WHERE id=?",(name,phone,skills,exp,loc,bio,salary,sid)); c.commit(); c.close()

def pf_score(s): return int(sum(1 for f in["name","phone","skills","bio","preferred_location","expected_salary"] if s.get(f))/6*100)

def job_post(cid,title,jtype,loc,salary,exp,desc,req,mob):
    c=_conn(); c.execute("INSERT INTO jobs(company_id,title,job_type,location,salary_range,experience_required,description,requirements,contact_mobile) VALUES(?,?,?,?,?,?,?,?,?)",(cid,title,jtype,loc,salary,exp,desc,req,mob)); c.commit(); c.close()

def jobs_get(q="",loc="",limit=200):
    c=_conn()
    sql="SELECT j.*,co.name AS company_name FROM jobs j JOIN companies co ON j.company_id=co.id WHERE 1=1"; p=[]
    if q: sql+=" AND (j.title LIKE ? OR j.description LIKE ?)"; p+=[f"%{q}%",f"%{q}%"]
    if loc: sql+=" AND j.location LIKE ?"; p.append(f"%{loc}%")
    sql+=" ORDER BY j.created_at DESC LIMIT ?"; p.append(limit)
    rows=c.execute(sql,p).fetchall(); c.close(); return [dict(r) for r in rows]

def job_get(jid):
    c=_conn(); r=c.execute("SELECT j.*,co.name AS company_name FROM jobs j JOIN companies co ON j.company_id=co.id WHERE j.id=?",(jid,)).fetchone(); c.close(); return dict(r) if r else None

def co_jobs(cid):
    c=_conn(); rows=c.execute("SELECT j.*,COUNT(a.id) AS appl FROM jobs j LEFT JOIN applications a ON a.job_id=j.id WHERE j.company_id=? GROUP BY j.id ORDER BY j.created_at DESC",(cid,)).fetchall(); c.close(); return [dict(r) for r in rows]

def job_delete(jid,cid):
    c=_conn(); c.execute("DELETE FROM applications WHERE job_id=?",(jid,)); c.execute("DELETE FROM jobs WHERE id=? AND company_id=?",(jid,cid)); c.commit(); c.close()

def co_applicants(cid,jid=None):
    c=_conn()
    sql="SELECT s.name,s.email,s.phone,s.skills,s.experience,a.applied_at,a.status,j.title AS job_title FROM applications a JOIN seekers s ON s.id=a.seeker_id JOIN jobs j ON j.id=a.job_id WHERE j.company_id=?"; p=[cid]
    if jid: sql+=" AND a.job_id=?"; p.append(jid)
    rows=c.execute(sql+" ORDER BY a.applied_at DESC",p).fetchall(); c.close(); return [dict(r) for r in rows]

def co_stats(cid):
    c=_conn(); t=c.execute("SELECT COUNT(*) FROM jobs WHERE company_id=?",(cid,)).fetchone()[0]; a=c.execute("SELECT COUNT(*) FROM applications a JOIN jobs j ON j.id=a.job_id WHERE j.company_id=?",(cid,)).fetchone()[0]; c.close(); return t,a

def ds_apply(idx,sid):
    c=_conn()
    try: c.execute("INSERT INTO ds_applications(ds_idx,seeker_id) VALUES(?,?)",(idx,sid)); c.commit(); return True
    except sqlite3.IntegrityError: return False
    finally: c.close()

def ds_applied(idx,sid):
    c=_conn(); r=c.execute("SELECT 1 FROM ds_applications WHERE ds_idx=? AND seeker_id=?",(idx,sid)).fetchone(); c.close(); return bool(r)

def ds_my_apps(sid):
    c=_conn(); rows=c.execute("SELECT * FROM ds_applications WHERE seeker_id=? ORDER BY applied_at DESC",(sid,)).fetchall(); c.close(); return [dict(r) for r in rows]

def sk_stats(sid):
    c=_conn(); a=c.execute("SELECT COUNT(*) FROM applications WHERE seeker_id=?",(sid,)).fetchone()[0]; b=c.execute("SELECT COUNT(*) FROM ds_applications WHERE seeker_id=?",(sid,)).fetchone()[0]; c.close(); return a+b,b


# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────
for k,v in [("page","home"),("user",None),("sel",None),("sq",""),("sl","")]:
    if k not in st.session_state: st.session_state[k]=v

def go(page, job=None):
    st.session_state.page = page
    if job is not None: st.session_state.sel = job
    elif page != "jobs": st.session_state.sel = None
    st.rerun()

def logout(): st.session_state.user=None; st.session_state.sel=None; go("home")


# ─────────────────────────────────────────────────────────────────────────────
# NAVBAR
# ─────────────────────────────────────────────────────────────────────────────
def navbar():
    u = st.session_state.user
    c0,c1,c2,c3,c4 = st.columns([2.5,1,1,1.3,1.3])
    c0.markdown('<span class="nav-logo">🛡️ True<em>Hire</em></span>', unsafe_allow_html=True)
    if c1.button("🏠 Home",  key="n_home", use_container_width=True): go("home")
    if c2.button("💼 Jobs",  key="n_jobs", use_container_width=True):
        st.session_state.sel=None; go("jobs")
    if u:
        dash = "dash_sk" if u["role"]=="seeker" else "dash_co"
        lbl  = f"👤 {u['name'].split()[0]}"
        if c3.button(lbl, key="n_dash", use_container_width=True): go(dash)
        if c4.button("🚪 Logout", key="n_lo", use_container_width=True): logout()
    else:
        if c3.button("🔑 Login",   key="n_li", use_container_width=True): go("login")
        if c4.button("✏️ Sign Up", key="n_su", use_container_width=True): go("register")
    st.markdown("<hr style='border:none;border-top:1.5px solid #e2ddd6;margin:.3rem 0 1rem;'>",
                unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# HOME PAGE
# ─────────────────────────────────────────────────────────────────────────────
def page_home():
    navbar()
    df, pac_vec, model, sv, sm = build_ml_engine()
    real_n = int((df["pac_pred"]==0).sum())
    fake_n = int((df["pac_pred"]==1).sum())

    st.markdown(f"""
    <div class="hero">
      <h1>Stop <em>Fake Jobs</em><br>Before They Stop You</h1>
      <p>Your PAC model (<code>pac_model.pkl</code>) analyses every listing.
         fraudulent label: <b>0 = Real · 1 = Fake · 2 = Irrelevant</b></p>
      <div class="hero-pills">
        <div class="hero-pill"><strong>{len(df):,}</strong> total listings</div>
        <div class="hero-pill"><strong style="color:#4ade80">{real_n:,}</strong> real</div>
        <div class="hero-pill"><strong style="color:#f87171">{fake_n:,}</strong> fake</div>
        <div class="hero-pill"><strong>pac_model.pkl</strong> loaded</div>
      </div>
    </div>""", unsafe_allow_html=True)

    # Search bar
    sc1,sc2,sc3 = st.columns([3,2,1])
    q   = sc1.text_input("Search",   placeholder="Python, Data Scientist, Finance…", label_visibility="collapsed", key="home_q")
    loc = sc2.text_input("Location", placeholder="City or Remote",                   label_visibility="collapsed", key="home_l")
    if sc3.button("🔍 Search", use_container_width=True, key="home_srch"):
        st.session_state.sq=q; st.session_state.sl=loc
        st.session_state.sel=None; go("jobs")

    # Live classifier
    st.markdown('<p class="sec-title" style="margin-top:1.6rem;">🔬 Live Job Classifier</p>', unsafe_allow_html=True)
    st.markdown('<p class="sec-sub">Paste any job text — your PAC model classifies it instantly</p>', unsafe_allow_html=True)
    with st.expander("▶ Open live classifier", expanded=False):
        lt = st.text_area("Job title + description", height=110, key="live_txt",
                           placeholder="Earn money fast! No investment needed. Work from home, guaranteed income…")
        lsal = st.text_input("Salary range (optional)", key="live_sal", placeholder="e.g. ₹5000/week")
        lcmp = st.text_input("Company info (optional)", key="live_cmp", placeholder="Brief company description")
        if st.button("⚡ Classify with PAC model", key="live_btn", use_container_width=True):
            if lt.strip():
                cls = pac_classify({"title":lt[:100],"description":lt,"salary_range":lsal,"company_profile":lcmp}, pac_vec, model)
                bg  = {"Real":"#dcfce7","Fake":"#fee2e2","Irrelevant":"#fef9c3"}[CLASS_LABEL[cls]]
                ico = {"Real":"✅","Fake":"⚠️","Irrelevant":"❓"}[CLASS_LABEL[cls]]
                st.markdown(f'<div style="background:{bg};border-radius:12px;padding:1rem 1.3rem;'
                             f'font-size:1rem;font-weight:700;margin-top:.5rem;">'
                             f'{ico} pac_model.pkl → <b>{CLASS_LABEL[cls]}</b> (fraudulent={cls})</div>',
                             unsafe_allow_html=True)
            else:
                st.warning("Please enter some job text first.")

    # Featured listings
    st.markdown('<p class="sec-title" style="margin-top:1.8rem;">Featured Listings</p>', unsafe_allow_html=True)
    st.markdown('<p class="sec-sub">Sampled from cleaned_jobs.csv — fraudulent value + PAC prediction shown</p>', unsafe_allow_html=True)
    featured = df.sample(min(6,len(df)), random_state=7)
    cols = st.columns(2)
    for i,(idx,row) in enumerate(featured.iterrows()):
        cls=int(row["pac_pred"]); lbl=int(row["label"])
        with cols[i%2]:
            st.markdown(f"""
            <div class="jcard">
              <h3>{row['title']}</h3>
              <div class="meta">{row.get('location','') or 'N/A'} · {row.get('industry','') or '—'}</div>
              <span class="tag">{row.get('employment_type','') or 'N/A'}</span> {pbadge(cls)}
              <div style="margin-top:.45rem;">{fpill(lbl)}</div>
              <div style="font-size:.68rem;color:#9ca3af;margin-top:.2rem;">
                fraudulent={lbl} | PAC={CLASS_LABEL[cls]}
              </div>
            </div>""", unsafe_allow_html=True)
            if st.button("View Details", key=f"hf_{idx}", use_container_width=True):
                go("jobs", job=("dataset",int(idx)))

    # Irrelevant jobs
    st.divider()
    st.markdown('<p class="sec-title">🎭 Irrelevant Listings</p>', unsafe_allow_html=True)
    st.markdown('<p class="sec-sub">Not real, not fake — just completely off the mark</p>', unsafe_allow_html=True)
    ic = st.columns(2)
    for i,job in enumerate(IRRELEVANT_JOBS):
        with ic[i%2]:
            st.markdown(f"""
            <div class="jcard" style="border-color:#fde047;">
              <h3>{job['title']}</h3>
              <div class="meta">{job['location']} · {job['industry']}</div>
              <span class="tag">{job['employment_type']}</span>
              <span class="b-irr">? Irrelevant</span>
              <div style="font-size:.68rem;color:#9ca3af;margin-top:.3rem;">{job['description'][:90]}…</div>
            </div>""", unsafe_allow_html=True)
            if st.button("View", key=f"hi_{i}", use_container_width=True):
                go("jobs", job=("irrelevant",i))

    st.divider()
    ca,cb = st.columns(2)
    with ca:
        st.markdown("""<div style="background:var(--navy);border-radius:16px;padding:1.8rem;color:#fff;">
        <h3 style="font-family:'Fraunces',serif;color:#fff;margin-bottom:.4rem;">🏢 Hiring?</h3>
        <p style="color:#93c5fd;margin-bottom:.8rem;">Post jobs. PAC model auto-classifies every listing.</p>
        </div>""", unsafe_allow_html=True)
        if st.button("Post a Job →", key="cta_co", use_container_width=True): go("register")
    with cb:
        st.markdown("""<div style="background:#fff8ed;border:1.5px solid #e8a020;border-radius:16px;padding:1.8rem;">
        <h3 style="font-family:'Fraunces',serif;color:var(--navy2);margin-bottom:.4rem;">🔍 Job Hunting?</h3>
        <p style="color:#6b6560;margin-bottom:.8rem;">Real jobs highlighted. Fakes blocked automatically.</p>
        </div>""", unsafe_allow_html=True)
        if st.button("Browse All Jobs →", key="cta_sk", use_container_width=True): go("jobs")


# ─────────────────────────────────────────────────────────────────────────────
# JOBS PAGE
# ─────────────────────────────────────────────────────────────────────────────
def page_jobs():
    navbar()
    df, pac_vec, model, sv, sm = build_ml_engine()

    if st.session_state.sel:
        src,idx = st.session_state.sel
        if src=="dataset":      detail_dataset(df,idx,pac_vec,model)
        elif src=="irrelevant": detail_irrelevant(IRRELEVANT_JOBS[idx])
        elif src=="posted":
            j=job_get(idx)
            if j: detail_posted(j,pac_vec,model)
        if st.button("← Back to listings", key="back_btn"):
            st.session_state.sel=None; st.rerun()
        return

    st.markdown('<p class="sec-title">Browse Jobs</p>', unsafe_allow_html=True)

    with st.expander("🔍 Search & Filter", expanded=True):
        jc1,jc2,jc3 = st.columns(3)
        q    = jc1.text_input("Keyword",  value=st.session_state.sq, placeholder="Python, Finance…", key="jq")
        loc  = jc2.text_input("Location", value=st.session_state.sl, placeholder="City or Remote…", key="jl")
        show = jc3.selectbox("Show",
            ["All","Real Only (label=0)","Fake Only (label=1)","Irrelevant Only"], key="jshow")
        if st.button("🔍 Search", use_container_width=True, key="jsrch"):
            st.session_state.sq=q; st.session_state.sl=loc; st.rerun()

    qt = f"{q} {loc}".strip() or "engineer developer analyst"
    res = do_search(qt, df, sv, sm)
    if loc.strip(): res=res[res["location"].str.contains(loc,case=False,na=False)]
    if show=="Real Only (label=0)":   res=res[res["label"]==0]
    elif show=="Fake Only (label=1)": res=res[res["label"]==1]
    elif show=="Irrelevant Only":     res=res.iloc[0:0]

    posted = jobs_get(q=q,loc=loc)
    irr_count = len(IRRELEVANT_JOBS) if show in("All","Irrelevant Only") else 0
    total = len(res)+len(posted)+irr_count

    st.markdown(f"**{total} listing(s)** &nbsp;"
                f"<span class='b-ai'>🤖 TF-IDF ranked · pac_model.pkl classified</span>",
                unsafe_allow_html=True)
    st.markdown("""<div class="box-ai" style="font-size:.8rem;margin:.6rem 0;">
      <b>Key:</b> <span class="b-real">✔ Real</span>=0 &nbsp;|&nbsp;
      <span class="b-fake">✘ Fake</span>=1 &nbsp;|&nbsp;
      <span class="b-irr">? Irrelevant</span>=2 &nbsp;&nbsp;
      <b>fraudulent</b> = CSV label (0 or 1)
    </div>""", unsafe_allow_html=True)

    # Company-posted
    if posted and show not in("Irrelevant Only",):
        st.markdown("#### 🏢 Company-Posted Jobs")
        for j in posted:
            cls=pac_classify({"title":j["title"],"description":j.get("description",""),
                               "requirements":j.get("requirements",""),"salary_range":j.get("salary_range",""),
                               "location":j.get("location",""),"company_profile":"","industry":""},pac_vec,model)
            st.markdown(f"""
            <div class="jcard">
              <h3>{j['title']}</h3>
              <div class="meta">{j['company_name']} · {j['location'] or 'Remote'}</div>
              <span class="tag-sal">{j['salary_range'] or 'Negotiable'}</span>
              <span class="tag">{j['job_type']}</span> {pbadge(cls)}
              <div style="font-size:.68rem;color:#9ca3af;margin-top:.25rem;">PAC model: <b>{CLASS_LABEL[cls]}</b></div>
            </div>""", unsafe_allow_html=True)
            _,b=st.columns([5,1])
            with b:
                if st.button("Apply", key=f"pa_{j['id']}", use_container_width=True):
                    go("jobs",job=("posted",j["id"]))

    # Dataset listings
    if show!="Irrelevant Only":
        st.markdown("#### 📊 cleaned_jobs.csv Listings")
        for _,row in res.head(80).iterrows():
            pct=max(5,min(99,int(row["score"]*100))); oi=int(row.name)
            cls=int(row["pac_pred"]); lbl=int(row["label"])
            st.markdown(f"""
            <div class="jcard">
              <h3>{row['title']}</h3>
              <div class="meta">{row.get('location','') or 'N/A'} · {row.get('industry','') or '—'}</div>
              <span class="tag">{row.get('employment_type','') or 'N/A'}</span>
              {pbadge(cls)} {fpill(lbl)}
              <span class="b-ai">🤖 {pct}% match</span>
              <div style="font-size:.68rem;color:#9ca3af;margin-top:.25rem;">
                fraudulent={lbl} | PAC={CLASS_LABEL[cls]}
              </div>
              <div class="bar-wrap"><div class="bar" style="width:{pct}%;"></div></div>
            </div>""", unsafe_allow_html=True)
            _,b=st.columns([5,1])
            with b:
                if st.button("Details", key=f"ds_{oi}", use_container_width=True):
                    go("jobs",job=("dataset",oi))

    # Irrelevant
    if show in("All","Irrelevant Only"):
        st.markdown("#### 🎭 Irrelevant Listings")
        for i,job in enumerate(IRRELEVANT_JOBS):
            st.markdown(f"""
            <div class="jcard" style="border-color:#fde047;">
              <h3>{job['title']}</h3>
              <div class="meta">{job['location']} · {job['industry']}</div>
              <span class="tag">{job['employment_type']}</span>
              <span class="b-irr">? Irrelevant</span>
              <div style="font-size:.68rem;color:#9ca3af;margin-top:.25rem;">{job['description'][:100]}…</div>
            </div>""", unsafe_allow_html=True)
            if st.button("View Details", key=f"irr_{i}", use_container_width=True):
                go("jobs",job=("irrelevant",i))


# ─────────────────────────────────────────────────────────────────────────────
# DETAIL VIEWS
# ─────────────────────────────────────────────────────────────────────────────
def detail_dataset(df,idx,pac_vec,model):
    row=df.iloc[idx]; u=st.session_state.user
    cls=int(row["pac_pred"]); lbl=int(row["label"])
    st.markdown(f"""
    <div class="detail-card">
      <div style="display:flex;align-items:center;gap:1.3rem;margin-bottom:1.2rem;">
        <div style="width:56px;height:56px;border-radius:14px;background:#0f2744;color:#fff;
          display:flex;align-items:center;justify-content:center;
          font-family:'Fraunces',serif;font-size:1.4rem;font-weight:900;flex-shrink:0;">
          {str(row['title'])[0].upper()}
        </div>
        <div>
          <h2>{row['title']}</h2>
          <p style="color:#6b6560;font-size:.87rem;margin:0;">
            {row.get('location','') or 'N/A'} · {row.get('industry','') or '—'}
          </p>
        </div>
      </div>
      <div style="margin-bottom:.8rem;">
        <span class="tag">{row.get('employment_type','') or 'N/A'}</span>
        <span class="tag">{row.get('required_experience','') or ''}</span>
        &nbsp;{pbadge(cls)}
      </div>
      <div style="background:#f5f3ef;border-radius:10px;padding:.75rem 1rem;margin-bottom:1rem;">
        {fpill(lbl)}
        <div style="font-size:.76rem;color:#6b6560;margin-top:.3rem;">
          CSV <code>fraudulent</code> = <b>{lbl}</b> → {'🟢 Real' if lbl==0 else '🔴 Fake'}
          &emsp;|&emsp; pac_model.pkl = <b>{cls}</b> → <b>{CLASS_LABEL[cls]}</b>
        </div>
      </div>
      <hr style="border:none;border-top:1px solid #e2ddd6;margin:1rem 0;">
      <h4 style="color:#0f2744;margin-bottom:.35rem;">Company Profile</h4>
      <p style="color:#374151;line-height:1.8;font-size:.91rem;">{row.get('company_profile','') or '—'}</p>
      <h4 style="color:#0f2744;margin-top:1.1rem;margin-bottom:.35rem;">Job Description</h4>
      <p style="color:#374151;line-height:1.8;font-size:.91rem;">{row.get('description','') or '—'}</p>
      <h4 style="color:#0f2744;margin-top:1.1rem;margin-bottom:.35rem;">Requirements</h4>
      <p style="color:#374151;line-height:1.8;font-size:.91rem;">{row.get('requirements','') or '—'}</p>
    </div>""", unsafe_allow_html=True)

    if cls==1: st.markdown('<div class="box-warn">⚠️ <b>Blocked:</b> PAC model flagged this as FAKE. Applications disabled.</div>', unsafe_allow_html=True)
    elif cls==2: st.markdown('<div class="box-irr">❓ Classified as Irrelevant — applications unavailable.</div>', unsafe_allow_html=True)
    elif u and u["role"]=="seeker":
        if ds_applied(idx,u["id"]): st.markdown('<div class="box-ok">✅ You have already applied.</div>', unsafe_allow_html=True)
        else:
            if st.button("✅ Apply Now", use_container_width=True, key="apply_ds"):
                if ds_apply(idx,u["id"]): st.success("🎉 Application submitted!")
                else: st.warning("Already applied.")
    else:
        st.markdown('<div class="box-info">Please <b>login as a Job Seeker</b> to apply.</div>', unsafe_allow_html=True)
        if st.button("🔑 Login to Apply", use_container_width=True, key="login_apply"): go("login")


def detail_irrelevant(job):
    st.markdown(f"""
    <div class="detail-card" style="border-color:#fde047;">
      <div style="display:flex;align-items:center;gap:1.3rem;margin-bottom:1.2rem;">
        <div style="width:56px;height:56px;border-radius:14px;background:#a16207;color:#fff;
          display:flex;align-items:center;justify-content:center;font-size:1.6rem;flex-shrink:0;">❓</div>
        <div>
          <h2>{job['title']}</h2>
          <p style="color:#6b6560;font-size:.87rem;margin:0;">{job['location']} · {job['industry']}</p>
        </div>
      </div>
      <div style="margin-bottom:.8rem;"><span class="tag">{job['employment_type']}</span> <span class="b-irr">? Irrelevant</span></div>
      <div class="box-irr">Classified as <b>Irrelevant</b> (fraudulent=2) — not a genuine job opportunity.</div>
      <h4 style="color:#0f2744;margin-bottom:.35rem;">Description</h4>
      <p style="color:#374151;line-height:1.8;font-size:.91rem;">{job['description']}</p>
      <h4 style="color:#0f2744;margin-top:1.1rem;margin-bottom:.35rem;">Requirements</h4>
      <p style="color:#374151;line-height:1.8;font-size:.91rem;">{job['requirements']}</p>
      <p style="color:#9ca3af;font-size:.82rem;margin-top:1rem;">💰 {job['salary_range']}</p>
    </div>""", unsafe_allow_html=True)
    st.markdown('<div class="box-irr">❓ Applications not available for irrelevant listings.</div>', unsafe_allow_html=True)


def detail_posted(j,pac_vec,model):
    u=st.session_state.user
    cls=pac_classify({"title":j["title"],"description":j.get("description",""),
                       "requirements":j.get("requirements",""),"salary_range":j.get("salary_range",""),
                       "location":j.get("location",""),"company_profile":"","industry":""},pac_vec,model)
    st.markdown(f"""
    <div class="detail-card">
      <div style="display:flex;align-items:center;gap:1.3rem;margin-bottom:1.2rem;">
        <div style="width:56px;height:56px;border-radius:14px;background:#0f2744;color:#fff;
          display:flex;align-items:center;justify-content:center;
          font-family:'Fraunces',serif;font-size:1.4rem;font-weight:900;flex-shrink:0;">
          {str(j['company_name'])[0].upper()}
        </div>
        <div>
          <h2>{j['title']}</h2>
          <p style="color:#6b6560;font-size:.87rem;margin:0;">{j['company_name']} · {j['location'] or 'Remote'}</p>
        </div>
      </div>
      <div style="margin-bottom:.8rem;">
        <span class="tag-sal">{j['salary_range'] or 'Negotiable'}</span>
        <span class="tag">{j['job_type']}</span>
        <span class="tag">{j['experience_required']} yrs exp</span>
        &nbsp;{pbadge(cls)}
      </div>
      <div style="background:#f5f3ef;border-radius:10px;padding:.7rem 1rem;margin-bottom:1rem;font-size:.76rem;color:#6b6560;">
        🤖 pac_model.pkl → <b>{cls}</b> → <b>{CLASS_LABEL[cls]}</b> (live classification)
      </div>
      <hr style="border:none;border-top:1px solid #e2ddd6;margin:1rem 0;">
      <h4 style="color:#0f2744;margin-bottom:.35rem;">Description</h4>
      <p style="color:#374151;line-height:1.8;font-size:.91rem;">{j['description'] or '—'}</p>
      <h4 style="color:#0f2744;margin-top:1.1rem;margin-bottom:.35rem;">Requirements</h4>
      <p style="color:#374151;line-height:1.8;font-size:.91rem;">{j['requirements'] or '—'}</p>
      {f"<p style='font-size:.84rem;color:#6b6560;margin-top:.8rem;'>📞 {j['contact_mobile']}</p>" if j.get('contact_mobile') else ""}
    </div>""", unsafe_allow_html=True)
    if cls==1: st.markdown('<div class="box-warn">⚠️ PAC model flagged this as FAKE. Applications blocked.</div>', unsafe_allow_html=True); return
    if u and u["role"]=="seeker":
        if st.button("✅ Apply Now", use_container_width=True, key="apply_posted"):
            c=_conn()
            try: c.execute("INSERT INTO applications(job_id,seeker_id) VALUES(?,?)",(j["id"],u["id"])); c.commit(); st.success("🎉 Application submitted!")
            except sqlite3.IntegrityError: st.warning("You've already applied.")
            finally: c.close()
    else:
        st.markdown('<div class="box-info">Please <b>login as a Job Seeker</b> to apply.</div>', unsafe_allow_html=True)
        if st.button("🔑 Login to Apply", use_container_width=True, key="login_apply_p"): go("login")


# ─────────────────────────────────────────────────────────────────────────────
# LOGIN PAGE — professional with @ validation
# ─────────────────────────────────────────────────────────────────────────────
def page_login():
    navbar()
    _,mc,_ = st.columns([1,1.8,1])
    with mc:
        st.markdown("""
        <div class="auth-card">
          <div style="text-align:center;margin-bottom:1.2rem;">
            <span style="font-size:2.5rem;">🛡️</span>
          </div>
          <h2>Welcome Back</h2>
          <p class="sub">Sign in to your TrueHire account</p>
        </div>""", unsafe_allow_html=True)

        role = st.radio("I am a:", ["Job Seeker","Company / Employer"],
                         horizontal=True, key="login_role",
                         label_visibility="visible")

        with st.form("lf", clear_on_submit=False):
            email = st.text_input("📧 Email address",
                                   placeholder="yourname@example.com",
                                   key="li_email")
            pw    = st.text_input("🔒 Password", type="password",
                                   placeholder="Enter your password",
                                   key="li_pw")
            sub   = st.form_submit_button("Sign In →", use_container_width=True)

        if sub:
            errs = []
            if not email.strip():
                errs.append("Email is required.")
            elif not is_valid_email(email):
                errs.append("Please enter a valid email address with @ symbol (e.g. name@domain.com).")
            if not pw:
                errs.append("Password is required.")
            if errs:
                for e in errs: st.error(e)
            else:
                u = sk_login(email.strip().lower(), pw) if role=="Job Seeker" else co_login(email.strip().lower(), pw)
                rk = "seeker" if role=="Job Seeker" else "company"
                if u:
                    st.session_state.user={"id":u["id"],"name":u["name"],"email":u["email"],"role":rk}
                    st.success(f"✅ Welcome back, {u['name'].split()[0]}!")
                    go("dash_sk" if rk=="seeker" else "dash_co")
                else:
                    st.error("❌ Invalid email or password. Please try again.")

        st.markdown('<div class="divider-text">or</div>', unsafe_allow_html=True)
        if st.button("Create a new account →", use_container_width=True, key="go_reg"):
            go("register")


# ─────────────────────────────────────────────────────────────────────────────
# REGISTER PAGE — professional with full validation
# ─────────────────────────────────────────────────────────────────────────────
def page_register():
    navbar()
    _,mc,_ = st.columns([0.8,2.4,0.8])
    with mc:
        st.markdown("""
        <div style="text-align:center;margin-bottom:1rem;">
          <span style="font-size:2.5rem;">✏️</span>
          <h2 style="font-family:'Fraunces',serif;color:var(--navy2);font-size:1.6rem;font-weight:900;margin:.3rem 0 .1rem;">Create Your Account</h2>
          <p style="color:var(--muted);font-size:.86rem;margin:0;">Join TrueHire — find real jobs, block fake ones</p>
        </div>""", unsafe_allow_html=True)

        role = st.radio("Register as:", ["Job Seeker","Company / Employer"],
                         horizontal=True, key="reg_role")
        st.markdown("<br>", unsafe_allow_html=True)

        if role=="Job Seeker":
            with st.form("rs", clear_on_submit=False):
                st.markdown("##### 👤 Personal Details")
                rc1,rc2 = st.columns(2)
                name  = rc1.text_input("Full Name *", placeholder="John Smith")
                email = rc2.text_input("Email Address *", placeholder="john@example.com")
                rc3,rc4 = st.columns(2)
                phone = rc3.text_input("Phone Number *", placeholder="9876543210")
                pw    = rc4.text_input("Password *", type="password", placeholder="Min 6 characters")
                rc5,rc6 = st.columns(2)
                pw2   = rc5.text_input("Confirm Password *", type="password", placeholder="Repeat password")
                exp   = rc6.number_input("Experience (years)", min_value=0, max_value=50)
                skills= st.text_input("Skills (comma-separated)", placeholder="Python, SQL, Machine Learning")
                sub   = st.form_submit_button("Create Account →", use_container_width=True)

            if sub:
                errs=[]
                if not name.strip(): errs.append("Full name is required.")
                if not email.strip(): errs.append("Email is required.")
                elif not is_valid_email(email): errs.append("Email must contain @ symbol (e.g. name@gmail.com).")
                if not phone.strip(): errs.append("Phone number is required.")
                elif not is_valid_phone(phone): errs.append("Phone number must be at least 10 digits.")
                if not pw: errs.append("Password is required.")
                elif len(pw)<6: errs.append("Password must be at least 6 characters.")
                elif pw!=pw2: errs.append("Passwords do not match.")
                if errs:
                    for e in errs: st.error(e)
                else:
                    ok,msg=sk_register(name.strip(), email.strip().lower(), pw, phone.strip(), skills, exp)
                    if ok: st.success(f"✅ {msg} Please sign in."); go("login")
                    else: st.error(f"❌ {msg}")

        else:
            with st.form("rc", clear_on_submit=False):
                st.markdown("##### 🏢 Company Details")
                rc1,rc2 = st.columns(2)
                name  = rc1.text_input("Company Name *", placeholder="Acme Corp")
                email = rc2.text_input("Work Email *", placeholder="hr@acmecorp.com")
                rc3,rc4 = st.columns(2)
                phone = rc3.text_input("Phone Number *", placeholder="9876543210")
                pw    = rc4.text_input("Password *", type="password", placeholder="Min 6 characters")
                rc5,rc6 = st.columns(2)
                pw2      = rc5.text_input("Confirm Password *", type="password")
                industry = rc6.selectbox("Industry", ["IT / Software","Finance","Healthcare",
                                                       "Manufacturing","Education","E-commerce","Other"])
                rc7,rc8 = st.columns(2)
                year     = rc7.number_input("Year Founded", min_value=1900, max_value=date.today().year, value=2010)
                website  = rc8.text_input("Website", placeholder="https://company.com")
                desc     = st.text_area("Company Description", placeholder="Brief company overview…")
                sub      = st.form_submit_button("Create Account →", use_container_width=True)

            if sub:
                errs=[]
                if not name.strip(): errs.append("Company name is required.")
                if not email.strip(): errs.append("Work email is required.")
                elif not is_valid_email(email): errs.append("Email must contain @ symbol (e.g. hr@company.com).")
                if not phone.strip(): errs.append("Phone number is required.")
                elif not is_valid_phone(phone): errs.append("Phone number must be at least 10 digits.")
                if not pw: errs.append("Password is required.")
                elif len(pw)<6: errs.append("Password must be at least 6 characters.")
                elif pw!=pw2: errs.append("Passwords do not match.")
                if errs:
                    for e in errs: st.error(e)
                else:
                    ok,msg=co_register(name.strip(), email.strip().lower(), pw, phone.strip(), industry, year, desc)
                    if ok: st.success(f"✅ {msg} Please sign in."); go("login")
                    else: st.error(f"❌ {msg}")

        st.markdown('<div class="divider-text">or</div>', unsafe_allow_html=True)
        if st.button("Already have an account? Sign In →", use_container_width=True, key="go_login"):
            go("login")


# ─────────────────────────────────────────────────────────────────────────────
# SEEKER DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────
def page_dash_seeker():
    u=st.session_state.user
    if not u or u["role"]!="seeker": go("login"); return
    df,*_=build_ml_engine()
    s=sk_get(u["id"]); total,_=sk_stats(u["id"]); sc=pf_score(s)

    with st.sidebar:
        st.markdown(f"### 👤 {u['name']}")
        st.markdown(f"📧 {u['email']}")
        st.progress(sc/100, text=f"Profile {sc}% complete")
        st.divider()
        section=st.radio("Navigate",
            ["📊 Overview","📋 My Applications","👤 Edit Profile"],
            label_visibility="collapsed", key="sk_sec")
        st.divider()
        if st.button("💼 Browse Jobs", use_container_width=True, key="sk_jobs"):
            st.session_state.sel=None; go("jobs")
        if st.button("🚪 Logout", use_container_width=True, key="sk_lo"): logout()

    real_n=int((df["pac_pred"]==0).sum()); fake_n=int((df["pac_pred"]==1).sum())
    st.markdown(f"""
    <div class="hero" style="padding:2rem;">
      <h1 style="font-size:1.8rem;">Welcome back, <em>{u['name'].split()[0]}</em>! 👋</h1>
      <p>Your AI-protected job hub. Fakes blocked by pac_model.pkl.</p>
    </div>""", unsafe_allow_html=True)

    if section=="📊 Overview":
        st.markdown(f"""
        <div class="stat-row">
          <div class="stat-card"><div class="n">{total}</div><div class="l">Applications</div></div>
          <div class="stat-card"><div class="n">{sc}%</div><div class="l">Profile Score</div></div>
          <div class="stat-card"><div class="n">{real_n:,}</div><div class="l">Real Jobs</div></div>
          <div class="stat-card"><div class="n">{fake_n:,}</div><div class="l">Fake Blocked</div></div>
        </div>""", unsafe_allow_html=True)
        apps=ds_my_apps(u["id"])[:5]; rows=[]
        for a in apps:
            i=a["ds_idx"]
            if 0<=i<len(df):
                r=df.iloc[i]
                rows.append({"Job":r["title"],"Location":r.get("location",""),"Applied":a["applied_at"][:10],"Status":a["status"]})
        st.markdown("#### 📋 Recent Applications")
        if rows: st.table(rows)
        else: st.info("No applications yet. Browse Jobs to get started!")

    elif section=="📋 My Applications":
        apps=ds_my_apps(u["id"]); rows=[]
        for a in apps:
            i=a["ds_idx"]
            if 0<=i<len(df):
                r=df.iloc[i]
                rows.append({"Job":r["title"],"Location":r.get("location",""),
                              "Applied":a["applied_at"][:10],"Status":a["status"],
                              "PAC Result":CLASS_LABEL.get(int(r["pac_pred"]),"?")})
        st.markdown("#### 📋 All My Applications")
        if rows: st.table(rows)
        else: st.info("No applications yet.")

    elif section=="👤 Edit Profile":
        st.markdown("#### 👤 Edit Profile")
        with st.form("sp"):
            ec1,ec2=st.columns(2)
            name  =ec1.text_input("Full Name",   value=s.get("name","") or "")
            phone =ec2.text_input("Phone",        value=s.get("phone","") or "")
            skills=st.text_input("Skills (comma-separated)", value=s.get("skills","") or "")
            ec3,ec4=st.columns(2)
            exp   =ec3.number_input("Experience (yrs)", min_value=0, value=int(s.get("experience") or 0))
            loc   =ec4.text_input("Preferred Location", value=s.get("preferred_location","") or "")
            bio   =st.text_area("About Me",      value=s.get("bio","") or "")
            salary=st.text_input("Expected Salary", value=s.get("expected_salary","") or "")
            if st.form_submit_button("💾 Save Profile", use_container_width=True):
                sk_update(u["id"],name,phone,skills,exp,loc,bio,salary)
                st.session_state.user["name"]=name; st.success("✅ Profile saved!")


# ─────────────────────────────────────────────────────────────────────────────
# COMPANY DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────
def page_dash_company():
    u=st.session_state.user
    if not u or u["role"]!="company": go("login"); return

    with st.sidebar:
        st.markdown(f"### 🏢 {u['name']}")
        st.markdown(f"📧 {u['email']}")
        st.divider()
        section=st.radio("Navigate",
            ["📊 Overview","➕ Post a Job","📋 My Postings","👥 Applicants","🏢 Profile"],
            label_visibility="collapsed", key="co_sec")
        st.divider()
        if st.button("🚪 Logout", use_container_width=True, key="co_lo"): logout()

    t,a=co_stats(u["id"])
    st.markdown(f"""
    <div class="hero" style="padding:2rem;">
      <h1 style="font-size:1.8rem;">Employer Dashboard 🏢</h1>
      <p>Manage your postings — <b>{u['name']}</b></p>
    </div>""", unsafe_allow_html=True)

    if section=="📊 Overview":
        st.markdown(f"""
        <div class="stat-row">
          <div class="stat-card"><div class="n">{t}</div><div class="l">Jobs Posted</div></div>
          <div class="stat-card"><div class="n">{a}</div><div class="l">Applicants</div></div>
        </div>""", unsafe_allow_html=True)
        jobs=co_jobs(u["id"])[:5]
        if not jobs: st.info("No jobs posted yet. Use ➕ Post a Job.")
        else: st.table([{"Title":j["title"],"Location":j["location"] or "Remote","Posted":j["created_at"][:10],"Applicants":j["appl"]} for j in jobs])

    elif section=="➕ Post a Job":
        st.markdown("#### ➕ Post a New Job")
        st.markdown('<div class="box-ai">🤖 pac_model.pkl will classify your job automatically when seekers view it.</div>', unsafe_allow_html=True)
        with st.form("pj"):
            pj1,pj2=st.columns(2)
            title=pj1.text_input("Job Title *", placeholder="e.g. Software Engineer")
            jtype=pj2.selectbox("Job Type",["Full-time","Part-time","Remote","Internship","Contract"])
            pj3,pj4=st.columns(2)
            loc  =pj3.text_input("Location",    placeholder="Chennai, India")
            salary=pj4.text_input("Salary Range", placeholder="₹6L–₹12L per annum")
            exp  =st.number_input("Experience Required (years)", min_value=0)
            desc =st.text_area("Job Description *", height=130,
                                placeholder="Describe the role, responsibilities, and company culture…")
            req  =st.text_area("Requirements / Skills", height=90,
                                placeholder="e.g. Python, 3+ years experience, team player…")
            mob  =st.text_input("Contact Mobile", placeholder="9876543210")
            sub  =st.form_submit_button("📤 Post Job", use_container_width=True)
        if sub:
            errs=[]
            if not title.strip(): errs.append("Job title is required.")
            if not desc.strip():  errs.append("Job description is required.")
            if errs:
                for e in errs: st.error(e)
            else:
                job_post(u["id"],title.strip(),jtype,loc,salary,exp,desc,req,mob)
                st.success("✅ Job posted successfully!")
                st.rerun()

    elif section=="📋 My Postings":
        jobs=co_jobs(u["id"])
        if not jobs: st.info("No jobs posted yet.")
        else:
            for j in jobs:
                with st.container():
                    pjc1,pjc2,pjc3=st.columns([4,1.2,1])
                    pjc1.markdown(f"**{j['title']}**  \n"
                                   f"`{j['job_type']}` · {j['location'] or 'Remote'} · {j['appl']} applicants")
                    pjc2.caption(f"Posted {j['created_at'][:10]}")
                    if pjc3.button("🗑 Delete", key=f"del_{j['id']}", use_container_width=True):
                        job_delete(j["id"],u["id"]); st.success("Job deleted."); st.rerun()
                    st.markdown("<hr style='border:none;border-top:1px solid #e2ddd6;margin:.3rem 0;'>", unsafe_allow_html=True)

    elif section=="👥 Applicants":
        jobs=co_jobs(u["id"]); jmap={"All Jobs":None}
        for j in jobs: jmap[f"{j['title']} ({j['created_at'][:10]})"]=j["id"]
        chosen=st.selectbox("Filter by job posting", list(jmap.keys()), key="appl_filter")
        appl=co_applicants(u["id"],jmap[chosen])
        st.markdown(f"**{len(appl)} applicant(s)**")
        if not appl: st.info("No applicants yet.")
        else:
            st.table([{"Name":a["name"],"Email":a["email"],"Phone":a.get("phone","—"),
                        "Skills":a["skills"] or "—","Experience":f"{a['experience'] or 0} yrs",
                        "Job":a["job_title"],"Applied":a["applied_at"][:10]} for a in appl])

    elif section=="🏢 Profile":
        co=co_get(u["id"])
        st.markdown("#### 🏢 Company Profile")
        with st.form("cp"):
            cp1,cp2=st.columns(2)
            name    =cp1.text_input("Company Name",  value=co.get("name","") or "")
            industry=cp2.selectbox("Industry",["IT / Software","Finance","Healthcare","Manufacturing","E-commerce","Other"])
            cp3,cp4=st.columns(2)
            website =cp3.text_input("Website",       value=co.get("website","") or "")
            year    =cp4.number_input("Year Founded", min_value=1900, max_value=date.today().year, value=int(co.get("year_founded") or 2010))
            phone   =st.text_input("Phone",           value=co.get("phone","") or "")
            desc    =st.text_area("Company Description", value=co.get("description","") or "")
            if st.form_submit_button("💾 Save Profile", use_container_width=True):
                co_update(u["id"],name,industry,website,year,phone,desc)
                st.session_state.user["name"]=name; st.success("✅ Profile saved!")


# ─────────────────────────────────────────────────────────────────────────────
# ROUTER
# ─────────────────────────────────────────────────────────────────────────────
PAGE = st.session_state.get("page","home")
if st.session_state.get("sel") and PAGE!="jobs":
    st.session_state.page="jobs"; PAGE="jobs"
{
    "home":     page_home,
    "jobs":     page_jobs,
    "login":    page_login,
    "register": page_register,
    "dash_sk":  page_dash_seeker,
    "dash_co":  page_dash_company,
}.get(PAGE, page_home)()