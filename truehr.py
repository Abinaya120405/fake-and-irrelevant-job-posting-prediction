"""
TrueHire — Verified Job Portal
Single-file Streamlit app (frontend + SQLite backend)
Run with: streamlit run truehire_app.py
"""

import streamlit as st
import sqlite3
import hashlib
import os
import json
from datetime import datetime, date

# ─────────────────────────────────────────────
# CONFIG & PAGE SETUP
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="TrueHire — Verified Job Portal",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="collapsed",
)

DB_PATH = "truehire.db"

# ─────────────────────────────────────────────
# CUSTOM CSS  (mirrors original design tokens)
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
  --primary: #1a3c5e;
  --accent:  #e8734a;
  --bg:      #f9f7f4;
  --text:    #1a1a2e;
  --border:  #e5e7eb;
  --success: #16a34a;
  --danger:  #dc2626;
}

html, body, [class*="css"] {
  font-family: 'DM Sans', sans-serif;
  background-color: var(--bg) !important;
  color: var(--text);
}

/* Hide default Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.5rem !important; max-width: 1100px; }

/* ── Branded header ── */
.th-header {
  background: linear-gradient(135deg, #1a3c5e 60%, #e8734a 100%);
  border-radius: 16px; padding: 2.5rem 2rem 2rem;
  margin-bottom: 1.5rem; color: #fff;
}
.th-header h1 { font-family: 'Playfair Display', serif; font-size: 2.6rem; margin: 0; }
.th-header h1 span { color: #e8734a; }
.th-header p  { opacity: 0.85; margin-top: 0.4rem; font-size: 1.05rem; }

/* ── Stat cards ── */
.stat-row { display:flex; gap:1rem; margin-bottom:1.5rem; flex-wrap:wrap; }
.stat-card {
  background:#fff; border-radius:12px; padding:1.2rem 1.5rem;
  border:1px solid var(--border); flex:1; min-width:130px; text-align:center;
  box-shadow:0 2px 12px rgba(26,60,94,0.07);
}
.stat-card .num { font-size:2rem; font-weight:700; color:var(--primary); line-height:1; }
.stat-card .lbl { font-size:0.8rem; color:#6b7280; margin-top:0.3rem; }

/* ── Job cards ── */
.job-card {
  background:#fff; border-radius:12px; padding:1.2rem 1.4rem;
  border:1px solid var(--border); margin-bottom:0.8rem;
  box-shadow:0 2px 8px rgba(26,60,94,0.06); transition: box-shadow .2s;
}
.job-card:hover { box-shadow:0 6px 24px rgba(26,60,94,0.13); }
.job-card h3 { margin:0 0 0.2rem; color:var(--primary); font-size:1rem; }
.job-card .meta { color:#6b7280; font-size:0.83rem; }
.tag {
  display:inline-block; background:#f3f4f6; border:1px solid var(--border);
  border-radius:50px; padding:0.18rem 0.7rem; font-size:0.75rem;
  color:#374151; margin-right:0.3rem; margin-top:0.3rem;
}
.badge-verified {
  background:#e8f5e9; color:#16a34a; border-radius:50px;
  padding:0.15rem 0.6rem; font-size:0.72rem; font-weight:600;
}

/* ── Section headers ── */
.sec-title {
  font-family:'Playfair Display',serif; color:var(--primary);
  font-size:1.5rem; margin-bottom:0.2rem;
}
.sec-sub { color:#6b7280; font-size:0.9rem; margin-bottom:1.2rem; }

/* ── Buttons ── */
div.stButton > button {
  background: var(--accent) !important;
  color: #fff !important;
  border: none !important;
  border-radius: 8px !important;
  font-weight: 600 !important;
  transition: background .2s, transform .15s !important;
}
div.stButton > button:hover {
  background: #cf5a32 !important;
  transform: translateY(-1px);
}
.stTextInput > div > input,
.stSelectbox > div > div,
.stTextArea > div > textarea,
.stNumberInput > div > input,
.stDateInput > div > input {
  border-radius: 8px !important;
  border: 1.5px solid var(--border) !important;
  font-family: 'DM Sans', sans-serif !important;
}
.stTextInput > div > input:focus,
.stTextArea > div > textarea:focus {
  border-color: var(--accent) !important;
  box-shadow: 0 0 0 3px rgba(232,115,74,.12) !important;
}

/* ── Nav pills ── */
.nav-pill {
  display:inline-block; padding:0.4rem 1.1rem; border-radius:50px;
  background:#fff; border:1.5px solid var(--border); margin-right:0.4rem;
  cursor:pointer; font-size:0.88rem; font-weight:500; color:var(--primary);
  text-decoration:none; transition:all .2s;
}
.nav-pill.active {
  background:var(--accent); border-color:var(--accent); color:#fff;
}

/* ── Toast / alert boxes ── */
.info-box {
  background:#eff6ff; border-left:4px solid #3b82f6;
  border-radius:8px; padding:0.8rem 1rem; margin-bottom:1rem; font-size:0.9rem;
}
.success-box {
  background:#f0fdf4; border-left:4px solid #16a34a;
  border-radius:8px; padding:0.8rem 1rem; margin-bottom:1rem; font-size:0.9rem;
}
.error-box {
  background:#fef2f2; border-left:4px solid #dc2626;
  border-radius:8px; padding:0.8rem 1rem; margin-bottom:1rem; font-size:0.9rem;
}

/* ── Table styles ── */
table { width:100%; border-collapse:collapse; font-size:0.88rem; }
th { background:#f3f4f6; color:#374151; padding:0.6rem 0.8rem; text-align:left; font-weight:600; }
td { padding:0.6rem 0.8rem; border-bottom:1px solid #f3f4f6; color:#374151; }
tr:hover td { background:#fffbf8; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
  background: var(--primary) !important;
}
section[data-testid="stSidebar"] * { color: #dbeafe !important; }
section[data-testid="stSidebar"] .stRadio label { font-size:0.95rem !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# DATABASE LAYER
# ─────────────────────────────────────────────
def get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_conn()
    c = conn.cursor()
    c.executescript("""
    CREATE TABLE IF NOT EXISTS companies (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        email TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        phone TEXT,
        industry TEXT,
        website TEXT,
        year_founded INTEGER,
        description TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    CREATE TABLE IF NOT EXISTS seekers (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        email TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        phone TEXT,
        skills TEXT,
        experience INTEGER DEFAULT 0,
        preferred_location TEXT,
        bio TEXT,
        expected_salary TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    CREATE TABLE IF NOT EXISTS jobs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        company_id INTEGER NOT NULL,
        title TEXT NOT NULL,
        job_type TEXT DEFAULT 'Full-time',
        location TEXT,
        salary_range TEXT,
        experience_required INTEGER DEFAULT 0,
        deadline TEXT,
        description TEXT,
        requirements TEXT,
        contact_mobile TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(company_id) REFERENCES companies(id)
    );
    CREATE TABLE IF NOT EXISTS applications (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        job_id INTEGER NOT NULL,
        seeker_id INTEGER NOT NULL,
        status TEXT DEFAULT 'Under Review',
        applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(job_id, seeker_id),
        FOREIGN KEY(job_id) REFERENCES jobs(id),
        FOREIGN KEY(seeker_id) REFERENCES seekers(id)
    );
    """)
    conn.commit()
    conn.close()

init_db()

def hash_pw(pw: str) -> str:
    return hashlib.sha256(pw.encode()).hexdigest()

# ─── Company helpers ───
def register_company(name, email, pw, phone, industry, year, desc):
    conn = get_conn()
    try:
        conn.execute(
            "INSERT INTO companies(name,email,password_hash,phone,industry,year_founded,description) VALUES(?,?,?,?,?,?,?)",
            (name, email, hash_pw(pw), phone, industry, year, desc)
        )
        conn.commit()
        return True, "Company registered!"
    except sqlite3.IntegrityError:
        return False, "Email already registered."
    finally:
        conn.close()

def login_company(email, pw):
    conn = get_conn()
    row = conn.execute("SELECT * FROM companies WHERE email=? AND password_hash=?",
                       (email, hash_pw(pw))).fetchone()
    conn.close()
    return dict(row) if row else None

def get_company(company_id):
    conn = get_conn()
    row = conn.execute("SELECT * FROM companies WHERE id=?", (company_id,)).fetchone()
    conn.close()
    return dict(row) if row else {}

def update_company(company_id, name, industry, website, year, phone, desc):
    conn = get_conn()
    conn.execute(
        "UPDATE companies SET name=?,industry=?,website=?,year_founded=?,phone=?,description=? WHERE id=?",
        (name, industry, website, year, phone, desc, company_id)
    )
    conn.commit(); conn.close()

# ─── Seeker helpers ───
def register_seeker(name, email, pw, phone, skills, exp):
    conn = get_conn()
    try:
        conn.execute(
            "INSERT INTO seekers(name,email,password_hash,phone,skills,experience) VALUES(?,?,?,?,?,?)",
            (name, email, hash_pw(pw), phone, skills, exp)
        )
        conn.commit()
        return True, "Account created!"
    except sqlite3.IntegrityError:
        return False, "Email already registered."
    finally:
        conn.close()

def login_seeker(email, pw):
    conn = get_conn()
    row = conn.execute("SELECT * FROM seekers WHERE email=? AND password_hash=?",
                       (email, hash_pw(pw))).fetchone()
    conn.close()
    return dict(row) if row else None

def get_seeker(seeker_id):
    conn = get_conn()
    row = conn.execute("SELECT * FROM seekers WHERE id=?", (seeker_id,)).fetchone()
    conn.close()
    return dict(row) if row else {}

def update_seeker(seeker_id, name, phone, skills, exp, location, bio, salary):
    conn = get_conn()
    conn.execute(
        "UPDATE seekers SET name=?,phone=?,skills=?,experience=?,preferred_location=?,bio=?,expected_salary=? WHERE id=?",
        (name, phone, skills, exp, location, bio, salary, seeker_id)
    )
    conn.commit(); conn.close()

def profile_score(s):
    fields = [s.get('name'), s.get('phone'), s.get('skills'), s.get('bio'),
              s.get('preferred_location'), s.get('expected_salary')]
    filled = sum(1 for f in fields if f)
    return int(filled / len(fields) * 100)

# ─── Job helpers ───
def post_job(company_id, title, job_type, location, salary, exp, deadline, desc, req, mobile):
    conn = get_conn()
    conn.execute(
        """INSERT INTO jobs(company_id,title,job_type,location,salary_range,experience_required,
           deadline,description,requirements,contact_mobile)
           VALUES(?,?,?,?,?,?,?,?,?,?)""",
        (company_id, title, job_type, location, salary, exp,
         str(deadline) if deadline else None, desc, req, mobile)
    )
    conn.commit(); conn.close()

def get_jobs(q="", location="", job_type="", min_salary="", experience="", industry="", limit=100):
    conn = get_conn()
    sql = """
        SELECT j.*, c.name AS company_name, c.industry
        FROM jobs j JOIN companies c ON j.company_id=c.id
        WHERE 1=1
    """
    params = []
    if q:
        sql += " AND (j.title LIKE ? OR j.description LIKE ?)"
        params += [f"%{q}%", f"%{q}%"]
    if location:
        sql += " AND j.location LIKE ?"
        params.append(f"%{location}%")
    if job_type:
        sql += " AND j.job_type=?"
        params.append(job_type)
    if experience:
        sql += " AND j.experience_required <= ?"
        params.append(int(experience))
    if industry:
        sql += " AND c.industry=?"
        params.append(industry)
    sql += " ORDER BY j.created_at DESC LIMIT ?"
    params.append(limit)
    rows = conn.execute(sql, params).fetchall()
    conn.close()
    return [dict(r) for r in rows]

def get_job(job_id):
    conn = get_conn()
    row = conn.execute(
        "SELECT j.*, c.name AS company_name FROM jobs j JOIN companies c ON j.company_id=c.id WHERE j.id=?",
        (job_id,)
    ).fetchone()
    conn.close()
    return dict(row) if row else None

def get_company_jobs(company_id):
    conn = get_conn()
    rows = conn.execute(
        """SELECT j.*, COUNT(a.id) AS applicant_count
           FROM jobs j LEFT JOIN applications a ON a.job_id=j.id
           WHERE j.company_id=? GROUP BY j.id ORDER BY j.created_at DESC""",
        (company_id,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]

def delete_job(job_id, company_id):
    conn = get_conn()
    conn.execute("DELETE FROM applications WHERE job_id=?", (job_id,))
    conn.execute("DELETE FROM jobs WHERE id=? AND company_id=?", (job_id, company_id))
    conn.commit(); conn.close()

def get_applicants(company_id, job_id=None):
    conn = get_conn()
    sql = """
        SELECT s.name, s.email, s.skills, s.experience, a.applied_at, a.status, j.title AS job_title
        FROM applications a
        JOIN seekers s ON s.id=a.seeker_id
        JOIN jobs j ON j.id=a.job_id
        WHERE j.company_id=?
    """
    params = [company_id]
    if job_id:
        sql += " AND a.job_id=?"; params.append(job_id)
    sql += " ORDER BY a.applied_at DESC"
    rows = conn.execute(sql, params).fetchall()
    conn.close()
    return [dict(r) for r in rows]

def apply_job(job_id, seeker_id):
    conn = get_conn()
    try:
        conn.execute("INSERT INTO applications(job_id,seeker_id) VALUES(?,?)", (job_id, seeker_id))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def get_seeker_applications(seeker_id):
    conn = get_conn()
    rows = conn.execute(
        """SELECT a.*, j.title, j.location, c.name AS company_name
           FROM applications a
           JOIN jobs j ON j.id=a.job_id
           JOIN companies c ON c.id=j.company_id
           WHERE a.seeker_id=? ORDER BY a.applied_at DESC""",
        (seeker_id,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]

def company_dashboard_stats(company_id):
    conn = get_conn()
    total_jobs = conn.execute("SELECT COUNT(*) FROM jobs WHERE company_id=?", (company_id,)).fetchone()[0]
    total_apps = conn.execute(
        "SELECT COUNT(*) FROM applications a JOIN jobs j ON j.id=a.job_id WHERE j.company_id=?",
        (company_id,)
    ).fetchone()[0]
    active_jobs = total_jobs  # all jobs treated as active
    conn.close()
    return total_jobs, total_apps, active_jobs

def seeker_dashboard_stats(seeker_id):
    conn = get_conn()
    total = conn.execute("SELECT COUNT(*) FROM applications WHERE seeker_id=?", (seeker_id,)).fetchone()[0]
    pending = conn.execute(
        "SELECT COUNT(*) FROM applications WHERE seeker_id=? AND status='Under Review'",
        (seeker_id,)
    ).fetchone()[0]
    conn.close()
    return total, pending


# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────
def ss():
    return st.session_state

if "page" not in st.session_state:
    st.session_state.page = "home"
if "user" not in st.session_state:
    st.session_state.user = None        # dict with id, name, role
if "selected_job" not in st.session_state:
    st.session_state.selected_job = None

def go(page):
    st.session_state.page = page
    st.session_state.selected_job = None
    st.rerun()

def logout():
    st.session_state.user = None
    go("home")


# ─────────────────────────────────────────────
# GLOBAL NAVBAR
# ─────────────────────────────────────────────
def render_navbar():
    user = st.session_state.user
    cols = st.columns([3, 1, 1, 1, 1])
    with cols[0]:
        if st.button("🏢 TrueHire", key="nav_home"):
            go("home")
    with cols[1]:
        if st.button("Browse Jobs", key="nav_jobs"):
            go("jobs")
    if user:
        with cols[2]:
            dash = "dashboard_seeker" if user["role"] == "seeker" else "dashboard_company"
            if st.button(f"👤 {user['name'].split()[0]}", key="nav_dash"):
                go(dash)
        with cols[3]:
            if st.button("Logout", key="nav_logout"):
                logout()
    else:
        with cols[2]:
            if st.button("Login", key="nav_login"):
                go("login")
        with cols[3]:
            if st.button("Sign Up", key="nav_register"):
                go("register")
    st.divider()


# ─────────────────────────────────────────────
# HOME PAGE
# ─────────────────────────────────────────────
def page_home():
    render_navbar()

    st.markdown("""
    <div class="th-header">
      <h1>True<span>Hire</span></h1>
      <p>Find Verified Jobs. Hire Real Talent. No Scams, No Middlemen.</p>
    </div>
    """, unsafe_allow_html=True)

    # Hero search
    c1, c2, c3 = st.columns([3, 2, 1])
    with c1:
        q = st.text_input("Job title or keyword", placeholder="e.g. Python Developer", label_visibility="collapsed")
    with c2:
        loc = st.text_input("Location", placeholder="e.g. Bangalore", label_visibility="collapsed")
    with c3:
        if st.button("🔍 Search", use_container_width=True):
            st.session_state["search_q"]   = q
            st.session_state["search_loc"] = loc
            go("jobs")

    # Stats
    st.markdown("""
    <div class="stat-row" style="margin-top:1rem;">
      <div class="stat-card"><div class="num">12,000+</div><div class="lbl">Verified Jobs</div></div>
      <div class="stat-card"><div class="num">3,200+</div><div class="lbl">Companies</div></div>
      <div class="stat-card"><div class="num">98%</div><div class="lbl">Legit Listings</div></div>
      <div class="stat-card"><div class="num">1L+</div><div class="lbl">Job Seekers</div></div>
    </div>
    """, unsafe_allow_html=True)

    # Featured jobs
    st.markdown('<p class="sec-title" style="margin-top:1rem;">Featured Jobs</p>', unsafe_allow_html=True)
    jobs = get_jobs(limit=6)
    if not jobs:
        st.info("No jobs yet. Companies can sign up and post jobs!")
    else:
        cols = st.columns(2)
        for i, j in enumerate(jobs):
            with cols[i % 2]:
                st.markdown(f"""
                <div class="job-card">
                  <div style="font-size:1.5rem;margin-bottom:0.4rem;">🏢</div>
                  <h3>{j['title']}</h3>
                  <div class="meta">{j['company_name']} · {j['location'] or 'Remote'}</div>
                  <div style="margin-top:0.5rem;">
                    <span class="tag">{j['salary_range'] or 'Negotiable'}</span>
                    <span class="tag">{j['job_type'] or 'Full-time'}</span>
                    <span class="badge-verified">✔ Verified</span>
                  </div>
                </div>
                """, unsafe_allow_html=True)
                if st.button("View & Apply", key=f"home_job_{j['id']}"):
                    st.session_state.selected_job = j['id']
                    go("jobs")

    # For Companies CTA
    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        <div style="background:#1a3c5e;border-radius:14px;padding:2rem;color:#fff;">
          <h2 style="font-family:'Playfair Display',serif;color:#fff;margin-bottom:0.5rem;">Hiring? Post a Job</h2>
          <p style="color:#93c5fd;">Reach thousands of verified candidates.</p>
          <ul style="color:#dbeafe;font-size:0.9rem;padding-left:1.2rem;">
            <li>Free job posting</li>
            <li>Direct applicant contacts</li>
            <li>Manage all applications in one place</li>
          </ul>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Post a Job →", key="cta_company"):
            go("register")
    with c2:
        st.markdown("""
        <div style="background:#fff3ed;border:1.5px solid #e8734a;border-radius:14px;padding:2rem;">
          <h2 style="font-family:'Playfair Display',serif;color:#1a3c5e;margin-bottom:0.5rem;">Looking for Work?</h2>
          <p style="color:#6b7280;">Browse thousands of verified openings across India.</p>
          <ul style="color:#374151;font-size:0.9rem;padding-left:1.2rem;">
            <li>No registration fee</li>
            <li>Transparent job details</li>
            <li>One-click apply</li>
          </ul>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Find Jobs →", key="cta_seeker"):
            go("jobs")


# ─────────────────────────────────────────────
# JOBS PAGE
# ─────────────────────────────────────────────
def page_jobs():
    render_navbar()
    st.markdown('<p class="sec-title">Browse Jobs</p>', unsafe_allow_html=True)
    st.markdown('<p class="sec-sub">Explore verified openings across India</p>', unsafe_allow_html=True)

    # If a specific job was pre-selected (from home)
    if st.session_state.selected_job:
        job = get_job(st.session_state.selected_job)
        if job:
            render_job_detail(job)
            if st.button("← Back to all jobs"):
                st.session_state.selected_job = None
                st.rerun()
            return

    # Search & Filters
    with st.expander("🔍 Search & Filter", expanded=True):
        c1, c2, c3 = st.columns(3)
        q       = c1.text_input("Keyword", value=st.session_state.get("search_q", ""), placeholder="Title / skill")
        loc     = c2.text_input("Location", value=st.session_state.get("search_loc", ""), placeholder="City")
        jtype   = c3.selectbox("Job Type", ["", "Full-time", "Part-time", "Remote", "Internship", "Contract"])
        c4, c5  = st.columns(2)
        max_exp = c4.number_input("Max Experience (yrs)", min_value=0, max_value=30, value=30)
        industry= c5.selectbox("Industry", ["", "IT / Software", "Finance", "Healthcare", "Manufacturing", "Education", "E-commerce", "Other"])
        if st.button("Search Jobs"):
            st.session_state["search_q"]   = q
            st.session_state["search_loc"] = loc

    jobs = get_jobs(
        q=q, location=loc, job_type=jtype,
        experience=max_exp if max_exp < 30 else "",
        industry=industry
    )

    if not jobs:
        st.info("No jobs match your search. Try different keywords.")
        return

    st.markdown(f"**{len(jobs)} job(s) found**")

    for j in jobs:
        with st.container():
            st.markdown(f"""
            <div class="job-card">
              <div style="display:flex;justify-content:space-between;align-items:flex-start;">
                <div>
                  <h3>{j['title']}</h3>
                  <div class="meta">{j['company_name']} · {j['location'] or 'Remote'}</div>
                  <div style="margin-top:0.5rem;">
                    <span class="tag">{j['job_type'] or 'Full-time'}</span>
                    <span class="tag">{j['salary_range'] or 'Negotiable'}</span>
                    <span class="tag">{j['experience_required'] or 0} yrs exp</span>
                    <span class="badge-verified">✔ Verified</span>
                  </div>
                </div>
              </div>
            </div>
            """, unsafe_allow_html=True)
            bc1, bc2 = st.columns([4, 1])
            with bc2:
                if st.button("View & Apply", key=f"jlist_{j['id']}"):
                    st.session_state.selected_job = j['id']
                    st.rerun()
        st.markdown("")


def render_job_detail(j):
    st.markdown(f"""
    <div class="job-card" style="padding:2rem;">
      <h2 style="font-family:'Playfair Display',serif;color:#1a3c5e;">{j['title']}</h2>
      <p style="color:#6b7280;">{j['company_name']} · {j['location'] or 'Remote'}</p>
      <div style="margin:0.8rem 0;">
        <span class="tag">{j['salary_range'] or 'Negotiable'}</span>
        <span class="tag">{j['job_type'] or 'Full-time'}</span>
        <span class="tag">{j['experience_required'] or 0} yrs exp</span>
      </div>
      <h4 style="margin-top:1.2rem;color:#1a3c5e;">Job Description</h4>
      <p style="color:#374151;line-height:1.7;">{j['description'] or 'No description provided.'}</p>
      <h4 style="margin-top:1rem;color:#1a3c5e;">Requirements</h4>
      <p style="color:#374151;line-height:1.7;">{j['requirements'] or 'See description.'}</p>
      {"<p style='margin-top:0.8rem;'><b>Contact:</b> " + j['contact_mobile'] + "</p>" if j.get('contact_mobile') else ""}
      {"<p><b>Deadline:</b> " + str(j['deadline']) + "</p>" if j.get('deadline') else ""}
    </div>
    """, unsafe_allow_html=True)

    user = st.session_state.user
    if user and user["role"] == "seeker":
        if st.button("✅ Apply Now", use_container_width=True, key="apply_detail"):
            ok = apply_job(j['id'], user["id"])
            if ok:
                st.success("🎉 Application submitted!")
            else:
                st.warning("You've already applied for this job.")
    elif not user:
        st.markdown('<div class="info-box">Please <b>login as a Job Seeker</b> to apply.</div>', unsafe_allow_html=True)
        if st.button("Login to Apply"):
            go("login")


# ─────────────────────────────────────────────
# LOGIN PAGE
# ─────────────────────────────────────────────
def page_login():
    render_navbar()
    st.markdown('<p class="sec-title">Sign In to TrueHire</p>', unsafe_allow_html=True)

    role = st.radio("I am a:", ["Job Seeker", "Company / Employer"], horizontal=True)
    st.divider()

    with st.form("login_form"):
        email = st.text_input("Email")
        pw    = st.text_input("Password", type="password")
        sub   = st.form_submit_button("Login", use_container_width=True)

    if sub:
        if not email or not pw:
            st.error("Please enter email and password.")
            return
        if role == "Job Seeker":
            user = login_seeker(email, pw)
            if user:
                st.session_state.user = {"id": user["id"], "name": user["name"],
                                          "email": user["email"], "role": "seeker"}
                go("dashboard_seeker")
            else:
                st.error("Invalid credentials.")
        else:
            user = login_company(email, pw)
            if user:
                st.session_state.user = {"id": user["id"], "name": user["name"],
                                          "email": user["email"], "role": "company"}
                go("dashboard_company")
            else:
                st.error("Invalid credentials.")

    st.markdown("Don't have an account?")
    if st.button("Create Account"):
        go("register")


# ─────────────────────────────────────────────
# REGISTER PAGE
# ─────────────────────────────────────────────
def page_register():
    render_navbar()
    st.markdown('<p class="sec-title">Create an Account</p>', unsafe_allow_html=True)

    role = st.radio("Register as:", ["Job Seeker", "Company / Employer"], horizontal=True)
    st.divider()

    if role == "Job Seeker":
        with st.form("reg_seeker"):
            c1, c2 = st.columns(2)
            name   = c1.text_input("Full Name *")
            email  = c2.text_input("Email *")
            c3, c4 = st.columns(2)
            phone  = c3.text_input("Phone")
            pw     = c4.text_input("Password *", type="password")
            skills = st.text_input("Skills (comma-separated)", placeholder="Python, React, SQL")
            exp    = st.number_input("Experience (years)", min_value=0, max_value=50)
            sub    = st.form_submit_button("Create Account", use_container_width=True)
        if sub:
            if not name or not email or not pw:
                st.error("Name, email and password are required.")
            else:
                ok, msg = register_seeker(name, email, pw, phone, skills, exp)
                if ok:
                    st.success(msg + " Please login.")
                    go("login")
                else:
                    st.error(msg)
    else:
        with st.form("reg_company"):
            c1, c2 = st.columns(2)
            name    = c1.text_input("Company Name *")
            email   = c2.text_input("Work Email *")
            c3, c4  = st.columns(2)
            phone   = c3.text_input("Phone")
            pw      = c4.text_input("Password *", type="password")
            industry= st.selectbox("Industry", ["IT / Software", "Finance", "Healthcare",
                                                 "Manufacturing", "Education", "E-commerce", "Other"])
            c5, c6  = st.columns(2)
            year    = c5.number_input("Year Founded", min_value=1900, max_value=date.today().year, value=2010)
            _       = c6.empty()
            desc    = st.text_area("Company Description")
            sub     = st.form_submit_button("Create Account", use_container_width=True)
        if sub:
            if not name or not email or not pw:
                st.error("Name, email and password are required.")
            else:
                ok, msg = register_company(name, email, pw, phone, industry, year, desc)
                if ok:
                    st.success(msg + " Please login.")
                    go("login")
                else:
                    st.error(msg)

    st.markdown("Already have an account?")
    if st.button("Login"):
        go("login")


# ─────────────────────────────────────────────
# SEEKER DASHBOARD
# ─────────────────────────────────────────────
def page_dashboard_seeker():
    user = st.session_state.user
    if not user or user["role"] != "seeker":
        go("login")
        return

    with st.sidebar:
        st.markdown(f"### 👤 {user['name']}")
        st.markdown(f"_{user['email']}_")
        st.divider()
        section = st.radio("Navigate", ["📊 Overview", "📋 My Applications", "👤 Edit Profile", "🔍 Browse Jobs"],
                           label_visibility="collapsed")
        st.divider()
        if st.button("🚪 Logout"):
            logout()

    # Header
    st.markdown(f"""
    <div class="th-header">
      <h1>True<span>Hire</span> Dashboard</h1>
      <p>Welcome back, {user['name']}!</p>
    </div>
    """, unsafe_allow_html=True)

    s = get_seeker(user["id"])
    total_apps, pending = seeker_dashboard_stats(user["id"])
    score = profile_score(s)

    if section == "📊 Overview":
        st.markdown(f"""
        <div class="stat-row">
          <div class="stat-card"><div class="num">{total_apps}</div><div class="lbl">Applications Sent</div></div>
          <div class="stat-card"><div class="num">{pending}</div><div class="lbl">Under Review</div></div>
          <div class="stat-card"><div class="num">{score}%</div><div class="lbl">Profile Complete</div></div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("#### Recent Applications")
        apps = get_seeker_applications(user["id"])[:5]
        if not apps:
            st.info("No applications yet. Browse jobs and apply!")
        else:
            rows = [{"Job": a["title"], "Company": a["company_name"],
                     "Applied": a["applied_at"][:10], "Status": a["status"]} for a in apps]
            st.table(rows)

    elif section == "📋 My Applications":
        st.markdown("#### All Applications")
        apps = get_seeker_applications(user["id"])
        if not apps:
            st.info("No applications yet.")
        else:
            rows = [{"Job": a["title"], "Company": a["company_name"],
                     "Location": a["location"] or "Remote",
                     "Applied": a["applied_at"][:10], "Status": a["status"]} for a in apps]
            st.table(rows)

    elif section == "👤 Edit Profile":
        st.markdown("#### Edit Profile")
        with st.form("seeker_profile"):
            c1, c2 = st.columns(2)
            name    = c1.text_input("Full Name", value=s.get("name", ""))
            phone   = c2.text_input("Phone", value=s.get("phone", "") or "")
            skills  = st.text_input("Skills (comma-separated)", value=s.get("skills", "") or "")
            c3, c4  = st.columns(2)
            exp     = c3.number_input("Experience (yrs)", min_value=0, value=int(s.get("experience") or 0))
            loc     = c4.text_input("Preferred Location", value=s.get("preferred_location", "") or "")
            bio     = st.text_area("About Me", value=s.get("bio", "") or "")
            salary  = st.text_input("Expected Salary (LPA)", value=s.get("expected_salary", "") or "")
            if st.form_submit_button("Save Profile", use_container_width=True):
                update_seeker(user["id"], name, phone, skills, exp, loc, bio, salary)
                st.session_state.user["name"] = name
                st.success("Profile saved!")

    elif section == "🔍 Browse Jobs":
        go("jobs")


# ─────────────────────────────────────────────
# COMPANY DASHBOARD
# ─────────────────────────────────────────────
def page_dashboard_company():
    user = st.session_state.user
    if not user or user["role"] != "company":
        go("login")
        return

    with st.sidebar:
        st.markdown(f"### 🏢 {user['name']}")
        st.markdown(f"_{user['email']}_")
        st.divider()
        section = st.radio("Navigate",
                           ["📊 Overview", "➕ Post a Job", "📋 My Job Posts",
                            "👥 Applicants", "🏢 Company Profile"],
                           label_visibility="collapsed")
        st.divider()
        if st.button("🚪 Logout"):
            logout()

    st.markdown(f"""
    <div class="th-header">
      <h1>True<span>Hire</span> — Employer</h1>
      <p>Welcome, {user['name']}!</p>
    </div>
    """, unsafe_allow_html=True)

    total_jobs, total_apps, active_jobs = company_dashboard_stats(user["id"])

    if section == "📊 Overview":
        st.markdown(f"""
        <div class="stat-row">
          <div class="stat-card"><div class="num">{total_jobs}</div><div class="lbl">Jobs Posted</div></div>
          <div class="stat-card"><div class="num">{total_apps}</div><div class="lbl">Total Applicants</div></div>
          <div class="stat-card"><div class="num">{active_jobs}</div><div class="lbl">Active Listings</div></div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("#### Recent Job Posts")
        jobs = get_company_jobs(user["id"])[:5]
        if not jobs:
            st.info("No jobs posted yet. Use 'Post a Job' to get started.")
        else:
            rows = [{"Title": j["title"], "Location": j["location"] or "Remote",
                     "Posted": j["created_at"][:10], "Applicants": j["applicant_count"]} for j in jobs]
            st.table(rows)

    elif section == "➕ Post a Job":
        st.markdown("#### Post a New Job")
        with st.form("post_job_form"):
            c1, c2 = st.columns(2)
            title   = c1.text_input("Job Title *")
            jtype   = c2.selectbox("Job Type", ["Full-time", "Part-time", "Remote", "Internship", "Contract"])
            c3, c4  = st.columns(2)
            loc     = c3.text_input("Location", placeholder="Bangalore, India")
            salary  = c4.text_input("Salary Range (LPA)", placeholder="8–15 LPA")
            c5, c6  = st.columns(2)
            exp     = c5.number_input("Experience Required (yrs)", min_value=0)
            deadline= c6.date_input("Application Deadline", value=None)
            desc    = st.text_area("Job Description *", height=140)
            req     = st.text_area("Requirements / Skills", height=100)
            mobile  = st.text_input("Contact Mobile")
            sub     = st.form_submit_button("Post Job", use_container_width=True)
        if sub:
            if not title or not desc:
                st.error("Title and description are required.")
            else:
                post_job(user["id"], title, jtype, loc, salary, exp, deadline, desc, req, mobile)
                st.success("✅ Job posted successfully!")

    elif section == "📋 My Job Posts":
        st.markdown("#### My Job Postings")
        jobs = get_company_jobs(user["id"])
        if not jobs:
            st.info("No jobs posted yet.")
        else:
            for j in jobs:
                c1, c2, c3 = st.columns([4, 1, 1])
                c1.markdown(f"**{j['title']}** — {j['location'] or 'Remote'}  \n"
                            f"<span class='tag'>{j['job_type']}</span> "
                            f"<span class='tag'>{j['applicant_count']} applicants</span>",
                            unsafe_allow_html=True)
                c2.caption(j["created_at"][:10])
                if c3.button("🗑 Delete", key=f"del_{j['id']}"):
                    delete_job(j["id"], user["id"])
                    st.success("Deleted."); st.rerun()

    elif section == "👥 Applicants":
        st.markdown("#### Applicants")
        jobs = get_company_jobs(user["id"])
        job_map = {"All Jobs": None}
        for j in jobs:
            job_map[j["title"]] = j["id"]
        chosen = st.selectbox("Filter by job", list(job_map.keys()))
        job_id = job_map[chosen]
        applicants = get_applicants(user["id"], job_id)
        if not applicants:
            st.info("No applicants yet.")
        else:
            rows = [{"Name": a["name"], "Email": a["email"],
                     "Skills": a["skills"] or "—", "Experience": f"{a['experience'] or 0} yrs",
                     "Job": a["job_title"], "Applied": a["applied_at"][:10]} for a in applicants]
            st.table(rows)

    elif section == "🏢 Company Profile":
        st.markdown("#### Company Profile")
        co = get_company(user["id"])
        with st.form("company_profile"):
            c1, c2 = st.columns(2)
            name    = c1.text_input("Company Name", value=co.get("name", ""))
            industry= c2.selectbox("Industry",
                                    ["IT / Software", "Finance", "Healthcare",
                                     "Manufacturing", "E-commerce", "Other"],
                                    index=0)
            c3, c4  = st.columns(2)
            website = c3.text_input("Website", value=co.get("website", "") or "")
            year    = c4.number_input("Year Founded", min_value=1900,
                                      max_value=date.today().year,
                                      value=int(co.get("year_founded") or 2010))
            phone   = st.text_input("Phone", value=co.get("phone", "") or "")
            desc    = st.text_area("Company Description", value=co.get("description", "") or "")
            if st.form_submit_button("Save Profile", use_container_width=True):
                update_company(user["id"], name, industry, website, year, phone, desc)
                st.session_state.user["name"] = name
                st.success("Profile saved!")


# ─────────────────────────────────────────────
# ROUTER
# ─────────────────────────────────────────────
PAGE_MAP = {
    "home":               page_home,
    "jobs":               page_jobs,
    "login":              page_login,
    "register":           page_register,
    "dashboard_seeker":   page_dashboard_seeker,
    "dashboard_company":  page_dashboard_company,
}

current = st.session_state.get("page", "home")
PAGE_MAP.get(current, page_home)()