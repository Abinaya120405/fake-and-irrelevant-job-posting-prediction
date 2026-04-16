"""
TrueHire — Verified Job Portal
Single-file Streamlit app: frontend + SQLite backend + TF-IDF search + PAC model
Run: pip install streamlit scikit-learn pandas numpy
     streamlit run truehire_app.py
"""

import streamlit as st
import sqlite3
import hashlib
from datetime import date
import re

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import Pipeline

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
# SYNTHETIC DATASET  (realistic Indian job market — replaces external CSV)
# columns: title, company, location, job_type, salary, exp, industry, skills, description
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
    ("Equity Research Analyst","Kotak Securities","Mumbai","Full-time","12-20 LPA",3,"Finance","Financial Modeling,Bloomberg,Excel,Sectoral Analysis,Python","Publish in-depth research reports on listed companies. Initiate coverage, track quarterly results, and provide buy/sell recommendations."),
    ("Doctor - General Physician","Apollo Hospitals","Chennai","Full-time","15-25 LPA",5,"Healthcare","Clinical Medicine,Patient Care,Medical Records,EMR,Diagnostics","Provide primary care to outpatients and inpatients. Diagnose and treat acute and chronic conditions, coordinate specialist referrals, and maintain medical records."),
    ("Nurse - ICU","Fortis Healthcare","Delhi","Full-time","5-8 LPA",2,"Healthcare","Patient Care,ICU,Ventilator Management,IV Therapy,BLS","Deliver high-quality nursing care in the ICU setting. Monitor critically ill patients, administer medications, and collaborate with the multidisciplinary team."),
    ("Medical Coder","Manipal Hospitals","Bangalore","Full-time","4-7 LPA",1,"Healthcare","ICD-10,CPT,Medical Terminology,CPC,Revenue Cycle","Assign accurate diagnostic and procedural codes to medical records. Ensure coding compliance and support billing operations."),
    ("Pharmacist","Sun Pharma","Mumbai","Full-time","5-9 LPA",2,"Healthcare","Pharmaceutical Knowledge,Drug Dispensing,Pharmacovigilance,QA,GMP","Dispense medications, counsel patients on drug usage, and maintain pharmacy inventory. Support clinical teams with medication therapy management."),
    ("Operations Research Analyst","Delhivery","Gurgaon","Full-time","10-16 LPA",3,"IT / Software","Python,Linear Programming,Simulation,SQL,OR-Tools","Model and optimize logistics and delivery network operations. Use operations research techniques to improve routing efficiency and reduce costs."),
    ("Python Developer","Zoho","Chennai","Full-time","8-14 LPA",2,"IT / Software","Python,Django,REST API,PostgreSQL,Redis","Develop backend services using Django. Write clean, testable code, design REST APIs, and optimize database queries for high-traffic applications."),
    ("Android Developer","PhonePe","Bangalore","Full-time","12-20 LPA",3,"IT / Software","Android,Kotlin,Jetpack Compose,MVVM,Firebase","Build and maintain Android applications used by millions. Implement new features, improve app performance, and collaborate closely with product and QA teams."),
    ("iOS Developer","Razorpay","Bangalore","Full-time","12-20 LPA",3,"IT / Software","Swift,Xcode,UIKit,SwiftUI,CoreData","Develop high-quality iOS applications. Write maintainable Swift code, integrate SDKs, and ensure smooth app releases through App Store guidelines."),
    ("NLP Engineer","Sarvam AI","Bangalore","Full-time","20-35 LPA",4,"IT / Software","NLP,Transformers,Python,HuggingFace,LLMs","Research and build NLP models for Indian languages. Fine-tune LLMs, build text classification and NER pipelines, and deploy models in production."),
    ("Data Engineer","Swiggy","Bangalore","Full-time","14-22 LPA",3,"IT / Software","Apache Spark,Kafka,Airflow,Python,AWS","Design and maintain large-scale data pipelines. Build real-time and batch processing workflows, ensure data quality, and optimize storage costs."),
    ("QA Engineer","Freshworks","Chennai","Full-time","7-12 LPA",2,"IT / Software","Selenium,TestNG,Python,API Testing,JIRA","Design and execute automated test suites. Identify defects early, maintain test documentation, and work closely with developers to improve product quality."),
    ("Scrum Master","Mphasis","Pune","Full-time","12-18 LPA",4,"IT / Software","Agile,Scrum,JIRA,Confluence,Stakeholder Management","Facilitate agile ceremonies, remove impediments, and coach teams on Scrum practices. Track sprint metrics and report progress to stakeholders."),
    ("Embedded Systems Engineer","Bosch India","Bangalore","Full-time","10-18 LPA",3,"Manufacturing","C,RTOS,CAN,Embedded Linux,AUTOSAR","Develop embedded software for automotive ECUs. Write low-level drivers, integrate RTOS, and perform hardware-software validation."),
    ("Manufacturing Engineer","Tata Motors","Pune","Full-time","8-14 LPA",3,"Manufacturing","AutoCAD,Lean Manufacturing,Six Sigma,PLC,FMEA","Optimize manufacturing processes for vehicle assembly. Apply lean principles, lead root-cause analysis, and manage production engineering projects."),
    ("Quality Control Engineer","Mahindra","Nashik","Full-time","7-12 LPA",2,"Manufacturing","Quality Control,Six Sigma,CMM,PPAP,APQP","Perform incoming, in-process, and final quality inspections. Maintain quality records, handle customer complaints, and drive corrective actions."),
    ("Industrial Designer","TVS Motor","Chennai","Full-time","7-13 LPA",2,"Manufacturing","SolidWorks,CATIA,Industrial Design,Rendering,Ergonomics","Design aesthetically appealing and functional products. Create 3D models, generate design proposals, and collaborate with engineering for design feasibility."),
    ("Content Writer","Byju's","Bangalore","Full-time","5-9 LPA",1,"Education","Content Writing,SEO,Curriculum Design,MS Word,Research","Create engaging educational content for K-12 students. Research topics, write scripts and study material, and optimize content for digital platforms."),
    ("Curriculum Developer","Unacademy","Bangalore","Full-time","7-12 LPA",3,"Education","Curriculum Design,Instructional Design,eLearning,LMS,Content Strategy","Design learning outcomes and course structures for online programs. Collaborate with subject matter experts and use instructional design models like ADDIE."),
    ("Teacher - Mathematics","CBSE School","Delhi","Full-time","4-8 LPA",2,"Education","Mathematics,Pedagogy,MS Office,Classroom Management,Assessment","Teach Mathematics to grades 9–12. Prepare lesson plans, conduct assessments, and provide academic support to students. CBSE curriculum expertise required."),
    ("HR Manager","Infosys BPM","Bangalore","Full-time","12-18 LPA",5,"IT / Software","HR Policies,Recruitment,HRIS,Employee Relations,Performance Management","Lead end-to-end HR operations for a 500+ employee business unit. Drive talent acquisition, manage performance cycles, and implement employee engagement programs."),
    ("Talent Acquisition Specialist","Wipro","Hyderabad","Full-time","7-12 LPA",2,"IT / Software","Recruitment,LinkedIn Recruiter,ATS,Stakeholder Management,Sourcing","Own full-cycle recruitment for technical roles. Source, screen, and close candidates while ensuring a great candidate experience and time-to-fill targets."),
    ("Digital Marketing Manager","Nykaa","Mumbai","Full-time","10-16 LPA",3,"E-commerce","SEO,SEM,Social Media,Google Analytics,Meta Ads","Drive online customer acquisition and retention. Manage paid campaigns, SEO strategy, and social media channels. Analyze performance and optimize ROI."),
    ("SEO Specialist","MakeMyTrip","Gurgaon","Full-time","6-10 LPA",2,"E-commerce","SEO,Google Search Console,Ahrefs,Content Strategy,HTML","Improve organic search rankings across key pages. Conduct keyword research, technical SEO audits, and build backlink strategies."),
    ("Graphic Designer","Canva India","Bangalore","Full-time","5-9 LPA",1,"IT / Software","Adobe Illustrator,Photoshop,Figma,Branding,Typography","Create compelling visual content for digital and print. Develop brand collaterals, social media creatives, and UI assets in line with brand guidelines."),
    ("Sales Manager","Salesforce India","Mumbai","Full-time","20-35 LPA",5,"IT / Software","CRM,B2B Sales,Negotiation,Salesforce,Pipeline Management","Drive enterprise software sales across assigned territory. Build relationships with C-suite stakeholders, manage the full sales cycle, and exceed quarterly targets."),
    ("Customer Success Manager","Freshdesk","Chennai","Full-time","8-14 LPA",3,"IT / Software","Customer Success,CRM,SaaS,Onboarding,Retention","Manage a portfolio of enterprise clients. Drive product adoption, reduce churn, conduct QBRs, and identify upsell opportunities."),
    ("Data Science Intern","Ola","Bangalore","Internship","20-30K/month",0,"IT / Software","Python,Pandas,Machine Learning,Statistics,Jupyter","Support data science projects across pricing and demand forecasting. Analyze datasets, build prototypes, and present findings to the team."),
    ("Software Engineer Intern","Google India","Hyderabad","Internship","60-80K/month",0,"IT / Software","Python,Algorithms,Data Structures,C++,Problem Solving","Work on real engineering projects under mentorship. Contribute to codebase, write tests, and present a project at the end of the internship."),
    ("Finance Intern","Morgan Stanley","Mumbai","Internship","40-60K/month",0,"Finance","Financial Modeling,Excel,PowerPoint,Research,Bloomberg","Assist analysts on financial research and client deliverables. Exposure to equity and fixed income markets."),
    ("Operations Manager","BigBasket","Bangalore","Full-time","12-18 LPA",5,"E-commerce","Operations,Vendor Management,Supply Chain,Excel,KPI Management","Oversee warehouse and last-mile delivery operations. Drive efficiency improvements, manage vendor relationships, and ensure SLA compliance."),
    ("Blockchain Developer","Polygon","Mumbai","Full-time","20-35 LPA",4,"IT / Software","Solidity,Ethereum,Web3.js,Smart Contracts,Python","Build and audit smart contracts on EVM-compatible blockchains. Develop DeFi protocol integrations and ensure contract security best practices."),
    ("AI Research Scientist","Microsoft India","Hyderabad","Full-time","40-70 LPA",6,"IT / Software","Deep Learning,Research,PyTorch,NLP,Computer Vision","Conduct original research in AI/ML. Publish papers, collaborate with product teams to transfer research to production, and mentor junior researchers."),
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

# ── Company DB ──
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

# ── Seeker DB ──
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

# ── Posted Jobs DB ──
def post_job(cid,title,jtype,loc,salary,exp,deadline,desc,req,mobile):
    conn=get_conn()
    conn.execute("INSERT INTO jobs(company_id,title,job_type,location,salary_range,experience_required,deadline,description,requirements,contact_mobile) VALUES(?,?,?,?,?,?,?,?,?,?)",(cid,title,jtype,loc,salary,exp,str(deadline) if deadline else None,desc,req,mobile))
    conn.commit(); conn.close()

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
    conn=get_conn(); conn.execute("DELETE FROM applications WHERE job_id=?",(jid,)); conn.execute("DELETE FROM jobs WHERE id=? AND company_id=?",(jid,cid)); conn.commit(); conn.close()

def get_applicants(cid,job_id=None):
    conn=get_conn()
    sql="SELECT s.name,s.email,s.skills,s.experience,a.applied_at,a.status,j.title AS job_title FROM applications a JOIN seekers s ON s.id=a.seeker_id JOIN jobs j ON j.id=a.job_id WHERE j.company_id=?"
    params=[cid]
    if job_id: sql+=" AND a.job_id=?"; params.append(job_id)
    rows=conn.execute(sql+" ORDER BY a.applied_at DESC",params).fetchall(); conn.close()
    return [dict(r) for r in rows]

# ── Dataset Applications DB ──
def apply_dataset_job(idx, sid):
    conn=get_conn()
    try:
        conn.execute("INSERT INTO dataset_applications(dataset_job_idx,seeker_id) VALUES(?,?)",(idx,sid))
        conn.commit(); return True
    except sqlite3.IntegrityError: return False
    finally: conn.close()

def already_applied_dataset(idx, sid):
    conn=get_conn(); row=conn.execute("SELECT 1 FROM dataset_applications WHERE dataset_job_idx=? AND seeker_id=?",(idx,sid)).fetchone()
    conn.close(); return bool(row)

def get_seeker_dataset_apps(sid):
    conn=get_conn(); rows=conn.execute("SELECT * FROM dataset_applications WHERE seeker_id=? ORDER BY applied_at DESC",(sid,)).fetchall()
    conn.close(); return [dict(r) for r in rows]

def seeker_dashboard_stats(sid):
    conn=get_conn()
    posted=conn.execute("SELECT COUNT(*) FROM applications WHERE seeker_id=?",(sid,)).fetchone()[0]
    ds=conn.execute("SELECT COUNT(*) FROM dataset_applications WHERE seeker_id=?",(sid,)).fetchone()[0]
    conn.close(); return posted+ds, ds

def company_dashboard_stats(cid):
    conn=get_conn()
    total=conn.execute("SELECT COUNT(*) FROM jobs WHERE company_id=?",(cid,)).fetchone()[0]
    apps=conn.execute("SELECT COUNT(*) FROM applications a JOIN jobs j ON j.id=a.job_id WHERE j.company_id=?",(cid,)).fetchone()[0]
    conn.close(); return total,apps,total


# ─────────────────────────────────────────────────────────────────────────────
# ML ENGINE  —  TF-IDF vectoriser  +  PAC classifier
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="🤖 Training AI job-matching engine…")
def build_ml_engine():
    cols = ["title","company","location","job_type","salary",
            "exp","industry","skills","description"]
    df = pd.DataFrame(RAW_JOBS, columns=cols)
    df["exp"] = pd.to_numeric(df["exp"], errors="coerce").fillna(0).astype(int)

    # Rich text corpus for TF-IDF
    df["corpus"] = (df["title"] + " " +
                    df["skills"].str.replace(",", " ") + " " +
                    df["description"])

    # 1️⃣ TF-IDF vectoriser (for cosine-similarity search & recommendations)
    tfidf = TfidfVectorizer(
        max_features=6000,
        ngram_range=(1, 2),
        stop_words="english",
        sublinear_tf=True,
    )
    tfidf_matrix = tfidf.fit_transform(df["corpus"])

    # 2️⃣ PAC pipeline (industry/category prediction from free text)
    le = LabelEncoder()
    y  = le.fit_transform(df["industry"])
    pac_pipe = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=4000, ngram_range=(1, 2),
                                   stop_words="english", sublinear_tf=True)),
        ("pac",   PassiveAggressiveClassifier(C=0.5, max_iter=1000,
                                               random_state=42, tol=1e-3)),
    ])
    pac_pipe.fit(df["corpus"], y)

    return df, tfidf, tfidf_matrix, pac_pipe, le


def tfidf_search(query: str, df, tfidf, tfidf_matrix):
    """Return full df sorted by cosine similarity to query."""
    if not query.strip():
        return df.copy().assign(score=1.0)
    qvec = tfidf.transform([query])
    sims = cosine_similarity(qvec, tfidf_matrix).flatten()
    return df.copy().assign(score=sims).sort_values("score", ascending=False)


def predict_category(text: str, pac_pipe, le) -> str:
    """Predict industry label from free text using trained PAC model."""
    if not text.strip():
        return "Unknown"
    return le.inverse_transform(pac_pipe.predict([text]))[0]


def recommend_jobs(seeker_text: str, df, tfidf, tfidf_matrix, top_n=5):
    """Top-N jobs ranked by cosine similarity to seeker's profile text."""
    return tfidf_search(seeker_text, df, tfidf, tfidf_matrix).head(top_n)


# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────
for _k, _v in [("page","home"),("user",None),("selected_job",None),
                ("search_q",""),("search_loc","")]:
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
# SHARED NAVBAR
# ─────────────────────────────────────────────────────────────────────────────
def render_navbar():
    user = st.session_state.user
    c0,c1,c2,c3,c4 = st.columns([2.5,1,1,1,1])
    c0.markdown('<span style="font-family:\'Playfair Display\',serif;font-size:1.4rem;font-weight:900;color:#1a3c5e;">True<span style=\'color:#e8734a\'>Hire</span></span>', unsafe_allow_html=True)
    if c1.button("🏠 Home",  key="nav_home"): go("home")
    if c2.button("💼 Jobs",  key="nav_jobs"): go("jobs")
    if user:
        dash = "dashboard_seeker" if user["role"]=="seeker" else "dashboard_company"
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
    df, tfidf, tfidf_matrix, pac_pipe, le = build_ml_engine()
    user = st.session_state.user

    st.markdown(f"""
    <div class="hero-banner">
      <h1>True<span>Hire</span></h1>
      <p>AI-powered job matching. {len(RAW_JOBS)}+ verified listings. Zero scams.</p>
      <div class="hero-stats">
        <div class="hero-stat"><div class="n">{len(RAW_JOBS)}+</div><div class="l">Verified Jobs</div></div>
        <div class="hero-stat"><div class="n">50+</div><div class="l">Top Companies</div></div>
        <div class="hero-stat"><div class="n">TF-IDF</div><div class="l">Smart Search</div></div>
        <div class="hero-stat"><div class="n">PAC</div><div class="l">AI Matching</div></div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Quick search bar
    c1,c2,c3 = st.columns([3,2,1])
    q   = c1.text_input("Search jobs", placeholder="e.g. Data Scientist, Python, AWS", label_visibility="collapsed")
    loc = c2.text_input("Location",    placeholder="City or Remote",                   label_visibility="collapsed")
    if c3.button("🔍 Search", use_container_width=True):
        st.session_state.search_q   = q
        st.session_state.search_loc = loc
        go("jobs")

    # AI personalised recommendations for logged-in seekers
    if user and user["role"] == "seeker":
        s = get_seeker(user["id"])
        profile_text = " ".join(filter(None,[s.get("skills",""),s.get("bio",""),s.get("preferred_location","")])).strip()
        if profile_text:
            predicted = predict_category(profile_text, pac_pipe, le)
            recs = recommend_jobs(profile_text, df, tfidf, tfidf_matrix, top_n=3)
            st.markdown(f"""
            <div class="ai-box">
              🤖 <b>AI Career Insight (PAC Model):</b> Your profile best fits
              <b>{predicted}</b> roles. Here are your top personalised matches:
            </div>
            """, unsafe_allow_html=True)
            cols = st.columns(3)
            for i, (orig_idx, row) in enumerate(recs.iterrows()):
                pct = max(5, min(99, int(row["score"]*100)))
                with cols[i]:
                    st.markdown(f"""
                    <div class="job-card">
                      <h3>{row['title']}</h3>
                      <div class="company">{row['company']} · {row['location']}</div>
                      <span class="tag-accent">{row['salary']}</span>
                      <span class="badge-ai">🤖 {pct}% match</span>
                      <div class="match-bar-wrap"><div class="match-bar" style="width:{pct}%;"></div></div>
                    </div>
                    """, unsafe_allow_html=True)
                    if st.button("View & Apply", key=f"home_rec_{i}_{orig_idx}"):
                        st.session_state.selected_job = ("dataset", int(orig_idx))
                        go("jobs")
            st.divider()

    # Featured jobs grid
    st.markdown('<p class="sec-title">Featured Opportunities</p>', unsafe_allow_html=True)
    st.markdown('<p class="sec-sub">Hand-picked verified roles from India\'s leading companies</p>', unsafe_allow_html=True)
    featured = df.sample(min(6, len(df)), random_state=7)
    cols = st.columns(2)
    for i, (idx, row) in enumerate(featured.iterrows()):
        with cols[i%2]:
            st.markdown(f"""
            <div class="job-card">
              <h3>{row['title']}</h3>
              <div class="company">{row['company']} · {row['location']}</div>
              <span class="tag">{row['salary']}</span>
              <span class="tag">{row['job_type']}</span>
              <span class="tag">{row['industry']}</span>
              <span class="badge-verified">✔ Verified</span>
            </div>
            """, unsafe_allow_html=True)
            if st.button("View Details", key=f"home_feat_{idx}"):
                st.session_state.selected_job = ("dataset", int(idx))
                go("jobs")

    # CTA section
    st.divider()
    ca,cb = st.columns(2)
    with ca:
        st.markdown("""<div style="background:#1a3c5e;border-radius:14px;padding:1.8rem;color:#fff;"><h3 style="font-family:'Playfair Display',serif;color:#fff;margin-bottom:0.5rem;">🏢 Hiring Talent?</h3><p style="color:#93c5fd;margin-bottom:1rem;">Post jobs & reach thousands of verified candidates for free.</p></div>""", unsafe_allow_html=True)
        if st.button("Post a Job →", key="cta_co"): go("register")
    with cb:
        st.markdown("""<div style="background:#fff5f0;border:1.5px solid #e8734a;border-radius:14px;padding:1.8rem;"><h3 style="font-family:'Playfair Display',serif;color:#1a3c5e;margin-bottom:0.5rem;">🔍 Looking for Work?</h3><p style="color:#6b7280;margin-bottom:1rem;">Browse AI-matched openings. One-click apply. No middlemen.</p></div>""", unsafe_allow_html=True)
        if st.button("Browse All Jobs →", key="cta_sk"): go("jobs")


# ─────────────────────────────────────────────────────────────────────────────
# JOBS PAGE
# ─────────────────────────────────────────────────────────────────────────────
def page_jobs():
    render_navbar()
    df, tfidf, tfidf_matrix, pac_pipe, le = build_ml_engine()

    # ── Job detail view ──
    if st.session_state.selected_job:
        src, idx = st.session_state.selected_job
        if src == "dataset":
            render_dataset_job_detail(df.iloc[idx], idx)
        else:
            job = get_posted_job(idx)
            if job: render_posted_job_detail(job)
        if st.button("← Back to all jobs"):
            st.session_state.selected_job = None
            st.rerun()
        return

    st.markdown('<p class="sec-title">Browse Jobs</p>', unsafe_allow_html=True)

    # Filters
    with st.expander("🔍 Search & Filter", expanded=True):
        c1,c2,c3 = st.columns(3)
        q       = c1.text_input("Keyword / Skills", value=st.session_state.search_q,  placeholder="Python, React, Finance…")
        loc     = c2.text_input("Location",          value=st.session_state.search_loc, placeholder="Bangalore, Remote…")
        jtype   = c3.selectbox("Job Type", ["All","Full-time","Part-time","Remote","Internship","Contract"])
        c4,c5,c6 = st.columns(3)
        industry = c4.selectbox("Industry", ["All","IT / Software","Finance","Healthcare","Manufacturing","Education","E-commerce","Other"])
        max_exp  = c5.slider("Max Experience (yrs)", 0, 15, 15)
        sort_by  = c6.selectbox("Sort by", ["AI Relevance","Salary (High→Low)"])
        if st.button("🔍 Search Jobs", use_container_width=True):
            st.session_state.search_q   = q
            st.session_state.search_loc = loc

    # PAC category prediction on query
    if q.strip():
        cat = predict_category(q, pac_pipe, le)
        st.markdown(f'<div class="ai-box">🤖 <b>PAC Model</b> classifies your search into: <b>{cat}</b></div>', unsafe_allow_html=True)

    # TF-IDF search
    search_text = f"{q} {loc}".strip() or "software engineer"
    results = tfidf_search(search_text, df, tfidf, tfidf_matrix)

    # Apply filters on dataset results
    if jtype != "All":     results = results[results["job_type"] == jtype]
    if industry != "All":  results = results[results["industry"] == industry]
    if loc.strip():        results = results[results["location"].str.contains(loc, case=False, na=False)]
    results = results[results["exp"] <= max_exp]
    if sort_by == "Salary (High→Low)":
        def _sal(s):
            nums = re.findall(r'\d+', str(s)); return int(nums[-1]) if nums else 0
        results = results.copy(); results["_s"] = results["salary"].apply(_sal)
        results = results.sort_values("_s", ascending=False)

    # Also get company-posted jobs
    posted = get_posted_jobs(q=q, location=loc,
                              job_type="" if jtype=="All" else jtype,
                              industry="" if industry=="All" else industry,
                              experience=max_exp if max_exp < 15 else "")

    total = len(results) + len(posted)
    st.markdown(f"**{total} job(s) found** &nbsp; <span class='badge-ai'>🤖 Ranked by TF-IDF</span>", unsafe_allow_html=True)

    # Company-posted jobs (priority banner)
    if posted:
        st.markdown("#### 🏢 Company-Posted Jobs")
        for j in posted:
            st.markdown(f"""
            <div class="job-card">
              <h3>{j['title']}</h3>
              <div class="company">{j['company_name']} · {j['location'] or 'Remote'}</div>
              <span class="tag-accent">{j['salary_range'] or 'Negotiable'}</span>
              <span class="tag">{j['job_type']}</span>
              <span class="tag">{j['experience_required']} yrs exp</span>
              <span class="badge-verified">✔ Verified</span>
            </div>
            """, unsafe_allow_html=True)
            b1,b2 = st.columns([5,1])
            with b2:
                if st.button("Apply", key=f"posted_{j['id']}"):
                    st.session_state.selected_job = ("posted", j['id']); st.rerun()

    # Dataset jobs
    st.markdown("#### 📊 Verified Dataset Listings")
    for _, row in results.iterrows():
        pct = max(5, min(99, int(row["score"]*100)))
        orig_idx = int(row.name)
        skills_preview = " ".join(f"<span class='tag'>{s.strip()}</span>" for s in str(row["skills"]).split(",")[:4])
        st.markdown(f"""
        <div class="job-card">
          <h3>{row['title']}</h3>
          <div class="company">{row['company']} · {row['location']}</div>
          <span class="tag-accent">{row['salary']}</span>
          <span class="tag">{row['job_type']}</span>
          <span class="tag">{row['exp']} yrs exp</span>
          <span class="badge-verified">✔ Verified</span>
          <span class="badge-ai">🤖 {pct}% match</span>
          <div style="margin-top:0.4rem;">{skills_preview}</div>
          <div class="match-bar-wrap"><div class="match-bar" style="width:{pct}%;"></div></div>
        </div>
        """, unsafe_allow_html=True)
        b1,b2 = st.columns([5,1])
        with b2:
            if st.button("Details", key=f"ds_{orig_idx}"):
                st.session_state.selected_job = ("dataset", orig_idx); st.rerun()


def render_dataset_job_detail(row, idx):
    user   = st.session_state.user
    skills = [s.strip() for s in str(row["skills"]).split(",")]
    st.markdown(f"""
    <div class="detail-card">
      <div style="display:flex;align-items:center;gap:1.2rem;margin-bottom:1.2rem;">
        <div style="width:56px;height:56px;border-radius:12px;background:#1a3c5e;color:#fff;
                    display:flex;align-items:center;justify-content:center;font-size:1.5rem;font-weight:700;flex-shrink:0;">
          {str(row['company'])[0].upper()}
        </div>
        <div>
          <h2>{row['title']}</h2>
          <p style="color:#6b7280;font-size:0.88rem;margin:0;">{row['company']} &nbsp;·&nbsp; {row['location']}</p>
        </div>
      </div>
      <div style="margin-bottom:1rem;">
        <span class="tag-accent">{row['salary']}</span>
        <span class="tag">{row['job_type']}</span>
        <span class="tag">{row['exp']} yrs experience</span>
        <span class="tag">{row['industry']}</span>
        <span class="badge-verified">✔ Verified</span>
      </div>
      <hr style="border:none;border-top:1px solid #e5e7eb;margin:1rem 0;">
      <h4 style="color:#1a3c5e;margin-bottom:0.5rem;">About the Role</h4>
      <p style="color:#374151;line-height:1.8;font-size:0.93rem;">{row['description']}</p>
      <h4 style="color:#1a3c5e;margin-top:1.2rem;margin-bottom:0.6rem;">Required Skills</h4>
      <div>{''.join(f"<span class='tag-accent'>{s}</span>" for s in skills)}</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("")
    if user and user["role"] == "seeker":
        if already_applied_dataset(idx, user["id"]):
            st.markdown('<div class="success-box">✅ You have already applied for this job.</div>', unsafe_allow_html=True)
        else:
            if st.button("✅ Apply Now", use_container_width=True, key="apply_ds"):
                if apply_dataset_job(idx, user["id"]): st.success("🎉 Application submitted!")
                else: st.warning("Already applied.")
    elif not user:
        st.markdown('<div class="info-box">Please <b>login as a Job Seeker</b> to apply.</div>', unsafe_allow_html=True)
        if st.button("Login to Apply"): go("login")


def render_posted_job_detail(j):
    user = st.session_state.user
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
        <span class="badge-verified">✔ Verified</span>
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
    st.markdown("")
    if user and user["role"] == "seeker":
        if st.button("✅ Apply Now", use_container_width=True, key="apply_posted"):
            conn=get_conn()
            try:
                conn.execute("INSERT INTO applications(job_id,seeker_id) VALUES(?,?)",(j['id'],user['id']))
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
    role = st.radio("I am a:", ["Job Seeker","Company / Employer"], horizontal=True)
    st.divider()
    with st.form("login_form"):
        email = st.text_input("Email Address")
        pw    = st.text_input("Password", type="password")
        sub   = st.form_submit_button("Sign In", use_container_width=True)
    if sub:
        if not email or not pw: st.error("Please fill all fields."); return
        if role == "Job Seeker":
            u = login_seeker(email, pw)
            if u: st.session_state.user={"id":u["id"],"name":u["name"],"email":u["email"],"role":"seeker"}; go("dashboard_seeker")
            else: st.error("❌ Invalid credentials.")
        else:
            u = login_company(email, pw)
            if u: st.session_state.user={"id":u["id"],"name":u["name"],"email":u["email"],"role":"company"}; go("dashboard_company")
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
    role = st.radio("Register as:", ["Job Seeker","Company / Employer"], horizontal=True)
    st.divider()
    if role == "Job Seeker":
        with st.form("reg_seeker"):
            c1,c2 = st.columns(2); name=c1.text_input("Full Name *"); email=c2.text_input("Email *")
            c3,c4 = st.columns(2); phone=c3.text_input("Phone"); pw=c4.text_input("Password *",type="password")
            skills=st.text_input("Skills (comma-separated)",placeholder="Python, SQL, Machine Learning")
            exp=st.number_input("Experience (years)",min_value=0,max_value=50)
            sub=st.form_submit_button("Create Account",use_container_width=True)
        if sub:
            if not name or not email or not pw: st.error("Name, email and password required.")
            else:
                ok,msg=register_seeker(name,email,pw,phone,skills,exp)
                if ok: st.success(msg+" Please login."); go("login")
                else:  st.error(msg)
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
                else:  st.error(msg)
    st.markdown("Already have an account?")
    if st.button("Login"): go("login")


# ─────────────────────────────────────────────────────────────────────────────
# SEEKER DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────
def page_dashboard_seeker():
    user = st.session_state.user
    if not user or user["role"] != "seeker": go("login"); return
    df, tfidf, tfidf_matrix, pac_pipe, le = build_ml_engine()
    s = get_seeker(user["id"])
    total_apps, ds_apps = seeker_dashboard_stats(user["id"])
    score = profile_score(s)

    with st.sidebar:
        st.markdown(f"### 👤 {user['name']}")
        st.markdown(f"*{user['email']}*")
        st.progress(score/100, text=f"Profile {score}% complete")
        st.divider()
        section = st.radio("Navigate",
                           ["📊 Overview","🤖 AI Recommendations","📋 My Applications","👤 Edit Profile"],
                           label_visibility="collapsed")
        st.divider()
        if st.button("🔍 Browse All Jobs"): go("jobs")
        if st.button("🚪 Logout"):          logout()

    st.markdown(f"""
    <div class="hero-banner" style="padding:2rem;">
      <h1 style="font-size:1.8rem;">Welcome back, {user['name'].split()[0]}! 👋</h1>
      <p>Your personalised AI-powered job hub.</p>
    </div>
    """, unsafe_allow_html=True)

    if section == "📊 Overview":
        st.markdown(f"""
        <div class="stat-row">
          <div class="stat-card"><div class="num">{total_apps}</div><div class="lbl">Total Applications</div></div>
          <div class="stat-card"><div class="num">{ds_apps}</div><div class="lbl">Jobs Applied</div></div>
          <div class="stat-card"><div class="num">{score}%</div><div class="lbl">Profile Complete</div></div>
          <div class="stat-card"><div class="num">{len(RAW_JOBS)}</div><div class="lbl">Jobs Available</div></div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("#### Recent Applications")
        apps = get_seeker_dataset_apps(user["id"])[:5]
        rows=[]
        for a in apps:
            idx=a["dataset_job_idx"]
            if 0<=idx<len(RAW_JOBS):
                j=RAW_JOBS[idx]; rows.append({"Job":j[0],"Company":j[1],"Applied":a["applied_at"][:10],"Status":a["status"]})
        if rows: st.table(rows)
        else: st.info("No applications yet. Use 🤖 AI Recommendations or Browse Jobs to get started!")

    elif section == "🤖 AI Recommendations":
        st.markdown('<p class="sec-title">🤖 AI Job Recommendations</p>', unsafe_allow_html=True)
        profile_text = " ".join(filter(None,[s.get("skills",""),s.get("bio",""),s.get("preferred_location","")])).strip()
        if not profile_text:
            st.markdown('<div class="ai-box">⚠️ Complete your <b>Skills</b>, <b>Bio</b>, and <b>Location</b> in Edit Profile to unlock personalised AI recommendations.</div>', unsafe_allow_html=True)
        else:
            predicted = predict_category(profile_text, pac_pipe, le)
            st.markdown(f"""
            <div class="ai-box">
              🤖 <b>PAC Model Prediction:</b> Your profile strongly fits <b>{predicted}</b> roles.<br>
              <small style="opacity:0.7;">Model: Passive Aggressive Classifier · TF-IDF bigrams (4,000 vocab) · cosine similarity ranking</small>
            </div>
            """, unsafe_allow_html=True)
            recs = recommend_jobs(profile_text, df, tfidf, tfidf_matrix, top_n=10)
            for i, (orig_idx, row) in enumerate(recs.iterrows()):
                pct     = max(5, min(99, int(row["score"]*100)))
                already = already_applied_dataset(int(orig_idx), user["id"])
                skills  = [sk.strip() for sk in str(row["skills"]).split(",")]
                st.markdown(f"""
                <div class="job-card">
                  <div style="display:flex;justify-content:space-between;align-items:flex-start;flex-wrap:wrap;gap:0.5rem;">
                    <div style="flex:1;">
                      <h3>{row['title']}</h3>
                      <div class="company">{row['company']} · {row['location']}</div>
                      <span class="tag-accent">{row['salary']}</span>
                      <span class="tag">{row['job_type']}</span>
                      <span class="badge-ai">🤖 {pct}% match</span>
                      {"<span class='badge-verified'>✔ Applied</span>" if already else ""}
                    </div>
                  </div>
                  <div style="margin-top:0.5rem;">{''.join(f"<span class='tag'>{sk}</span>" for sk in skills[:5])}</div>
                  <div class="match-bar-wrap" style="margin-top:0.6rem;"><div class="match-bar" style="width:{pct}%;"></div></div>
                </div>
                """, unsafe_allow_html=True)
                b1,b2 = st.columns([3,1])
                with b1:
                    if st.button("View Details", key=f"ai_view_{i}_{orig_idx}"):
                        st.session_state.selected_job=("dataset",int(orig_idx)); go("jobs")
                with b2:
                    if not already:
                        if st.button("⚡ Quick Apply", key=f"ai_apply_{i}_{orig_idx}"):
                            apply_dataset_job(int(orig_idx), user["id"]); st.success(f"Applied to {row['title']}!"); st.rerun()
                    else: st.caption("✅ Applied")

    elif section == "📋 My Applications":
        st.markdown("#### All My Applications")
        apps = get_seeker_dataset_apps(user["id"])
        rows = []
        for a in apps:
            idx=a["dataset_job_idx"]
            if 0<=idx<len(RAW_JOBS):
                j=RAW_JOBS[idx]
                rows.append({"Job Title":j[0],"Company":j[1],"Location":j[2],"Type":j[3],"Applied On":a["applied_at"][:10],"Status":a["status"]})
        if rows: st.table(rows)
        else: st.info("No applications yet.")

    elif section == "👤 Edit Profile":
        st.markdown("#### Edit Profile")
        st.markdown('<div class="ai-box">💡 A richer profile (skills + bio + location) significantly improves your AI match score.</div>', unsafe_allow_html=True)
        with st.form("seeker_profile"):
            c1,c2  = st.columns(2)
            name   = c1.text_input("Full Name",   value=s.get("name","") or "")
            phone  = c2.text_input("Phone",        value=s.get("phone","") or "")
            skills = st.text_input("Skills (comma-separated)", value=s.get("skills","") or "",
                                   placeholder="Python, Machine Learning, SQL, React…")
            c3,c4  = st.columns(2)
            exp    = c3.number_input("Experience (yrs)", min_value=0, value=int(s.get("experience") or 0))
            loc    = c4.text_input("Preferred Location", value=s.get("preferred_location","") or "")
            bio    = st.text_area("Career Summary / About Me", value=s.get("bio","") or "",
                                  placeholder="Briefly describe your experience, goals, and what you're looking for…")
            salary = st.text_input("Expected Salary (LPA)", value=s.get("expected_salary","") or "")
            if st.form_submit_button("Save Profile", use_container_width=True):
                update_seeker(user["id"],name,phone,skills,exp,loc,bio,salary)
                st.session_state.user["name"]=name; st.success("✅ Profile saved!")


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
                           ["📊 Overview","➕ Post a Job","📋 My Job Posts","👥 Applicants","🏢 Company Profile"],
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
        else: st.table([{"Title":j["title"],"Location":j["location"] or "Remote","Posted":j["created_at"][:10],"Applicants":j["applicant_count"]} for j in jobs])

    elif section == "➕ Post a Job":
        st.markdown("#### Post a New Job")
        with st.form("post_job"):
            c1,c2  = st.columns(2); title=c1.text_input("Job Title *"); jtype=c2.selectbox("Job Type",["Full-time","Part-time","Remote","Internship","Contract"])
            c3,c4  = st.columns(2); loc=c3.text_input("Location",placeholder="Bangalore, India"); salary=c4.text_input("Salary Range (LPA)",placeholder="8–15 LPA")
            c5,c6  = st.columns(2)
            exp    = c5.number_input("Experience Required (yrs)", min_value=0)
            has_dl = c6.checkbox("Set Application Deadline")
            deadline = c6.date_input("Deadline Date", value=date.today()) if has_dl else None
            desc   = st.text_area("Job Description *", height=140)
            req    = st.text_area("Requirements / Skills", height=100)
            mobile = st.text_input("Contact Mobile")
            sub    = st.form_submit_button("Post Job", use_container_width=True)
        if sub:
            if not title or not desc: st.error("Title and description are required.")
            else: post_job(user["id"],title,jtype,loc,salary,exp,deadline,desc,req,mobile); st.success("✅ Job posted successfully!")

    elif section == "📋 My Job Posts":
        st.markdown("#### My Job Postings")
        jobs = get_company_jobs(user["id"])
        if not jobs: st.info("No jobs posted yet.")
        else:
            for j in jobs:
                c1,c2,c3 = st.columns([4,1,1])
                c1.markdown(f"**{j['title']}** — {j['location'] or 'Remote'}  \n`{j['job_type']}` · {j['applicant_count']} applicants")
                c2.caption(j["created_at"][:10])
                if c3.button("🗑 Delete", key=f"del_{j['id']}"): delete_job(j["id"],user["id"]); st.success("Deleted."); st.rerun()

    elif section == "👥 Applicants":
        st.markdown("#### Job Applicants")
        jobs    = get_company_jobs(user["id"])
        job_map = {"All Jobs":None}
        for j in jobs: job_map[j["title"]]=j["id"]
        chosen     = st.selectbox("Filter by job", list(job_map.keys()))
        applicants = get_applicants(user["id"], job_map[chosen])
        if not applicants: st.info("No applicants yet.")
        else:
            st.table([{"Name":a["name"],"Email":a["email"],"Skills":a["skills"] or "—",
                       "Experience":f"{a['experience'] or 0} yrs","Job":a["job_title"],"Applied":a["applied_at"][:10]} for a in applicants])

    elif section == "🏢 Company Profile":
        st.markdown("#### Company Profile")
        co = get_company(user["id"])
        with st.form("co_profile"):
            c1,c2   = st.columns(2); name=c1.text_input("Company Name",value=co.get("name","") or "")
            industry= c2.selectbox("Industry",["IT / Software","Finance","Healthcare","Manufacturing","E-commerce","Other"])
            c3,c4   = st.columns(2); website=c3.text_input("Website",value=co.get("website","") or ""); year=c4.number_input("Year Founded",min_value=1900,max_value=date.today().year,value=int(co.get("year_founded") or 2010))
            phone   = st.text_input("Phone", value=co.get("phone","") or "")
            desc    = st.text_area("Company Description", value=co.get("description","") or "")
            if st.form_submit_button("Save Profile", use_container_width=True):
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
