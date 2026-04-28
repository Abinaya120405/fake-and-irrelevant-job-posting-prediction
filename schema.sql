-- =====================================================
-- TrueHire Database Schema — MySQL
-- Run: mysql -u root -p < schema.sql
-- =====================================================

CREATE DATABASE IF NOT EXISTS truehire_db
  CHARACTER SET utf8mb4
  COLLATE utf8mb4_unicode_ci;

USE truehire_db;

-- ── USERS ──────────────────────────────────────────
CREATE TABLE IF NOT EXISTS users (
  id            INT AUTO_INCREMENT PRIMARY KEY,
  name          VARCHAR(200) NOT NULL,
  email         VARCHAR(255) NOT NULL UNIQUE,
  password_hash VARCHAR(255) NOT NULL,
  role          ENUM('seeker','company','admin') NOT NULL DEFAULT 'seeker',
  phone         VARCHAR(20),
  is_active     TINYINT(1) DEFAULT 1,
  created_at    DATETIME DEFAULT CURRENT_TIMESTAMP,
  updated_at    DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  INDEX idx_email (email),
  INDEX idx_role  (role)
) ENGINE=InnoDB;

-- ── SEEKER PROFILES ──────────────────────────────
CREATE TABLE IF NOT EXISTS seeker_profiles (
  id                 INT AUTO_INCREMENT PRIMARY KEY,
  user_id            INT NOT NULL UNIQUE,
  skills             TEXT,
  experience         DECIMAL(4,1) DEFAULT 0,
  bio                TEXT,
  preferred_location VARCHAR(200),
  expected_salary    VARCHAR(50),
  resume_url         VARCHAR(500),
  created_at         DATETIME DEFAULT CURRENT_TIMESTAMP,
  updated_at         DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
  INDEX idx_sp_user (user_id)
) ENGINE=InnoDB;

-- ── COMPANY PROFILES ──────────────────────────────
CREATE TABLE IF NOT EXISTS company_profiles (
  id           INT AUTO_INCREMENT PRIMARY KEY,
  user_id      INT NOT NULL UNIQUE,
  industry     VARCHAR(100),
  year_founded YEAR,
  description  TEXT,
  website      VARCHAR(500),
  logo_url     VARCHAR(500),
  created_at   DATETIME DEFAULT CURRENT_TIMESTAMP,
  updated_at   DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
  INDEX idx_cp_user (user_id)
) ENGINE=InnoDB;

-- ── JOBS ──────────────────────────────────────────
CREATE TABLE IF NOT EXISTS jobs (
  id                  INT AUTO_INCREMENT PRIMARY KEY,
  company_id          INT NOT NULL,
  title               VARCHAR(300) NOT NULL,
  description         TEXT,
  requirements        TEXT,
  location            VARCHAR(200),
  salary_range        VARCHAR(100),
  job_type            ENUM('Full-time','Part-time','Remote','Internship','Contract') DEFAULT 'Full-time',
  experience_required DECIMAL(4,1) DEFAULT 0,
  contact_mobile      VARCHAR(20),
  deadline            DATE,
  ml_label            ENUM('genuine','fake','irrelevant','pending') DEFAULT 'pending',
  ml_confidence       DECIMAL(5,2) DEFAULT 0.00,
  is_active           TINYINT(1) DEFAULT 1,
  created_at          DATETIME DEFAULT CURRENT_TIMESTAMP,
  updated_at          DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  FOREIGN KEY (company_id) REFERENCES users(id) ON DELETE CASCADE,
  INDEX idx_job_company  (company_id),
  INDEX idx_job_label    (ml_label),
  INDEX idx_job_active   (is_active),
  FULLTEXT INDEX ft_job_search (title, description)
) ENGINE=InnoDB;

-- ── APPLICATIONS ──────────────────────────────────
CREATE TABLE IF NOT EXISTS applications (
  id           INT AUTO_INCREMENT PRIMARY KEY,
  job_id       INT NOT NULL,
  seeker_id    INT NOT NULL,
  status       ENUM('pending','reviewed','shortlisted','rejected','hired') DEFAULT 'pending',
  match_status ENUM('matched','mismatch','pending') DEFAULT 'pending',
  cover_note   TEXT,
  applied_at   DATETIME DEFAULT CURRENT_TIMESTAMP,
  updated_at   DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  UNIQUE KEY  uq_application (job_id, seeker_id),
  FOREIGN KEY (job_id)    REFERENCES jobs(id)  ON DELETE CASCADE,
  FOREIGN KEY (seeker_id) REFERENCES users(id) ON DELETE CASCADE,
  INDEX idx_app_job    (job_id),
  INDEX idx_app_seeker (seeker_id),
  INDEX idx_app_status (status)
) ENGINE=InnoDB;

-- ── SEED DEMO DATA ──────────────────────────────
INSERT INTO users (name, email, password_hash, role) VALUES
('Demo Seeker',  'seeker@demo.com',  '$2b$12$demohashedpassword1111111111111111111111111111', 'seeker'),
('TechCorp Ltd', 'hr@techcorp.com', '$2b$12$demohashedpassword2222222222222222222222222222', 'company');

INSERT INTO company_profiles (user_id, industry, year_founded, description) VALUES
(2, 'IT / Software', 2015, 'Leading software company building scalable web solutions.');

INSERT INTO seeker_profiles (user_id, skills, experience, bio) VALUES
(1, 'Python, React, SQL', 2, 'Passionate developer looking for growth opportunities.');

SELECT 'Schema created successfully.' AS status;
