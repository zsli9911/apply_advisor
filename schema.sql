-- Schema for the apply_advisor knowledge base.
--
-- Structured knowledge is normalized across six tables so that every
-- admission requirement, tuition figure and deadline is (a) stored with its
-- own scale / applicability and (b) traceable to an official source.
-- Unstructured policy text lives in policy_chunks with vector embeddings.

CREATE EXTENSION IF NOT EXISTS vector;

-- ===========================================================================
-- 1. Schools
-- ===========================================================================
CREATE TABLE IF NOT EXISTS universities (
    id                    SERIAL PRIMARY KEY,
    name                  TEXT NOT NULL UNIQUE,
    country               TEXT NOT NULL,
    city                  TEXT,
    university_type       TEXT,                 -- public_university | grande_ecole | institute
    official_website      TEXT,
    ranking_qs            INTEGER,              -- NULL when unknown (do not invent)
    ranking_times         INTEGER,
    language_environment  TEXT,                 -- e.g. "french; english widely used"
    living_cost_min       INTEGER,              -- EUR / year
    living_cost_max       INTEGER,              -- EUR / year
    last_verified_at      DATE
);

CREATE INDEX IF NOT EXISTS idx_universities_country ON universities (country);

-- ===========================================================================
-- 2. Programs
-- ===========================================================================
CREATE TABLE IF NOT EXISTS programs (
    id                    SERIAL PRIMARY KEY,
    university_id         INTEGER NOT NULL REFERENCES universities (id) ON DELETE CASCADE,
    name                  TEXT NOT NULL,        -- e.g. "MSc Data Science"
    degree_type           TEXT NOT NULL,        -- bachelor | master | phd
    discipline            TEXT NOT NULL,        -- broad field, e.g. "computer science"
    sub_discipline        TEXT,
    teaching_language     TEXT NOT NULL,        -- english | french | ...
    duration_months       INTEGER,
    tuition_fee           INTEGER,
    currency              TEXT DEFAULT 'EUR',
    intake                TEXT,                 -- e.g. "Fall"
    application_open_date  TEXT,                -- ISO date (nullable)
    application_deadline   TEXT,                -- ISO date
    official_url          TEXT,
    application_platform  TEXT,                 -- "Mon Master" | "Études en France" | portal
    program_status        TEXT DEFAULT 'open',  -- open | closed | suspended
    -- ranking / recommendation signals (used by the tiering engine)
    selectivity           INTEGER DEFAULT 3,    -- 1 (open) .. 5 (elite) → 冲刺/匹配/保底
    career_tags           TEXT,                 -- comma list, e.g. "data science, ai research"
    scholarship_available BOOLEAN DEFAULT FALSE,
    last_verified_at      DATE
);

CREATE INDEX IF NOT EXISTS idx_programs_university ON programs (university_id);
CREATE INDEX IF NOT EXISTS idx_programs_discipline ON programs (discipline);
CREATE INDEX IF NOT EXISTS idx_programs_language   ON programs (teaching_language);
CREATE INDEX IF NOT EXISTS idx_programs_degree     ON programs (degree_type);

-- ===========================================================================
-- 3. Admission requirements  (成绩体系 + 适用对象, not a bare number)
-- ===========================================================================
CREATE TABLE IF NOT EXISTS admission_requirements (
    id                       SERIAL PRIMARY KEY,
    program_id               INTEGER NOT NULL REFERENCES programs (id) ON DELETE CASCADE,
    minimum_grade            REAL,              -- on `grade_scale`, NOT normalized
    grade_scale              REAL,              -- 4.0 | 5.0 | 20 | 100
    source_country           TEXT,              -- whose grading system the min is stated in
    requirement_type         TEXT DEFAULT 'minimum',  -- minimum | recommended
    confidence               TEXT DEFAULT 'high',     -- high | medium | low
    recommended_grade        REAL,              -- competitive grade on the same scale
    accepted_degrees         TEXT,              -- e.g. "bachelor (3 or 4 year)"
    accepts_cross_discipline BOOLEAN,
    accepts_three_year_bachelor BOOLEAN,
    required_background      TEXT DEFAULT 'any',
    prerequisite_courses     TEXT,
    work_experience_required BOOLEAN DEFAULT FALSE,
    portfolio_required       BOOLEAN DEFAULT FALSE,
    gre_required             BOOLEAN DEFAULT FALSE,
    gmat_required            BOOLEAN DEFAULT FALSE,
    interview_required       BOOLEAN DEFAULT FALSE
);

CREATE INDEX IF NOT EXISTS idx_admission_program ON admission_requirements (program_id);

-- ===========================================================================
-- 4. Language requirements  (per-band, with waiver policy)
-- ===========================================================================
CREATE TABLE IF NOT EXISTS language_requirements (
    id                SERIAL PRIMARY KEY,
    program_id        INTEGER NOT NULL REFERENCES programs (id) ON DELETE CASCADE,
    language          TEXT NOT NULL,            -- english | french
    test_type         TEXT,                     -- IELTS | TOEFL | DELF/DALF/TCF
    minimum_total     REAL,
    minimum_listening REAL,
    minimum_reading   REAL,
    minimum_writing   REAL,
    minimum_speaking  REAL,
    minimum_cefr      TEXT,                     -- for CEFR-based (French) requirements
    waiver_available  BOOLEAN DEFAULT FALSE,
    waiver_conditions TEXT
);

CREATE INDEX IF NOT EXISTS idx_language_program ON language_requirements (program_id);

-- ===========================================================================
-- 5. Document requirements  (材料清单入库)
-- ===========================================================================
CREATE TABLE IF NOT EXISTS document_requirements (
    id                     SERIAL PRIMARY KEY,
    program_id             INTEGER NOT NULL REFERENCES programs (id) ON DELETE CASCADE,
    document_type          TEXT NOT NULL,       -- passport | transcripts | motivation_letter | ...
    required               BOOLEAN DEFAULT TRUE,
    conditions             TEXT,
    format_requirement     TEXT,
    translation_required   BOOLEAN DEFAULT FALSE,
    certification_required BOOLEAN DEFAULT FALSE
);

CREATE INDEX IF NOT EXISTS idx_document_program ON document_requirements (program_id);

-- ===========================================================================
-- 6. Sources  (可解释性: every fact traces to an official source)
-- ===========================================================================
CREATE TABLE IF NOT EXISTS sources (
    id              SERIAL PRIMARY KEY,
    entity_type     TEXT NOT NULL,              -- program | admission | language | document | university
    entity_id       INTEGER NOT NULL,           -- id in the corresponding table
    source_url      TEXT,
    source_title    TEXT,
    retrieved_at    DATE,
    effective_date  DATE,
    expiration_date DATE,
    source_type     TEXT,                        -- program_page | admission_page | official_policy
    authority_level TEXT,                        -- official | aggregator | community
    content_hash    TEXT
);

CREATE INDEX IF NOT EXISTS idx_sources_entity ON sources (entity_type, entity_id);

-- ===========================================================================
-- Unstructured knowledge: policy / guide documents (metadata + chunks)
-- ===========================================================================
-- Document-level metadata (文档元数据), one row per source file.
CREATE TABLE IF NOT EXISTS policy_documents (
    id             SERIAL PRIMARY KEY,
    document_key   TEXT NOT NULL UNIQUE,   -- filename / stable id
    university_id  INTEGER,
    program_id     INTEGER,
    country        TEXT,
    document_type  TEXT,                   -- admission_policy | visa_policy | ...
    source_url     TEXT,
    official       BOOLEAN DEFAULT TRUE,
    retrieved_at   DATE,
    effective_year TEXT,                   -- e.g. "2026-2027"
    language       TEXT,
    content_hash   TEXT
);

-- NOTE: the embedding dimension MUST match EMBEDDING_DIM in your .env
--       (1536 for text-embedding-3-small). Change it here if you switch models.
CREATE TABLE IF NOT EXISTS policy_chunks (
    id            SERIAL PRIMARY KEY,
    document_id   INTEGER REFERENCES policy_documents (id) ON DELETE CASCADE,
    source        TEXT NOT NULL,
    title         TEXT,
    section       TEXT,
    heading_path  TEXT,                    -- JSON array of headings (结构感知切分)
    chunk_type    TEXT,                    -- academic_requirement | language_requirement | ...
    chunk_index   INTEGER NOT NULL,
    content       TEXT NOT NULL,
    embedding     vector(1536)
);

CREATE INDEX IF NOT EXISTS idx_policy_source ON policy_chunks (source);
CREATE INDEX IF NOT EXISTS idx_policy_doc    ON policy_chunks (document_id);
CREATE INDEX IF NOT EXISTS idx_policy_type   ON policy_chunks (chunk_type);

-- Approximate nearest-neighbour index (cosine distance).
CREATE INDEX IF NOT EXISTS idx_policy_embedding
    ON policy_chunks USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);
