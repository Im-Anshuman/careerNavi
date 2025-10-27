# Overview

This is an AI-powered Career Roadmap Generator that helps users discover suitable career paths based on their skills and interests. The application uses semantic embeddings and cosine similarity to match user profiles with career options from a curated dataset. It provides personalized learning roadmaps at different skill levels (Beginner, Intermediate, Advanced) and identifies missing skills needed for each career path.

The application is built with Streamlit for the web interface and uses the SentenceTransformer model (all-MiniLM-L6-v2) for semantic matching between user input and career profiles. Users can also opt for a rule-based matching approach as an alternative to AI-powered matching.

# User Preferences

Preferred communication style: Simple, everyday language.

# System Architecture

## Frontend Architecture

**Problem**: Need an intuitive, interactive web interface for users to input skills/interests and view career recommendations.

**Solution**: Streamlit-based single-page application with form-based input and dynamic result rendering.

**Rationale**: Streamlit provides rapid prototyping capabilities with minimal frontend code, allowing focus on the core matching logic. The framework handles state management and UI updates automatically.

**Key Design Decisions**:
- Form-based input with comma-separated values for simplicity
- Checkbox toggle between AI-powered and rule-based matching
- Real-time progress indicators during embedding computation
- Wide layout mode for better readability of roadmap content

## Backend Architecture

**Problem**: Match user profiles to careers efficiently and provide personalized recommendations.

**Solution**: Two-tier matching system with both semantic embeddings and rule-based scoring.

**Components**:

1. **Embedding-based Matching** (Primary):
   - Pre-computed career vectors stored in `careers_vectorized.json`
   - User input vectorized on-demand using SentenceTransformer
   - Cosine similarity calculation for ranking
   - Pros: Captures semantic relationships, handles variations in terminology
   - Cons: Requires ML model loading, slightly slower on first run

2. **Rule-based Scoring** (Alternative):
   - Direct set intersection between user skills/interests and career requirements
   - Weighted scoring (70% skills, 30% interests)
   - Pros: Fast, deterministic, no model dependency
   - Cons: Exact matches only, misses semantic relationships

3. **Personalization Engine**:
   - Skill coverage ratio determines recommended phase (Beginner/Intermediate/Advanced)
   - Missing skills identification by set difference
   - Thresholds: <30% = Beginner, 30-70% = Intermediate, >70% = Advanced

## Data Storage

**Problem**: Store career data with both structured information and vector embeddings efficiently.

**Solution**: JSON file-based storage with separate raw and vectorized datasets.

**Structure**:
- `careers_data.json`: Human-readable career profiles with skills, interests, roadmaps, and resources
- `careers_vectorized.json`: Same data augmented with `_vector` field containing 384-dimensional embeddings

**Rationale**: JSON provides simplicity for this MVP-scale application (30 careers). No database overhead needed. Vectorized file enables fast loading without re-computation.

**Alternatives Considered**: SQLite or PostgreSQL could be used for larger datasets with filtering capabilities, but adds unnecessary complexity for current scale.

## Processing Pipeline

**Offline Vectorization** (`vectorize.py`):
- Converts career profiles to text representations
- Generates embeddings using SentenceTransformer
- Augments JSON with vector data
- One-time preprocessing step, not part of runtime

**Runtime Logic** (`logic.py`):
- Input preprocessing and normalization
- Vector similarity computation
- Rule-based scoring fallback
- Phase determination and skill gap analysis

# External Dependencies

## Machine Learning Framework

**SentenceTransformer (sentence-transformers)**:
- Model: `all-MiniLM-L6-v2` (384-dimensional embeddings)
- Purpose: Convert text to semantic vectors for similarity matching
- First-time download ~90MB, then cached locally

## Scientific Computing

**NumPy**: Array operations and vector storage
**scikit-learn**: Cosine similarity calculations (sklearn.metrics.pairwise)

## Web Framework

**Streamlit**: Complete web application framework including:
- Form rendering and state management
- Layout components and styling
- Session state handling
- Deployment capabilities via Streamlit Cloud

## Development Tools

**tqdm**: Progress bars during vectorization process (offline script only)

## Data Format

All career data stored as JSON with the following schema:
- `career`: Career title (string)
- `required_skills`: List of skill strings
- `interest_tags`: List of interest area strings
- `roadmap`: Object with Beginner/Intermediate/Advanced arrays of learning steps
- `resources`: List of recommended learning resource strings
- `_vector`: 384-element array (in vectorized version only)