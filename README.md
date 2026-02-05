# AI-Integration with Job Board Platform

This project is an AI-integrated job board platform designed to provide advanced job recommendations, resume analysis, job description improvement, and intelligent RAG-based question-answering. It leverages embeddings, LLMs, and AI agent capabilities to enhance the recruitment and job search experience.

## Project and Environment Setup:

- Environment Setup:

1. Clone the repo:
 ```bash
git clone https://github.com/het-zignuts/AI-Integration-in-Job-Board-Platform.git
```

Create a new env in project folder (ensure python 3.11.x):
```bash
python -m venv .venv
```

2. Activate the environemnt:
```bash
source .venv/bin/activate
```

3. Intsall the dependencies:
```bash
pip install -r requirements.txt
```

4. Running the server:
```bash
uvicorn app.main:app --reload
```
The server starts running on (https://127.0.0.1:8000)

#### Interactive docs:
- Swagger UI -> https://127.0.0.1:8000/docs
- ReDoc -> https://127.0.0.1:8000/redoc

## Database Setup (Persistent Database: PostgreSQL):

1. Install pgvector:
```bash
brew install pgvector
```
2. Create pgvector extension for PostgreSQL.
```psql
CREATE EXTENSION IF NOT EXISTS vector;
```

3. Create embeddings table and index. (Script in app.db.sql > embeddings.sql)
```psql
CREATE TABLE IF NOT EXISTS embeddings (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    entity_type TEXT NOT NULL,       
    entity_id UUID NOT NULL,
    content TEXT NOT NULL,
    embedding VECTOR(384) NOT NULL,
    created_at TIMESTAMP DEFAULT now(),
    updated_at TIMESTAMP
);

CREATE INDEX IF NOT EXISTS embeddings_vector_idx
ON embeddings
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);
```

## Folder Structure:

```text
Job_board_major_project (repo)
├── app/
│   ├── ai/ .................... (AI integration)
│   │   ├── agent
│   │   ├── schema
│   │   ├── embeddings
│   │   ├── rag
│   │   ├── llms
│   │   ├── job_recommender
│   │   └── jd_improviser
│   ├── api/ .................... (API endppoints defined here)
│   │   ├── application.py
│   │   ├── company.py 
│   │   ├── job.py 
│   │   └── user.py 
│   ├── auth/ .................... (Dependencies and authentication routes (registration, login, token refresh))
│   │   ├── deps.py 
│   │   └── routes.py
│   ├── core/ .................... (Core settings)
│   │   ├── config.py  (gives env variables)
│   │   ├── security.py (gives token management and password security utils)
│   │   └── enum.py (enum classes)
│   ├── crud/ ..................... (Business Logic (CRUD operations))
│   │   ├── application.py 
│   │   ├── job.py 
│   │   ├── company.py 
│   │   └── user.py
│   ├── models/ ................... (database models (SQLModel))
│   │   ├── application.py 
│   │   ├── job.py 
│   │   ├── company.py 
│   │   ├── refreshtoken.py 
│   │   └── user.py
│   ├── schema/ .....................(Pydantic Schemas)
│   │   ├── application.py 
│   │   ├── job.py 
│   │   ├── company.py 
│   │   ├── token.py 
│   │   └── user.py
│   ├── db/
│   │   ├── sql
│   │   ├── session.py ..............(Session Management)
│   │   └── init_db.py ..............(Database Initializaton)
│   └── tests/ ......................(Unit tests)
│       ├── conftest.py 
│       ├── factory.py 
│       ├── test_application.py 
│       ├── test_job.py 
│       ├── test_company.py 
│       ├── test_token.py 
│       └── test_user.py
│   
├── requirements.txt 
├──README.md
├──pytest.ini 
└── main.py .........................(Entry point)
```

## API Summary:

| Endpoint                  | Method | Description                                                                          | Input                                        | Response Model / Output                                                 | Access Control      |
| ------------------------- | ------ | ------------------------------------------------------------------------------------ | -------------------------------------------- | ----------------------------------------------------------------------- | ------------------- |
| `/ai/ask-ai`              | GET    | Ask a general query to the AI Assistant and get context-based answer.                | `search_query` (query string)                | JSON (generated answer with sources)                                    | Authenticated users |
| `/ai/job-recommendations` | GET    | Generate top job recommendations based on candidate’s resume.                        | `resume_txt` (string)                        | `RecommendationResponse` (list of job matches with confidence & reason) | Authenticated users |
| `/ai/improve-job-rec`     | POST   | Improve a job description text using AI.                                             | `ImprovementRequest` (`mode`, `description`) | `ImprovementResponse` (improved JD)                                     | Recruiters only     |
| `/ai/ai-agent`            | GET    | Run the AI agent for complex tasks (multi-step reasoning, API calls, vector search). | `query` (query string)                       | JSON (agent result, context, or error)                                  | Authenticated users |

## Features:

### 1. Embedding Creation
- Converts jobs, companies, and resumes into semantic vectors for AI search, recommendations, and reasoning.
- Job/company embeddings: combine metadata (title, location, tags, description).
- Resume embeddings: load PDF/TXT -> normalize -> chunk -> embed.
- Stores all embeddings in a central DB with upserts for efficient updates.
- Uses HuggingFace all-MiniLM-L6-v2 for lightweight, high-quality embeddings.
- Preprocessing and chunking avoid token limits for downstream AI agents.
- Enables agents to summarize, reason, and query without reprocessing raw data.


### 2. RAG-Based Q&A
- Purpose: Answer user questions using only pre-indexed job, company, and resume data.
- Context Retrieval: Uses vector embeddings (pgvector + HuggingFace MiniLM) to fetch top-k semantically relevant chunks from the database.
- RAG Chain: Combines retrieved context with a reusable prompt template -> passes to Groq LLM -> parsed using Pydantic schema.
- Output Constraints:
    - Strictly JSON format.
    - Includes question, answer (string), sources (entity_id/type + quoted content), and confidence (0–1).
    - No external knowledge; fallback message if context insufficient.
- Technical Highlights:
    - LangChain pipeline: PromptTemplate | LLM | PydanticOutputParser.
    - Efficient semantic retrieval via embeddings.
    - Ensures concise, factual, and source-traceable answers.


### 3. Job Recommendations:
- Purpose: Rate top job matches for a candidate based on their resume.
- Workflow:
    - Embed candidate resume using sentence-transformers embeddings.
    - Retrieve top-K semantically similar jobs from embeddings DB using pgvector cosine similarity.
- For each matched job:
    - Construct context string including entity_type, entity_id, and job content.
    - Pass context + a prompt to Groq LLM through LangChain.
    - Parse output using Pydantic into AIRatingResponse (confidence + short reason).
    - Return structured results as JobMatch objects containing job_id, title, confidence, and reason.
- Technical Highlights:
    - LangChain pipeline: PromptTemplate | Groq LLM | PydanticOutputParser.
    - Vector similarity retrieval ensures semantic relevance.
    - Each recommendation includes a confidence score and concise reasoning, making it traceable and actionable.

### 4. Job Description Improvement:

- Purpose: Automatically enhance job descriptions (shorten, clarify, or reformat) for better readability and impact.
- Workflow:
    - Accept job description and improvement mode (e.g., concise, technical, persuasive).
    - Pass inputs into a LangChain pipeline:
    - PromptTemplate -> Groq LLM -> PydanticOutputParser.
    - Parse LLM output into ImprovementResponse (structured improved description).
    - Return improved text for use in UI or job postings.
- Technical Highlights:
    - LangChain + Groq LLM enables controlled, structured text improvement.
    - PromptTemplate allows flexible modes of improvement.
    - Output is validated with Pydantic, ensuring consistent response format.

### 5. AI Agent Feature:

- Purpose: Acts as an AI assistant for the job board platform, automating reasoning, recommendations, and data retrieval.
- Key Functionalities:
    - Executes tasks using multi-step reasoning (MAX_STEPS=10).
    - Performs actions as instructed by LLM output:
    - api_call -> call internal APIs.
    - search_vector_db -> fetch top-k semantic matches from embeddings.
    - llm_reasoning_tool -> perform reasoning over context.
    - Maintains context and conversation history (self.context, self.messages).
    - Safety check to prevent unauthorized actions (safety: not_allowed).
- Technical Highlights:
    - LLM-driven workflow via agent_llm (Groq LLM).
    - Structured action inputs using Pydantic (AgentResponse).
    - Vector-based retrieval using pgvector embeddings.
    - Observations stored in context and logged in messages with tool_call_id.
    - Modular prompt system: separate user/system prompts for flexible task handling.
- Scalability & Control:
    - Limits agent iterations to prevent infinite loops.
    - Supports multiple entity types (job, company, candidate_resume) for reasoning and search.
