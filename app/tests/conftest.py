"""
Pytest configuration and shared fixtures for Job Portal API tests.
"""
import os
import pytest
from sqlmodel import Session, create_engine, SQLModel
from app.db.session import DatabaseSession
from fastapi.testclient import TestClient
from main import app
from app.models.user import User
from app.core.security import Security
from uuid import uuid4 
from app.core.config import Config
from jose import jwt
from app.db.session import db_session_manager
from app.tests.factory import user_payload
from sqlalchemy import text
from app.core.enum import UserRole, ApplicationStatus, ModeOfWork, EmploymentType

# Force application to run in test mode
os.environ["ENV"] = "test"

# using test database url
TEST_DATABASE_URL=Config.TEST_DATABASE_URL

# Create a separate engine for tests
engine=create_engine(TEST_DATABASE_URL, echo=True)

# Drops and recreates the public schema. Resets DB before every test
def reset_db(engine):
    with engine.begin() as conn:
        conn.execute(text("DROP SCHEMA public CASCADE"))
        conn.execute(text("CREATE SCHEMA public"))

# Initializes database schema once per test session.
@pytest.fixture(scope="session", autouse=True)
def db_engine():
    reset_db(engine)
    SQLModel.metadata.create_all(engine)
    yield engine
    reset_db(engine)

# Provides a SQLModel session bound to the test database.
@pytest.fixture(scope="session")
def db_session(db_engine):
    with Session(db_engine) as session:
        yield session       

# Overrides the application's database dependency to use the test database instead of production.
def override_get_session():
    with Session(engine) as session:
        yield session

# Apply dependency override for all tests
app.dependency_overrides[db_session_manager.get_session] = override_get_session

# Provides a FastAPI TestClient instance for API testing.
@pytest.fixture(scope="session")
def client():
    return TestClient(app)

#  Factory fixture to generate user registration/login payloads.
@pytest.fixture
def user_data_factory():
    def _factory(role=UserRole.CANDIDATE, current_organization=None, **overrides):
        return user_payload(role=role, current_organization=current_organization, **overrides)
    return _factory

# Registers a user via API and returns the created user response.
@pytest.fixture
def get_registered_user(client, user_data_factory):
    def _factory(role=UserRole.CANDIDATE, **overrides):
        payload = user_data_factory(role=role, **overrides)
        response = client.post("/auth/register", json=payload)
        assert response.status_code == 201
        return response.json()
    return _factory

# Logs in a registered user and returns access & refresh tokens.
@pytest.fixture
def login_and_get_tokens(client, get_registered_user, user_data_factory):
    def _factory(role=UserRole.CANDIDATE, **overrides):
        user = get_registered_user(role=role, **overrides)
        login_payload = user_data_factory(uname= user["user_name"], email= user["email"], role= user["role"], **overrides)
        response = client.post("/auth/login", json=login_payload)
        assert response.status_code == 200
        return response.json()
    return _factory

# Generates Authorization headers containing a valid access token.
@pytest.fixture
def auth_headers(login_and_get_tokens):
    def _factory(role=UserRole.CANDIDATE, **overrides):
        tokens = login_and_get_tokens(role=role, **overrides)
        return {"Authorization": f"Bearer {tokens['access_token']}"}
    return _factory

# Returns a sample payload for company creation.
@pytest.fixture
def company_payload():
    return {
        "name": f"TestCompany-{uuid4()}",
        "description": "Test description",
        "website": "test.com",
        "location": "Gandhinagar",
        "domain": "Python",
        "company_size": 50,
    }

# Returns a sample payload for job creation.
@pytest.fixture
def job_payload():
    return {
    "title" : f"SDE-Intern-{uuid4().hex}" ,
    "description" : "That's my job...",
    "location" : "Gandhinagar",
    "mode": ModeOfWork.ONSITE,
    "employment_type" : EmploymentType.INTERN,
    "remuneration_range" : "4-5LPA",
    "tags": ["Python", "AI", "ML"]
    }

# Returns multiple job payloads for bulk job creation tests.
@pytest.fixture
def jobs_payload():
    return [
            {
        "title" : f"SDE-Intern-{uuid4().hex}" ,
        "description" : "That's my job...",
        "location" : "Gandhinagar",
        "mode": ModeOfWork.ONSITE,
        "employment_type" : EmploymentType.INTERN,
        "remuneration_range" : "4-5LPA",
        "tags": ["Python", "AI", "ML", ModeOfWork.ONSITE]
        },
        {
        "title" : f"Digital-Marketing-{uuid4().hex}" ,
        "description" : "That's my job...",
        "location" : "Ahmedabad",
        "mode": ModeOfWork.ONSITE,
        "employment_type" : EmploymentType.FULL_TIME,
        "remuneration_range" : "4-8LPA",
        "tags": ["SEO", "Marketing", "Digital Marketing", ModeOfWork.ONSITE]
        },
        {
        "title" : f"QA-Intern-{uuid4().hex}" ,
        "description" : "That's my job...",
        "location" : "Vadodara",
        "mode": ModeOfWork.REMOTE,
        "employment_type" : EmploymentType.INTERN,
        "remuneration_range" : "4-7LPA",
        "tags": ["BDA", "testing", "QA"]
        }
    ]

# Returns a payload for job application submission.
@pytest.fixture
def application_payload():
    return {"message": f"A new application - {uuid4().hex}"}


# Creates multiple jobs and returns the list of created jobs.
@pytest.fixture
def get_created_jobs_list(client, auth_headers, get_created_company, jobs_payload):
    company_id=get_created_company["id"]
    headers=auth_headers(role=UserRole.RECRUITER, current_organization=company_id)
    jobs_list=[]
    for job in jobs_payload:
        response = client.post("/jobs/", json=job, headers=headers)
        assert response.status_code == 201
        jobs_list.append(response.json())
    return jobs_list

# Creates and returns a company as a recruiter.
@pytest.fixture
def get_created_company(client, auth_headers, company_payload):
    headers = auth_headers(UserRole.RECRUITER)
    response = client.post("/companies/", json=company_payload, headers=headers)
    assert response.status_code == 201
    return response.json()

# Creates and returns a job under an existing company.
@pytest.fixture
def get_created_job(client, auth_headers, job_payload, get_created_company):
    company_id=get_created_company["id"]
    headers=auth_headers(role=UserRole.RECRUITER, current_organization=company_id)
    response = client.post("/jobs/", json=job_payload, headers=headers)
    assert response.status_code == 201
    return response.json()

# Creates a temporary directory for resume uploads during tests, uses tmp_path and monkeypatch from pytest
@pytest.fixture
def temp_upload_dir(tmp_path, monkeypatch):
    temp_dir = tmp_path / "resumes"
    temp_dir.mkdir()
    monkeypatch.setenv("UPLOAD_RESUME_DIR", str(temp_dir))
    return temp_dir

# Applies to a job with a resume upload and returns the created application.
@pytest.fixture
def get_created_application(client, auth_headers, application_payload, get_created_job, temp_upload_dir):
    job_id=get_created_job["id"]
    headers=auth_headers(UserRole.CANDIDATE)
    content=b"test resume content"
    response=client.post(f"/applications/jobs/{job_id}/apply", headers=headers, data=application_payload, files={"resume":("test_resume.pdf", content, "application/pdf")})
    assert response.status_code==201
    return response.json()