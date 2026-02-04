"""
API endpoints for crud operations on jobs.
Creation, updation and deletion of jobs is restricted to just recruiters and endpoints. However, all 3 user types can access company retrieval.
Role based access control implemented through dependencies...
SQLModel Session passed as dependency
"""

from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlmodel import Session
from uuid import UUID
from fastapi_pagination import Page, paginate
from app.db.session import db_session_manager
from app.auth.deps import *
from app.models.user import User
from app.core.enum import UserRole
from app.schemas.job import JobCreate, JobUpdate, JobResponse
from app.crud.job import *
from app.crud.company import *
from app.ai.embeddings.embed_job import *

router = APIRouter(prefix="/jobs", tags=["Jobs"]) # router creation for jobs APIs

# Job Creation API
@router.post("/", response_model=JobResponse, status_code=status.HTTP_201_CREATED)
def create_job_api(job: JobCreate, current_user: User = Depends(get_current_user), session: Session = Depends(db_session_manager.get_session)):
    if not is_recruiter(current_user) and not is_admin(current_user): # only recruiter and admin can create jobs
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Only recruiters can create job postings")
    company = get_company_by_id(current_user.current_organization, session)
    if not company:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Company not found")
    created_job = create_job(job, company.id, session) # call to business logic for job creation
    embed_jobs(session)
    return created_job

# Job retrieval API by ID
@router.get("/{job_id}", response_model=JobResponse, status_code=status.HTTP_200_OK)
def get_job_api(job_id: UUID, session: Session = Depends(db_session_manager.get_session)):
    job = get_job_by_id(job_id, session)
    if not job:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")
    return job

# API to list all jobs, list can be filtered by loction, mode, employment type, tags and searched by title and description (case insensitive search). Ordering by date of job creation is also applied.
# Use of query and search params.
# Paginated response model.
@router.get("/", response_model=Page[JobResponse], status_code=status.HTTP_200_OK)
def list_jobs_api(
    session: Session = Depends(db_session_manager.get_session),
    search_query: Optional[str] = Query(None),
    location: Optional[str] = None,
    mode: Optional[ModeOfWork] = None,
    employment_type: Optional[EmploymentType] = None,
    tags: Optional[list[str]] = Query(None),
    order_by: str = "posted_at",
    order_type : str = "desc",
):
    jobs = list_jobs(
        session,
        search_query=search_query,
        location=location,
        mode=mode,
        employment_type=employment_type,
        tags=tags,
        order_by=order_by,
        order_type=order_type,
    ) # call to business logic
    return paginate(jobs) # pagination applied to the jobs

# job updation endpoint... only recruiters and admin allowed.
@router.put("/{job_id}", response_model=JobResponse, status_code=status.HTTP_200_OK)
def update_job_api(job_id: UUID, job: JobUpdate, current_user: User = Depends(get_current_user), session: Session = Depends(db_session_manager.get_session)):
    if not is_recruiter(current_user) and not is_admin(current_user): # prevent candidates from applying to jobs
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Only recruiters can update job postings")
    current_job = get_job_by_id(job_id, session)
    if not current_job:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")
    company = get_company_by_id(current_user.current_organization, session)
    if not company or current_job.company_id != company.id: # prevent other compnay's employee to update the job of this company
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Insufficient permissions to update this job")
    updated_job = update_job(job_id, job, session)
    embed_jobs(session)
    return updated_job

# job deletion endpoint.. for only recruiters and admin
@router.delete("/{job_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_job_api(job_id: UUID, current_user: User = Depends(get_current_user), session: Session = Depends(db_session_manager.get_session)):
    if not is_recruiter(current_user) and not is_admin(current_user): # prevent candidates from job deletion
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Only recruiters can delete job postings")
    current_job = get_job_by_id(job_id, session)
    if not current_job:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")
    company = get_company_by_id(current_user.current_organization, session)
    if not company or current_job.company_id != company.id: # prevent other compnay's employee to delete the job of this company
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Insufficient permissions to delete this job")
    success = delete_job(job_id, session) # call to business logic
    if not success:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")
    embed_jobs(session)
    return {
        "success_status": success # sen the success status as the response
    }