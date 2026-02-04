"""
CRUD operations for jobs.

This module handles creation, retrieval, updating, deletion,
and relationship management of jobs.
"""

from sqlmodel import Session, select
from typing import Optional
from uuid import UUID
from datetime import datetime, timezone
from sqlalchemy import or_
from app.models.job import Job
from app.models.application import Application
from app.schemas.job import JobCreate, JobUpdate, JobResponse
from app.core.enum import ModeOfWork, EmploymentType

# Job creation business logic
def create_job(job: JobCreate, company_id: UUID, session: Session) -> JobResponse:
    job_instance=Job(**job.model_dump()) # setting fields sent as JobCreate object in model instance
    # setting remaining fields
    job_instance.company_id=company_id
    job_instance.applications=[]
    session.add(job_instance)
    session.commit()
    session.refresh(job_instance)
    return JobResponse(id=job_instance.id, title=job_instance.title, description=job_instance.description, location=job_instance.location, mode=job_instance.mode, employment_type=job_instance.employment_type, remuneration_range=job_instance.remuneration_range, company_id=job_instance.company_id, tags=job_instance.tags, posted_at=job_instance.posted_at, applications=job_instance.applications)

# job retrieval by id
def get_job_by_id(job_id: UUID, session: Session) -> Optional[JobResponse]:
    job=session.exec(select(Job).where(Job.id==job_id)).first()
    if job:
        return JobResponse(id=job.id, title=job.title, description=job.description, location=job.location, mode=job.mode, employment_type=job.employment_type, remuneration_range=job.remuneration_range, company_id=job.company_id, tags=job.tags, posted_at=job.posted_at, applications=job.applications)
    return None

# getting list of jobs with proper search, filter, order and pagination specifications
def list_jobs(session: Session, # SQLModel session object
              search_query: Optional[str] = None, # search query
              location: Optional[str] = None, # loication filter value
              mode: Optional[ModeOfWork] = None, # mode of work filter
              employment_type: Optional[EmploymentType] = None, # employment type filter
              tags: Optional[list[str]] = None, # tags to fiter jobs with
              order_by: str = "posted_at",  # ordering filed of jobs list
              order_type : str = "desc", # order type of job list
              ) -> list[JobResponse]: 
    query = select(Job)
    if search_query: # case insensiitive search of jobs wit given query
        query = query.where(or_(Job.title.ilike(f"%{search_query}%"), Job.description.ilike(f"%{search_query}%")))
    if location: # filter based on location
        query = query.where(Job.location.ilike(f"%{location}%"))
    if mode: # filter based on mode of work
        query = query.where(Job.mode==mode)
    if employment_type: # filter based on employment type
        query = query.where(Job.employment_type==employment_type)
    if tags: # filter based on tags
        query = query.where(Job.tags.contains(tags))
    if order_by == "posted_at": # setting order based on field provided
        if order_type == "asc": # setting ascending order
            query = query.order_by(Job.posted_at.asc())
        else: # setting descending order
            query = query.order_by(Job.posted_at.desc())
    jobs=session.exec(query).all() # retrieve all jobs matching above criteria
    return [JobResponse(id=job.id, title=job.title, description=job.description, location=job.location, mode=job.mode, employment_type=job.employment_type, remuneration_range=job.remuneration_range, company_id=job.company_id, tags=job.tags, posted_at=job.posted_at, applications=job.applications) for job in jobs]

# job Update business logic
def update_job(job_id: UUID, new_job: JobUpdate, session: Session) -> Optional[JobResponse]:
    job=session.exec(select(Job).where(Job.id==job_id)).first()
    if not job:
        return None
    job_data = new_job.model_dump()
    for key, value in job_data.items():
        setattr(job, key, value) # updating job instance fields with newly passed fields
    session.add(job)
    session.commit()
    session.refresh(job)
    return JobResponse(id=job.id, title=job.title, description=job.description, location=job.location, mode=job.mode, employment_type=job.employment_type, remuneration_range=job.remuneration_range, company_id=job.company_id, tags=job.tags, posted_at=job.posted_at, applications=job.applications)

# job deletion logic
def delete_job(job_id: UUID, session: Session) -> bool:
    job=session.exec(select(Job).where(Job.id==job_id)).first() # retrieve job
    applications=session.exec(select(Application).where(Application.job_id == job_id)).all() # retrieve associated applications
    for application in applications:
        session.delete(application) # delete every appliation associated with that job
    if not job:
        return False
    session.delete(job) # delete the job
    session.commit()
    return True