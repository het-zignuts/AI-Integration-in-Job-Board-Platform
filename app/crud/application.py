"""
CRUD operations for job applications.

This module handles creation, retrieval, updating, deletion,
and relationship management of job applications.
"""

from sqlmodel import Session, select
from typing import Optional
from uuid import UUID
from datetime import datetime, timezone
from app.models.application import Application
from app.models.user import User
from app.schemas.application import ApplicationCreate, ApplicationUpdate, ApplicationResponse
from app.models.job import Job
from app.crud.job import get_job_by_id
from app.core.enum import ApplicationStatus
import os

# Business Logic to create application
def create_application(application: ApplicationCreate, user_id: UUID, job_id: UUID, resume_filename: str, resume_path: str, session: Session) -> ApplicationResponse:
    job=get_job_by_id(job_id, session) # retrieve the job from job id passed to the function
    if not job:
        return None # return none if job not found
    application_instance=Application(**application.model_dump()) # create application instance 
    # set various fields as per the default/request data
    application_instance.user_id=user_id
    application_instance.job_id=job_id
    application_instance.resume_filename=resume_filename
    application_instance.resume_path=resume_path
    application_instance.status=ApplicationStatus.APPLIED
    session.add(application_instance) # add data to db sesion
    session.commit() # commit the db session
    session.refresh(application_instance) # reload the data with latest persisted state
    return ApplicationResponse(id=application_instance.id, user_id=application_instance.user_id, job_id=application_instance.job_id, resume_filename=application_instance.resume_filename, message=application_instance.message, status=application_instance.status, applied_at=application_instance.applied_at, updated_at=application_instance.updated_at)

# retrieve application from db based on the application id and send it as the response model object
def get_application_by_id(application_id: UUID, session: Session) -> Optional[ApplicationResponse]:
    application=session.exec(select(Application).where(Application.id==application_id)).first()
    if application:
        return ApplicationResponse(id=application.id, user_id=application.user_id, job_id=application.job_id, resume_filename=application.resume_filename, message=application.message, status=application.status, applied_at=application.applied_at, updated_at=application.updated_at)
    return None

# retrieve all application pertainng to a particualr job
def get_application_by_job_id(job_id: UUID, session: Session) -> list[ApplicationResponse]:
    applications=session.exec(select(Application).where(Application.job_id==job_id)).all()
    return [ApplicationResponse(id=application.id, user_id=application.user_id, job_id=application.job_id, resume_filename=application.resume_filename, message=application.message, status=application.status, applied_at=application.applied_at, updated_at=application.updated_at) for application in applications]

#  retrieve all applications pertaining to a particular user
def get_application_by_user_id(user_id: UUID, session: Session) -> list[ApplicationResponse]:
    applications=session.exec(select(Application).where(Application.user_id==user_id)).all()
    return [ApplicationResponse(id=application.id, user_id=application.user_id, job_id=application.job_id, resume_filename=application.resume_filename, message=application.message, status=application.status, applied_at=application.applied_at, updated_at=application.updated_at) for application in applications]

# retrieve operation to give list of all application objects present in the system
def list_applications(session: Session) -> list[ApplicationResponse]: 
    applications=session.exec(select(Application)).all()
    return [ApplicationResponse(id=application.id, user_id=application.user_id, job_id=application.job_id, resume_filename=application.resume_filename, message=application.message, status=application.status, applied_at=application.applied_at, updated_at=application.updated_at) for application in applications]

# update operation on application (status update)
def update_application(application_id: UUID, new_application: ApplicationUpdate, session: Session) -> Optional[ApplicationResponse]:
    application=session.exec(select(Application).where(Application.id==application_id)).first()
    if not application:
        return None
    application.status=new_application.status # update status
    application.updated_at=datetime.now(timezone.utc) # set updated_at field in application instance
    session.add(application)
    session.commit()
    session.refresh(application)
    return ApplicationResponse(id=application.id, user_id=application.user_id, job_id=application.job_id, resume_filename=application.resume_filename, message=application.message, status=application.status, applied_at=application.applied_at, updated_at=application.updated_at)

# delete application crud business logic
def delete_application(application_id: UUID, session: Session) -> bool: 
    application=session.exec(select(Application).where(Application.id==application_id)).first() # retriving application by the id
    user_id=application.user_id # getting the user id
    user=session.exec(select(User).where(User.id==user_id)).first() # getting the user
    resume_path=application.resume_path # gettig resume path
    if not application:
        return False
    session.delete(application) # deleting application
    session.commit()
    if resume_path and os.path.exists(resume_path): # deleting associated resume
        os.remove(resume_path)
    return True

# linking application to job, adding it to applications field of the job
def add_application_to_job(application_id: UUID, job_id: UUID, session: Session) -> bool:
    application=session.exec(select(Application).where(Application.id==application_id)).first()
    job=session.exec(select(Job).where(Job.id==job_id)).first()
    if not application or not job:
        return False
    if application not in job.applications:
        job.applications.append(application)
    session.add(job)
    session.commit()
    return True

# removing application from applications list of the job
def remove_application_from_job(application: Application, job_id: UUID, session: Session) -> bool:
    job=session.exec(select(Job).where(Job.id==job_id)).first()
    if not application or not job:
        return False
    if application in job.applications:
        job.applications.remove(application)
    session.add(job)
    session.commit()
    return True

# adding application to the user applications field
def add_application_to_user(application_id: UUID, user_id: UUID, session: Session) -> bool:
    application=session.exec(select(Application).where(Application.id==application_id)).first()
    user=session.exec(select(User).where(User.id==user_id)).first()
    if not application or not user:
        return False
    if application not in user.applications:
        user.applications.append(application)
    session.add(user)
    session.commit()
    return True

# removiung application from the applicxations list in user object
def remove_application_from_user(application: Application, user_id: UUID, session: Session) -> bool:
    user=session.exec(select(User).where(User.id==user_id)).first()
    if not application or not user:
        return False
    if application in user.applications:
        user.applications.remove(application)
    session.add(user)
    session.commit()
    return True
    