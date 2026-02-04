"""
API routes for application CRUD defined here.

Role based access control implemented through dependencies...
SQLModel Session passed as dependency
"""

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, status
from sqlmodel import Session
from uuid import UUID
import os
import shutil
from app.db.session import db_session_manager
from app.auth.deps import *
from app.models.user import User
from app.models.job import Job
from app.core.enum import UserRole
from app.schemas.application import ApplicationResponse
from app.crud.application import *
from app.crud.job import get_job_by_id
from app.crud.company import get_company_by_id
from app.core.config import Config

# router instance for the application API endpoints.
router = APIRouter(prefix="/applications", tags=["Applications"])

# Application creation API
@router.post("/jobs/{job_id}/apply", response_model=ApplicationResponse, status_code=status.HTTP_201_CREATED)
def create_application_api(
    job_id: UUID,
    message: str = "",
    resume: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    session: Session = Depends(db_session_manager.get_session),
):
    if is_candidate(current_user): # only a user who is candidate can apply for any job.
        job = get_job_by_id(job_id, session)
        if not job:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")
        for application in job.applications:
            if current_user.id == application.user_id: # prevent duplicate applications from same user.
                raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="User already applied...")
        company = get_company_by_id(job.company_id, session)
        if not company:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Company not found")
        uploads_dir = Config.UPLOAD_RESUME_DIR
        os.makedirs(uploads_dir, exist_ok=True)
        resume_filename = f"{current_user.id}_{job_id}_{resume.filename}" # generating resume filename
        resume_path = os.path.join(uploads_dir, resume_filename) # generating storage path
        with open(resume_path, "wb") as buffer:
            shutil.copyfileobj(resume.file, buffer) # store actual resume
        application_data = ApplicationCreate(message=message)
        application = create_application(application_data, current_user.id, job_id, resume_filename, resume_path, session) # crete application object
        if not application:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Failed to create application")
        added_to_job= add_application_to_job(application.id, job_id, session) # link application to the job
        if not added_to_job:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Failed to link application to job")
        added_to_user= add_application_to_user(application.id, current_user.id, session) # link application to the user
        if not added_to_user:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Failed to link application to user")
        return application
    else:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Only candidate can create application...")

# Application access endpoint
@router.get("/{application_id}", response_model=ApplicationResponse, status_code=status.HTTP_200_OK)
def get_application_api(application_id: UUID, current_user: User = Depends(get_current_user), session: Session = Depends(db_session_manager.get_session)):
    application = get_application_by_id(application_id, session)
    if not application:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Application not found")
    if is_candidate(current_user):
        print("Is candidate, checking ID...")
        if application.user_id != current_user.id:  # allow the user to access its own application only.
            print("ID mismatch of candidate...")
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="A candidate can only access its own application...")
        return application
    if is_recruiter(current_user): # allow recruiter access
        print("Is recruiter.... Getting Job...")
        job=get_job_by_id(application.job_id, session)
        print("Checking company ID...")
        print("User organization: ", current_user.current_organization)
        print("Company ID: ", job.company_id)
        if current_user.current_organization != job.company_id:
            print("Company ID mismatch...")
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="A recruiter can only view applications of its own organization...") 
        return application
    if is_admin(current_user):
        print("Is Admin... sending application...") # allow admin access
        return application
    else:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Only rcruiter, the candidate itself or admin can retrieve application")

@router.get("/jobs/{job_id}", response_model=list[ApplicationResponse], status_code=status.HTTP_200_OK)
def get_applications_by_job_api(job_id: UUID, current_user: User = Depends(get_current_user), session: Session = Depends(db_session_manager.get_session)):
    if is_recruiter(current_user) or is_admin(current_user):
        job = get_job_by_id(job_id, session)
        if not job:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")
        applications = get_application_by_job_id(job_id, session)
        return applications
    else:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Only recruiters or admin can view applications for this job")

# Access application through user id
@router.get("/users/{user_id}", response_model=list[ApplicationResponse], status_code=status.HTTP_200_OK)
def get_applications_by_user_api(user_id: UUID, current_user: User = Depends(get_current_user), session: Session = Depends(db_session_manager.get_session)):
    if not is_recruiter(current_user) and not is_admin(current_user): # prevent candidates from accessing created applications
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Insufficient permissions")
    applications = get_application_by_user_id(user_id, session)
    return applications

# Application Update logic (Only status update happens)
@router.put("/{application_id}", response_model=ApplicationResponse, status_code=status.HTTP_200_OK)
def update_application_status_api(application_id: UUID, new_status: ApplicationStatus, current_user: User = Depends(get_current_user), session: Session = Depends(db_session_manager.get_session)):
    if not is_recruiter(current_user) and not is_admin(current_user): # allow only recruiters and admin to update application status
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Only recruiters can update application status")
    application = get_application_by_id(application_id, session)
    if not application:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Application not found")
    job = get_job_by_id(application.job_id, session) # get jpob for which the application was created
    if not job:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")
    updated_application = update_application(application_id, ApplicationUpdate(status=new_status), session) # update application throught the respective CRUD operation
    if not updated_application:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Failed to update application")
    return updated_application

# Application Deletion Endpoint
@router.delete("/{application_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_application_api(application_id: UUID, current_user: User = Depends(get_current_user), session: Session = Depends(db_session_manager.get_session)):
    if not is_recruiter(current_user) and not is_admin(current_user): # allow only recruiter and admin to handle application deletion
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Insufficient permissions")
    application = get_application_by_id(application_id, session)
    if not application:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Application not found")
    removed_from_job= remove_application_from_job(application, application.job_id, session) # unlink application from job
    removed_from_user=remove_application_from_user(application, application.user_id, session) # unlink application from user
    if not removed_from_job:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Failed to unlink application from job")
    if not removed_from_user:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Failed to unlink application from user")
    success = delete_application(application_id, session)
    if not success:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Failed to delete application")
    return {
        "success_status": success # send deletion success status in response
    }

