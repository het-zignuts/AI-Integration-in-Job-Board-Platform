"""
Pydantic schemas for job applications.

These schemas define request and response payloads
for application-related API endpoints.
"""

from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime
from uuid import UUID
from app.core.enum import ApplicationStatus

class ApplicationCreate(BaseModel):
    # SCHEMA FOR APPLICATION CREATION PAYLOAD, ONLY MESSAGE TO BE SENT
    message: Optional[str] = None

class ApplicationUpdate(BaseModel):
    # SCHEMA FOR APPLICATION UPDATE PAYLOAD, ONLY STATUS TO BE SENT IN REQUEST
    status: ApplicationStatus

class ApplicationResponse(BaseModel):
    # SCHEMA FOR RESPONSE RECIEVED FROM APPLICATION APIS
    id: UUID
    user_id: UUID
    job_id: UUID
    resume_filename: str
    message: Optional[str] = None
    status: ApplicationStatus
    applied_at: datetime
    updated_at: Optional[datetime] = None