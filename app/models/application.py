"""
SQLModel Model for representing Application object.
"""

from sqlmodel import SQLModel, Field, Relationship
from typing import Optional, TYPE_CHECKING
from app.core.enum import ApplicationStatus
from datetime import datetime, timezone
from uuid import UUID, uuid4
from sqlalchemy import Column, Enum as SAEnum

if TYPE_CHECKING: # to prevent circular imports
    from app.models.job import Job
    from app.models.user import User

# Model class, inheriting from SQLModel
class Application(SQLModel, table=True):
    id: UUID = Field(default_factory=uuid4, primary_key=True, index=True) # application id
    user_id: UUID = Field(foreign_key="user.id", nullable=False) # user id
    job_id: UUID = Field(foreign_key="job.id", nullable=False) # job id associated with
    job: Optional["Job"] = Relationship(back_populates="applications") # job it is related to
    resume_filename: str = Field(nullable=False) # name of resume file
    resume_path: str = Field(nullable=False) # path at which the file is stored
    message : Optional[str] = Field(default=None, nullable=True) # message shared by user allong with the application
    status: ApplicationStatus = Field(sa_column=SAEnum(ApplicationStatus, name="applicationstatus", native_enum=True, validate_strings=True, nullable=False), default=ApplicationStatus.APPLIED) # Application status, default and initial is APPLIED, values taken from ApplicationStatus enum class
    applied_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), nullable=False) # timestamp of application
    updated_at: Optional[datetime] = Field(default=None, nullable=True) # timestamp of update
    user: Optional["User"] = Relationship(back_populates="applications") # user it is associated with