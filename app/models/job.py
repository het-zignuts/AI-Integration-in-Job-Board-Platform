from sqlmodel import SQLModel, Field, Relationship
from typing import Optional, List, TYPE_CHECKING
from datetime import datetime, timezone
from uuid import UUID, uuid4
from sqlalchemy import Column, Enum as SAEnum
from sqlalchemy.dialects.postgresql import JSONB
from app.core.enum import ModeOfWork, EmploymentType

if TYPE_CHECKING: # prevents circular imports
    from app.models.application import Application

#  Model for Job objcts
class Job(SQLModel, table=True):
    id : UUID = Field(default_factory=uuid4, primary_key=True, index=True) # id of job
    title : str = Field(index=True, nullable=False) # job title
    description : Optional[str] = Field(default=None, nullable=True) # job description
    location : Optional[str] = Field(default=None, nullable=True) # job location
    mode: ModeOfWork = Field(sa_column=SAEnum(ModeOfWork, name="modeofwork", native_enum=True, validate_strings=True, nullable=False),default=ModeOfWork.ONSITE) # mode of work of job. default ONSITE, takes its values from ModeOfWork enum class
    employment_type : EmploymentType = Field(sa_column=SAEnum(EmploymentType, name="employmenttype", native_enum=True, validate_strings=True, nullable=False),default=EmploymentType.FULL_TIME) # emplyment category of job, default is FULL_TIME, takes values from EmploymentType
    remuneration_range : Optional[str] = Field(default=None, nullable=True) # remuneration range
    company_id : UUID = Field(foreign_key="company.id", nullable=False) # compnay for which the job is to be done
    tags: List[str] = Field(sa_column=Column(JSONB, nullable=True), default_factory=list) # tags associated with the job
    posted_at : datetime = Field(default_factory=lambda:datetime.now(timezone.utc), nullable=False) # time created 
    applications: Optional[List["Application"]] = Relationship(back_populates="job") # list of applications for the job