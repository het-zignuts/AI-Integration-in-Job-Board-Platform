"""
Pydantic schemas for job-related operations.

These schemas define request and response payloads
used by job API endpoints.
"""

from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime
from uuid import UUID
from app.core.enum import EmploymentType, ModeOfWork
from app.models.application import Application

class JobCreate(BaseModel):
    # SCHEMA FOR JOB CREATION
    title : str 
    description : Optional[str] = None
    location : Optional[str] = None
    mode: ModeOfWork
    employment_type : EmploymentType 
    remuneration_range : Optional[str] = None
    tags: List[str] = []

class JobUpdate(BaseModel):
    # SCHEMA FOR JOB UPDATE
    title : str 
    description : Optional[str] = None
    location : Optional[str] = None
    mode: ModeOfWork
    employment_type : EmploymentType 
    remuneration_range : Optional[str] = None
    tags: List[str] = []

class JobResponse(BaseModel):
    # SCHEMA FOR JOB API RESPONSES
    id : UUID
    title : str 
    description : Optional[str] = None
    location : Optional[str] = None
    mode: ModeOfWork
    employment_type : EmploymentType 
    remuneration_range : Optional[str] = None
    company_id : UUID
    tags: List[str] = []
    posted_at : datetime
    applications: List["Application"]=[]