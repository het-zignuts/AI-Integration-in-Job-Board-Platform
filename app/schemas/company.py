"""
Pydantic schemas for company-related operations.

These schemas define the request and response payloads
used by company API endpoints.
"""

from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime
from uuid import UUID

class CompanyCreate(BaseModel):
    # SCHEMA FOR COMPANY CREATION
    name: str
    description: Optional[str] = None
    website: Optional[str] = None
    location: Optional[str] = None
    domain: Optional[str] = None
    company_size: int = Field(default=0)

class CompanyUpdate(BaseModel):
    # SCHEMA FOR COMPANY UPDATE
    name: str
    description: Optional[str] = None
    website: Optional[str] = None
    location: Optional[str] = None
    domain: Optional[str] = None
    company_size: int

class CompanyResponse(BaseModel):
    # SCHEMA FOR RESPONSES IN COMPANY CRUD APIS
    id: UUID
    name: str
    description: Optional[str] = None
    website: Optional[str] = None
    location: Optional[str] = None
    domain: Optional[str] = None
    company_size: int
    owner_id: UUID