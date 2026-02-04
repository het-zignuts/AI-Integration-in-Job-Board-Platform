from sqlmodel import SQLModel, Field
from typing import Optional, List
from app.core.enum import UserRole
from datetime import datetime
from uuid import UUID, uuid4

class Company(SQLModel, table=True):
    id: UUID = Field(default_factory=uuid4, primary_key=True, index=True)  # company id
    name: str = Field(index=True, unique=True, nullable=False) # name of company
    description: Optional[str] = Field(default=None, nullable=True) # description of company
    website: Optional[str] = Field(default=None, nullable=True) # website name of the company
    location: Optional[str] = Field(default=None, nullable=True) # location of th company
    domain: Optional[str] = Field(default=None, nullable=True) # domain the coompany works in
    company_size: int = Field(default=0, nullable=False) # total workforce ofthe company
    owner_id: UUID = Field(foreign_key="user.id", nullable=False) # user_id of the company owner (the one who created the company)