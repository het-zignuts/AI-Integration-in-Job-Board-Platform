from sqlmodel import SQLModel, Field, Relationship
from typing import Optional, TYPE_CHECKING, List
from app.core.enum import UserRole
from datetime import datetime, timezone
from uuid import UUID, uuid4
from sqlalchemy import Enum as SAEnum

if TYPE_CHECKING: # prevent circular import 
    from app.models.application import Application

class User(SQLModel, table=True):
    id: UUID = Field(default_factory=uuid4, primary_key=True, index=True) # id field to generate and store user_id
    user_name: str = Field(index=True, unique=True, nullable=False) # stores a unique user name
    email: str = Field(index=True, unique=True, nullable=False) # store unique user email
    password: str = Field(nullable=False) # user password (hashed)
    role: UserRole = Field(sa_column=SAEnum(UserRole, name="userrole", native_enum=True, validate_strings=True, nullable=False), default=UserRole.CANDIDATE) # user role, default CANDIDATE, takes value from UserRole enum class
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), nullable=False) # timestamp of creation
    updated_at: Optional[datetime] = Field(default=None, nullable=True) # timestamp of update
    current_organization: Optional[UUID] = Field(default=None, nullable=True, foreign_key="company.id") # id of company user is currently associated with
    applications: List["Application"] = Relationship(back_populates="user") # List of applications associated with the user