"""
Database initialization module.

This module is responsible for creating all database tables defined using SQLModel metadata. Invoked once during application startup.
"""

from .session import db_session_manager
from sqlmodel import SQLModel
from app.models.user import User
from app.models.company import Company
from app.models.application import Application
from app.models.job import Job

def init_db():
    # Initializing database. Creating all tables (imported as models) if they dont already exist in database
    SQLModel.metadata.create_all(db_session_manager.engine)