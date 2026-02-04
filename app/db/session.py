"""
Creating SQLAlchemy engine and providing session dependency for api endpoints 
"""

from sqlmodel import Session, create_engine
from typing import Generator
from app.core.config import Config

class DatabaseSession:
    # wraps SQLModel engine and provides session generator
    def __init__(self, engine):
        self.engine = engine
    def get_session(self) -> Generator[Session, None, None]:
        # yield database session
        with Session(self.engine) as session:
            yield session

# creae sqlalchemy engine using database url passed
engine= create_engine(Config.DATABASE_URL, echo=True)

# shared session manager instance across application
db_session_manager = DatabaseSession(engine)