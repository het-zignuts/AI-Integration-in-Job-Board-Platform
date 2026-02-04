from sqlmodel import SQLModel, Field
from sqlalchemy import Column
from pgvector.sqlalchemy import Vector
from uuid import UUID, uuid4
from datetime import datetime

class Embedding(SQLModel, table=True):
    __tablename__ = "embeddings"
    id: UUID = Field(default_factory=uuid4, primary_key=True)
    entity_type: str = Field(index=True, nullable=False)
    entity_id: UUID = Field(index=True, nullable=False)
    content: str = Field(nullable=False)
    embedding: list[float] = Field(
        sa_column=Column(Vector(384), nullable=False)
    )
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime | None = Field(default=None)
