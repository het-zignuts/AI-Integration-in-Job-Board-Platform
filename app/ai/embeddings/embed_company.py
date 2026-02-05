from app.ai.embeddings.embed_utils import *
from sqlmodel import Session, select
from app.models.company import Company
from sqlalchemy import text
from datetime import datetime

def embed_companies(session: Session):
    """
    Generate embeddings for all companies and upsert them into the database.
    Converts company info into a semantic vector for AI search and reasoning.
    """
    companies=session.exec(select(Company)).all()
    entity_type="company"
    embed_model=get_embed_model()
    for company in companies:
        entity_id=company.id
        content=build_company_embedding_content(company) # Prepare text for embedding
        embedding=embed_model.embed_query(content) # Generate vector
        # Check if embedding already exists
        existing=session.execute(text("SELECT id FROM embeddings WHERE entity_type = :entity_type AND entity_id = :entity_id"), {"entity_type": entity_type, "entity_id": entity_id}).first()
        if existing:
             # Update existing embedding
            session.execute(text("UPDATE embeddings SET content = :content, embedding = :embedding, updated_at = :updated_at WHERE entity_type = :entity_type AND entity_id = :entity_id"), 
                {
                    "content": content,
                    "embedding": embedding,
                    "updated_at": datetime.utcnow(),
                    "entity_type": entity_type,
                    "entity_id": entity_id
                }
            )
        else:
            # Insert new embedding
            session.execute(text("INSERT INTO embeddings (entity_type, entity_id, content, embedding) VALUES (:entity_type, :entity_id, :content, :embedding)"),
                {
                    "entity_type": entity_type,
                    "entity_id": entity_id,
                    "content": content,
                    "embedding": embedding
                }
            )
    session.commit() # Save all changes to DB