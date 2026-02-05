from app.ai.embeddings.embed_utils import *
from sqlmodel import Session, select
from app.models.job import Job
from app.models.company import Company
from sqlalchemy import text
from datetime import datetime

def embed_jobs(session: Session):
    """
    Generate embeddings for all jobs and store/update them in the database.
    Combines job and company info into semantic vectors for AI search/recommendation.
    """
    jobs=session.exec(select(Job)).all()
    entity_type="job"
    embed_model=get_embed_model()
    for job in jobs:
        company=session.get(Company, job.company_id)
        content=build_job_embedding_content(job, company) # Prepare job + company text
        embedding=embed_model.embed_query(content)  # Generate embedding vector
        entity_id=job.id
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
    session.commit()  # Save all changes