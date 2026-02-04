from app.ai.embeddings.embed utils import *
from sqlmodel import Session, select
from app.models.job import Job
from app.models.company import Company

def embed_jobs(session: Session):
    jobs=session.exec(select(Job)).all()
    entity_type="job"
    for job in jobs:
        company=session.get(Company, job.company_id)
        content=build_job_content(job, company)
        embedding=embed_model.embed_query(content)
        entity_id=job.id
        existing=session.exec(select(Embedding).where(Embedding.entity_type == entity_type, Embedding.entity_id == entity_id)).first()
        if existing:
            existing.content=content
            existing.embedding=embedding
            existing.updated_at=datetime.utcnow()
        else:
            session.add(Embedding(entity_type=entity_type, entity_id=entity_id, content=content, embedding=embedding))
    session.commit()