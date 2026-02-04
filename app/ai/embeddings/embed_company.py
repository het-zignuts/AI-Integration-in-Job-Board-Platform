from app.ai.embeddings.embed utils import *
from sqlmodel import Session, select
from app.models.company import Company

def embed_companies(session: Session):
    companies=session.exec(select(Company)).all()
    entity_type="company"
    for compnay in companies:
        content=build_company_content(company)
        embedding=embed_model.embed_query(content)
        entity_id=company.id
        existing=session.exec(select(Embedding).where(Embedding.entity_type == entity_type, Embedding.entity_id == entity_id)).first()
        if existing:
            existing.content=content
            existing.embedding=embedding
            existing.updated_at=datetime.utcnow()
        else:
            session.add(Embedding(entity_type=entity_type, entity_id=entity_id, content=content, embedding=embedding))
    session.commit()