from app.ai.embeddings.embed_utils import *
from sqlmodel import Session, select
from app.models.company import Company
from app.models.user import User
from app.core.enum import UserRole
from sqlalchemy import text
from datetime import datetime

def embed_jobs(session: Session):
    users=session.exec(select(User).where(User.role==UserRole.CANDIDATE)).all()
    entity_type="candidate_resume"
    embed_model=get_embed_model()
    for user in users:
        applications=session.exec(select(Application).where(Application.user_id==user.id)).all()
        resume_texts=[]
        for appliction in applications:
            text=load_document(app.resume_path)
            if text:
                resume_texts.append(text)
        if not resume_texts:
            continue 
        combined_resume="\n\n".join(resume_texts)
        content=build_candidate_embedding_content(user=user, resume_text=combined_resume)
        embedding=embeddings_model.embed_query(content)
        entity_id=user.id
        existing=session.execute(text("SELECT id FROM embeddings WHERE entity_type = :entity_type AND entity_id = :entity_id"), {"entity_type": entity_type, "entity_id": entity_id}).first()
        if existing:
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
            session.execute(text("INSERT INTO embeddings (entity_type, entity_id, content, embedding) VALUES (:entity_type, :entity_id, :content, :embedding)"),
                {
                    "entity_type": entity_type,
                    "entity_id": entity_id,
                    "content": content,
                    "embedding": embedding
                }
            )
    session.commit()