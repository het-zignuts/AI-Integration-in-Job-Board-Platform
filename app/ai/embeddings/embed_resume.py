from app.ai.embeddings.embed utils import *
from sqlmodel import Session, select
from app.models.company import Company
from app.models.user import User
from app.core.enum import UserRole

def embed_jobs(session: Session):
    users=session.exec(select(User).where(User.role==UserRole.CANDIDATE)).all()
    entity_type="candidate_resume"
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
        existing=session.exec(select(Embedding).where(Embedding.entity_type == entity_type, Embedding.entity_id == entity_id)).first()
        if existing:
            existing.content=content
            existing.embedding=embedding
            existing.updated_at=datetime.utcnow()
        else:
            session.add(Embedding(entity_type=entity_type, entity_id=entity_id, content=content, embedding=embedding))
    session.commit()