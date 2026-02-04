from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlmodel import Session
from app.db.session import db_session_manager
from app.auth.deps import *
from app.core.config import Config
from app.ai.rag.retriever import retrieve_context
from app.ai.rag.chain import generate_answer
import json
from app.ai.ai_schemas.response import *
from uuid import UUID, uuid4
from app.ai.job_recommender.job_recommender import get_recs
from app.ai.jd_improviser.improviser import get_improved_jd

# router instance for the AI API endpoints.
ai_router=APIRouter(prefix="/ai", tags=["AI-Assistant"])

@ai_router.get("/ask-ai", status_code=status.HTTP_200_OK)
def ai_assistance(session: Session = Depends(db_session_manager.get_session), current_user: User = Depends(get_current_user), search_query: str = Query(..., description="Ask your query to AI Assistant...")):
    retrieved_context=retrieve_context(session, search_query, 5)
    response=generate_answer(retrieved_context, search_query)
    return response

@ai_router.get("/job-recommendations", status_code=status.HTTP_200_OK, response_model=RecommendationResponse)
def get_job_recs(resume_txt: str, current_user: User = Depends(get_current_user), session: Session = Depends(db_session_manager.get_session)):
    matches=get_recs(resume_text, session)
    return matches

@ai_router.get("/improve-job-rec", status_code=status.HTTP_200_OK, response_model=ImprovementResponse)
def get_improved_jd(request: ImprovementRequest, current_user: User = Depends(get_current_user), session: Session = Depends(db_session_manager.get_session)):
    if not is_recruiter(current_user):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Only recruiter can improve JD...")
    improved_jd=get_improved_jd(request.mode, request.description, session)
    return improved_jd