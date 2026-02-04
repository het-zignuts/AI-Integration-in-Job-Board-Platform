from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlmodel import Session
from app.db.session import db_session_manager
from app.auth.deps import *
from app.core.config import Config
from app.ai.rag.retriever import retrieve_context
from app.ai.rag.chain import generate_answer
import json

# router instance for the AI API endpoints.
ai_router=APIRouter(prefix="/ai", tags=["AI-Assistant"])

@ai_router.get("/ask-ai", status_code=status.HTTP_200_OK)
def ai_assistance(session: Session = Depends(db_session_manager.get_session), search_query: str = Query(..., description="Ask your query to AI Assistant...")):
    retrieved_context=retrieve_context(session, search_query, 5)
    response=generate_answer(retrieved_context, search_query)
    return response