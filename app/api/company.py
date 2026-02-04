"""
APIs to handle CRUD for companies on the portal.

Only Admin and Recruiter allowed to access them.
Role based access control implemented through dependencies...
SQLModel Session passed as dependency
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlmodel import Session
from uuid import UUID
from app.db.session import db_session_manager
from app.auth.deps import *
from app.models.user import User
from app.core.enum import UserRole
from app.schemas.company import *
from app.crud.company import *
from app.ai.embeddings.embed_company import *

router = APIRouter(prefix="/companies", tags=["Companies"]) # router instance for company APIs

# API for company creation
@router.post("/", response_model=CompanyResponse, status_code=status.HTTP_201_CREATED)
def create_company_api(company: CompanyCreate, current_user: User = Depends(get_current_user), session: Session = Depends(db_session_manager.get_session)):
    if not is_recruiter(current_user) and not is_admin(current_user): # Prevent candidates to create a company
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not allowed to create company")
    created_company = create_company(company, current_user.id, session)
    if created_company is None: # prevent creation of dupicate companies
        raise HTTPException(
            status_code=400,
            detail="Company already exists"
        )
    embed_companies(session)
    return created_company

# Retrive company through id, all users allowed to see companies
@router.get("/{company_id}", response_model=CompanyResponse, status_code=status.HTTP_200_OK)
def get_company_api(company_id: UUID, session: Session = Depends(db_session_manager.get_session)):
    company = get_company_by_id(company_id, session)
    if not company:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Company not found")
    return company

# Updation endpoint, candidate restricted
@router.put("/{company_id}", response_model=CompanyResponse, status_code=status.HTTP_200_OK)
def update_company_api(company_id: UUID, company: CompanyUpdate, current_user: User = Depends(get_current_user), session: Session = Depends(db_session_manager.get_session)):
    if not check_ownership(current_user, company_id, session) and not is_admin(current_user):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Only owner or admin can update company")
    updated_company = update_company(company_id, company, session) # sending new data to be put
    if not updated_company:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Company not found")
    embed_companies(session)
    return updated_company

# Compay Deletion enpoint, resticted to admin and recruiter.
@router.delete("/{company_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_company_api(company_id: UUID, current_user: User = Depends(get_current_user), session: Session = Depends(db_session_manager.get_session)):
    if not check_ownership(current_user, company_id, session) and not is_admin(current_user):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Only owner or admin can delete company")
    success = delete_company(company_id, session)
    if not success:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Company not found")
    embed_companies(session)
    return {
        "success_status": success
    }

# get list of companies, endpoint open to all kinds of users]
@router.get("/", response_model=list[CompanyResponse], status_code=status.HTTP_200_OK)
def list_companies_api(session: Session = Depends(db_session_manager.get_session)):
    companies = list_companies(session)
    return companies
