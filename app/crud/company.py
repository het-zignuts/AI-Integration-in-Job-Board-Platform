"""
CRUD operations for companies.

This module handles creation, retrieval, updating, deletion,
and relationship management of companies.
"""

from sqlmodel import Session, select
from typing import Optional
from uuid import UUID
from datetime import datetime, timezone
from app.models.company import Company
from app.models.user import User
from app.schemas.company import CompanyCreate, CompanyUpdate, CompanyResponse
from app.core.security import Security

# company creation business logic
def create_company(company: CompanyCreate, owner_id: UUID, session: Session) -> CompanyResponse:
    if session.exec(select(Company).where(Company.name == company.name)).first():
        return None
    company_instance=Company(**company.model_dump())
    company_instance.owner_id=owner_id
    session.add(company_instance)
    session.commit()
    session.refresh(company_instance)
    return CompanyResponse(id=company_instance.id, name=company_instance.name, description=company_instance.description, website=company_instance.website, location=company_instance.location, domain=company_instance.domain, company_size=company_instance.company_size, owner_id=company_instance.owner_id)

# company retrieval business logic
def get_company_by_id(company_id: UUID, session: Session) -> Optional[CompanyResponse]:
    company=session.exec(select(Company).where(Company.id==company_id)).first()
    if company:
        return CompanyResponse(id=company.id, name=company.name, description=company.description, website=company.website, location=company.location, domain=company.domain, company_size=company.company_size, owner_id=company.owner_id)
    return None

#  retriving list of companies from the database
def list_companies(session: Session) -> list[CompanyResponse]: 
    companies=session.exec(select(Company)).all()
    return [CompanyResponse(id=company.id, name=company.name, description=company.description, website=company.website, location=company.location, domain=company.domain, company_size=company.company_size, owner_id=company.owner_id) for company in companies]

# updating the company data with new data sent to the update route 
def update_company(company_id: UUID, new_company: CompanyUpdate, session: Session) -> Optional[CompanyResponse]:
    company=session.exec(select(Company).where(Company.id==company_id)).first()
    if not company:
        return None
    company_data = new_company.model_dump()
    for key, value in company_data.items(): # updating each field with the new value 
        setattr(company, key, value)
    session.add(company)
    session.commit()
    session.refresh(company)
    return CompanyResponse(id=company.id, name=company.name, description=company.description, website=company.website, location=company.location, domain=company.domain, company_size=company.company_size, owner_id=company.owner_id)

# company deletion 
def delete_company(company_id: UUID, session: Session) -> bool:
    # setting all employees' current organisation as none
    users = session.exec(select(User).where(User.current_organization == company_id)).all()
    for user in users:
        user.current_organization = None
    company=session.exec(select(Company).where(Company.id==company_id)).first()
    if not company:
        return False
    session.delete(company)
    session.commit()
    return True
