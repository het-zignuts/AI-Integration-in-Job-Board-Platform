"""
Authentication and authorization dependencies.
Provides:
- JWT-based authentication using HTTP Bearer tokens
- Current user resolution from access tokens
- Role-based access helpers (admin, recruiter, candidate)
- Ownership checks for company-level authorization
"""

from fastapi import Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from app.core.config import Config
from app.core.security import Security
from uuid import UUID
from app.models.user import User
from app.core.enum import UserRole
from app.db.session import db_session_manager
from sqlmodel import Session
import time
from datetime import datetime, timezone
from app.crud.user import *
from app.crud.company import *

# HTTP Bearer Authentication Scheme (Authentication: Bearer <Token>)
auth_header_scheme = HTTPBearer()

def get_cred(session : Session = Depends(db_session_manager.get_session), creds: HTTPAuthorizationCredentials = Depends(auth_header_scheme)):
    token=creds.credentials
    return token

# Retrieve the current user from decoding of JWT access token
def get_current_user(session : Session = Depends(db_session_manager.get_session), creds: HTTPAuthorizationCredentials = Depends(auth_header_scheme)) -> User:
    try:
        print("Inside get_current_user... getting creds")
        token=creds.credentials # get auth credentials from containing JWT tokens from the Header
        print("Token received: " + token)
        print("Now (time.time):", time.time())
        print("Now (utc):", datetime.now(timezone.utc).timestamp())
        print("Token exp:", payload_exp := jwt.get_unverified_claims(token)["exp"])
        print("Diff (exp - now):", payload_exp - time.time())
        payload = jwt.decode(token, Config.SECRET_KEY,algorithms=[Config.ALGORITHM]) # JWT decoding through SECRET key, checking of expiry time, etc.  
        print("Payload decoded: " + str(payload))
        if payload: # check successful decoding
            print("Payload exists...")
            user_id: UUID = UUID(payload.get("sub")) # access user id in sub field
            tkn_type: str|None=payload.get("type") # get token type : access or refresh
            if user_id is None or tkn_type is None:
                print("user id:" + str(user_id)+" this one..")
                if tkn_type != "access": print("token type:" + tkn_type)
                if user_id is None: print("user id is none")
                raise HTTPException(status_code=401, detail="Invalid authentication credentials")
            user=get_user_model_instance(user_id, session) # retrieve user from db
            if not user:
                raise HTTPException(status_code=401, detail="User not found...")
            return user
        else:
            raise HTTPException(status_code=401, detail="Authentication failed...")
    except JWTError as e: # raise JWT Exception for failure in decoding
        print("JWT Error occurred...")
        print("JWT ERROR TYPE:", type(e))
        print("JWT ERROR:", e)
        raise HTTPException(status_code=401, detail="Invalid authentication credentials")

# Authorization - RBAC 

#  check if user is admin
def is_admin(current_user: User = Depends(get_current_user)) -> bool:
    if current_user.role != UserRole.ADMIN:
        return False
    return True

#  check if user is candidate
def is_candidate(current_user: User = Depends(get_current_user)) -> bool:
    if current_user.role != UserRole.CANDIDATE:
        return False
    return True

#  check if user is recruiter
def is_recruiter(current_user: User = Depends(get_current_user)) -> bool:
    if current_user.role != UserRole.RECRUITER:
        return False
    return True

# check if the user owns the company
def check_ownership(current_user: User, company_id: UUID, session: Session) -> bool:
    company=get_company_by_id(company_id, session)
    if current_user.id == company.owner_id:
        return True
    return False
        
  