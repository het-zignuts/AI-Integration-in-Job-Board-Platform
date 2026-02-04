"""
API endpoints to CRUD users... user can only crud itself... list users only through admin creds.
Role based access control implemented through dependencies...
SQLModel Session passed as dependency
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlmodel import Session
from app.db.session import db_session_manager
from app.auth.deps import *
from app.models.user import User
from app.schemas.user import UserUpdate, UserResponse
from app.crud.user import *
from app.core.enum import UserRole

router = APIRouter(prefix="/users", tags=["Users"]) # router creation for users crud

# User can retrieve its own details through this api.
@router.get("/me", response_model=UserResponse, status_code=status.HTTP_200_OK)
def get_current_user_api(current_user: User = Depends(get_current_user), session: Session=Depends(db_session_manager.get_session)):
    user= get_user_by_id(current_user.id, session) # call to business logic
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    return user

# User can update its own profile from this api
@router.put("/me", response_model=UserResponse, status_code=status.HTTP_200_OK)
def update_current_user_api(user: UserUpdate, current_user: User = Depends(get_current_user), session: Session = Depends(db_session_manager.get_session)):
    updated_user = update_user(current_user.id, user, session) # call to business lpgic
    if not updated_user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    return updated_user

# API Endpoint to delete own's account
@router.delete("/me", status_code=status.HTTP_204_NO_CONTENT)
def delete_current_user_api(current_user: User = Depends(get_current_user), session: Session = Depends(db_session_manager.get_session)):
    success = delete_user(current_user.id, session) # call to busines logic
    if not success:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    return  {
        "success_status": success
    }

# list all users.. only admin can do that...
@router.get("/", response_model=list[UserResponse], status_code=status.HTTP_200_OK)
def list_all_users_api(current_user: User = Depends(get_current_user), session: Session = Depends(db_session_manager.get_session)):
    if not is_admin(current_user):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Insufficient permissions")
    users = list_users(session) # call to crud operation
    return users