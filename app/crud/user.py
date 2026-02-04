"""
User CRUD operations.

This module contains database interaction logic for creating,
retrieving, updating, and deleting user records, along with
related authentication data cleanup.
"""

from sqlmodel import Session, select
from typing import Optional
from uuid import UUID
from datetime import datetime, timezone
from app.models.user import User
from app.models.refreshtoken import RefreshToken
from app.schemas.user import UserCreate, UserUpdate, UserResponse
from app.core.security import Security

# business logic for user creation, used by register api
def create_user(session: Session, user: UserCreate) -> UserResponse:
    if (session.exec(select(User).where(User.email == user.email)).first()): # checking if user with email already exists 
        return None # return none if it already does
    hashed_password = Security.hash_password(user.password) # hash the password otherwise using the security util
    user_instance=User(email=user.email, user_name=user.user_name,password=hashed_password, role=user.role, current_organization=user.current_organization) # create a user model instance
    session.add(user_instance) # add user instance to db sesion
    session.commit() # commit the change
    session.refresh(user_instance) # reload instance for latest data
    return UserResponse(id=user_instance.id, user_name=user_instance.user_name, email=user_instance.email, role=user_instance.role, created_at=user_instance.created_at, updated_at=user_instance.updated_at, current_organization=user_instance.current_organization)

# user retrieval by email
def get_user_by_email(session: Session, email: str) -> Optional[UserResponse]:
    user=session.exec(select(User).where(User.email==email)).first()
    if user:
        return UserResponse(id=user.id, user_name=user.user_name, email=user.email, role=user.role, created_at=user.created_at, updated_at=user.updated_at, current_organization=user.current_organization)
    return None

# user retrival through id passed as parameter
def get_user_by_id(user_id: UUID, session: Session) -> Optional[UserResponse]:
    # user=session.exec(select(User).where(User.id==user_id)).first()
    user=session.get(User, user_id)
    if user:
        return UserResponse(id=user.id, user_name=user.user_name, email=user.email, role=user.role, created_at=user.created_at, updated_at=user.updated_at, current_organization=user.current_organization)
    return None

# get the model instance of user, not the pydantic schema 
def get_user_model_instance(user_id: UUID, session: Session) -> Optional[User]:
    user=session.exec(select(User).where(User.id==user_id)).first()
    if user:
        return user
    return None

#  API to list all users
def list_users(session: Session) -> list[UserResponse]: 
    users=session.exec(select(User)).all()
    return [UserResponse(id=user.id, user_name=user.user_name, email=user.email, role=user.role, created_at=user.created_at, updated_at=user.updated_at, current_organization=user.current_organization) for user in users]

# Business Logic to update a user
def update_user(user_id: UUID, new_user: UserUpdate, session: Session) -> Optional[UserResponse]:
    user=session.exec(select(User).where(User.id==user_id)).first()
    if not user:
        return None
    # set only those fields which are passed as request
    if new_user.user_name != None:
        user.user_name=new_user.user_name
    if new_user.email != None:
        user.email=new_user.email
    if new_user.password != None:
        user.password=Security.hash_password(new_user.password)
    if new_user.role != None:
        user.role=new_user.role
    if new_user.current_organization != None:
        user.current_organization=new_user.current_organization
    user.updated_at=datetime.now(timezone.utc) # set updated time for user
    session.add(user)
    session.commit()
    session.refresh(user)
    return UserResponse(id=user.id, user_name=user.user_name, email=user.email, role=user.role, created_at=user.created_at, updated_at=user.updated_at, current_organization=user.current_organization)

# Business logic to delete a user
def delete_user(user_id: UUID, session: Session) -> bool:
    user=session.exec(select(User).where(User.id==user_id)).first() # retrieve user through id
    if not user:
        return False
    session.query(RefreshToken).filter(RefreshToken.user_id == user_id).delete(synchronize_session=False) # delete all refresh tokens belonging to the user
    session.delete(user) # delete the user
    session.commit()
    return True