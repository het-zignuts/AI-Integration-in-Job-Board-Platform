"""
Factory utilities for generating test payloads.
"""

from uuid import UUID, uuid4
from app.core.config import Config
from app.core.enum import UserRole

# Generate a user payload for registration or login tests.
def user_payload(uname:str|None=None, email:str|None=None, password:str="UserTest@Password123", role:UserRole|None=None, current_organization=None):
    if uname is None:
        uname=str(uuid4().hex)
    if email is None:
        email=f"user_{uname}@pytest.com"
    if role is None:
        role=UserRole.CANDIDATE
    return {
        "user_name": uname,
        "email": email,
        "password": password,
        "role": role,
        "current_organization": current_organization
    }