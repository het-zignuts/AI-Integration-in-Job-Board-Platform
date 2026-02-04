"""
Security utils for authentication and authorization written in this file.
"""

from passlib.context import CryptContext
from app.core.config import Config
from datetime import datetime, timedelta, timezone
from jose import JWTError, jwt
from enum import Enum
from uuid import uuid4, UUID
from app.models.refreshtoken import RefreshToken
from sqlmodel import Session
from fastapi import HTTPException
from app.models.user import User
from sqlmodel import select

# Password hasing context (bcrypt)
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

class Security:
    # All helper methods defined as a collection of static methods in this class

    # method to hash the plaintext password using bcrypt hashing algorithm 
    @staticmethod
    def hash_password(password: str) -> str:
        return pwd_context.hash(password)

    # verify user's password against stored hash.
    @staticmethod
    def verify_password(email: str, plain_password: str, session: Session) -> bool:
        user=session.exec(select(User).where(User.email==email)).first() # retrieving user
        if not user:
            raise HTTPException(status_code=400, detail="User not found")
        return pwd_context.verify(plain_password, user.password) # verify password using the specified context

    # method to create access token based on the payload passed as the argument ( a dictionary )
    @staticmethod
    def create_access_token(data: dict) -> str:
        payload = data.copy() # copy the data as payload
        expires_at=payload.get('iat') + int(Config.TOKEN_EXPIRY_TIME) * 60 # get expiry time from time of issue of token in payload 
        print("Token issued at (iat):", int(payload.get('iat')))
        payload.update({"exp": int(expires_at)}) # add expiry time to payload
        print("Token expiration time (exp):", int(expires_at))
        payload.update({"type": "access"}) # as we are creating access token, we set the type to access, in payload and update it.
        encoded_jwt = jwt.encode(payload, Config.SECRET_KEY, Config.ALGORITHM) # finally we create the token using jwt.encode(), as per the algorith and key specified in configuration file.
        return encoded_jwt

    # method to verify access token
    @staticmethod
    def verify_access_token(token: str) -> dict | None:
        try:
            payload = jwt.decode(token, Config.SECRET_KEY, algorithms=[Config.ALGORITHM]) # decode the token
            return payload # return the payload if valid
        except JWTError: # else throw exception
            return None

    # method to create refresh token
    @staticmethod
    def create_refresh_token(user_id: UUID, role: str):
        created_at=datetime.now(timezone.utc) # getting the token issue timestamp
        exp_time=created_at + timedelta(days=20) # getting the expiry time
        token_id=str(uuid4()) # creatingthe id of refresh token (jti)
        payload={
            "sub": user_id,
            "token_id": token_id,
            "created_at": int(created_at.timestamp()),
            "exp": int(exp_time.timestamp()),
            "type": "refresh",
            "role": role
        } # generate the payload
        encoded_refresh_jwt=jwt.encode(payload, Config.REFRESH_SECRET_KEY, Config.ALGORITHM) # encode the payload to get the refresh token
        return {
            "ref_token": encoded_refresh_jwt,
            "token_id": token_id,
            "created_at": created_at,
            "exp": exp_time
            } # send a dict containing required details

    # a utility to store refresh token 
    @staticmethod
    def store_refresh_token(token_id: str, exp_time: datetime, user_id: UUID, session: Session):
        token=RefreshToken(token_id=token_id, exp=exp_time, user_id=user_id) # create a model instance
        session.add(token) # adding the token instance
        session.commit() # committing the action 
        session.refresh(token) # refreshing the session to get all latest set fileds of the token