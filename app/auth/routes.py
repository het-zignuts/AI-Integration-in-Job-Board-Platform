from fastapi import APIRouter, Depends, HTTPException, status
from sqlmodel import Session
from app.db.session import db_session_manager
from app.crud.user import *
from app.auth.deps import get_current_user
from app.schemas.user import UserCreate, UserResponse
from app.schemas.token import AccessToken, RefreshToken
from app.core.security import Security
from app.core.config import Config
from jose import jwt, JWTError
from app.models.refreshtoken import RefreshToken as RefreshTokenModel
from app.core.config import Config
import time
from uuid import UUID
"""
User creation and authentication routes...
Session passes as dependency.
"""


# Create router for authentication routes
auth_router = APIRouter(prefix="/auth", tags=["auth"])

# User Registration (creation) API
@auth_router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
def register_user(user: UserCreate, session: Session = Depends(db_session_manager.get_session)):
    user= create_user(session, user) #call to business logic for user creation
    if not user:
        raise HTTPException(status_code=400, detail="User already registered...")
    return user

# User Login Route
@auth_router.post("/login", response_model=AccessToken, status_code=status.HTTP_200_OK)
def login_user(user: UserCreate, session: Session = Depends(db_session_manager.get_session)):
    if Security.verify_password(user.email, user.password, session): # password verification through security util.
        user_db=get_user_by_email(session, user.email) # retrieve user through email passed at the time of login
        if not user_db:
            raise HTTPException(status_code=400, detail="User not found, please register")
        access_token = Security.create_access_token({"sub": str(user_db.id), "role": str(user_db.role), "iat": time.time()}) # create access token through call to approriate security util.
        print("LOGIN TOKEN ISSUED AT:", int(time.time()))
        refresh_jwt_token = Security.create_refresh_token(str(user_db.id), user_db.role) # creation of refresh token
        Security.store_refresh_token(token_id=refresh_jwt_token["token_id"], exp_time=refresh_jwt_token["exp"], user_id=user_db.id, session=session) # storing the refreah token metadata through proper util call. 
        refresh_token=refresh_jwt_token["ref_token"]
        return {"access_token": access_token, "refresh_token": refresh_token, "token_type": "bearer"} # return auth tokens as succesful login response
    else:
        raise HTTPException(status_code=401, detail="Invalid password") # raise exception for invalid password.

# token refresh route
@auth_router.post("/refresh", response_model=AccessToken, status_code=status.HTTP_200_OK)
def refresh_access_token(refresh_token: RefreshToken, session: Session = Depends(db_session_manager.get_session)):
    token_data = jwt.decode(refresh_token.refresh_token, Config.REFRESH_SECRET_KEY, algorithms=[Config.ALGORITHM]) # decoding of refresh token sent by client
    if not token_data or token_data.get("type") != "refresh": # raise exfeption for failure in decoding
        raise HTTPException(status_code=401, detail="Invalid token")
    ref_token_db= session.exec(select(RefreshTokenModel).where(RefreshTokenModel.token_id == token_data["token_id"])).first() # access stored refresh token metadata from db based on decoded creds...
    if not ref_token_db: # failure in access raises token invalid or revoked exception
        raise HTTPException(status_code=401, detail="Refresh token not found or revoked") 
    new_access_token = Security.create_access_token({"sub": token_data["sub"], "role": token_data["role"], "iat": time.time()}) # else create new access token
    new_refresh_jwt_token = Security.create_refresh_token(token_data["sub"], token_data["role"]) # create refresh token
    Security.store_refresh_token(token_id=new_refresh_jwt_token["token_id"], exp_time=new_refresh_jwt_token["exp"], user_id=UUID(token_data["sub"]), session=session) # store new refesh token metadata
    session.delete(ref_token_db) # delete old ref token
    session.commit()
    return {"access_token": new_access_token, "refresh_token": new_refresh_jwt_token["ref_token"], "token_type": "bearer"} # send new auth tokens