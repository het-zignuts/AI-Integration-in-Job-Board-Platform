from sqlmodel import SQLModel, Field
from uuid import UUID, uuid4
from datetime import datetime, timedelta, timezone
from app.core.config import Config

# Model for storing refresh token metadata
class RefreshToken(SQLModel, table=True):
    id: UUID=Field(default_factory=uuid4, primary_key=True, index=True) # id of the model instance
    token_id: str=str(uuid4()) # jti (jwt token id ) stored
    exp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc) + timedelta(days=Config.REFRESH_TOKEN_EXPIRY_TIME), nullable=False) # time stamp of token expiry
    user_id: UUID = Field(foreign_key="user.id", nullable=False) # user_id for the user who created the tokens
    revoked: bool=Field(default=False) #boolean ield to check whether the token is recvoked or not