from pydantic import BaseModel

class Token(BaseModel):
    # BASE TOKEN SCHEMA
    token_type: str = "bearer"

class AccessToken(Token):
    # SCHEMA FOR ACCESS TOKEN RESPONSES, INHERITS TOKEN MODEL
    access_token: str
    refresh_token: str

class RefreshToken(Token):
    # SCHEMA FOR REFRESH TOKEN RESPONSES, INHERITS TOKEN MODEL
    refresh_token: str