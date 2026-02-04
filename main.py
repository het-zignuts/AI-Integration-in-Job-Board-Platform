"""
Main entry point for the Job Board FastAPI application.

"""

from app.db.init_db import init_db
from app.api.user import router as user_router
from app.api.company import router as company_router
from app.api.job import router as job_router
from app.api.application import router as application_router
from app.auth.routes import auth_router  
from app.api.ai_routes import ai_router
from fastapi import FastAPI
from fastapi_pagination import add_pagination

app = FastAPI() # Initializes the FastAPI app

@app.on_event("startup") # Application startup hook.
def on_startup():
    init_db() # Establish database connections on startup

# Registering all API routers
app.include_router(user_router)
app.include_router(company_router)
app.include_router(job_router)
app.include_router(application_router)
app.include_router(auth_router)
app.include_router(ai_router)

# Enable pagination globally for supported endpoints
add_pagination(app)