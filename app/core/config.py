"""
Application configuration module.

This module is responsible for:
- Loading environment variables
- Managing environment-specific behavior (development vs production)
- Centralizing all configuration values
- Validating required settings at startup
"""

import os
from dotenv import load_dotenv


# Determine current environment (default: development)
ENV = os.getenv("ENV", "development")


# Load environment variables from `.env` file
# Only used for local development and testing
if ENV != "production":
    load_dotenv()


class Config:
    """
    Central configuration class.

    All application-wide settings should be accessed
    through this class to ensure consistency.
    """

    # JWT secrets
    SECRET_KEY = os.getenv("SECRET_KEY")
    REFRESH_SECRET_KEY = os.getenv("REFRESH_SECRET_KEY")

    # Database connection URLs
    DATABASE_URL = os.getenv("DATABASE_URL")
    TEST_DATABASE_URL = os.getenv("TEST_DATABASE_URL")

    # File upload settings
    UPLOAD_RESUME_DIR = os.getenv("UPLOAD_RESUME_DIR", "uploads/resumes")

    # Token expiry settings
    TOKEN_EXPIRY_TIME = int(os.getenv("TOKEN_EXPIRY_TIME", "30"))  # minutes
    REFRESH_TOKEN_EXPIRY_TIME = int(os.getenv("REFRESH_TOKEN_EXPIRY_TIME", "20"))  # days

    # JWT signing algorithm
    ALGORITHM = os.getenv("ALGORITHM", "HS256")


# Validate required environment variables in production
# Fail fast to prevent misconfigured deployments
if ENV == "production":
    required = [
        "SECRET_KEY",
        "REFRESH_SECRET_KEY",
        "DATABASE_URL",
    ]

    missing = [var for var in required if not os.getenv(var)]
    if missing:
        raise RuntimeError(f"Missing required env vars: {missing}")


# Fix Render / SQLAlchemy Postgres driver prefix
# Render provides `postgresql://` but SQLAlchemy expects
# `postgresql+psycopg2://`
if Config.DATABASE_URL and Config.DATABASE_URL.startswith("postgresql://"):
    Config.DATABASE_URL = Config.DATABASE_URL.replace(
        "postgresql://",
        "postgresql+psycopg2://",
        1,
    )
