# ============================================================
# app/core/config.py — Application Configuration
# ============================================================
# All configuration lives here. Never hardcode values like API keys,
# model paths, or environment flags directly in your code files.
# Why? Because when you switch from dev to production (or Colab to
# a real server), you only change ONE file or environment variable,
# not hunt through 10 different files.
#
# We use Pydantic's BaseSettings which automatically reads values
# from environment variables AND a .env file. Order of priority:
#   1. Actual environment variables (highest priority)
#   2. Values in the .env file
#   3. Default values defined here (fallback)
# ============================================================

from pydantic_settings import BaseSettings
from typing import List


class Settings(BaseSettings):

    # ── Environment ───────────────────────────────────────
    # Tells the app whether it's running locally, in Colab, or deployed.
    # Used to toggle features like /docs visibility, debug logging, etc.
    ENVIRONMENT: str = "development"

    # ── API ───────────────────────────────────────────────
    API_VERSION: str = "1.0.0"

    # ALLOWED_ORIGINS defines which frontend URLs can call this API.
    # In development, it's localhost. In production, your deployed domain.
    # The List[str] type means .env should have comma-separated values:
    #   ALLOWED_ORIGINS=http://localhost:5173,https://myfashionapp.com
    ALLOWED_ORIGINS: List[str] = [
        "http://localhost:5173",   # Vite dev server default port
        "http://localhost:3000",   # Alternative React dev port
    ]

    # ── Google Gemini ─────────────────────────────────────
    # Your Google AI API key. Never commit this to GitHub.
    # It's read from the .env file at runtime.
    # We use Gemini 2.0 Flash — fast, cheap, excellent at structured JSON.
    GOOGLE_API_KEY: str = ""
    GEMINI_MODEL: str = "gemini-2.0-flash"

    # ── ML Models ─────────────────────────────────────────
    # Paths to the trained .pth files. In Colab, these will point to
    # Google Drive mount paths. Locally, they're relative to the project root.
    AGE_GENDER_MODEL_PATH: str = "model_weights/age_gender_model.pth"
    FACIAL_FEATURE_MODEL_PATH: str = "model_weights/celeba_imbalance_aware_classifier.pth"
    AGE_SCALER_PATH: str = "model_weights/age_scaler.pkl"

    # ── Shopping API ──────────────────────────────────────
    # We'll use SerpApi's Google Shopping endpoint.
    # Free tier: 100 searches/month — enough for development and demo.
    # Sign up at serpapi.com — it's free, no credit card needed initially.
    SERPAPI_KEY: str = ""

    # ── File Upload ───────────────────────────────────────
    UPLOAD_FOLDER: str = "uploads"
    MAX_FILE_SIZE_MB: int = 10          # Slightly tighter than original's 16MB
    ALLOWED_EXTENSIONS: List[str] = ["jpg", "jpeg", "png", "webp"]

    # ── Pydantic Settings Config ──────────────────────────
    # This tells BaseSettings to look for a .env file in the project root.
    # `extra = "ignore"` means unknown .env variables don't cause errors.
    model_config = {
        "env_file": ".env",
        "extra": "ignore",
    }


# ── Singleton Instance ────────────────────────────────────
# We create ONE instance of Settings here.
# Every other file imports THIS instance — they don't create their own.
# This ensures all config reads from the same source.
#
# Usage in other files:
#   from app.core.config import settings
#   print(settings.GOOGLE_API_KEY)
settings = Settings()
