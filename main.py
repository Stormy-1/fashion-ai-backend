# ============================================================
# main.py — FastAPI Application Entry Point
# ============================================================
# This is the root of the entire backend application.
# FastAPI is chosen over Flask for three key reasons:
#   1. Native async support — no request blocking during LLM calls
#   2. Automatic API documentation at /docs (Swagger UI)
#   3. Built-in request/response validation via Pydantic models
# ============================================================

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.core.config import settings
from app.routers import recommend, health


# ── Lifespan Handler ──────────────────────────────────────
# @asynccontextmanager turns this into a startup/shutdown manager.
# Code before `yield` runs on startup, code after runs on shutdown.
# This is where we load heavy ML models ONCE at startup so that
# every request reuses the same loaded model (not reload per request).
# In the original project, models were reloaded on every API call — wasteful.
@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup ──
    print("🚀 Fashion AI Backend starting up...")
    print(f"   Environment : {settings.ENVIRONMENT}")
    print(f"   API version : {settings.API_VERSION}")

    # We import the model loader here so it only runs at startup.
    # The loaded models are stored in a shared `model_store` dict
    # that all request handlers can access without reloading.
    from app.services.model_service import model_store, load_all_models
    await load_all_models(model_store)
    print("✅ Models loaded and cached in memory")

    yield  # Application runs here — handles all incoming requests

    # ── Shutdown ──
    # Clean up GPU memory when the server stops.
    print("🛑 Shutting down — clearing model cache...")
    model_store.clear()


# ── App Initialization ────────────────────────────────────
# We pass the lifespan handler so FastAPI knows about startup/shutdown.
# title, description, version show up in the auto-generated /docs page.
app = FastAPI(
    title="Fashion AI API",
    description="AI-powered fashion recommendation backend",
    version=settings.API_VERSION,
    lifespan=lifespan,
    # Only show /docs in development — not in production
    docs_url="/docs" if settings.ENVIRONMENT == "development" else None,
    redoc_url=None,
)


# ── CORS Middleware ───────────────────────────────────────
# CORS (Cross-Origin Resource Sharing) allows our React frontend
# (running on localhost:5173 or a deployed domain) to call this API.
# Without this, browsers block cross-origin requests for security.
# In the original Flask app, CORS(app) allowed ALL origins forever —
# that's a security risk in production. Here we restrict it properly.
app.add_middleware(
    CORSMiddleware,
    # settings.ALLOWED_ORIGINS is a list defined in config.py
    # e.g. ["http://localhost:5173"] in dev, ["https://yourapp.com"] in prod
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],   # Only the methods we actually use
    allow_headers=["*"],
)


# ── Router Registration ───────────────────────────────────
# Routers are FastAPI's way of splitting endpoints into separate files.
# Instead of one giant api.py with everything, each feature gets its own file.
# prefix="/api/v1" means all routes in these routers start with /api/v1/...
# This is API versioning — if we ever break the API, we add /api/v2 instead
# of breaking existing clients.
app.include_router(health.router, prefix="/api/v1", tags=["Health"])
app.include_router(recommend.router, prefix="/api/v1", tags=["Recommendations"])


# ── Root Route ────────────────────────────────────────────
@app.get("/")
async def root():
    """Minimal root response — confirms the server is alive."""
    return {
        "message": "Fashion AI API is running",
        "docs": "/docs",
        "version": settings.API_VERSION,
    }
