# ============================================================
# app/routers/health.py — Health Check Endpoint
# ============================================================
# A health check endpoint is a standard practice in any API.
# Its job: confirm the server is alive AND that critical dependencies
# (models, API keys) are loaded and ready.
#
# This is useful for:
#   - Debugging: "is the server running?" before testing other endpoints
#   - Colab: quick confirm that everything loaded correctly at startup
#   - Production: load balancers ping /health to decide if traffic should route here
# ============================================================

from fastapi import APIRouter
from app.services.model_service import model_store
from app.core.config import settings

# APIRouter is FastAPI's equivalent of Flask's Blueprint.
# It's a mini-app that holds a group of related routes.
# In main.py we attach it with app.include_router(health.router)
router = APIRouter()


@router.get("/health")
async def health_check():
    """
    Returns the operational status of the API and all its dependencies.
    A 200 response means everything is ready to serve recommendations.
    """

    # Check which models have been successfully loaded into model_store.
    # model_store is a dict populated during startup in model_service.py.
    # If a model failed to load, its key won't be in model_store.
    models_loaded = {
        "age_gender_model":     "age_gender" in model_store,
        "facial_feature_model": "facial_features" in model_store,
        "age_scaler":           "age_scaler" in model_store,
    }

    # Check that API keys are configured (not empty strings).
    # We check length > 10 to avoid flagging obviously wrong placeholder values.
    api_keys_configured = {
        "gemini":  len(settings.GOOGLE_API_KEY) > 10,
        "serpapi": len(settings.SERPAPI_KEY) > 10,
    }

    # Overall status: everything must be green for "healthy"
    all_models_ready = all(models_loaded.values())
    all_keys_ready   = all(api_keys_configured.values())
    overall_status   = "healthy" if (all_models_ready and all_keys_ready) else "degraded"

    return {
        "status":              overall_status,
        "environment":         settings.ENVIRONMENT,
        "api_version":         settings.API_VERSION,
        "models":              models_loaded,
        "api_keys_configured": api_keys_configured,
    }
