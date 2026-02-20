# ============================================================
# app/routers/recommend.py — The Main Recommendation Pipeline
# ============================================================
# This is the orchestrator. It:
#   1. Receives the uploaded image + user inputs
#   2. Runs ML inference (age/gender + facial features) concurrently
#   3. Calls Gemini for outfit recommendations
#   4. Searches for real products concurrently
#   5. Packages everything into a typed response
#
# The original api.py did all of this synchronously in one giant
# function, wrote intermediate results to JSON files, then read them
# back. This version keeps everything in memory and uses async
# concurrency to cut total processing time significantly.
#
# PROCESSING TIME COMPARISON (estimated):
#   Original:  model load + inference + LLM + scraping = 45-90 seconds (sequential)
#   This file: inference (concurrent) + LLM + shopping (concurrent) = ~15-25 seconds
# ============================================================

import asyncio
import time
import logging
from fastapi import APIRouter, File, Form, UploadFile, HTTPException

from app.models.schemas import (
    RecommendationResponse,
    UserProfile,
    FacialFeatures,
)
from app.services.predict_service import (
    run_age_gender_prediction,
    run_facial_feature_prediction,
)
from app.services.llm_service import get_outfit_recommendations
from app.services.shopping_service import get_products_for_outfits
from app.core.config import settings

logger = logging.getLogger(__name__)
router = APIRouter()


def _validate_image_file(file: UploadFile) -> None:
    """
    Validates the uploaded file before processing.
    Raises HTTPException if validation fails.
    """
    allowed_content_types = {
        "image/jpeg", "image/jpg", "image/png",
        "image/webp", "image/bmp"
    }
    if file.content_type not in allowed_content_types:
        raise HTTPException(
            status_code=400,
            detail=f"File type '{file.content_type}' not supported. Use JPEG, PNG, or WebP."
        )


@router.post(
    "/recommend",
    response_model=RecommendationResponse,
    summary="Get personalized fashion recommendations",
    description="Upload a face image and provide body measurements to receive AI-powered outfit recommendations with real product links.",
)
async def get_recommendations(
    image: UploadFile = File(..., description="Face photo (JPEG, PNG, or WebP)"),
    height: float = Form(..., ge=100, le=250, description="Height in cm"),
    weight: float = Form(..., ge=30,  le=300, description="Weight in kg"),
    occasion: str = Form(default="casual", description="Event occasion"),
):
    """
    The complete fashion recommendation pipeline.

    Accepts multipart/form-data with image + body measurements.
    Returns outfit recommendations with real products.
    """
    start_time = time.time()

    # ── Step 0: Validate ──────────────────────────────────
    _validate_image_file(image)

    allowed_occasions = {
        "casual", "formal", "party", "sports",
        "traditional", "beach", "work", "date"
    }
    occasion_lower = occasion.lower().strip()
    if occasion_lower not in allowed_occasions:
        raise HTTPException(
            status_code=400,
            detail=f"Occasion must be one of: {', '.join(sorted(allowed_occasions))}"
        )

    # ── Step 1: Read image into memory ───────────────────
    # Read all bytes once. Pass to both models — no disk I/O needed.
    image_bytes = await image.read()

    if len(image_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    max_bytes = settings.MAX_FILE_SIZE_MB * 1024 * 1024
    if len(image_bytes) > max_bytes:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Maximum size is {settings.MAX_FILE_SIZE_MB}MB"
        )

    logger.info(
        f"Processing: height={height}cm weight={weight}kg "
        f"occasion={occasion_lower} size={len(image_bytes)/1024:.1f}KB"
    )

    # ── Step 2: Run both ML models concurrently ───────────
    # Age/gender and facial features are independent — run simultaneously.
    # asyncio.gather() runs both coroutines concurrently.
    # Time = max(model_A_time, model_B_time) instead of sum.
    logger.info("Running ML inference concurrently...")
    (age, gender, _), (facial_probs, facial_readable) = await asyncio.gather(
        run_age_gender_prediction(image_bytes),
        run_facial_feature_prediction(image_bytes),
    )

    bmi = round(weight / ((height / 100) ** 2), 2)
    logger.info(f"Inference done: {age}yo {gender} BMI={bmi} features='{facial_readable}'")

    # ── Step 3: Gemini outfit recommendations ────────────
    # Passes human-readable features ("black hair, oval face")
    # instead of raw float scores — Gemini reasons about these properly.
    logger.info("Calling Gemini...")
    outfits = await get_outfit_recommendations(
        age=age, gender=gender, height=height, weight=weight,
        bmi=bmi, occasion=occasion_lower,
        facial_features_readable=facial_readable,
    )

    if not outfits:
        raise HTTPException(
            status_code=503,
            detail="Could not generate outfit recommendations. Please try again."
        )

    # ── Step 4: Product search (concurrent per outfit) ───
    # 3 SerpApi searches run simultaneously — ~3s instead of ~9s.
    logger.info("Searching products...")
    products = await get_products_for_outfits(outfits, max_per_outfit=4)

    # ── Step 5: Return typed response ────────────────────
    elapsed = round(time.time() - start_time, 2)
    logger.info(f"Done in {elapsed}s — {len(outfits)} outfits, {len(products)} products")

    return RecommendationResponse(
        success=True,
        user_profile=UserProfile(
            height=height, weight=weight, bmi=bmi,
            occasion=occasion_lower, age=age, gender=gender,
        ),
        facial_features=FacialFeatures(
            raw_scores=facial_probs,
            readable_summary=facial_readable,
        ),
        outfits=outfits,
        products=products,
        total_products=len(products),
        processing_time_seconds=elapsed,
    )
