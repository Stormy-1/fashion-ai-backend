# ============================================================
# app/models/schemas.py — Request & Response Data Models
# ============================================================
# Pydantic models define the exact shape of data coming IN to the API
# and going OUT of the API. This is one of FastAPI's biggest advantages
# over Flask — validation happens automatically before your code runs.
#
# If a request is missing `height` or sends `height: "tall"`, FastAPI
# rejects it with a clear error message before it ever reaches your logic.
# In the original Flask app, this validation was written manually.
# ============================================================

from pydantic import BaseModel, Field, field_validator
from typing import List, Optional


# ── Request Models ────────────────────────────────────────
# These define what the frontend must send to the API.

class RecommendationRequest(BaseModel):
    """
    The data the frontend sends when requesting fashion recommendations.
    All fields are validated automatically by FastAPI before our code runs.
    """

    # Field() lets us add metadata: description shows in /docs, ge/le set numeric bounds.
    # ge = greater than or equal, le = less than or equal
    height: float = Field(
        ...,                          # `...` means required (no default)
        ge=100, le=250,               # Valid human heights in cm
        description="User height in centimeters",
        example=170.0
    )

    weight: float = Field(
        ...,
        ge=30, le=300,                # Valid human weights in kg
        description="User weight in kilograms",
        example=65.0
    )

    # Optional with a default — if frontend doesn't send it, we use "casual"
    occasion: str = Field(
        default="casual",
        description="Occasion for the outfit",
        example="formal"
    )

    # Validator runs after the field is parsed.
    # Ensures occasion is one of our accepted values.
    # In the original, invalid occasion values silently passed through to the LLM.
    @field_validator("occasion")
    @classmethod
    def validate_occasion(cls, v: str) -> str:
        allowed = {
            "casual", "formal", "party", "sports",
            "traditional", "beach", "work", "date"
        }
        v_lower = v.lower().strip()
        if v_lower not in allowed:
            raise ValueError(f"Occasion must be one of: {', '.join(sorted(allowed))}")
        return v_lower


# ── Response Models ───────────────────────────────────────
# These define the exact structure of what the API sends back.
# The frontend can rely on these shapes without defensive checks.

class FacialFeatures(BaseModel):
    """Human-readable facial features extracted from the image."""
    # We store the raw probability scores AND a human-readable list.
    # Raw scores are useful for debugging; readable list is what the LLM sees.
    raw_scores:       dict              # e.g. {"Black_Hair": 0.91, "Oval_Face": 0.73}
    readable_summary: str               # e.g. "black hair, oval face, arched eyebrows"


class UserProfile(BaseModel):
    """Computed user profile combining image analysis + user inputs."""
    height:   float
    weight:   float
    bmi:      float
    occasion: str
    age:      int
    gender:   str


class OutfitRecommendation(BaseModel):
    """A single outfit recommendation from Gemini."""
    product_name:  str
    color_palette: List[str]           # ["Cream", "Gold", "Maroon"]
    fit:           str                 # "Regular Fit - M"
    gender:        str
    style_notes:   Optional[str] = None  # Extra context Gemini might add


class ShoppingProduct(BaseModel):
    """A real product found via the Shopping API."""
    title:       str
    brand:       Optional[str] = None
    price:       Optional[str] = None
    rating:      Optional[float] = None
    reviews:     Optional[int] = None
    image_url:   Optional[str] = None
    product_url: Optional[str] = None
    source:      Optional[str] = None   # Which platform (e.g. "Amazon", "Flipkart")


class RecommendationResponse(BaseModel):
    """
    The complete response sent back to the frontend.
    Every field is typed so the frontend TypeScript types can match exactly.
    """
    success:          bool
    user_profile:     UserProfile
    facial_features:  FacialFeatures
    outfits:          List[OutfitRecommendation]   # 3 AI-generated outfit ideas
    products:         List[ShoppingProduct]         # Real products from shopping API
    total_products:   int
    processing_time_seconds: float                  # Useful for debugging slowness


class ErrorResponse(BaseModel):
    """Standard error response shape."""
    success: bool = False
    error:   str
    detail:  Optional[str] = None
