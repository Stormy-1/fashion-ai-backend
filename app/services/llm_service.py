# ============================================================
# app/services/llm_service.py — Gemini Fashion Recommendation Service
# ============================================================
# This replaces the original llm.py completely.
#
# What changed and why:
#
#   ORIGINAL: LangChain + zephyr-7b-beta + regex to parse output
#     - zephyr-7b is a general model, not great at structured output
#     - LangChain added a heavy dependency for a single prompt call
#     - Regex parsing broke on any formatting variation from the model
#     - Raw float scores passed to LLM ("jawline_strength: 0.81")
#
#   NEW: google-generativeai SDK + Gemini Flash + JSON mode
#     - Gemini is significantly better at following format instructions
#     - JSON mode guarantees parseable output — no regex needed
#     - Direct SDK call, no middleware
#     - Natural language features ("black hair, oval face") passed instead
#     - Richer outfit schema: includes style_notes and color reasoning
# ============================================================

import json
import logging
import google.generativeai as genai
from typing import List

from app.core.config import settings
from app.models.schemas import OutfitRecommendation

logger = logging.getLogger(__name__)

# ── Gemini Client Initialization ──────────────────────────
# Configure the SDK with our API key once at module import time.
# All calls to genai.GenerativeModel() after this will use this key.
genai.configure(api_key=settings.GOOGLE_API_KEY)

# GenerativeModel is the client for a specific model version.
# We specify the model here, not per-call, for consistency.
# "gemini-2.0-flash" — fast, cheap, excellent instruction following.
# If you have 2.5 Pro access, change to "gemini-2.5-pro-preview-0325"
_gemini_client = genai.GenerativeModel(settings.GEMINI_MODEL)


def _build_system_prompt() -> str:
    """
    The system prompt tells Gemini its role and output rules.
    Separated from the user prompt so the model treats it as
    persistent instructions rather than part of the conversation.
    """
    return """You are an expert fashion stylist AI for a premium fashion recommendation platform.
Your job is to recommend outfits based on a user's physical profile and detected facial features.

CRITICAL RULES:
1. Always respond with VALID JSON only — no markdown, no explanation, no preamble
2. The JSON must match the exact schema provided
3. Recommend exactly 3 outfits
4. Color choices must genuinely complement the user's detected skin tone and hair color
5. Fit recommendations must account for the user's BMI
6. Occasion must strongly influence the style (formal ≠ casual ≠ party)
7. Style notes should be specific, not generic ("choose navy blue to complement your black hair" not "looks good")"""


def _build_user_prompt(
    age: int,
    gender: str,
    height: float,
    weight: float,
    bmi: float,
    occasion: str,
    facial_features_readable: str,
) -> str:
    """
    Builds the per-request user prompt with all user data injected.
    Uses f-string formatting — cleaner than LangChain's PromptTemplate
    for a single prompt with no complex conditional logic.

    Args:
        facial_features_readable: The human-readable string from
            predict_service.features_to_human_readable()
            e.g. "black hair, oval face, arched eyebrows, full lips"
    """

    # BMI category helps the LLM suggest appropriate fits without
    # us having to hard-code fit rules ourselves
    if bmi < 18.5:
        bmi_category = "underweight — suggest structured/layered fits to add visual volume"
    elif bmi < 25:
        bmi_category = "healthy weight — most fits work well, can experiment with all silhouettes"
    elif bmi < 30:
        bmi_category = "overweight — suggest relaxed/straight fits, avoid very slim cuts"
    else:
        bmi_category = "obese — suggest oversized/relaxed fits with vertical patterns to elongate"

    return f"""Recommend 3 outfits for this user:

USER PROFILE:
- Age: {age} years old
- Gender: {gender}
- Height: {height} cm
- Weight: {weight} kg
- BMI: {bmi:.1f} ({bmi_category})
- Occasion: {occasion}
- Detected facial features: {facial_features_readable}

Respond with this EXACT JSON structure (no other text):
{{
  "outfits": [
    {{
      "product_name": "Specific searchable product name (e.g. 'Slim Fit Cotton Oxford Shirt')",
      "color_palette": ["Color1", "Color2", "Color3"],
      "fit": "Fit type and size (e.g. 'Slim Fit - M')",
      "gender": "{gender}",
      "style_notes": "1-2 sentences explaining WHY this works for this specific user's features and occasion",
      "search_query": "Short Google Shopping search query to find this product (e.g. 'slim fit oxford shirt navy men')"
    }},
    {{ ... }},
    {{ ... }}
  ]
}}

The 'search_query' field is critical — it will be used to find real products.
Make it specific enough to return relevant results but not so narrow it returns nothing."""


async def get_outfit_recommendations(
    age: int,
    gender: str,
    height: float,
    weight: float,
    bmi: float,
    occasion: str,
    facial_features_readable: str,
) -> List[OutfitRecommendation]:
    """
    Calls Gemini to generate 3 outfit recommendations.

    Returns:
        List of OutfitRecommendation objects (validated Pydantic models).
        Returns an empty list if Gemini call fails — caller handles fallback.

    The `search_query` field in the raw Gemini response is extracted here
    and passed through as part of the outfit data. The shopping service
    uses these queries instead of the full product_name for better
    search results.
    """
    system_prompt = _build_system_prompt()
    user_prompt   = _build_user_prompt(
        age, gender, height, weight, bmi, occasion, facial_features_readable
    )

    try:
        logger.info(f"Calling Gemini ({settings.GEMINI_MODEL}) for {gender}, {age}yo, {occasion}")

        # generation_config controls the model's output behavior:
        #   response_mime_type="application/json" — JSON mode. The model
        #     is constrained to output valid JSON. This replaces all regex parsing.
        #   temperature=0.7 — some creativity for varied recommendations,
        #     but not so high that the format breaks. Original used 0.0 (too rigid).
        #   max_output_tokens=1024 — 3 outfits with style notes fits comfortably.
        response = await _gemini_client.generate_content_async(
            contents=[
                {"role": "user", "parts": [system_prompt + "\n\n" + user_prompt]}
            ],
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json",
                temperature=0.7,
                max_output_tokens=1024,
            ),
        )

        # With JSON mode, response.text is guaranteed valid JSON.
        # No regex, no cleanup, no strip() hacks.
        raw_json = json.loads(response.text)
        raw_outfits = raw_json.get("outfits", [])

        if not raw_outfits:
            logger.warning("Gemini returned empty outfits list")
            return []

        # Validate each outfit against our Pydantic schema.
        # If Gemini hallucinates an extra field or misnames one,
        # Pydantic catches it here rather than crashing downstream.
        outfits = []
        for raw in raw_outfits[:3]:  # Cap at 3 even if model returns more
            try:
                outfit = OutfitRecommendation(
                    product_name  = raw.get("product_name", "Fashion Item"),
                    color_palette = raw.get("color_palette", []),
                    fit           = raw.get("fit", "Regular Fit"),
                    gender        = raw.get("gender", gender),
                    style_notes   = raw.get("style_notes"),
                    # Pass search_query through as extra data on the model
                    # We'll use it in shopping_service.py
                    search_query  = raw.get("search_query", raw.get("product_name", "")),
                )
                outfits.append(outfit)
            except Exception as e:
                logger.warning(f"Skipping malformed outfit from Gemini: {e}")
                continue

        logger.info(f"Gemini returned {len(outfits)} valid outfit recommendations")
        return outfits

    except json.JSONDecodeError as e:
        # This shouldn't happen with JSON mode enabled, but handle defensively
        logger.error(f"Gemini response was not valid JSON: {e}")
        return []

    except Exception as e:
        logger.error(f"Gemini API call failed: {e}")
        return []
