# ============================================================
# app/services/shopping_service.py — Product Discovery via SerpApi
# ============================================================
# This replaces the original web_scrapping.py completely.
#
# Why SerpApi instead of scraping:
#   - Scraping breaks every time a site updates its HTML structure
#   - Scraping violates most e-commerce sites' Terms of Service
#   - SerpApi is a paid API service that wraps Google Shopping legally
#   - Free tier gives 100 searches/month — enough for dev + demo
#   - Returns structured JSON — no HTML parsing, no fragile selectors
#
# Design: For each of the 3 outfits Gemini recommends, we run ONE
# search using the outfit's `search_query` field. Results are tagged
# with their `outfit_index` so the frontend can group them visually.
# ============================================================

import asyncio
import logging
from typing import List

import httpx

from app.core.config import settings
from app.models.schemas import OutfitRecommendation, ShoppingProduct

logger = logging.getLogger(__name__)

# SerpApi Google Shopping endpoint
SERPAPI_ENDPOINT = "https://serpapi.com/search"


async def _search_single_outfit(
    client: httpx.AsyncClient,
    outfit: OutfitRecommendation,
    outfit_index: int,
    max_results: int = 4,
) -> List[ShoppingProduct]:
    """
    Searches Google Shopping for one outfit recommendation.

    We use the `search_query` field that Gemini generated — it's
    specifically crafted to be a good shopping search (concise, specific).
    Falls back to product_name if search_query is missing.

    Args:
        client:       Shared async HTTP client (connection pooling)
        outfit:       The outfit recommendation from Gemini
        outfit_index: 0, 1, or 2 — which outfit this belongs to
        max_results:  How many products to return per outfit

    Returns:
        List of ShoppingProduct objects, tagged with outfit_index
    """
    query = outfit.search_query or outfit.product_name

    # SerpApi parameters for Google Shopping
    # Full docs: https://serpapi.com/google-shopping-api
    params = {
        "engine":   "google_shopping",   # Use Google Shopping specifically
        "q":        query,
        "api_key":  settings.SERPAPI_KEY,
        "num":      max_results,
        "hl":       "en",                # Language: English
        "gl":       "in",                # Country: India (matches target audience)
        # "gl": "us" — change to "us" for US products
    }

    try:
        logger.info(f"Shopping search [{outfit_index}]: '{query}'")
        response = await client.get(SERPAPI_ENDPOINT, params=params, timeout=10.0)
        response.raise_for_status()  # Raises exception for 4xx/5xx responses

        data = response.json()

        # SerpApi returns results under "shopping_results" key
        # Each result has: title, price, source, rating, reviews, thumbnail, link
        raw_results = data.get("shopping_results", [])

        products = []
        for item in raw_results[:max_results]:
            # Extract rating — SerpApi returns it as a float string like "4.2"
            rating = None
            if item.get("rating"):
                try:
                    rating = float(item["rating"])
                except (ValueError, TypeError):
                    pass

            # Extract review count — SerpApi returns it as int or string like "1,234"
            reviews = None
            if item.get("reviews"):
                try:
                    reviews = int(str(item["reviews"]).replace(",", ""))
                except (ValueError, TypeError):
                    pass

            product = ShoppingProduct(
                title       = item.get("title", "Unknown Product"),
                brand       = item.get("source"),       # "source" = brand/store name
                price       = item.get("price"),         # Already formatted: "₹1,199"
                rating      = rating,
                reviews     = reviews,
                image_url   = item.get("thumbnail"),     # Product image URL
                product_url = item.get("link"),           # Direct product link
                source      = item.get("source"),
                outfit_index = outfit_index,             # Tag with which outfit
            )
            products.append(product)

        logger.info(f"Found {len(products)} products for outfit {outfit_index}")
        return products

    except httpx.TimeoutException:
        logger.warning(f"Shopping search timed out for outfit {outfit_index}: '{query}'")
        return []
    except httpx.HTTPStatusError as e:
        logger.error(f"SerpApi returned {e.response.status_code} for outfit {outfit_index}")
        return []
    except Exception as e:
        logger.error(f"Shopping search failed for outfit {outfit_index}: {e}")
        return []


async def get_products_for_outfits(
    outfits: List[OutfitRecommendation],
    max_per_outfit: int = 4,
) -> List[ShoppingProduct]:
    """
    Searches for real products for all outfit recommendations concurrently.

    KEY DESIGN DECISION — concurrent vs sequential:
      SEQUENTIAL (original approach): search outfit 1 → wait → search 2 → wait → search 3
        Time: ~3 × 3 seconds = 9 seconds minimum
      CONCURRENT (our approach): search all 3 simultaneously with asyncio.gather()
        Time: ~3 seconds (limited by slowest single search)

    asyncio.gather() runs all coroutines concurrently within the same
    event loop. This is NOT threading — it's cooperative multitasking.
    While one HTTP request is waiting for SerpApi to respond, the other
    requests are also running. Total time ≈ single request time.

    Args:
        outfits:        List of 3 OutfitRecommendation objects from Gemini
        max_per_outfit: Max products to fetch per outfit (default: 4 → 12 total)

    Returns:
        Flat list of all ShoppingProduct objects across all outfits,
        each tagged with outfit_index for frontend grouping.
    """
    if not outfits:
        return []

    if not settings.SERPAPI_KEY or len(settings.SERPAPI_KEY) < 10:
        logger.warning("SerpApi key not configured — skipping product search")
        return []

    # httpx.AsyncClient is an async HTTP session.
    # Using `async with` ensures the connection pool is properly closed
    # after all searches complete, even if some fail.
    # limits= sets max concurrent connections to SerpApi — be a good API citizen.
    async with httpx.AsyncClient(
        limits=httpx.Limits(max_connections=5, max_keepalive_connections=3)
    ) as client:

        # Create one coroutine per outfit search
        search_tasks = [
            _search_single_outfit(client, outfit, index, max_per_outfit)
            for index, outfit in enumerate(outfits)
        ]

        # asyncio.gather() runs all tasks concurrently and collects results.
        # return_exceptions=True means if one search fails, the others still complete.
        # Without it, one failure would cancel all pending searches.
        results = await asyncio.gather(*search_tasks, return_exceptions=True)

    # Flatten results: [[product, product], [product], [product, product, product]]
    # → [product, product, product, product, product, product]
    all_products: List[ShoppingProduct] = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            # gather() with return_exceptions=True returns the exception itself
            # instead of raising it. Log and skip.
            logger.error(f"Outfit {i} search raised exception: {result}")
        elif isinstance(result, list):
            all_products.extend(result)

    logger.info(f"Total products found: {len(all_products)} across {len(outfits)} outfits")
    return all_products
