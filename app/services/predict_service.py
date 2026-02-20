# ============================================================
# app/services/predict_service.py — ML Inference Service
# ============================================================
# This file does two jobs:
#   1. Run the age/gender model on an uploaded image
#   2. Run the facial feature model on the same image
#
# The results feed directly into llm_service.py.
#
# Key improvement over the original:
#   - No file I/O between steps (no JSON files written to disk)
#   - All functions are pure: image in → predictions out
#   - Suppressed print() replaced with proper logging
#   - feature_to_human_readable() converts raw floats → natural language
#     so Gemini actually understands what it's looking at
# ============================================================

import io
import torch
import logging
import numpy as np
from PIL import Image
from torchvision import transforms
from typing import Tuple

# run_in_threadpool is Starlette's utility for offloading blocking (synchronous)
# code to a worker thread WITHOUT blocking the FastAPI event loop.
#
# The pattern we use throughout this file:
#   _sync_*  → plain `def`, contains all the blocking PyTorch logic
#   async *  → `async def` wrapper that calls run_in_threadpool(_sync_*)
#
# This way asyncio.gather() in recommend.py genuinely runs both models
# concurrently in separate threads, while the event loop stays free
# to handle other incoming requests (e.g. /health from another user).
from starlette.concurrency import run_in_threadpool

from app.services.model_service import model_store

logger = logging.getLogger(__name__)


# ── Image Preprocessing ───────────────────────────────────
# EfficientNet-B4 was trained on ImageNet-normalized images.
# We must apply the EXACT same transforms during inference
# that were used during training, or predictions will be wrong.
# ImageNet mean/std are standard values used for all EfficientNet models.
INFERENCE_TRANSFORM = transforms.Compose([
    transforms.Resize((380, 380)),           # EfficientNet-B4's native input size
    transforms.CenterCrop(380),
    transforms.ToTensor(),                   # Converts PIL [0,255] → Tensor [0.0,1.0]
    transforms.Normalize(                    # ImageNet normalization
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

# ── CelebA Attribute Labels ───────────────────────────────
# These must be in the EXACT same order they were used during training.
# The model outputs a vector of 16 values — index 0 = Straight_Hair, etc.
# If the order changes, all predictions are misaligned.
CELEBA_ATTRS = [
    'Straight_Hair', 'Wavy_Hair', 'No_Beard', 'Oval_Face', 'Pale_Skin',
    'Pointy_Nose', 'Receding_Hairline', 'Mustache', 'Big_Lips', 'Big_Nose',
    'Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Bushy_Eyebrows',
    'Arched_Eyebrows', 'Bald'
]

# Categorical groups: within each group, only ONE attribute applies.
# e.g., hair can't be both Black AND Blond — we pick the highest probability.
# Carried over from the original ml.py (this logic was correct there).
CATEGORICAL_GROUPS = [
    ['Black_Hair', 'Blond_Hair', 'Brown_Hair'],
    ['Bushy_Eyebrows', 'Arched_Eyebrows'],
    ['Straight_Hair', 'Wavy_Hair'],
]


def _bytes_to_tensor(image_bytes: bytes) -> torch.Tensor:
    """
    Converts raw image bytes (from the uploaded file) into a
    preprocessed tensor ready for model inference.

    Args:
        image_bytes: Raw bytes from the uploaded image file

    Returns:
        Tensor of shape [1, 3, 380, 380] on the model's device
    """
    # io.BytesIO wraps bytes in a file-like object so PIL can read them.
    # We never write to disk — the image lives entirely in memory.
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # Apply the same transforms used during training
    tensor = INFERENCE_TRANSFORM(image)

    # Add batch dimension: [3, 380, 380] → [1, 3, 380, 380]
    # Models expect a batch even when processing single images
    tensor = tensor.unsqueeze(0)

    # Move to the same device as the model (GPU or CPU)
    device = model_store.get("device", torch.device("cpu"))
    return tensor.to(device)


def _apply_categorical_thresholding(
    raw_probs: dict[str, float],
    single_threshold: float = 0.3
) -> dict[str, float]:
    """
    Custom thresholding for facial features — kept from original ml.py.
    This is the one piece of the original code that was genuinely smart.

    For categorical groups (e.g. hair color): picks the highest probability
    attribute and zeros out the others. Hair can't be Black AND Blond.

    For single attributes (e.g. Oval_Face): zeros out anything below threshold.
    Threshold 0.3 is intentionally lower than 0.5 — facial features are
    subtle, and 0.5 causes too many false negatives.

    Args:
        raw_probs: Dict of {attribute_name: probability_score}
        single_threshold: Minimum probability to include a single attribute

    Returns:
        Processed probabilities with categorical logic applied
    """
    processed = {}

    # Track which attributes are part of categorical groups
    categorical_attrs = {attr for group in CATEGORICAL_GROUPS for attr in group}

    # For each categorical group, keep only the winner
    for group in CATEGORICAL_GROUPS:
        group_probs = {attr: raw_probs[attr] for attr in group if attr in raw_probs}
        if group_probs:
            winner = max(group_probs, key=group_probs.get)
            for attr in group:
                if attr in raw_probs:
                    processed[attr] = raw_probs[attr] if attr == winner else 0.0

    # For non-categorical attributes, apply simple threshold
    for attr, prob in raw_probs.items():
        if attr not in categorical_attrs:
            processed[attr] = prob if prob > single_threshold else 0.0

    return processed


def features_to_human_readable(processed_probs: dict[str, float]) -> str:
    """
    Converts raw probability scores into natural language for the LLM.

    THIS IS THE KEY IMPROVEMENT over the original.
    Original sent: "jawline_strength: 0.81, cheekbone_prominence: 0.76"
    We send:       "black hair, oval face, arched eyebrows, big lips"

    Gemini can reason about natural language. It cannot meaningfully
    interpret "Arched_Eyebrows: 0.63" in a fashion context.

    Only includes attributes with probability > 0 (i.e., above threshold
    or the winner in their categorical group).

    Args:
        processed_probs: Output of _apply_categorical_thresholding()

    Returns:
        Comma-separated natural language string of detected features
    """
    # Human-readable labels for each CelebA attribute
    label_map = {
        'Straight_Hair':      'straight hair',
        'Wavy_Hair':          'wavy hair',
        'No_Beard':           'clean-shaven',
        'Oval_Face':          'oval face shape',
        'Pale_Skin':          'fair/pale skin tone',
        'Pointy_Nose':        'pointed nose',
        'Receding_Hairline':  'receding hairline',
        'Mustache':           'mustache',
        'Big_Lips':           'full lips',
        'Big_Nose':           'broad nose',
        'Black_Hair':         'black hair',
        'Blond_Hair':         'blonde hair',
        'Brown_Hair':         'brown hair',
        'Bushy_Eyebrows':     'thick eyebrows',
        'Arched_Eyebrows':    'arched eyebrows',
        'Bald':               'bald/shaved head',
    }

    # Only include features that were detected (probability > 0)
    detected = [
        label_map[attr]
        for attr, prob in processed_probs.items()
        if prob > 0 and attr in label_map
    ]

    if not detected:
        return "no distinctive facial features detected"

    return ", ".join(detected)


def _sync_age_gender_prediction(image_bytes: bytes) -> Tuple[int, str, float]:
    """
    SYNCHRONOUS core — all blocking PyTorch logic lives here.
    Plain `def` (not async) so it's safe to run in a thread pool.
    Never call this directly from async code — use the wrapper below.
    """
    model = model_store.get("age_gender")
    scaler = model_store.get("age_scaler")

    if model is None:
        logger.warning("Age/gender model not loaded — returning defaults")
        return 25, "Unknown", 0.0

    try:
        tensor = _bytes_to_tensor(image_bytes)

        with torch.no_grad():
            if model_store["device"].type == "cuda":
                with torch.amp.autocast("cuda"):
                    output = model(tensor)
            else:
                output = model(tensor)

        output     = output.squeeze(0).cpu().numpy()
        age_raw    = float(output[0])
        gender_raw = float(output[1])

        if scaler is not None:
            age = int(round(scaler.inverse_transform([[age_raw]])[0][0]))
            age = max(1, min(age, 100))
        else:
            age = int(round(age_raw * 100))
            age = max(1, min(age, 100))

        gender_prob = float(torch.sigmoid(torch.tensor(gender_raw)))
        gender      = "Male" if gender_prob >= 0.5 else "Female"
        confidence  = gender_prob if gender == "Male" else 1 - gender_prob

        logger.info(f"Age/gender prediction: {age}yo {gender} (conf: {confidence:.2f})")
        return age, gender, confidence

    except Exception as e:
        logger.error(f"Age/gender inference failed: {e}")
        return 25, "Unknown", 0.0


async def run_age_gender_prediction(image_bytes: bytes) -> Tuple[int, str, float]:
    """
    ASYNC wrapper — the function your router actually calls.

    Offloads _sync_age_gender_prediction to a worker thread via
    run_in_threadpool. This frees the event loop immediately so
    FastAPI can handle other requests while PyTorch is running.

    When asyncio.gather() calls this alongside run_facial_feature_prediction,
    both thread-pool jobs execute concurrently — genuine parallelism,
    not the fake async that a bare `async def` with blocking code gives you.
    """
    return await run_in_threadpool(_sync_age_gender_prediction, image_bytes)


def _sync_facial_feature_prediction(image_bytes: bytes) -> Tuple[dict, str]:
    """
    SYNCHRONOUS core — all blocking PyTorch logic lives here.
    Plain `def` (not async) so it's safe to run in a thread pool.
    Never call this directly from async code — use the wrapper below.
    """
    model = model_store.get("facial_features")

    if model is None:
        logger.warning("Facial feature model not loaded — returning empty features")
        return {}, "no facial features available"

    try:
        tensor = _bytes_to_tensor(image_bytes)

        with torch.no_grad():
            if model_store["device"].type == "cuda":
                with torch.amp.autocast("cuda"):
                    output = model(tensor)
            else:
                output = model(tensor)

        probs      = torch.sigmoid(output).squeeze(0).cpu().numpy()
        raw_probs  = {attr: float(probs[i]) for i, attr in enumerate(CELEBA_ATTRS)}
        processed  = _apply_categorical_thresholding(raw_probs)
        readable   = features_to_human_readable(processed)

        logger.info(f"Facial features: {readable}")
        return processed, readable

    except Exception as e:
        logger.error(f"Facial feature inference failed: {e}")
        return {}, "facial feature detection failed"


async def run_facial_feature_prediction(image_bytes: bytes) -> Tuple[dict, str]:
    """
    ASYNC wrapper — the function your router actually calls.

    Same pattern as run_age_gender_prediction: offloads the blocking
    _sync_facial_feature_prediction to a worker thread.

    When asyncio.gather() runs this alongside run_age_gender_prediction,
    both execute concurrently in separate threads — true parallelism.
    """
    return await run_in_threadpool(_sync_facial_feature_prediction, image_bytes)
