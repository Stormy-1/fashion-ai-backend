# ============================================================
# app/services/model_service.py — ML Model Loading & Caching
# ============================================================
# This file is responsible for ONE thing: loading the trained PyTorch
# models from disk into GPU/CPU memory, and keeping them there.
#
# The critical insight vs the original code:
#   ORIGINAL: Models were reloaded from .pth files on every API request.
#             First request takes 10-15 seconds. Wastes GPU memory constantly.
#   NEW:      Models load ONCE at startup. Every request reuses the same
#             in-memory model. Response time drops to seconds, not minutes.
#
# `model_store` is the shared dictionary that holds all loaded models.
# Think of it as the application's in-memory model registry.
# ============================================================

import torch
import pickle
import logging
from pathlib import Path

from app.core.config import settings

# Use Python's logging module instead of print() statements.
# This is the professional standard — logs have levels (INFO, WARNING, ERROR)
# and can be configured to write to files, external services, etc.
# In the original code, print() was used everywhere.
logger = logging.getLogger(__name__)

# ── Model Store ───────────────────────────────────────────
# A plain dictionary that acts as our in-memory model registry.
# Keys are model names, values are the loaded model/scaler objects.
# Populated at startup, read by all service functions.
#
# Example contents after startup:
# {
#   "age_gender":      <loaded EfficientNet model on CUDA>,
#   "facial_features": <loaded CelebA classifier on CUDA>,
#   "age_scaler":      <loaded sklearn StandardScaler>,
#   "device":          device("cuda:0")
# }
model_store: dict = {}


def _get_device() -> torch.device:
    """
    Determines the best available compute device.
    In Colab with T4 GPU selected: returns cuda:0
    In Colab CPU runtime or locally: returns cpu
    """
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info(f"GPU found: {gpu_name} ({vram_gb:.1f} GB VRAM)")
    else:
        device = torch.device("cpu")
        logger.warning("No GPU found — running on CPU. Inference will be slow.")
    return device


def _build_age_gender_model() -> torch.nn.Module:
    """
    Rebuilds the EfficientNet-B4 model architecture.
    We must recreate the same architecture that was used during training
    before we can load the saved weights (.pth file) into it.

    The original project trained a custom head on top of EfficientNet-B4:
    - EfficientNet-B4 backbone (pretrained on ImageNet, then fine-tuned)
    - Custom classification head for age (regression) + gender (binary)
    """
    # Import here to avoid loading torchvision at module import time
    # (keeps startup faster if this function isn't called yet)
    from torchvision import models
    import torch.nn as nn

    # Load EfficientNet-B4 without pretrained weights —
    # we're about to overwrite all weights from our .pth file anyway
    backbone = models.efficientnet_b4(weights=None)

    # The original model replaced EfficientNet's classifier head.
    # EfficientNet-B4's final feature dimension is 1792.
    # We recreate the same custom head: dropout → linear → (age_out, gender_out)
    in_features = backbone.classifier[1].in_features  # 1792
    backbone.classifier = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(512, 2),   # Output: [age_prediction, gender_logit]
    )

    return backbone


def _build_facial_feature_model(num_attributes: int = 16) -> torch.nn.Module:
    """
    Rebuilds the CelebA facial feature classifier architecture.
    Multi-label classifier: predicts 16 binary attributes simultaneously.
    """
    from torchvision import models
    import torch.nn as nn

    backbone = models.efficientnet_b4(weights=None)
    in_features = backbone.classifier[1].in_features

    # Multi-label classification head — no softmax, we use sigmoid per attribute
    # (each attribute is independent, unlike multi-class where they sum to 1)
    backbone.classifier = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(512, num_attributes),  # One output per facial attribute
    )

    return backbone


async def load_all_models(store: dict) -> None:
    """
    Loads all ML models into memory at application startup.
    Called once from main.py's lifespan handler.

    Args:
        store: The model_store dict to populate.
               Passed in explicitly so this function is testable.
    """
    device = _get_device()
    store["device"] = device

    # ── Age & Gender Model ────────────────────────────────
    age_gender_path = Path(settings.AGE_GENDER_MODEL_PATH)
    if age_gender_path.exists():
        try:
            logger.info(f"Loading age/gender model from {age_gender_path}...")
            model = _build_age_gender_model()

            # map_location ensures the model loads to the right device
            # even if it was saved on a different device (e.g., saved on GPU, loading on CPU)
            state_dict = torch.load(age_gender_path, map_location=device)

            # Some saved models wrap state_dict in a checkpoint dict.
            # Handle both formats gracefully.
            if "model_state_dict" in state_dict:
                state_dict = state_dict["model_state_dict"]

            model.load_state_dict(state_dict)
            model.to(device)

            # eval() switches off dropout and batch norm training behavior.
            # ALWAYS call this before inference — forgetting it causes inconsistent predictions.
            model.eval()

            store["age_gender"] = model
            logger.info("✅ Age/gender model loaded successfully")
        except Exception as e:
            logger.error(f"❌ Failed to load age/gender model: {e}")
    else:
        logger.warning(f"Age/gender model not found at {age_gender_path}")

    # ── Facial Feature Model ──────────────────────────────
    facial_feature_path = Path(settings.FACIAL_FEATURE_MODEL_PATH)
    if facial_feature_path.exists():
        try:
            logger.info(f"Loading facial feature model from {facial_feature_path}...")
            model = _build_facial_feature_model()
            state_dict = torch.load(facial_feature_path, map_location=device)

            if "model_state_dict" in state_dict:
                state_dict = state_dict["model_state_dict"]

            model.load_state_dict(state_dict)
            model.to(device)
            model.eval()

            store["facial_features"] = model
            logger.info("✅ Facial feature model loaded successfully")
        except Exception as e:
            logger.error(f"❌ Failed to load facial feature model: {e}")
    else:
        logger.warning(f"Facial feature model not found at {facial_feature_path}")

    # ── Age Scaler ────────────────────────────────────────
    # The original model predicts a normalized age value.
    # The scaler reverses that normalization back to actual years.
    scaler_path = Path(settings.AGE_SCALER_PATH)
    if scaler_path.exists():
        try:
            with open(scaler_path, "rb") as f:
                store["age_scaler"] = pickle.load(f)
            logger.info("✅ Age scaler loaded successfully")
        except Exception as e:
            logger.error(f"❌ Failed to load age scaler: {e}")
    else:
        logger.warning(f"Age scaler not found at {scaler_path}")
