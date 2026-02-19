# ============================================================
# notebooks/colab_setup.py
# ============================================================
# This is your Colab notebook written as a .py file so it can
# live in the repo. When you're in Colab, create a new notebook
# and copy each cell block (between # CELL markers) into a cell.
#
# COLAB SETUP: Before running, go to:
#   Runtime → Change runtime type → T4 GPU
# ============================================================


# ─────────────────────────────────────────────────────────────
# CELL 1 — Mount Google Drive
# Run this first. It will prompt you to authorize access.
# Your Drive is mounted at /content/drive/MyDrive/
# ─────────────────────────────────────────────────────────────
from google.colab import drive
drive.mount('/content/drive')

# Create the project folder structure in your Drive.
# Files here persist between Colab sessions — unlike /content/ which resets.
import os
DRIVE_PROJECT = '/content/drive/MyDrive/fashion-ai'
os.makedirs(f'{DRIVE_PROJECT}/models', exist_ok=True)
os.makedirs(f'{DRIVE_PROJECT}/uploads', exist_ok=True)
print(f"✅ Drive mounted. Project folder: {DRIVE_PROJECT}")


# ─────────────────────────────────────────────────────────────
# CELL 2 — Clone your backend repo
# Replace YOUR_GITHUB_USERNAME with your actual username after
# you create the new repo.
# ─────────────────────────────────────────────────────────────
import os

REPO_URL = "https://github.com/YOUR_GITHUB_USERNAME/fashion-ai-backend.git"
REPO_DIR = "/content/fashion-ai-backend"

if not os.path.exists(REPO_DIR):
    os.system(f"git clone {REPO_URL} {REPO_DIR}")
    print(f"✅ Repo cloned to {REPO_DIR}")
else:
    # If already cloned, just pull latest changes
    os.system(f"cd {REPO_DIR} && git pull")
    print(f"✅ Repo updated")

os.chdir(REPO_DIR)
print(f"Working directory: {os.getcwd()}")


# ─────────────────────────────────────────────────────────────
# CELL 3 — Install dependencies
# --quiet suppresses the wall of pip output
# --break-system-packages is needed in newer Colab environments
# ─────────────────────────────────────────────────────────────
os.system("pip install -r requirements.txt -q")
print("✅ Dependencies installed")

# Verify GPU is available
import torch
if torch.cuda.is_available():
    print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
else:
    print("⚠️  No GPU detected — switch runtime to T4 GPU for faster inference")


# ─────────────────────────────────────────────────────────────
# CELL 4 — Copy model weights from Drive to Colab local storage
# We copy to /content/ for faster disk I/O during inference.
# Your original .pth files go in /content/drive/MyDrive/fashion-ai/models/
# ─────────────────────────────────────────────────────────────
import shutil

MODELS_SRC = '/content/drive/MyDrive/fashion-ai/models'
MODELS_DST = '/content/fashion-ai-backend/model_weights'
os.makedirs(MODELS_DST, exist_ok=True)

model_files = [
    'age_gender_model.pth',
    'celeba_imbalance_aware_classifier.pth',
    'age_scaler.pkl'
]

for model_file in model_files:
    src = f'{MODELS_SRC}/{model_file}'
    dst = f'{MODELS_DST}/{model_file}'
    if os.path.exists(src):
        shutil.copy2(src, dst)
        size_mb = os.path.getsize(dst) / 1024**2
        print(f"✅ Copied {model_file} ({size_mb:.1f} MB)")
    else:
        print(f"⚠️  {model_file} not found in Drive — upload it first")


# ─────────────────────────────────────────────────────────────
# CELL 5 — Create .env file in Colab
# We write the .env directly in this cell so secrets never touch GitHub.
# Replace the placeholder values with your actual keys.
# ─────────────────────────────────────────────────────────────

env_content = """
ENVIRONMENT=development
GOOGLE_API_KEY=your_gemini_api_key_here
SERPAPI_KEY=your_serpapi_key_here
AGE_GENDER_MODEL_PATH=model_weights/age_gender_model.pth
FACIAL_FEATURE_MODEL_PATH=model_weights/celeba_imbalance_aware_classifier.pth
AGE_SCALER_PATH=model_weights/age_scaler.pkl
ALLOWED_ORIGINS=http://localhost:5173,https://your-frontend-domain.com
""".strip()

with open('/content/fashion-ai-backend/.env', 'w') as f:
    f.write(env_content)

print("✅ .env file created")
print("⚠️  Remember to replace API key placeholders with your real keys!")


# ─────────────────────────────────────────────────────────────
# CELL 6 — Install ngrok and expose the FastAPI server
# ngrok creates a public HTTPS tunnel to your Colab server.
# This is how your React frontend (running locally or deployed)
# can reach the FastAPI backend running in Colab.
# ─────────────────────────────────────────────────────────────
os.system("pip install pyngrok -q")

from pyngrok import ngrok, conf

# Set your ngrok authtoken — get it free at https://dashboard.ngrok.com/
NGROK_TOKEN = "your_ngrok_authtoken_here"
conf.get_default().auth_token = NGROK_TOKEN

# Start the tunnel on port 8000 (FastAPI's default)
public_url = ngrok.connect(8000)
print(f"✅ Public API URL: {public_url}")
print(f"   Paste this into your frontend's VITE_API_URL environment variable")
print(f"   API docs: {public_url}/docs")


# ─────────────────────────────────────────────────────────────
# CELL 7 — Start the FastAPI server
# This is the last cell. Run it AFTER all others succeed.
# The server runs in the foreground — the cell stays "running"
# as long as the server is alive. That's expected behavior.
# ─────────────────────────────────────────────────────────────
import subprocess
import sys

# We run uvicorn as a subprocess so Colab doesn't freeze.
# --host 0.0.0.0 makes it accessible from outside localhost
# --port 8000 matches our ngrok tunnel
# --reload auto-restarts when code changes (great for development)
process = subprocess.Popen(
    [sys.executable, "-m", "uvicorn", "app.main:app",
     "--host", "0.0.0.0", "--port", "8000", "--reload"],
    cwd='/content/fashion-ai-backend'
)

print(f"✅ Server started (PID: {process.pid})")
print(f"   Health check: GET {public_url}/api/v1/health")
print(f"   API docs:     GET {public_url}/docs")
