# src/config.py
from pathlib import Path
import os

# --- Directory Paths ---
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
MODELS_DIR = ROOT / "models"  # <-- ADD THIS LINE
METADATA_CSV = DATA_DIR / "ravdess_metadata.csv"

# --- Download Settings ---
# Default download url (Zenodo speech audio). You can override with env var DOWNLOAD_URL.
DEFAULT_DOWNLOAD_URL = os.getenv("DOWNLOAD_URL", "https://zenodo.org/record/1188976/files/Audio_Speech_Actors_01-24.zip?download=1")

# Temporary download filename
ZIP_NAME = "ravdess.zip"

# Chunk size for streaming downloads
CHUNK_SIZE = 1024 * 1024
