# src/dataset_downloader.py
import os
import requests
import zipfile
from pathlib import Path
from tqdm import tqdm
from .config import RAW_DIR, DATA_DIR, ZIP_NAME, CHUNK_SIZE, DEFAULT_DOWNLOAD_URL

def download_zip(url: str, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        print(f"[download_zip] Found existing file, skipping download: {out_path}")
        return out_path

    print(f"[download_zip] Downloading {url} -> {out_path}")
    resp = requests.get(url, stream=True, timeout=60)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    with open(out_path, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=out_path.name) as pbar:
        for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))
    return out_path

def safe_extract(zip_path: Path, target_dir: Path):
    """Extract zip safely and skip if already extracted."""
    if target_dir.exists() and any(target_dir.iterdir()):
        print(f"[safe_extract] Target directory {target_dir} already exists and is non-empty. Skipping extraction.")
        return target_dir

    print(f"[safe_extract] Extracting {zip_path} -> {target_dir}")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(target_dir)
    return target_dir

def find_base_extracted_dir(extract_dir: Path):
    # Many zips contain a top folder like "Audio_Speech_Actors_01-24"
    for child in extract_dir.iterdir():
        if child.is_dir() and any(child.glob("*.wav")):
            return child
    # fallback: return extract_dir itself
    return extract_dir

def ensure_dataset(download_url: str = None):
    """
    Ensure the dataset is downloaded and extracted.
    Returns path to the directory containing .wav files.
    """
    download_url = download_url or DEFAULT_DOWNLOAD_URL
    zip_out = RAW_DIR / ZIP_NAME
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    zip_path = download_zip(download_url, zip_out)
    extracted_root = safe_extract(zip_path, RAW_DIR / "extracted")

    wav_root = find_base_extracted_dir(extracted_root)
    print(f"[ensure_dataset] WAV root directory: {wav_root}")
    return wav_root

if __name__ == "__main__":
    print("Run ensure_dataset() to download + extract.")
    ensure_dataset()
