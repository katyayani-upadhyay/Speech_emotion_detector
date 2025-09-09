# src/build_metadata.py
import csv
from pathlib import Path
import pandas as pd
from .dataset_downloader import ensure_dataset
from .config import METADATA_CSV

EMOTION_MAP = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}

def parse_filename(fname: str):
    """
    RAVDESS filename format example:
    03-01-05-02-02-02-12.wav
    returns dict of parsed metadata
    """
    base = Path(fname).stem
    parts = base.split("-")
    if len(parts) < 7:
        return None
    modality, channel, emotion_code, intensity_code, statement_code, repetition, actor = parts[:7]
    emotion = EMOTION_MAP.get(emotion_code, "unknown")
    intensity = "normal" if intensity_code == "01" else "strong"
    statement = "kids are talking by the door" if statement_code == "01" else "dogs are sitting by the door"
    gender = "male" if int(actor) % 2 == 1 else "female"
    return {
        "filename": fname,
        "emotion": emotion,
        "intensity": intensity,
        "statement": statement,
        "repetition": repetition,
        "actor": actor,
        "gender": gender
    }

def build_metadata(download_url: str = None, out_csv: Path = METADATA_CSV):
    wav_root = ensure_dataset(download_url)
    rows = []
    for p in sorted(wav_root.rglob("*.wav")):
        parsed = parse_filename(p.name)
        if parsed is None:
            # skip non-standard files
            continue
        parsed["path"] = str(p)
        rows.append(parsed)

    df = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"[build_metadata] Wrote metadata for {len(df)} files to {out_csv}")
    return df

if __name__ == "__main__":
    df = build_metadata()
    print(df.head())
