# src/feature_extractor.py
import librosa
import numpy as np

def extract_features(file_path: str, n_mfcc: int = 40):
    """
    Extracts MFCC features from an audio file.

    Args:
        file_path (str): Path to the audio file.
        n_mfcc (int): Number of MFCCs to return.

    Returns:
        np.ndarray: A numpy array of the mean of the MFCCs.
                   Returns None if the file cannot be processed.
    """
    try:
        # Load the audio file
        audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')

        # Get MFCCs
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc)

        # To get a consistent feature vector size, we take the mean of the MFCCs over time
        mfccs_scaled_features = np.mean(mfccs.T, axis=0)
        return mfccs_scaled_features

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

if __name__ == '__main__':
    # This is a quick test to see if it works.
    # First, make sure you have run the dataset downloader to get the data.
    from pathlib import Path
    from .config import DATA_DIR
    
    # Use the metadata CSV to find a sample file
    metadata_path = DATA_DIR / "ravdess_metadata.csv"
    if metadata_path.exists():
        import pandas as pd
        df = pd.read_csv(metadata_path)
        if not df.empty:
            sample_file = df.iloc[0]['path']
            print(f"Extracting features for a sample file: {sample_file}")
            features = extract_features(sample_file)
            if features is not None:
                print("Features extracted successfully!")
                print("Shape:", features.shape)
                print("First 5 features:", features[:5])
        else:
            print("Metadata CSV is empty. Cannot test.")
    else:
        print("Metadata CSV not found. Please run build_metadata.py first.")