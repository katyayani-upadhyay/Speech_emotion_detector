# src/train_model.py
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt

from .config import METADATA_CSV, ROOT
from .feature_extractor import extract_features

# Create a directory to save models and results
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def train():
    """
    Main function to load data, extract features, train, evaluate, and save the model.
    """
    # 1. Load Metadata
    print("[INFO] Loading metadata...")
    if not METADATA_CSV.exists():
        print(f"[ERROR] Metadata file not found at {METADATA_CSV}. Please run `src.build_metadata` first.")
        return
    df = pd.read_csv(METADATA_CSV)
    print(f"Loaded metadata with {len(df)} entries.")

    # 2. Extract Features
    print("[INFO] Extracting features for all audio files. This may take a while...")
    # Use tqdm for a progress bar
    features = [extract_features(row['path']) for _, row in tqdm(df.iterrows(), total=df.shape[0])]
    
    # Filter out any files that failed to process
    valid_indices = [i for i, f in enumerate(features) if f is not None]
    X = np.array([features[i] for i in valid_indices])
    y = df['emotion'].iloc[valid_indices]
    print(f"[INFO] Successfully extracted features for {len(X)} files.")

    # 3. Prepare Data for Modeling
    print("[INFO] Preparing data for modeling...")
    # Encode the string labels ('happy', 'sad') into numbers
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.25, random_state=42, stratify=y_encoded)
    print(f"Training set size: {len(X_train)}")
    print(f"Testing set size: {len(X_test)}")

    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 4. Train the Model
    print("[INFO] Training the MLP Classifier...")
    model = MLPClassifier(
        alpha=0.01,
        batch_size=256,
        epsilon=1e-08,
        hidden_layer_sizes=(300,),
        learning_rate='adaptive',
        max_iter=500,
        random_state=42
    )
    model.fit(X_train_scaled, y_train)

    # 5. Evaluate the Model
    print("[INFO] Evaluating the model...")
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")
    
    # Print detailed report
    print("\nClassification Report:")
    report = classification_report(y_test, y_pred, target_names=le.classes_)
    print(report)

    # Save the classification report
    with open(RESULTS_DIR / "classification_report.txt", "w") as f:
        f.write(report)

    # Generate and save confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=le.classes_, yticklabels=le.classes_, cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig(RESULTS_DIR / "confusion_matrix.png")
    print(f"[INFO] Saved confusion matrix to {RESULTS_DIR / 'confusion_matrix.png'}")

    # 6. Save the Model and Preprocessing Objects
    print("[INFO] Saving model and preprocessing objects...")
    joblib.dump(model, MODELS_DIR / "emotion_classifier.pkl")
    joblib.dump(scaler, MODELS_DIR / "scaler.pkl")
    joblib.dump(le, MODELS_DIR / "label_encoder.pkl")
    print(f"[SUCCESS] Model and objects saved to {MODELS_DIR}")


if __name__ == "__main__":
    train()