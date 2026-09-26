"""
Model Training and Evaluation Pipeline
1. Loads or generates benchmark dataset
2. Extracts feature vectors for each assignment pair
3. Trains and benchmarks ML classifiers:
   - Logistic Regression
   - Random Forest
4. Saves the best model and evaluation metrics to models/classifier.joblib
"""

import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm

# Add parent directory to path so src can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.ml_classifier import extract_pair_features, DuplicateDetectorModel
from data.generate_dataset import generate_dataset


def run_training_pipeline():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_dir = os.path.join(base_dir, "data")
    models_dir = os.path.join(base_dir, "models")
    os.makedirs(models_dir, exist_ok=True)

    csv_path = os.path.join(data_dir, "assignments_dataset.csv")

    if not os.path.exists(csv_path):
        print("[1/4] Generating assignment dataset...")
        df = generate_dataset(num_pairs_per_class=200)
        df.to_csv(csv_path, index=False, encoding="utf-8")
    else:
        print("[1/4] Loading existing assignment dataset...")
        df = pd.read_csv(csv_path)

    print(f"Dataset loaded: {len(df)} sample pairs across classes:")
    print(df["label_name"].value_counts().to_string())

    print("\n[2/4] Extracting NLP feature vectors (TF-IDF, SBERT, Jaccard, Overlap, Length)...")
    X_list = []
    y_list = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting features"):
        t1 = str(row["assignment1"])
        t2 = str(row["assignment2"])
        lbl = int(row["label"])

        res = extract_pair_features(t1, t2)
        X_list.append(res["feature_vector"])
        y_list.append(lbl)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)

    print(f"\nFeature matrix shape: {X.shape}")

    # Train Random Forest
    print("\n[3/4] Training and evaluating Random Forest Classifier...")
    rf_model = DuplicateDetectorModel(model_type="rf")
    rf_metrics = rf_model.train(X, y, test_size=0.20)

    # Train Logistic Regression
    print("Training and evaluating Logistic Regression Classifier...")
    lr_model = DuplicateDetectorModel(model_type="lr")
    lr_metrics = lr_model.train(X, y, test_size=0.20)

    print("\n" + "=" * 60)
    print("               MODEL PERFORMANCE COMPARISON")
    print("=" * 60)
    print(f"Random Forest  -> Accuracy: {rf_metrics['accuracy']}% | F1: {rf_metrics['f1_score']}% | Precision: {rf_metrics['precision']}% | Recall: {rf_metrics['recall']}%")
    print(f"Logistic Regr. -> Accuracy: {lr_metrics['accuracy']}% | F1: {lr_metrics['f1_score']}% | Precision: {lr_metrics['precision']}% | Recall: {lr_metrics['recall']}%")
    print("=" * 60)

    # Select best model
    best_model = rf_model if rf_metrics["f1_score"] >= lr_metrics["f1_score"] else lr_model
    best_name = "Random Forest" if best_model == rf_model else "Logistic Regression"
    print(f"\n[4/4] Saving best model ({best_name}) to {os.path.join(models_dir, 'classifier.joblib')}...")
    best_model.save(os.path.join(models_dir, "classifier.joblib"))

    print("Pipeline completed successfully! Model ready for Streamlit deployment.")


if __name__ == "__main__":
    run_training_pipeline()
