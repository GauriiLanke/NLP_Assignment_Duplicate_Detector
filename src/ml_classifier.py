"""
Machine Learning Classifier Module
Features:
- Extracts multimodal feature vectors from assignment pairs
- Rule-based thresholding (Composite score)
- Supervised ML classification (Logistic Regression, Random Forest, SVM)
- Model training, saving, loading, and cross-evaluation
"""

import os
from typing import Dict, List, Tuple, Any
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.model_selection import train_test_split

from src.preprocessing import preprocess_text
from src.feature_extraction import (
    compute_tfidf_similarity,
    compute_jaccard_similarity,
    compute_word_overlap_ratio,
    compute_length_features
)
from src.semantic_similarity import compute_semantic_similarity

# Constants
LABEL_MAPPING = {0: "Different", 1: "Similar", 2: "Duplicate"}
INV_LABEL_MAPPING = {v: k for k, v in LABEL_MAPPING.items()}

DEFAULT_MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


def extract_pair_features(text1: str, text2: str) -> Dict[str, Any]:
    """
    Extracts complete numerical feature vector and metadata for an assignment pair.
    """
    tokens1, clean1 = preprocess_text(text1)
    tokens2, clean2 = preprocess_text(text2)

    # 1. Traditional Lexical Features
    tfidf_sim = compute_tfidf_similarity(clean1, clean2)
    jaccard_sim = compute_jaccard_similarity(tokens1, tokens2)
    word_overlap = compute_word_overlap_ratio(tokens1, tokens2)
    length_meta = compute_length_features(text1, text2, tokens1, tokens2)

    # 2. Modern Semantic Features
    semantic_sim = compute_semantic_similarity(text1, text2)

    # 3. Rule-based Composite Score (0-100%)
    # 40% TF-IDF + 60% Semantic gives high weight to semantic meaning while respecting exact words
    composite_score = (0.40 * tfidf_sim + 0.60 * semantic_sim) * 100.0

    # Rule-based status
    if composite_score >= 82.0:
        rule_status = "Duplicate"
        rule_color = "red"
        rule_badge = "🔴 HIGHLY SIMILAR / DUPLICATE"
    elif composite_score >= 58.0:
        rule_status = "Similar"
        rule_color = "orange"
        rule_badge = "🟠 POSSIBLY SIMILAR"
    else:
        rule_status = "Different"
        rule_color = "green"
        rule_badge = "🟢 DIFFERENT"

    feature_vector = np.array([
        tfidf_sim,
        semantic_sim,
        jaccard_sim,
        word_overlap,
        length_meta["token_ratio"]
    ], dtype=np.float32)

    return {
        "feature_vector": feature_vector,
        "tfidf_sim": round(tfidf_sim * 100, 2),
        "semantic_sim": round(semantic_sim * 100, 2),
        "jaccard_sim": round(jaccard_sim * 100, 2),
        "word_overlap": round(word_overlap * 100, 2),
        "token_ratio": round(length_meta["token_ratio"] * 100, 2),
        "tokens1_count": length_meta["tokens1_count"],
        "tokens2_count": length_meta["tokens2_count"],
        "composite_score": round(composite_score, 1),
        "rule_status": rule_status,
        "rule_badge": rule_badge,
        "rule_color": rule_color
    }


class DuplicateDetectorModel:
    """
    Supervised Machine Learning Classifier wrapper.
    Supports Logistic Regression and Random Forest.
    """

    def __init__(self, model_type: str = "rf"):
        self.model_type = model_type
        if model_type == "lr":
            self.model = LogisticRegression(max_iter=1000)
        else:
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.is_trained = False
        self.metrics = {}

    def train(self, X: np.ndarray, y: np.ndarray, test_size: float = 0.2) -> Dict[str, Any]:
        """Trains the classifier and calculates evaluation metrics."""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        self.model.fit(X_train, y_train)
        self.is_trained = True

        # Predictions on test set
        y_pred = self.model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average="weighted", zero_division=0
        )
        cm = confusion_matrix(y_test, y_pred)

        self.metrics = {
            "accuracy": round(float(acc) * 100, 2),
            "precision": round(float(precision) * 100, 2),
            "recall": round(float(recall) * 100, 2),
            "f1_score": round(float(f1) * 100, 2),
            "confusion_matrix": cm.tolist(),
            "test_sample_count": len(y_test),
            "train_sample_count": len(y_train)
        }
        return self.metrics

    def predict(self, feature_vector: np.ndarray) -> Dict[str, Any]:
        """Predicts class and probability for a given feature vector."""
        if not self.is_trained:
            # Fallback heuristic if ML model artifact not yet trained
            vec = feature_vector.flatten()
            comp = (0.4 * vec[0] + 0.6 * vec[1]) * 100
            if comp >= 82.0:
                pred_label = "Duplicate"
                conf = comp
            elif comp >= 58.0:
                pred_label = "Similar"
                conf = comp
            else:
                pred_label = "Different"
                conf = 100 - comp
            return {
                "prediction": pred_label,
                "confidence": round(conf, 1),
                "probabilities": {"Different": 0.0, "Similar": 0.0, "Duplicate": 0.0}
            }

        X_in = feature_vector.reshape(1, -1)
        pred_idx = self.model.predict(X_in)[0]
        pred_label = LABEL_MAPPING.get(pred_idx, "Similar")

        probs = self.model.predict_proba(X_in)[0]
        classes = self.model.classes_
        prob_dict = {
            LABEL_MAPPING.get(cls_idx, str(cls_idx)): round(float(prob) * 100, 1)
            for cls_idx, prob in zip(classes, probs)
        }
        confidence = prob_dict.get(pred_label, round(float(np.max(probs)) * 100, 1))

        return {
            "prediction": pred_label,
            "confidence": confidence,
            "probabilities": prob_dict
        }

    def save(self, filepath: str):
        """Saves model to disk."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump({"model": self.model, "metrics": self.metrics, "type": self.model_type}, filepath)

    @classmethod
    def load(cls, filepath: str) -> "DuplicateDetectorModel":
        """Loads model from disk."""
        data = joblib.load(filepath)
        instance = cls(model_type=data.get("type", "rf"))
        instance.model = data["model"]
        instance.metrics = data.get("metrics", {})
        instance.is_trained = True
        return instance


def get_default_model() -> DuplicateDetectorModel:
    """Attempts to load pre-trained model from disk, otherwise returns fallback instance."""
    model_path = os.path.join(DEFAULT_MODEL_DIR, "classifier.joblib")
    if os.path.exists(model_path):
        try:
            return DuplicateDetectorModel.load(model_path)
        except Exception as e:
            print(f"[Warning] Failed to load trained model: {e}")
    return DuplicateDetectorModel()
