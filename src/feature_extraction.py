"""
Feature Extraction Module
Calculates lexical and traditional NLP similarity features:
- TF-IDF with Cosine Similarity
- Jaccard Similarity (word sets)
- Word Overlap Ratio
- Document Length Ratios
- Key Shared Terms & Top Keywords
"""

from typing import Dict, List, Set, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def compute_tfidf_similarity(text1: str, text2: str) -> float:
    """
    Computes Cosine Similarity between TF-IDF representations of two texts.
    Returns float in range [0.0, 1.0].
    """
    if not text1.strip() or not text2.strip():
        return 0.0

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        sublinear_tf=True
    )
    try:
        tfidf_matrix = vectorizer.fit_transform([text1, text2])
        sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        return float(np.clip(sim, 0.0, 1.0))
    except Exception:
        return 0.0


def compute_batch_tfidf_matrix(texts: List[str]) -> np.ndarray:
    """
    Computes pairwise TF-IDF cosine similarity matrix for a list of texts.
    Returns N x N numpy matrix.
    """
    if not texts:
        return np.array([[]])
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
    matrix = vectorizer.fit_transform(texts)
    sim_matrix = cosine_similarity(matrix, matrix)
    return np.clip(sim_matrix, 0.0, 1.0)


def compute_jaccard_similarity(tokens1: List[str], tokens2: List[str]) -> float:
    """
    Computes Jaccard Similarity on token sets:
    J(A, B) = |A ∩ B| / |A ∪ B|
    """
    set1, set2 = set(tokens1), set(tokens2)
    if not set1 or not set2:
        return 0.0
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return float(intersection / union) if union > 0 else 0.0


def compute_word_overlap_ratio(tokens1: List[str], tokens2: List[str]) -> float:
    """
    Computes Dice coefficient / Word Overlap ratio:
    Overlap = (2 * |A ∩ B|) / (|A| + |B|)
    """
    set1, set2 = set(tokens1), set(tokens2)
    if not set1 or not set2:
        return 0.0
    common = len(set1.intersection(set2))
    total = len(set1) + len(set2)
    return float(2 * common / total) if total > 0 else 0.0


def compute_length_features(text1: str, text2: str, tokens1: List[str], tokens2: List[str]) -> Dict[str, float]:
    """
    Computes length symmetry and differences:
    - Char ratio: min(len1, len2) / max(len1, len2)
    - Token ratio: min(t1, t2) / max(t1, t2)
    - Absolute difference in tokens
    """
    len_char1, len_char2 = len(text1), len(text2)
    max_char = max(len_char1, len_char2, 1)
    min_char = min(len_char1, len_char2)
    char_ratio = min_char / max_char

    t1, t2 = len(tokens1), len(tokens2)
    max_t = max(t1, t2, 1)
    min_t = min(t1, t2)
    token_ratio = min_t / max_t
    token_diff = abs(t1 - t2)

    return {
        "char_ratio": float(char_ratio),
        "token_ratio": float(token_ratio),
        "token_diff": float(token_diff),
        "tokens1_count": t1,
        "tokens2_count": t2
    }


def extract_shared_top_keywords(text1: str, text2: str, top_k: int = 8) -> List[Tuple[str, float]]:
    """
    Finds top overlapping terms between two texts weighted by their combined TF-IDF importance.
    """
    if not text1.strip() or not text2.strip():
        return []

    try:
        vectorizer = TfidfVectorizer(ngram_range=(1, 1), stop_words="english")
        tfidf_matrix = vectorizer.fit_transform([text1, text2])
        feature_names = vectorizer.get_feature_names_out()

        v1 = tfidf_matrix[0].toarray().flatten()
        v2 = tfidf_matrix[1].toarray().flatten()

        shared_scores = []
        for i, word in enumerate(feature_names):
            if v1[i] > 0 and v2[i] > 0:
                # Combined importance score
                score = (v1[i] + v2[i]) / 2.0
                shared_scores.append((word, float(score)))

        shared_scores.sort(key=lambda x: x[1], reverse=True)
        return shared_scores[:top_k]
    except Exception:
        return []
