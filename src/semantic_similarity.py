"""
Semantic Similarity Module
Uses Dense Sentence Embeddings (Sentence-Transformers / SBERT)
Model: all-MiniLM-L6-v2 (fast, lightweight, highly accurate for paraphrasing & semantics)
Provides fallback to spaCy embeddings if sentence-transformers is loading or offline.
"""

from typing import List, Union
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

_SBERT_MODEL = None
_SBERT_FAILED = False


def get_sbert_model():
    """Lazily load SentenceTransformer model with fallback."""
    global _SBERT_MODEL, _SBERT_FAILED
    if _SBERT_MODEL is not None:
        return _SBERT_MODEL

    if _SBERT_FAILED:
        return None

    try:
        from sentence_transformers import SentenceTransformer
        # all-MiniLM-L6-v2 is compact (~80MB), fast, and state-of-the-art for sentence similarity
        _SBERT_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
        return _SBERT_MODEL
    except Exception as e:
        print(f"[Notice] SentenceTransformer not ready or downloading ({e}). Using spaCy fallback.")
        _SBERT_FAILED = True
        return None


def get_spacy_embedding(text: str) -> np.ndarray:
    """Fallback semantic vector using spaCy average token vectors."""
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(text)
        return doc.vector
    except Exception:
        # Emergency dummy vector
        return np.zeros(96, dtype=np.float32)


def encode_texts(texts: List[str]) -> np.ndarray:
    """Encodes a list of texts into dense vectors."""
    if not texts:
        return np.array([])

    model = get_sbert_model()
    if model is not None:
        try:
            return model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        except Exception:
            pass

    # Fallback to spaCy vectors
    vectors = [get_spacy_embedding(t) for t in texts]
    return np.array(vectors)


def compute_semantic_similarity(text1: str, text2: str) -> float:
    """
    Computes Semantic Cosine Similarity between dense embeddings of two documents.
    Returns float in range [0.0, 1.0].
    """
    if not text1.strip() or not text2.strip():
        return 0.0

    model = get_sbert_model()
    if model is not None:
        try:
            emb = model.encode([text1, text2], convert_to_numpy=True, show_progress_bar=False)
            sim = cosine_similarity(emb[0:1], emb[1:2])[0][0]
            return float(np.clip(sim, 0.0, 1.0))
        except Exception as e:
            print(f"[Warning] SBERT encoding failed, using fallback: {e}")

    # Fallback computation
    v1 = get_spacy_embedding(text1).reshape(1, -1)
    v2 = get_spacy_embedding(text2).reshape(1, -1)
    norm1, norm2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if norm1 > 0 and norm2 > 0:
        sim = np.dot(v1, v2.T)[0][0] / (norm1 * norm2)
        return float(np.clip(sim, 0.0, 1.0))
    return 0.0


def compute_batch_semantic_matrix(texts: List[str]) -> np.ndarray:
    """
    Computes pairwise semantic cosine similarity matrix for a list of texts.
    Returns N x N numpy matrix.
    """
    if not texts:
        return np.array([[]])

    embeddings = encode_texts(texts)
    if embeddings.size == 0 or np.all(embeddings == 0):
        # Fallback identity
        n = len(texts)
        return np.eye(n)

    sim_matrix = cosine_similarity(embeddings, embeddings)
    return np.clip(sim_matrix, 0.0, 1.0)
