"""
Sentence Analyzer Module
Finds sentence-level similarities, matches, and paraphrased sections between two documents.
Generates matched sentence pairs and visual HTML highlighting.
"""

from typing import List, Dict, Any, Tuple
import numpy as np
from src.preprocessing import split_into_sentences, preprocess_text
from src.feature_extraction import compute_tfidf_similarity, compute_jaccard_similarity
from src.semantic_similarity import compute_semantic_similarity, encode_texts
from sklearn.metrics.pairwise import cosine_similarity


def analyze_sentence_matches(
    text1: str,
    text2: str,
    threshold: float = 0.55
) -> List[Dict[str, Any]]:
    """
    Compares all sentence pairs between text1 and text2.
    Returns list of matched sentence objects sorted from highest similarity to lowest.
    """
    sents1 = split_into_sentences(text1)
    sents2 = split_into_sentences(text2)

    if not sents1 or not sents2:
        return []

    # Try dense embeddings for sentence cross-matching
    try:
        emb1 = encode_texts(sents1)
        emb2 = encode_texts(sents2)
        sim_matrix = cosine_similarity(emb1, emb2)
    except Exception:
        # Fallback to TF-IDF matrix
        sim_matrix = np.zeros((len(sents1), len(sents2)))
        for i, s1 in enumerate(sents1):
            for j, s2 in enumerate(sents2):
                sim_matrix[i, j] = compute_tfidf_similarity(s1, s2)

    matched_pairs = []
    used_s2_indices = set()

    # For each sentence in sents1, find best match in sents2
    for i, s1 in enumerate(sents1):
        best_j = int(np.argmax(sim_matrix[i]))
        best_score = float(sim_matrix[i, best_j])

        # Also calculate lexical TF-IDF and Jaccard for detailed inspection
        _, p1 = preprocess_text(s1)
        _, p2 = preprocess_text(sents2[best_j])
        lexical_sim = compute_tfidf_similarity(p1, p2)
        jaccard_sim = compute_jaccard_similarity(p1.split(), p2.split())

        # Composite sentence score
        composite = 0.65 * best_score + 0.35 * lexical_sim

        if composite >= threshold:
            sim_pct = round(composite * 100, 1)

            # Categorize severity cleanly
            if sim_pct >= 85.0:
                severity = "Exact / High Plagiarism"
                color = "#1d4ed8"  # Deep Royal Blue
                bg_color = "rgba(37, 99, 235, 0.12)"
                tag = "Exact"
            elif sim_pct >= 60.0:
                severity = "Paraphrased Match"
                color = "#0284c7"  # Ocean Cyan Blue
                bg_color = "rgba(2, 132, 199, 0.12)"
                tag = "Paraphrased"
            else:
                severity = "Weak / Conceptual Overlap"
                color = "#0369a1"  # Steel Sky Blue
                bg_color = "rgba(14, 165, 233, 0.10)"
                tag = "Weak"

            matched_pairs.append({
                "sent1_index": i + 1,
                "sent2_index": best_j + 1,
                "sentence_a": s1,
                "sentence_b": sents2[best_j],
                "semantic_score": round(best_score * 100, 1),
                "lexical_score": round(lexical_sim * 100, 1),
                "jaccard_score": round(jaccard_sim * 100, 1),
                "similarity_score": sim_pct,
                "severity": severity,
                "tag": tag,
                "color": color,
                "bg_color": bg_color
            })
            used_s2_indices.add(best_j)

    # Sort descending by similarity
    matched_pairs.sort(key=lambda x: x["similarity_score"], reverse=True)
    return matched_pairs


def generate_highlighted_html(
    text: str,
    matched_sentences: List[str],
    highlight_color: str = "rgba(37, 99, 235, 0.20)"
) -> str:
    """
    Renders text with matching sentences highlighted in styled <span> tags.
    """
    if not text:
        return ""

    escaped = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    for sent in matched_sentences:
        clean_sent = sent.strip()
        if len(clean_sent) > 8 and clean_sent in text:
            # Wrap with blue highlight cap
            replacement = (
                f'<span style="background-color: {highlight_color}; '
                f'border-bottom: 2px solid #2563eb; color: #0f172a; border-radius: 4px; padding: 2px 6px; font-weight: 500;">'
                f'{clean_sent}</span>'
            )
            escaped = escaped.replace(clean_sent, replacement)

    # Convert newlines to breaks
    formatted = escaped.replace("\n", "<br>")
    return f'<div style="font-family: inherit; line-height: 1.75; font-size: 0.96rem;">{formatted}</div>'
