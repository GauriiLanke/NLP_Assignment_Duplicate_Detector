"""
Report Generator Module
Generates human-readable, professional audit and similarity reports
for student assignment duplicate detection.
"""

from datetime import datetime
from typing import Dict, List, Any


def generate_text_report(
    doc1_name: str,
    doc2_name: str,
    results: Dict[str, Any],
    matched_sentences: List[Dict[str, Any]]
) -> str:
    """
    Builds a cleanly formatted markdown / text report suitable for saving or printing.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    rule_status = results.get("rule_status", "N/A")
    comp_score = results.get("composite_score", 0.0)
    tfidf = results.get("tfidf_sim", 0.0)
    semantic = results.get("semantic_sim", 0.0)
    jaccard = results.get("jaccard_sim", 0.0)
    overlap = results.get("word_overlap", 0.0)
    t1 = results.get("tokens1_count", 0)
    t2 = results.get("tokens2_count", 0)

    ml_info = results.get("ml_pred", {})
    ml_pred = ml_info.get("prediction", "N/A")
    ml_conf = ml_info.get("confidence", 0.0)

    report_lines = [
        "=" * 70,
        "          ACADEMIC ASSIGNMENT DUPLICATE DETECTION AUDIT REPORT",
        "=" * 70,
        f"Generated At    : {timestamp}",
        f"Assignment 1    : {doc1_name} ({t1} clean tokens)",
        f"Assignment 2    : {doc2_name} ({t2} clean tokens)",
        "-" * 70,
        "1. DETECTION SUMMARY",
        f"  Overall Similarity Score : {comp_score}%",
        f"  Rule-based Verdict       : {rule_status.upper()}",
        f"  ML Model Prediction      : {ml_pred.upper()} (Confidence: {ml_conf}%)",
        "-" * 70,
        "2. DETAILED NLP METRIC BREAKDOWN",
        f"  • TF-IDF Cosine Similarity  : {tfidf}% (Lexical & keyword overlap)",
        f"  • Semantic SBERT Similarity : {semantic}% (Dense conceptual meaning)",
        f"  • Jaccard Set Similarity    : {jaccard}% (Unique vocabulary intersection)",
        f"  • Word Overlap Ratio        : {overlap}% (Dice token agreement)",
        "-" * 70,
        f"3. SUSPICIOUS / MATCHED SENTENCES DETECTED ({len(matched_sentences)} found)",
    ]

    if not matched_sentences:
        report_lines.append("  No matching or heavily paraphrased sentences detected above threshold.")
    else:
        for idx, match in enumerate(matched_sentences[:10], 1):
            report_lines.extend([
                f"\n  [Match #{idx}] Severity: {match['severity']} (Score: {match['similarity_score']}%)",
                f"    Assignment A: \"{match['sentence_a']}\"",
                f"    Assignment B: \"{match['sentence_b']}\"",
                f"    Breakdown   : Semantic {match['semantic_score']}% | Lexical {match['lexical_score']}%"
            ])

    report_lines.extend([
        "\n" + "-" * 70,
        "4. ACADEMIC EVALUATION & DISCLAIMER",
        "  Note: This report identifies textual and semantic similarity using NLP.",
        "  High similarity indicates potential duplication or shared templates.",
        "  Final academic integrity decisions must be verified through manual faculty review.",
        "=" * 70
    ])

    return "\n".join(report_lines)
