"""
Unit and Integration Tests for Assignment Duplicate Detector
Verifies:
1. Document parsing (TXT, DOCX)
2. NLP preprocessing (tokens, lemmatization)
3. TF-IDF, SBERT, and Jaccard similarity metrics
4. ML classifier prediction & confidence
5. Sentence-level matching
6. Classroom batch matrix computation
7. Audit report generation
"""

import os
import sys

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, BASE_DIR)

from src.document_parser import parse_document
from src.preprocessing import preprocess_text, split_into_sentences
from src.feature_extraction import (
    compute_tfidf_similarity,
    compute_jaccard_similarity,
    compute_word_overlap_ratio,
    compute_batch_tfidf_matrix
)
from src.semantic_similarity import compute_semantic_similarity, compute_batch_semantic_matrix
from src.ml_classifier import extract_pair_features, get_default_model
from src.sentence_analyzer import analyze_sentence_matches, generate_highlighted_html
from src.report_generator import generate_text_report


def test_end_to_end_pipeline():
    print("=== RUNNING END-TO-END PIPELINE TESTS ===")

    samples_dir = os.path.join(BASE_DIR, "data", "samples")
    f_orig = os.path.join(samples_dir, "Student_01_AI_Original.txt")
    f_dup = os.path.join(samples_dir, "Student_02_AI_Duplicate.txt")
    f_para = os.path.join(samples_dir, "Student_03_AI_Paraphrased.docx")
    f_diff = os.path.join(samples_dir, "Student_06_DBMS_Different.docx")

    # 1. Test Document Parser
    print("[Test 1] Testing Document Parser...")
    text_orig = parse_document(f_orig)
    text_dup = parse_document(f_dup)
    text_para = parse_document(f_para)
    text_diff = parse_document(f_diff)
    assert len(text_orig) > 50, "Original text too short"
    assert len(text_para) > 50, "Paraphrased text too short"
    assert len(text_diff) > 50, "DBMS text too short"
    print("  -> Passed: Parsed TXT and DOCX files successfully.")

    # 2. Test Preprocessing
    print("[Test 2] Testing NLP Preprocessing...")
    tokens, cleaned = preprocess_text(text_orig)
    assert len(tokens) > 10, "Preprocessing produced too few tokens"
    assert "machine" in tokens or "learn" in tokens, "Missing expected tokens"
    sents = split_into_sentences(text_orig)
    assert len(sents) >= 4, f"Expected >= 4 sentences, got {len(sents)}"
    print(f"  -> Passed: {len(tokens)} clean tokens, {len(sents)} sentences.")

    # 3. Test Similarity on Duplicate Pair
    print("[Test 3] Testing Similarity on Duplicate Pair...")
    res_dup = extract_pair_features(text_orig, text_dup)
    assert res_dup["composite_score"] >= 98.0, f"Expected duplicate >= 98%, got {res_dup['composite_score']}%"
    assert res_dup["rule_status"] == "Duplicate", f"Expected Duplicate, got {res_dup['rule_status']}"
    print(f"  -> Passed: Duplicate score = {res_dup['composite_score']}%")

    # 4. Test Similarity on Paraphrased Pair
    print("[Test 4] Testing Similarity on Paraphrased Pair...")
    res_para = extract_pair_features(text_orig, text_para)
    assert res_para["semantic_sim"] >= 75.0, f"Expected high semantic similarity, got {res_para['semantic_sim']}%"
    print(f"  -> Passed: Paraphrase Composite = {res_para['composite_score']}%, Semantic = {res_para['semantic_sim']}%")

    # 5. Test Similarity on Different Pair
    print("[Test 5] Testing Similarity on Different Pair...")
    res_diff = extract_pair_features(text_orig, text_diff)
    assert res_diff["composite_score"] < 60.0, f"Expected different < 60%, got {res_diff['composite_score']}%"
    assert res_diff["rule_status"] == "Different", f"Expected Different, got {res_diff['rule_status']}"
    print(f"  -> Passed: Different topic score = {res_diff['composite_score']}%")

    # 6. Test ML Model
    print("[Test 6] Testing ML Model Inference...")
    model = get_default_model()
    pred_dup = model.predict(res_dup["feature_vector"])
    pred_diff = model.predict(res_diff["feature_vector"])
    assert pred_dup["prediction"] == "Duplicate", f"Expected Duplicate, got {pred_dup['prediction']}"
    assert pred_diff["prediction"] == "Different", f"Expected Different, got {pred_diff['prediction']}"
    print(f"  -> Passed: ML predicted Duplicate ({pred_dup['confidence']}%) & Different ({pred_diff['confidence']}%)")

    # 7. Test Sentence Analyzer
    print("[Test 7] Testing Sentence-level Matches...")
    matches = analyze_sentence_matches(text_orig, text_para, threshold=0.60)
    assert len(matches) > 0, "Expected at least 1 matched sentence"
    print(f"  -> Passed: Detected {len(matches)} matching sentence pairs.")

    # 8. Test Classroom Batch Matrix
    print("[Test 8] Testing Classroom Batch Matrix...")
    texts = [text_orig, text_dup, text_para, text_diff]
    mat_tfidf = compute_batch_tfidf_matrix(texts)
    mat_sem = compute_batch_semantic_matrix(texts)
    assert mat_tfidf.shape == (4, 4), "TF-IDF matrix shape error"
    assert mat_sem.shape == (4, 4), "Semantic matrix shape error"
    print(f"  -> Passed: 4x4 matrix computed successfully.")

    # 9. Test Report Generator
    print("[Test 9] Testing Audit Report Generation...")
    res_dup["ml_pred"] = pred_dup
    report = generate_text_report("Orig.txt", "Dup.txt", res_dup, matches)
    assert "ACADEMIC ASSIGNMENT DUPLICATE DETECTION" in report
    assert "Overall Similarity Score" in report
    print("  -> Passed: Report generated with length " + str(len(report)) + " chars.")

    print("\n" + "=" * 50)
    print("ALL TESTS PASSED SUCCESSFULLY!")
    print("=" * 50)


if __name__ == "__main__":
    test_end_to_end_pipeline()
