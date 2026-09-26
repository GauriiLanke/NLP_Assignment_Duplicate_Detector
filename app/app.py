"""
Assignment Duplicate Detector - Streamlit Web Application
Modern, high-performance NLP & Machine Learning system for academic duplicate detection.
"""

import os
import sys
import io
import time
from typing import List, Dict, Any
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Ensure project root is in sys.path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.document_parser import parse_document
from src.preprocessing import preprocess_text, get_preprocessing_steps_breakdown
from src.feature_extraction import (
    compute_tfidf_similarity,
    compute_batch_tfidf_matrix,
    compute_jaccard_similarity,
    compute_word_overlap_ratio,
    extract_shared_top_keywords
)
from src.semantic_similarity import compute_semantic_similarity, compute_batch_semantic_matrix
from src.sentence_analyzer import analyze_sentence_matches, generate_highlighted_html
from src.ml_classifier import extract_pair_features, get_default_model, DuplicateDetectorModel
from src.report_generator import generate_text_report

# Page Configuration
st.set_page_config(
    page_title="Assignment Duplicate Detector",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load Custom Blue Shade Design System CSS
css_file = os.path.join(os.path.dirname(__file__), "style.css")
if os.path.exists(css_file):
    with open(css_file, "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# Helper: Load Sample Assignments
@st.cache_data
def get_sample_files() -> Dict[str, str]:
    samples_dir = os.path.join(BASE_DIR, "data", "samples")
    samples = {}
    if os.path.exists(samples_dir):
        for fname in sorted(os.listdir(samples_dir)):
            fpath = os.path.join(samples_dir, fname)
            try:
                samples[fname] = parse_document(fpath)
            except Exception:
                pass
    return samples


# Header Banner
st.markdown("""
<div class="main-header">
    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 15px;">
        <div>
            <h1>📚 Assignment Duplicate Detector</h1>
            <p>Smart Textual & Semantic Similarity Detection for Academic Submissions</p>
        </div>
        <div>
            <span class="header-badge">
                ✨ AI & NLP Powered
            </span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Tight Executive Dark Navy Sidebar Configuration
st.sidebar.markdown("## Detection Settings")
st.sidebar.caption("Threshold sensitivity:")

dup_thresh = st.sidebar.slider("Duplicate Threshold (%)", min_value=70, max_value=95, value=82, step=1)
sim_thresh = st.sidebar.slider("Similar Threshold (%)", min_value=40, max_value=75, value=58, step=1)

st.sidebar.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
st.sidebar.markdown("### Preprocessing Controls")
use_lemmatization = st.sidebar.checkbox("Enable Lemmatization (spaCy)", value=True)
remove_stopwords = st.sidebar.checkbox("Remove Stopwords", value=True)

# Load Trained ML Model
ml_model = get_default_model()

# Clean 2-Tab Layout
tab1, tab2 = st.tabs([
    "🔍 Pairwise Assignment Detector",
    "👥 Classroom Batch Detector"
])

# ==============================================================================
# TAB 1: PAIRWISE DETECTOR
# ==============================================================================
with tab1:
    st.subheader("Compare Two Student Assignments")
    st.caption("Upload files (PDF, DOCX, TXT) or paste assignment text directly.")

    input_mode = st.radio(
        "Select Input Source:",
        ["📁 Upload Documents (PDF / DOCX / TXT)", "✍️ Paste Text Directly"],
        horizontal=True
    )

    text_a, text_b = "", ""
    name_a, name_b = "Assignment 1", "Assignment 2"

    if input_mode == "📁 Upload Documents (PDF / DOCX / TXT)":
        col1, col2 = st.columns(2)
        with col1:
            file_a = st.file_uploader("Upload Assignment 1", type=["pdf", "docx", "txt"], key="file_a")
            if file_a is not None:
                try:
                    text_a = parse_document(file_a)
                    name_a = file_a.name
                    st.success(f"Successfully loaded `{name_a}` ({len(text_a.split())} words)")
                except Exception as e:
                    st.error(f"Error parsing file 1: {e}")

        with col2:
            file_b = st.file_uploader("Upload Assignment 2", type=["pdf", "docx", "txt"], key="file_b")
            if file_b is not None:
                try:
                    text_b = parse_document(file_b)
                    name_b = file_b.name
                    st.success(f"Successfully loaded `{name_b}` ({len(text_b.split())} words)")
                except Exception as e:
                    st.error(f"Error parsing file 2: {e}")

    else:
        col1, col2 = st.columns(2)
        with col1:
            text_a = st.text_area(
                "Assignment A Content:",
                height=220,
                placeholder="Paste the first assignment text here...",
                value="Machine learning is a subset of artificial intelligence that enables computers to learn from data and improve their accuracy without explicit programming."
            )
            name_a = "Student_A_DirectText"
        with col2:
            text_b = st.text_area(
                "Assignment B Content:",
                height=220,
                placeholder="Paste the second assignment text here...",
                value="Machine learning is a branch within artificial intelligence where computational systems learn patterns from training data to boost precision."
            )
            name_b = "Student_B_DirectText"

    # Action Run Button without emoji
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Run Duplicate Detection", type="primary", use_container_width=True):
        if not text_a.strip() or not text_b.strip():
            st.error("Please provide text or files for both assignments to run the analysis.")
        else:
            with st.spinner("Analyzing assignment text with TF-IDF and SBERT semantic embeddings..."):
                start_time = time.time()
                pair_features = extract_pair_features(text_a, text_b)
                comp_score = pair_features["composite_score"]
                tfidf_score = pair_features["tfidf_sim"]
                semantic_score = pair_features["semantic_sim"]
                jaccard_score = pair_features["jaccard_sim"]
                overlap_score = pair_features["word_overlap"]

                # Custom threshold evaluation
                if comp_score >= dup_thresh:
                    status_class = "status-duplicate"
                    status_text = "🔴 HIGHLY SIMILAR / DUPLICATE"
                elif comp_score >= sim_thresh:
                    status_class = "status-similar"
                    status_text = "🟠 POSSIBLY SIMILAR (PARAPHRASED)"
                else:
                    status_class = "status-different"
                    status_text = "🟢 DIFFERENT / UNIQUE"

                # ML Classifier prediction
                ml_pred_res = ml_model.predict(pair_features["feature_vector"])
                pair_features["ml_pred"] = ml_pred_res
                pair_features["rule_status"] = "Duplicate" if comp_score >= dup_thresh else ("Similar" if comp_score >= sim_thresh else "Different")

                # Sentence matches
                matched_sentences = analyze_sentence_matches(text_a, text_b, threshold=0.55)
                elapsed = time.time() - start_time

                # Save results in session state for seamless filtering
                st.session_state["analysis_done"] = True
                st.session_state["pair_features"] = pair_features
                st.session_state["status_class"] = status_class
                st.session_state["status_text"] = status_text
                st.session_state["matched_sentences"] = matched_sentences
                st.session_state["text_a"] = text_a
                st.session_state["text_b"] = text_b
                st.session_state["name_a"] = name_a
                st.session_state["name_b"] = name_b

    # Render Results Section if analysis has been executed
    if st.session_state.get("analysis_done", False):
        pair_features = st.session_state["pair_features"]
        status_class = st.session_state["status_class"]
        status_text = st.session_state["status_text"]
        matched_sentences = st.session_state["matched_sentences"]
        text_a = st.session_state["text_a"]
        text_b = st.session_state["text_b"]
        name_a = st.session_state["name_a"]
        name_b = st.session_state["name_b"]

        comp_score = pair_features["composite_score"]
        tfidf_score = pair_features["tfidf_sim"]
        semantic_score = pair_features["semantic_sim"]
        jaccard_score = pair_features["jaccard_sim"]
        overlap_score = pair_features["word_overlap"]
        ml_pred_res = pair_features.get("ml_pred", {})

        st.markdown("---")
        # Attractive Result Banner
        st.markdown(f"""
        <div class="status-badge {status_class}">
            <span>{status_text}</span>
            <span>Overall Similarity: <strong>{comp_score}%</strong></span>
        </div>
        """, unsafe_allow_html=True)

        # Metric Cards Grid with Top Accent Borders
        mcol1, mcol2, mcol3, mcol4, mcol5 = st.columns(5)
        with mcol1:
            st.markdown(f"""
            <div class="metric-card metric-card-composite">
                <div class="label">Composite Match</div>
                <div class="value" style="color: #2563eb;">{comp_score}%</div>
            </div>
            """, unsafe_allow_html=True)
        with mcol2:
            st.markdown(f"""
            <div class="metric-card metric-card-tfidf">
                <div class="label">TF-IDF Similarity</div>
                <div class="value" style="color: #4f46e5;">{tfidf_score}%</div>
            </div>
            """, unsafe_allow_html=True)
        with mcol3:
            st.markdown(f"""
            <div class="metric-card metric-card-semantic">
                <div class="label">Semantic (SBERT)</div>
                <div class="value" style="color: #0284c7;">{semantic_score}%</div>
            </div>
            """, unsafe_allow_html=True)
        with mcol4:
            st.markdown(f"""
            <div class="metric-card metric-card-jaccard">
                <div class="label">Jaccard Index</div>
                <div class="value" style="color: #0ea5e9;">{jaccard_score}%</div>
            </div>
            """, unsafe_allow_html=True)
        with mcol5:
            st.markdown(f"""
            <div class="metric-card metric-card-overlap">
                <div class="label">Word Overlap</div>
                <div class="value" style="color: #0369a1;">{overlap_score}%</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Visual Gauge & ML Model Verdict
        gcol1, gcol2 = st.columns([1, 1])

        with gcol1:
            st.markdown("#### 🎯 Overall Similarity Dial")
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=comp_score,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Composite Match Score (%)", 'font': {'size': 16, 'color': '#64748b'}},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#94a3b8"},
                    'bar': {'color': "#2563eb"},
                    'bgcolor': "white",
                    'borderwidth': 1,
                    'bordercolor': "#cbd5e1",
                    'steps': [
                        {'range': [0, sim_thresh], 'color': 'rgba(14, 165, 233, 0.15)'},
                        {'range': [sim_thresh, dup_thresh], 'color': 'rgba(2, 132, 199, 0.20)'},
                        {'range': [dup_thresh, 100], 'color': 'rgba(37, 99, 235, 0.25)'}
                    ],
                    'threshold': {
                        'line': {'color': "#1d4ed8", 'width': 4},
                        'thickness': 0.75,
                        'value': dup_thresh
                    }
                }
            ))
            fig_gauge.update_layout(height=250, margin=dict(l=20, r=20, t=30, b=20), paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_gauge, use_container_width=True)

        with gcol2:
            st.markdown("#### 🤖 ML Classification Verdict")
            st.markdown(f"**Predicted Status:** `{ml_pred_res.get('prediction', 'N/A').upper()}`")
            st.markdown(f"**Confidence Level:** `{ml_pred_res.get('confidence', 0)}%`")

            # Probability Distribution Bar Chart
            probs = ml_pred_res.get("probabilities", {})
            if probs and any(probs.values()):
                df_probs = pd.DataFrame({
                    "Class": list(probs.keys()),
                    "Probability (%)": list(probs.values())
                })
                fig_bar = px.bar(
                    df_probs,
                    x="Class",
                    y="Probability (%)",
                    color="Class",
                    color_discrete_map={"Different": "#0ea5e9", "Similar": "#0284c7", "Duplicate": "#2563eb"},
                    text="Probability (%)"
                )
                fig_bar.update_layout(height=190, margin=dict(l=20, r=20, t=10, b=10), showlegend=False, paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_bar, use_container_width=True)

        # Key Shared Keywords
        shared_keywords = extract_shared_top_keywords(text_a, text_b, top_k=8)
        if shared_keywords:
            st.markdown("#### 🔑 Top Shared Terms (TF-IDF Weighted)")
            kw_html = " ".join([
                f"<span style='background:#eff6ff; color:#1e40af; border:1px solid #93c5fd; padding:6px 14px; border-radius:20px; margin-right:8px; font-weight:600; font-size:0.88rem; display:inline-block;'>{kw} ({score:.2f})</span>"
                for kw, score in shared_keywords
            ])
            st.markdown(kw_html, unsafe_allow_html=True)

        st.markdown("---")

        # SENTENCE-LEVEL SIMILARITY INSPECTOR
        st.markdown("### 🔍 Sentence-Level Plagiarism Inspector")
        st.caption(f"Found **{len(matched_sentences)}** matching or paraphrased sentences above threshold.")

        if matched_sentences:
            filter_choice = st.radio(
                "Filter Match Severity:",
                ["All Matches", "🔴 Exact / High Plagiarism (>= 85%)", "🔵 Paraphrased Matches (60% - 84%)", "⚪ Conceptual Matches (< 60%)"],
                horizontal=True,
                key="sentence_filter_radio"
            )

            filtered_matches = matched_sentences
            if "Exact" in filter_choice:
                filtered_matches = [m for m in matched_sentences if m["similarity_score"] >= 85.0]
            elif "Paraphrased" in filter_choice:
                filtered_matches = [m for m in matched_sentences if 60.0 <= m["similarity_score"] < 85.0]
            elif "Conceptual" in filter_choice:
                filtered_matches = [m for m in matched_sentences if m["similarity_score"] < 60.0]

            if filtered_matches:
                for m in filtered_matches:
                    card_class = (
                        "match-card-high" if m["similarity_score"] >= 85.0
                        else ("match-card-medium" if m["similarity_score"] >= 60.0 else "match-card-low")
                    )
                    st.markdown(f"""
                    <div class="match-card {card_class}">
                        <div style="display:flex; justify-content:space-between; font-weight:700; font-size:0.95rem; margin-bottom:0.5rem;">
                            <span style="color:{m['color']};">{m['severity']}</span>
                            <span>Score: {m['similarity_score']}% (Semantic: {m['semantic_score']}%, TF-IDF: {m['lexical_score']}%)</span>
                        </div>
                        <div style="font-size:0.92rem; margin-bottom:0.4rem;">
                            <strong>Assignment A (Sentence #{m['sent1_index']}):</strong> "{m['sentence_a']}"
                        </div>
                        <div style="font-size:0.92rem;">
                            <strong>Assignment B (Sentence #{m['sent2_index']}):</strong> "{m['sentence_b']}"
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info(f"ℹ️ No sentence matches found matching criteria: '{filter_choice}'.")

            # Side-by-side Highlighted Documents View
            with st.expander("📄 View Side-by-Side Highlighted Documents"):
                hcol1, hcol2 = st.columns(2)
                matched_a = [m["sentence_a"] for m in matched_sentences]
                matched_b = [m["sentence_b"] for m in matched_sentences]

                with hcol1:
                    st.markdown(f"**{name_a} (Highlighted Matches)**")
                    html_a = generate_highlighted_html(text_a, matched_a)
                    st.markdown(f"<div class='doc-box'>{html_a}</div>", unsafe_allow_html=True)

                with hcol2:
                    st.markdown(f"**{name_b} (Highlighted Matches)**")
                    html_b = generate_highlighted_html(text_b, matched_b)
                    st.markdown(f"<div class='doc-box'>{html_b}</div>", unsafe_allow_html=True)
        else:
            st.success("✅ No matching sentences found between these two assignments!")

        # Download Audit Report Button
        st.markdown("<br>", unsafe_allow_html=True)
        report_text = generate_text_report(name_a, name_b, pair_features, matched_sentences)
        st.download_button(
            label="📥 Download Detailed Audit Report (.txt)",
            data=report_text,
            file_name=f"Duplicate_Report_{name_a}_vs_{name_b}.txt",
            mime="text/plain",
            use_container_width=True
        )

# ==============================================================================
# TAB 2: CLASSROOM BATCH DETECTOR
# ==============================================================================
with tab2:
    st.subheader("Classroom Batch Duplicate Detector (All-vs-All)")
    st.caption("Upload multiple student submissions (PDF, DOCX, TXT) to generate a complete similarity matrix heatmap and uncover duplicate clusters.")

    uploaded_files = st.file_uploader(
        "Upload Classroom Assignments (3 to 20 files):",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True
    )

    batch_docs = {}
    if uploaded_files:
        for uf in uploaded_files:
            try:
                batch_docs[uf.name] = parse_document(uf)
            except Exception as e:
                st.warning(f"Could not parse `{uf.name}`: {e}")
        if batch_docs:
            st.success(f"Successfully loaded {len(batch_docs)} student files.")

    if len(batch_docs) >= 2:
        if st.button("Generate Classroom Similarity Matrix", type="primary", use_container_width=True):
            with st.spinner("Computing pairwise similarities across all student assignments..."):
                doc_names = list(batch_docs.keys())
                doc_texts = list(batch_docs.values())
                n = len(doc_names)

                # Compute pairwise composite matrix
                tfidf_matrix = compute_batch_tfidf_matrix(doc_texts)
                semantic_matrix = compute_batch_semantic_matrix(doc_texts)
                composite_matrix = (0.40 * tfidf_matrix + 0.60 * semantic_matrix) * 100.0

                # Form table of ranked pairs
                pair_records = []
                for i in range(n):
                    for j in range(i + 1, n):
                        score = round(float(composite_matrix[i, j]), 1)
                        if score >= dup_thresh:
                            v = "🔴 Duplicate"
                        elif score >= sim_thresh:
                            v = "🟠 Similar"
                        else:
                            v = "🟢 Different"

                        pair_records.append({
                            "Student 1": doc_names[i],
                            "Student 2": doc_names[j],
                            "Composite Similarity (%)": score,
                            "TF-IDF (%)": round(float(tfidf_matrix[i, j]) * 100, 1),
                            "Semantic (%)": round(float(semantic_matrix[i, j]) * 100, 1),
                            "Verdict": v
                        })

                df_pairs = pd.DataFrame(pair_records).sort_values(by="Composite Similarity (%)", ascending=False)

            # Interactive Heatmap in Blue Scale
            st.markdown("### 🗺️ Pairwise Similarity Matrix Heatmap")
            short_labels = [name.replace(".txt", "").replace(".docx", "").replace(".pdf", "") for name in doc_names]

            fig_heat = px.imshow(
                composite_matrix,
                x=short_labels,
                y=short_labels,
                color_continuous_scale="Blues",
                zmin=0,
                zmax=100,
                text_auto=".1f",
                aspect="auto",
                labels=dict(color="Similarity %")
            )
            fig_heat.update_layout(
                height=500,
                xaxis_title="Assignments",
                yaxis_title="Assignments",
                margin=dict(l=40, r=40, t=30, b=40),
                paper_bgcolor="rgba(0,0,0,0)"
            )
            st.plotly_chart(fig_heat, use_container_width=True)

            # Collusion / Duplicate Group Clustering
            st.markdown("### 👥 Collusion & Duplicate Group Clusters")
            parent = {i: i for i in range(n)}

            def find(i):
                if parent[i] == i:
                    return i
                parent[i] = find(parent[i])
                return parent[i]

            def union(i, j):
                root_i = find(i)
                root_j = find(j)
                if root_i != root_j:
                    parent[root_i] = root_j

            for i in range(n):
                for j in range(i + 1, n):
                    if composite_matrix[i, j] >= dup_thresh:
                        union(i, j)

            groups = {}
            for i in range(n):
                root = find(i)
                groups.setdefault(root, []).append(doc_names[i])

            cluster_num = 1
            has_collusion = False
            for root, members in groups.items():
                if len(members) > 1:
                    has_collusion = True
                    st.markdown(f"""
                    <div style="background:#eff6ff; border-left:5px solid #2563eb; padding:0.9rem 1.2rem; border-radius:12px; margin-bottom:0.75rem; box-shadow: 0 4px 12px rgba(37,99,235,0.12);">
                        <strong style="color:#1e40af;">⚠️ Duplicate Group #{cluster_num}:</strong> &nbsp;
                        {" ".join([f"<span class='cluster-pill'>{m}</span>" for m in members])}
                        <br><span style="font-size:0.85rem; color:#1e3a8a;">These students share identical or heavily overlapping assignment submissions.</span>
                    </div>
                    """, unsafe_allow_html=True)
                    cluster_num += 1

            if not has_collusion:
                st.success("✅ No multi-student duplicate clusters detected! Submissions are unique.")

            # Ranked Suspect Pairs Table
            st.markdown("### 📋 Ranked Suspect Assignment Pairs")
            st.dataframe(df_pairs, use_container_width=True)

            # Export CSV Button
            csv_batch = df_pairs.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Export Batch Matrix Results as CSV",
                data=csv_batch,
                file_name="Classroom_Duplicate_Matrix_Results.csv",
                mime="text/csv",
                use_container_width=True
            )
    else:
        st.info("Please upload at least 2 assignment files for classroom batch analysis.")
