# 📚 NLP Mini Project: Assignment Duplicate Detector

An end-to-end Natural Language Processing (NLP) and Machine Learning system designed to detect **exact duplication, paraphrased copying, and semantic similarity** across student assignments.

Built with **Streamlit**, **scikit-learn**, **spaCy**, **NLTK**, and **Sentence-Transformers (SBERT)**.

---

## 🎯 1. Project Overview & Motivation

In academic institutions, students frequently submit assignments that are:
1. 🔴 **Exact Duplicates** (word-for-word copy / minor character insertions)
2. 🟠 **Paraphrased / Semantic Duplicates** (rewritten sentences, synonym substitutions, passive voice)
3. 🟢 **Unique / Different Topics** (distinct concepts and independent writing)

Traditional plagiarism checkers rely primarily on literal string matching or basic N-grams, failing to catch smart paraphrasing. This project combines **traditional lexical NLP (TF-IDF + Jaccard)** with **modern deep contextual embeddings (Sentence-Transformers `all-MiniLM-L6-v2`)** and **supervised Machine Learning classification (Random Forest & Logistic Regression)** to create an explainable and reliable duplicate detector.

---

## 🏗️ 2. System Architecture

```text
                  STUDENT ASSIGNMENTS
          (PDF / DOCX / TXT / Classroom Batch)
                           │
                           ▼
              [Document Text Extraction]
               (PyPDF2 & python-docx)
                           │
                           ▼
               [NLP Preprocessing Pipeline]
        • Case Normalization & Noise Removal
        • Sentence Segmentation
        • Word Tokenization
        • Stopword Removal
        • Lemmatization (spaCy `en_core_web_sm`)
                           │
            ┌──────────────┴──────────────┐
            ▼                             ▼
    [Traditional NLP]             [Modern Deep NLP]
 • TF-IDF Vectorizer (1-2 ngrams) • Dense Sentence-Transformers
 • Cosine Similarity              • `all-MiniLM-L6-v2` SBERT
 • Jaccard Index (Word sets)      • Semantic Cosine Similarity
 • Word Overlap Ratio (Dice)
            │                             │
            └──────────────┬──────────────┘
                           ▼
                [Feature Vector Assembly]
       [TF-IDF, SBERT, Jaccard, Overlap, LenRatio]
                           │
            ┌──────────────┴──────────────┐
            ▼                             ▼
    [Rule-Based Scorer]           [ML Classifier]
  Weighted Ensemble: 40% TF-IDF  Random Forest / Logistic Reg.
     + 60% SBERT Embeddings       Trained on 600+ Pair Dataset
            │                             │
            └──────────────┬──────────────┘
                           ▼
                  [Detection Verdict]
    🔴 DUPLICATE (>=82%) | 🟠 SIMILAR (58-81%) | 🟢 DIFFERENT (<58%)
                           │
         ┌─────────────────┼─────────────────┐
         ▼                 ▼                 ▼
[Sentence Inspector]  [Classroom Matrix]  [Audit Report]
Side-by-side color    N×N interactive     Downloadable
highlighting & diff   heatmap & clusters  text/PDF export
```

---

## 📂 3. Project Directory Structure

```text
E:\Assignment_Duplicate_Detector\
│
├── app\
│   ├── app.py                     # Streamlit web application
│   └── style.css                  # Custom styling
│
├── data\
│   ├── generate_dataset.py        # Benchmark dataset generator
│   ├── assignments_dataset.csv    # 600+ labeled assignment pairs
│   ├── create_sample_files.py     # Test assignment creator
│   └── samples\                   # Preloaded sample files (.txt, .docx)
│       ├── Student_01_AI_Original.txt
│       ├── Student_02_AI_Duplicate.txt
│       ├── Student_03_AI_Paraphrased.docx
│       ├── Student_04_OS_Different.txt
│       ├── Student_05_Cloud_Different.txt
│       └── Student_06_DBMS_Different.docx
│
├── models\
│   └── classifier.joblib          # Trained Random Forest / Logistic Regression model
│
├── src\
│   ├── __init__.py
│   ├── document_parser.py         # Multi-format document parser (PDF, DOCX, TXT)
│   ├── preprocessing.py           # NLP cleaning, tokenization, lemmatization
│   ├── feature_extraction.py      # TF-IDF, Jaccard, overlap, length ratios
│   ├── semantic_similarity.py     # Sentence Transformers dense embeddings
│   ├── ml_classifier.py           # Feature engineering, ML models, evaluation
│   ├── sentence_analyzer.py       # Fine-grained sentence matching & highlighting
│   └── report_generator.py        # Professional audit report generator
│
├── scripts\
│   ├── train_models.py            # Train & evaluate classifiers
│   └── run_app.bat                # 1-click Windows batch launcher
│
├── requirements.txt               # Dependencies
└── README.md                      # Project documentation & viva guide
```

---

## 🚀 4. How to Run the Project

### Method 1: One-Click Batch Launcher (Easiest)
Simply double click:
```text
E:\Assignment_Duplicate_Detector\scripts\run_app.bat
```

### Method 2: From Terminal
```bash
cd E:\Assignment_Duplicate_Detector
python -m streamlit run app/app.py
```
Then open `http://localhost:8501` in your web browser.

---

## 📊 5. Key Features Demonstrated in UI

1. **Pairwise Assignment Comparison**:
   - Supports upload of `.pdf`, `.docx`, and `.txt` files or manual text input.
   - 1-Click preloaded demo test cases for viva demonstrations.
   - Dual-engine verdict: **Weighted NLP Ensemble** and **Trained Supervised ML Model**.
2. **Sentence-Level Plagiarism Inspector**:
   - Matches individual sentences across documents and ranks them by severity.
   - Side-by-side visual document preview with color-highlighted matching sentences.
3. **Classroom Batch Detector**:
   - Analyzes submissions from an entire classroom (3 to 20+ assignments).
   - Generates an interactive **Pairwise Similarity Heatmap**.
   - Discovers **Duplicate Collusion Clusters** (groups of students sharing content).
   - Exports batch matrix to CSV.
4. **Model Analytics & Evaluation**:
   - Confusion Matrix and Accuracy/Precision/Recall/F1 metrics.
   - Feature Importance breakdown.
5. **Downloadable Audit Report**:
   - Generates a timestamped markdown/text report with line-by-line matches.

---

## 🎓 6. College Viva Q&A Guide

### Q1: What is the main drawback of TF-IDF in plagiarism detection?
> **Answer**: TF-IDF is a bag-of-words / lexical approach. It relies on exact word overlap. If a student paraphrases sentences using synonyms (e.g. replacing *"Machine learning enables computers to learn from data"* with *"AI models extract patterns from training datasets"*), TF-IDF produces a low similarity score because there are few shared words. This is why we integrate Sentence Transformers (SBERT) to capture semantic meaning.

### Q2: Why did you use `all-MiniLM-L6-v2` as your Sentence Transformer model?
> **Answer**: `all-MiniLM-L6-v2` maps sentences to a 384-dimensional dense vector space. It is specifically tuned for semantic textual similarity (STS). It is compact (~80 MB), fast on CPU inference (under 100ms per pair), and provides state-of-the-art cosine similarity accuracy for paraphrased content.

### Q3: What is the difference between Cosine Similarity and Jaccard Similarity?
> **Answer**:
> - **Cosine Similarity** evaluates the angle between vectors: $\text{Cosine}(A, B) = \frac{A \cdot B}{\|A\| \|B\|}$. It considers both presence and magnitude/weight of features (such as TF-IDF scores).
> - **Jaccard Similarity** is set-based: $J(A, B) = \frac{|A \cap B|}{|A \cup B|}$. It measures the proportion of shared unique vocabulary tokens without weighting.

### Q4: Why is Lemmatization used instead of Stemming?
> **Answer**: Stemming uses heuristic chopping of word affixes (e.g., Porter Stemmer truncates *"computing"* to *"comput"*), which often results in non-dictionary roots. Lemmatization uses morphological analysis with Part-of-Speech tags (via spaCy) to identify the true root word (e.g., *"ran"*, *"running"* $\to$ *"run"*), ensuring meaningful vector representations.

### Q5: Why is the project titled "Assignment Duplicate Detector" instead of "Plagiarism Detector"?
> **Answer**: Academic similarity can arise from common assignment templates, standard definitions, or shared lab prompt questions. An automated tool cannot determine moral intent. Therefore, flagging assignments as "Duplicate" or "Similar" serves as an explainable diagnostic aid for faculty review rather than an accusatory plagiarism verdict.

---

## 👨‍💻 Technologies Used
- **Python 3.13**
- **Streamlit** (User Interface)
- **spaCy** (`en_core_web_sm` for Lemmatization)
- **NLTK** (Tokenization & Stopwords)
- **scikit-learn** (TF-IDF, Logistic Regression, Random Forest, Metrics)
- **Sentence-Transformers & PyTorch** (`all-MiniLM-L6-v2` Embeddings)
- **PyPDF2 & python-docx** (Document Parsing)
- **Plotly & Seaborn** (Data Visualization)
