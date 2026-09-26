"""
Text Preprocessing Module
Implements standard NLP pipeline:
- Lowercasing & noise removal
- Sentence segmentation
- Tokenization
- Stopword removal
- Lemmatization (using spaCy or NLTK WordNet)
"""

import re
import string
from typing import List, Tuple

# Try loading spaCy, fall back gracefully if unavailable
_SPACY_NLP = None
try:
    import spacy
    _SPACY_NLP = spacy.load("en_core_web_sm", disable=["parser", "ner"])
except Exception:
    _SPACY_NLP = None

# Fallback English stopwords
DEFAULT_STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are",
    "aren't", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both",
    "but", "by", "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't",
    "doing", "don't", "down", "during", "each", "few", "for", "from", "further", "had", "hadn't",
    "has", "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here",
    "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll",
    "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself", "let's",
    "me", "more", "most", "mustn't", "my", "myself", "no", "nor", "not", "of", "off", "on", "once",
    "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "shan't",
    "she", "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such", "than", "that",
    "that's", "the", "their", "theirs", "them", "themselves", "then", "there", "there's", "these",
    "they", "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too",
    "under", "until", "up", "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've",
    "were", "weren't", "what", "what's", "when", "when's", "where", "where's", "which", "while",
    "who", "who's", "whom", "why", "why's", "with", "won't", "would", "wouldn't", "you", "you'd",
    "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves"
}

try:
    import nltk
    from nltk.corpus import stopwords
    NLTK_STOPWORDS = set(stopwords.words("english"))
except Exception:
    NLTK_STOPWORDS = DEFAULT_STOPWORDS

STOPWORDS = NLTK_STOPWORDS if NLTK_STOPWORDS else DEFAULT_STOPWORDS


def clean_raw_text(text: str) -> str:
    """Removes URLs, non-ASCII formatting artifacts, extra spaces, and lowercases text."""
    if not text:
        return ""
    # Normalize unicode and lower
    text = text.lower()
    # Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    # Remove email addresses
    text = re.sub(r"\S+@\S+", "", text)
    # Normalize excessive newlines and whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def split_into_sentences(text: str) -> List[str]:
    """Splits a document text into individual sentences for fine-grained matching."""
    if not text:
        return []
    # Clean whitespace first
    clean = re.sub(r"\s+", " ", text).strip()
    try:
        from nltk.tokenize import sent_tokenize
        sentences = sent_tokenize(clean)
    except Exception:
        # Regex fallback for sentence splitting
        sentences = re.split(r"(?<=[.!?])\s+", clean)

    # Filter out empty or very short snippets
    return [s.strip() for s in sentences if len(s.strip()) > 5]


def preprocess_text(
    text: str,
    remove_stop: bool = True,
    lemmatize: bool = True
) -> Tuple[List[str], str]:
    """
    Full text preprocessing pipeline:
    1. Clean & lowercase text
    2. Tokenize words
    3. Lemmatize words (via spaCy or fallback)
    4. Remove stopwords and punctuation
    
    Returns:
        tokens: List of processed word tokens
        processed_str: Space-separated string of processed tokens
    """
    cleaned = clean_raw_text(text)
    if not cleaned:
        return [], ""

    tokens = []

    # Priority 1: spaCy (fast, accurate POS-based lemmatization)
    if _SPACY_NLP is not None:
        doc = _SPACY_NLP(cleaned)
        for token in doc:
            # Skip punctuation, spaces, and numbers/symbols
            if token.is_punct or token.is_space or not token.text.isalnum():
                continue
            word = token.lemma_.lower() if lemmatize else token.text.lower()
            if remove_stop and (word in STOPWORDS or token.is_stop):
                continue
            if len(word) > 1:
                tokens.append(word)
    else:
        # Priority 2: NLTK / Regex fallback
        words = re.findall(r"\b[a-z0-9]+\b", cleaned)
        for word in words:
            if remove_stop and word in STOPWORDS:
                continue
            if len(word) > 1:
                tokens.append(word)

    processed_str = " ".join(tokens)
    return tokens, processed_str


def get_preprocessing_steps_breakdown(text: str) -> dict:
    """
    Returns step-by-step transformation for educational/UI visualization in the report.
    """
    raw = text.strip()
    lowered = raw.lower()
    no_punct = re.sub(r"[^\w\s]", " ", lowered)
    no_punct = re.sub(r"\s+", " ", no_punct).strip()
    raw_tokens = no_punct.split()
    no_stops = [w for w in raw_tokens if w not in STOPWORDS and len(w) > 1]
    final_tokens, final_str = preprocess_text(text, remove_stop=True, lemmatize=True)

    return {
        "1_original": raw,
        "2_lowercased": lowered,
        "3_punctuation_removed": no_punct,
        "4_tokenized_count": len(raw_tokens),
        "5_stopwords_removed_count": len(no_stops),
        "6_lemmatized_count": len(final_tokens),
        "final_cleaned_string": final_str
    }
