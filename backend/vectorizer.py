"""
TF-IDF Vectorizer for feature extraction
Extracts top keywords/features from text for better model input
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NewsVectorizer:
    def __init__(self, max_features=512):
        """Initialize TF-IDF vectorizer"""
        self.max_features = max_features
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words='english',
            lowercase=True,
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95
        )
        self.is_fitted = False
    
    def fit(self, texts):
        """Fit vectorizer on training texts"""
        self.vectorizer.fit(texts)
        self.is_fitted = True
        logger.info(f"✓ TF-IDF vectorizer fitted with {len(self.vectorizer.get_feature_names_out())} features")
    
    def get_top_keywords(self, text, n=20):
        """Extract top N keywords from text using TF-IDF scores"""
        if not self.is_fitted:
            logger.warning("Vectorizer not fitted. Using basic tokenization...")
            return text.split()[:n]
        
        tfidf_matrix = self.vectorizer.transform([text])
        feature_names = self.vectorizer.get_feature_names_out()
        scores = tfidf_matrix.toarray()[0]
        
        # Get indices of top n features
        top_indices = np.argsort(scores)[-n:][::-1]
        top_keywords = [(feature_names[i], scores[i]) for i in top_indices if scores[i] > 0]
        
        return top_keywords
    
    def get_feature_importance_text(self, text, n=10):
        """Generate importance-weighted text snippet"""
        keywords = self.get_top_keywords(text, n)
        if keywords:
            return " ".join([kw for kw, _ in keywords])
        return text[:200]


def chunk_text(text, max_tokens=512, overlap=50):
    """
    Split text into chunks for processing
    Preserves context with overlap between chunks
    """
    words = text.split()
    chunks = []
    
    # Approximate tokens (1 token ≈ 1.3 words)
    chunk_size = int(max_tokens / 1.3)
    
    i = 0
    while i < len(words):
        chunk = ' '.join(words[i:i + chunk_size])
        chunks.append(chunk)
        i += chunk_size - int(overlap / 1.3)
    
    return chunks if chunks else [text]


if __name__ == '__main__':
    # Test
    sample_texts = [
        "Breaking news about economy and markets today",
        "Scientists discover new medical treatment breakthrough"
    ]
    
    vectorizer = NewsVectorizer()
    vectorizer.fit(sample_texts)
    
    test_text = "economy markets news today"
    keywords = vectorizer.get_top_keywords(test_text, n=5)
    print(f"Top keywords: {keywords}")
    
    long_text = "A" * 1000
    chunks = chunk_text(long_text, max_tokens=512)
    print(f"Text chunked into {len(chunks)} pieces")
