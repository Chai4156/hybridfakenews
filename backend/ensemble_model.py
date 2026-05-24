"""
Ensemble Model: RoBERTa + SBERT Embeddings + Naive Bayes
Soft voting: RoBERTa (0.8) + SBERT-NB (0.2)
"""

import torch
import numpy as np
import logging
import os
import pickle
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer
from sklearn.naive_bayes import GaussianNB
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Paths
MODELS_DIR = os.path.join(os.path.dirname(__file__), 'saved_model')
ROBERTA_PATH = os.path.join(MODELS_DIR, 'fine_tuned')
SBERT_NB_PATH = os.path.join(MODELS_DIR, 'sbert_nb_model.pkl')
SBERT_MODEL_NAME = 'all-MiniLM-L6-v2'  # Lightweight SBERT model


class EnsembleModel:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.roberta_tokenizer = None
        self.roberta_model = None
        self.sbert_model = None
        self.nb_model = None
        self.roberta_weight = 0.8
        self.sbert_nb_weight = 0.2
        
    def load_models(self):
        """Load RoBERTa and SBERT models"""
        logger.info("Loading RoBERTa model...")
        try:
            self.roberta_tokenizer = AutoTokenizer.from_pretrained(ROBERTA_PATH)
            self.roberta_model = AutoModelForSequenceClassification.from_pretrained(ROBERTA_PATH, num_labels=2)
        except:
            logger.warning("Could not load fine-tuned RoBERTa, using base model...")
            self.roberta_tokenizer = AutoTokenizer.from_pretrained('roberta-base')
            self.roberta_model = AutoModelForSequenceClassification.from_pretrained('roberta-base', num_labels=2)
        
        self.roberta_model.to(self.device)
        self.roberta_model.eval()
        
        logger.info("Loading SBERT model...")
        self.sbert_model = SentenceTransformer(SBERT_MODEL_NAME, device=self.device)
        
        logger.info("Loading Naive Bayes model trained on SBERT embeddings...")
        if os.path.exists(SBERT_NB_PATH):
            with open(SBERT_NB_PATH, 'rb') as f:
                self.nb_model = pickle.load(f)
            logger.info("✓ SBERT-NB model loaded")
        else:
            logger.warning(f"SBERT-NB model not found at {SBERT_NB_PATH}. Run train_ensemble() first.")
            self.nb_model = None
    
    def get_roberta_predictions(self, texts, batch_size=32):
        """Get RoBERTa predictions (probabilities)"""
        roberta_probs = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc="RoBERTa inference"):
            batch_texts = texts[i:i+batch_size]
            
            encodings = self.roberta_tokenizer(
                batch_texts,
                add_special_tokens=True,
                max_length=256,
                padding=True,
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt'
            )
            
            input_ids = encodings['input_ids'].to(self.device)
            attention_mask = encodings['attention_mask'].to(self.device)
            
            with torch.no_grad():
                outputs = self.roberta_model(input_ids, attention_mask=attention_mask)
                probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()
                roberta_probs.append(probs)
        
        return np.vstack(roberta_probs)
    
    def get_sbert_nb_predictions(self, texts, batch_size=32):
        """Get SBERT embeddings + NB predictions (probabilities)"""
        if self.nb_model is None:
            logger.warning("SBERT-NB model not loaded. Run load_models() and train first.")
            return None
        
        logger.info("Generating SBERT embeddings...")
        embeddings = self.sbert_model.encode(texts, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True)
        
        logger.info("Getting NB predictions on SBERT embeddings...")
        # Get probabilities from Naive Bayes
        sbert_nb_probs = self.nb_model.predict_proba(embeddings)
        
        return sbert_nb_probs
    
    def predict(self, texts, batch_size=32, return_confidence=True):
        """
        Ensemble prediction with soft voting:
        - RoBERTa: 0.8 weight
        - SBERT + Naive Bayes: 0.2 weight
        
        Returns:
            predictions: Binary predictions (0=Fake, 1=Real)
            confidences: Confidence scores
        """
        roberta_probs = self.get_roberta_predictions(texts, batch_size)
        
        if self.nb_model is not None:
            sbert_nb_probs = self.get_sbert_nb_predictions(texts, batch_size)
            # Soft voting: weighted average
            ensemble_probs = (self.roberta_weight * roberta_probs + 
                            self.sbert_nb_weight * sbert_nb_probs)
        else:
            logger.warning("Using only RoBERTa (SBERT-NB not available)")
            ensemble_probs = roberta_probs
        
        predictions = np.argmax(ensemble_probs, axis=1)
        confidences = np.max(ensemble_probs, axis=1)
        
        if return_confidence:
            return predictions, confidences, ensemble_probs
        return predictions
    
    def generate_reasoning(self, text, prediction, confidence, roberta_prob, sbert_prob):
        """
        Generate human-readable reasoning for predictions
        
        Args:
            text: Input text
            prediction: 0=Fake, 1=Real
            confidence: Confidence score (0-1)
            roberta_prob: RoBERTa probability for this class
            sbert_prob: SBERT-NB probability for this class
        
        Returns:
            reasoning_text: Human-readable explanation
        """
        label = "Real" if prediction == 1 else "Fake"
        conf_pct = confidence * 100
        
        # Confidence assessment
        if conf_pct >= 90:
            confidence_text = "very high confidence"
        elif conf_pct >= 75:
            confidence_text = "high confidence"
        elif conf_pct >= 60:
            confidence_text = "moderate confidence"
        else:
            confidence_text = "low confidence"
        
        # Model agreement assessment
        roberta_class = 1 if roberta_prob[1] > 0.5 else 0
        sbert_class = 1 if sbert_prob[1] > 0.5 else 0
        agreement = roberta_class == sbert_class
        
        agreement_text = "Both models agree" if agreement else "Models partially disagree"
        
        # Text characteristics
        text_len = len(text.split())
        if text_len < 50:
            length_text = "short text"
        elif text_len < 200:
            length_text = "medium-length text"
        else:
            length_text = "lengthy article"
        
        reasoning = (
            f"Classified as {label} with {confidence_text} ({conf_pct:.1f}%). "
            f"{agreement_text} (RoBERTa: {roberta_prob[1]:.1%}, SBERT-NB: {sbert_prob[1]:.1%}). "
            f"Analysis of {length_text} with hybrid ensemble model (RoBERTa 80% + SBERT-NB 20%)."
        )
        
        return reasoning
    
    def save_nb_model(self):
        """Save trained Naive Bayes model"""
        os.makedirs(MODELS_DIR, exist_ok=True)
        with open(SBERT_NB_PATH, 'wb') as f:
            pickle.dump(self.nb_model, f)
        logger.info(f"✓ SBERT-NB model saved to {SBERT_NB_PATH}")


def train_ensemble_on_data(data_path, epochs=1):
    """
    Train the ensemble model:
    1. RoBERTa is already fine-tuned (use existing)
    2. Train Naive Bayes on SBERT embeddings
    """
    import pandas as pd
    from sklearn.model_selection import train_test_split
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ensemble = EnsembleModel(device=device)
    ensemble.load_models()
    
    # Load data
    logger.info(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    df = df.dropna(subset=['text', 'label'])
    
    # Map labels
    label_map = {'fake': 0, 'FAKE': 0, '0': 0, 'real': 1, 'REAL': 1, '1': 1}
    df['label'] = df['label'].astype(str).map(label_map)
    df = df.dropna(subset=['label'])
    df['label'] = df['label'].astype(int)
    
    # Split: 70% train, 15% val, 15% test
    train_df, temp_df = train_test_split(df, test_size=0.30, random_state=42, stratify=df['label'])
    val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=42, stratify=temp_df['label'])
    
    logger.info(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    
    # Generate SBERT embeddings for training data
    logger.info("Generating SBERT embeddings for training data...")
    train_texts = train_df['text'].values
    train_labels = train_df['label'].values
    train_embeddings = ensemble.sbert_model.encode(train_texts, batch_size=32, show_progress_bar=True, convert_to_numpy=True)
    
    # Train Naive Bayes on SBERT embeddings
    logger.info("Training Naive Bayes on SBERT embeddings...")
    ensemble.nb_model = GaussianNB()
    ensemble.nb_model.fit(train_embeddings, train_labels)
    
    # Evaluate on test set
    logger.info("Evaluating ensemble on test set...")
    test_texts = test_df['text'].values
    test_labels = test_df['label'].values
    
    test_preds, test_confs, _ = ensemble.predict(test_texts, batch_size=32)
    
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
    
    accuracy = accuracy_score(test_labels, test_preds)
    precision = precision_score(test_labels, test_preds, zero_division=0)
    recall = recall_score(test_labels, test_preds, zero_division=0)
    f1 = f1_score(test_labels, test_preds, zero_division=0)
    
    logger.info(f"\n=== Ensemble Model Performance ===")
    logger.info(f"Accuracy:  {accuracy:.4f}")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall:    {recall:.4f}")
    logger.info(f"F1-Score:  {f1:.4f}")
    logger.info(f"\nClassification Report:\n{classification_report(test_labels, test_preds, target_names=['Fake', 'Real'])}")
    
    # Save the trained NB model
    ensemble.save_nb_model()
    
    return ensemble, {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'predictions': test_preds,
        'confidences': test_confs
    }


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='data/news.csv', help='Path to training data')
    parser.add_argument('--train', action='store_true', help='Train the ensemble model')
    args = parser.parse_args()
    
    if args.train:
        ensemble, results = train_ensemble_on_data(args.data_path)
    else:
        # Just load and test
        ensemble = EnsembleModel()
        ensemble.load_models()
        test_texts = [
            "Breaking: Scientists discover groundbreaking cure for cancer",
            "Massive government conspiracy uncovered - wake up sheeple!"
        ]
        preds, confs, probs = ensemble.predict(test_texts)
        for text, pred, conf in zip(test_texts, preds, confs):
            label = "Real" if pred == 1 else "Fake"
            print(f"Text: {text[:50]}... => {label} ({conf:.2%})")
