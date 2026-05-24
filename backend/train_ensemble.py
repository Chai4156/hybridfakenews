"""
Training script for Ensemble Model
Trains Naive Bayes on SBERT embeddings while using pre-trained RoBERTa
Soft voting: RoBERTa (0.8) + SBERT-NB (0.2)
"""

import argparse
import sys
import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from ensemble_model import train_ensemble_on_data, EnsembleModel

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main(args):
    logger.info("=" * 80)
    logger.info("ENSEMBLE MODEL TRAINING")
    logger.info("Architecture: RoBERTa (0.8) + SBERT-NB (0.2)")
    logger.info("=" * 80)
    
    if not args.data_path or not os.path.exists(args.data_path):
        logger.error(f"Dataset not found at: {args.data_path}")
        sys.exit(1)
    
    # Train ensemble
    ensemble, results = train_ensemble_on_data(args.data_path)
    
    # Optionally generate visualizations
    if args.visualize:
        logger.info("Generating confusion matrix visualization...")
        df_data = pd.read_csv(args.data_path)
        label_map = {'fake': 0, 'FAKE': 0, '0': 0, 'real': 1, 'REAL': 1, '1': 1}
        df_data['label'] = df_data['label'].astype(str).map(label_map)
        df_data = df_data.dropna(subset=['label'])
        
        _, test_df = train_test_split(df_data, test_size=0.30, random_state=42, stratify=df_data['label'])
        _, test_df = train_test_split(test_df, test_size=0.50, random_state=42, stratify=test_df['label'])
        
        test_labels = test_df['label'].values
        test_preds = results['predictions']
        
        cm = confusion_matrix(test_labels, test_preds)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Fake', 'Real'], yticklabels=['Fake', 'Real'], cbar=True)
        plt.title('Confusion Matrix - Ensemble Model (RoBERTa 0.8 + SBERT-NB 0.2)')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('confusion_matrix_ensemble.png', dpi=300, bbox_inches='tight')
        logger.info("✓ Confusion matrix saved to: confusion_matrix_ensemble.png")
    
    logger.info("\n" + "=" * 80)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 80)
    logger.info(f"SBERT-NB Model saved: saved_model/sbert_nb_model.pkl")
    logger.info(f"Ready to use ensemble for predictions!")


if __name__ == '__main__':
    import os
    
    parser = argparse.ArgumentParser(
        description='Train Ensemble Model: RoBERTa + SBERT-NB'
    )
    parser.add_argument(
        '--data_path',
        type=str,
        default='data/news.csv',
        help='Path to training dataset (CSV with "text" and "label" columns)'
    )
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Generate confusion matrix visualization'
    )
    
    args = parser.parse_args()
    
    try:
        main(args)
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        sys.exit(1)
