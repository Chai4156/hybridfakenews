import os
import pandas as pd
import numpy as np
import logging
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

FINE_TUNED_MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'roberta_fine_tuned')

def load_roberta_model():
    """Load fine-tuned RoBERTa model."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    logger.info(f"Loading fine-tuned RoBERTa from: {FINE_TUNED_MODEL_PATH}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(FINE_TUNED_MODEL_PATH)
        model = AutoModelForSequenceClassification.from_pretrained(FINE_TUNED_MODEL_PATH, num_labels=2)
        model.to(device)
        model.eval()
        logger.info("✓ Model loaded successfully")
        return tokenizer, model, device
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return None, None, None


def get_predictions(texts, tokenizer, model, device, batch_size=8):
    """Get RoBERTa predictions."""
    predictions = []
    logger.info(f"Generating predictions for {len(texts)} samples...")
    
    try:
        for i in tqdm(range(0, len(texts), batch_size), desc="Inference"):
            batch_texts = list(texts[i:i+batch_size])
            
            encodings = tokenizer(
                batch_texts,
                add_special_tokens=True,
                max_length=256,
                return_token_type_ids=False,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt'
            )
            
            input_ids = encodings['input_ids'].to(device)
            attention_mask = encodings['attention_mask'].to(device)
            
            with torch.no_grad():
                outputs = model(input_ids, attention_mask=attention_mask)
                preds = torch.argmax(outputs.logits, dim=1).cpu().tolist()
                predictions.extend(preds)
        
        return np.array(predictions)
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        return None


def plot_confusion_matrix(cm, save_path='confusion_matrix_roberta_finetuned.png'):
    """Plot and save confusion matrix."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Fake', 'Real'], yticklabels=['Fake', 'Real'], cbar=True)
    plt.title('Confusion Matrix - Fine-Tuned RoBERTa')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    logger.info(f"✓ Saved: {save_path}")
    plt.close()


def test_roberta():
    """Test the fine-tuned RoBERTa model."""
    
    # Load data
    logger.info("Loading dataset...")
    df = pd.read_csv('../WELFake_Dataset.csv')
    df = df.dropna(subset=['text', 'label'])
    
    # Map labels
    label_map = {'fake': 0, 'FAKE': 0, '0': 0, 'real': 1, 'REAL': 1, '1': 1}
    df['label'] = df['label'].astype(str).map(label_map)
    df = df.dropna(subset=['label'])
    df['label'] = df['label'].astype(int)
    
    logger.info(f"Total samples: {len(df)}")
    
    # Split data
    train_df, test_df = train_test_split(
        df, test_size=0.30, random_state=42, stratify=df['label']
    )
    
    # Use all test set for proper evaluation
    test_texts = test_df['text'].values
    test_labels = test_df['label'].values
    
    logger.info(f"Test set size: {len(test_df)}")
    
    # Load model
    tokenizer, model, device = load_roberta_model()
    if model is None:
        logger.error("Could not load model. Exiting.")
        return
    
    # Get predictions
    predictions = get_predictions(test_texts, tokenizer, model, device, batch_size=8)
    
    if predictions is None:
        logger.error("Could not generate predictions. Exiting.")
        return
    
    # Calculate metrics
    logger.info("\n" + "="*70)
    logger.info("FINE-TUNED ROBERTA EVALUATION RESULTS")
    logger.info("="*70)
    
    accuracy = accuracy_score(test_labels, predictions)
    precision = precision_score(test_labels, predictions, average='weighted', zero_division=0)
    recall = recall_score(test_labels, predictions, average='weighted', zero_division=0)
    f1 = f1_score(test_labels, predictions, average='weighted', zero_division=0)
    cm = confusion_matrix(test_labels, predictions)
    
    logger.info(f"\nAccuracy:  {accuracy:.4f}")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall:    {recall:.4f}")
    logger.info(f"F1 Score:  {f1:.4f}")
    
    logger.info(f"\nConfusion Matrix:\n{cm}")
    
    # Classification report
    report = classification_report(test_labels, predictions, target_names=['Fake', 'Real'])
    logger.info(f"\nClassification Report:\n{report}")
    
    # Plot confusion matrix
    plot_confusion_matrix(cm)
    
    # Save metrics to CSV
    metrics_df = pd.DataFrame({
        'Model': ['Fine-Tuned RoBERTa'],
        'Accuracy': [round(accuracy, 4)],
        'Precision': [round(precision, 4)],
        'Recall': [round(recall, 4)],
        'F1 Score': [round(f1, 4)],
        'Correct_Predictions': [np.sum(test_labels == predictions)],
        'Total_Samples': [len(test_labels)]
    })
    
    metrics_df.to_csv('roberta_finetuned_metrics.csv', index=False)
    logger.info("\n✓ Saved metrics to: roberta_finetuned_metrics.csv")
    
    logger.info("\n" + "="*70)
    logger.info("COMPLETE!")
    logger.info("="*70)


if __name__ == '__main__':
    test_roberta()
