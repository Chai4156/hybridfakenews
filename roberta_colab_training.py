# Fine-tune RoBERTa for Fake News Detection on Google Colab
# Install required packages
import os
import sys

# Install transformers and torch (Colab has torch pre-installed, but we need latest transformers)
!pip install -q transformers torch scikit-learn pandas tqdm

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Set working directory
import os
os.chdir('/content/drive/MyDrive')

# Setup imports
import logging
import pandas as pd
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import random

# Set seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Check GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")
if torch.cuda.is_available():
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# Define Dataset class
class NewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


def evaluate(model, loader, device):
    """Evaluate model on validation/test set."""
    model.eval()
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc='Evaluating'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().tolist())

    return {
        'accuracy':  round(accuracy_score(all_labels, all_preds), 4),
        'precision': round(precision_score(all_labels, all_preds, zero_division=0), 4),
        'recall':    round(recall_score(all_labels, all_preds, zero_division=0), 4),
        'f1':        round(f1_score(all_labels, all_preds, zero_division=0), 4),
    }, all_preds, all_labels


def train_roberta(
    data_path='WELFake_Dataset.csv',
    model_name='roberta-base',
    epochs=3,
    batch_size=16,
    learning_rate=2e-5,
    max_len=256,
    output_dir='./roberta_fine_tuned'
):
    """Fine-tune RoBERTa on news classification task."""
    
    logger.info("="*80)
    logger.info("ROBERTA FINE-TUNING FOR FAKE NEWS DETECTION")
    logger.info("="*80)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    logger.info(f"\nLoading data from {data_path}...")
    df = pd.read_csv(data_path)
    df = df.dropna(subset=['text', 'label'])
    
    logger.info(f"Total samples: {len(df)}")
    
    # Map labels
    label_map = {'fake': 0, 'FAKE': 0, '0': 0, 'real': 1, 'REAL': 1, '1': 1}
    df['label'] = df['label'].astype(str).map(label_map)
    df = df.dropna(subset=['label'])
    df['label'] = df['label'].astype(int)
    
    logger.info(f"Samples after cleaning: {len(df)}")
    logger.info(f"Label distribution:\n{df['label'].value_counts()}")
    
    # Split data
    train_df, temp_df = train_test_split(
        df, test_size=0.30, random_state=42, stratify=df['label']
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, random_state=42, stratify=temp_df['label']
    )
    
    logger.info(f"\nData split:")
    logger.info(f"  Train: {len(train_df)}")
    logger.info(f"  Val:   {len(val_df)}")
    logger.info(f"  Test:  {len(test_df)}")
    
    # Load tokenizer and model
    logger.info(f"\nLoading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    model.to(device)
    
    # Create datasets and dataloaders
    logger.info("\nCreating datasets...")
    train_dataset = NewsDataset(train_df['text'].values, train_df['label'].values, tokenizer, max_len)
    val_dataset = NewsDataset(val_df['text'].values, val_df['label'].values, tokenizer, max_len)
    test_dataset = NewsDataset(test_df['text'].values, test_df['label'].values, tokenizer, max_len)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Setup optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    total_steps = len(train_loader) * epochs
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1.0, end_factor=0.0, total_iters=total_steps
    )
    
    # Training loop
    logger.info("\n" + "="*80)
    logger.info("TRAINING")
    logger.info("="*80)
    
    best_f1 = 0
    for epoch in range(epochs):
        logger.info(f"\n{'='*80}")
        logger.info(f"EPOCH {epoch+1}/{epochs}")
        logger.info(f"{'='*80}")
        
        # Training phase
        model.train()
        total_loss = 0
        
        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        avg_loss = total_loss / len(train_loader)
        
        # Validation phase
        logger.info(f"\nValidating...")
        val_metrics, _, _ = evaluate(model, val_loader, device)
        
        logger.info(f"Epoch {epoch+1} Summary:")
        logger.info(f"  Loss:      {avg_loss:.4f}")
        logger.info(f"  Accuracy:  {val_metrics['accuracy']:.4f}")
        logger.info(f"  Precision: {val_metrics['precision']:.4f}")
        logger.info(f"  Recall:    {val_metrics['recall']:.4f}")
        logger.info(f"  F1 Score:  {val_metrics['f1']:.4f}")

        # Save best model
        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            logger.info(f"✓ Best model saved (F1={best_f1:.4f})")
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)

    # Test set evaluation
    logger.info("\n" + "="*80)
    logger.info("TEST SET EVALUATION")
    logger.info("="*80)
    
    test_metrics, test_preds, test_labels = evaluate(model, test_loader, device)
    
    logger.info(f"\nTest Metrics:")
    logger.info(f"  Accuracy:  {test_metrics['accuracy']:.4f}")
    logger.info(f"  Precision: {test_metrics['precision']:.4f}")
    logger.info(f"  Recall:    {test_metrics['recall']:.4f}")
    logger.info(f"  F1 Score:  {test_metrics['f1']:.4f}")
    
    # Confusion matrix
    logger.info(f"\nConfusion Matrix:")
    cm = confusion_matrix(test_labels, test_preds)
    logger.info(f"{cm}")
    
    # Classification report
    logger.info(f"\nClassification Report:")
    report = classification_report(test_labels, test_preds, target_names=['Fake', 'Real'])
    logger.info(f"{report}")
    
    logger.info("\n" + "="*80)
    logger.info("TRAINING COMPLETE!")
    logger.info("="*80)
    logger.info(f"Model saved to: {output_dir}")
    
    return model, tokenizer, test_metrics


# Main execution
if __name__ == '__main__':
    # Configure these parameters
    DATA_PATH = 'WELFake_Dataset.csv'  # Upload to Google Drive or use this path
    MODEL_NAME = 'roberta-base'
    EPOCHS = 3
    BATCH_SIZE = 16  # Increase if you have more GPU memory
    LEARNING_RATE = 2e-5
    MAX_LEN = 256
    OUTPUT_DIR = './roberta_fine_tuned'
    
    # Run training
    model, tokenizer, metrics = train_roberta(
        data_path=DATA_PATH,
        model_name=MODEL_NAME,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        max_len=MAX_LEN,
        output_dir=OUTPUT_DIR
    )
    
    logger.info("\n✓ Model training and evaluation completed successfully!")
    logger.info(f"✓ Fine-tuned model saved to: {OUTPUT_DIR}")
    logger.info(f"✓ Best F1 Score on Test Set: {metrics['f1']:.4f}")
