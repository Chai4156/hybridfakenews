import os
import argparse
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from model import NewsDataset, SAVED_MODEL_PATH
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def evaluate(model, loader, device):
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
    }


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")

    df = pd.read_csv(args.data_path)
    df = df.dropna(subset=['text', 'label'])

    label_map = {'fake': 0, 'FAKE': 0, '0': 0, 'real': 1, 'REAL': 1, '1': 1}
    df['label'] = df['label'].astype(str).map(label_map)
    df = df.dropna(subset=['label'])
    df['label'] = df['label'].astype(int)

    train_df, temp_df = train_test_split(df, test_size=0.30, random_state=42, stratify=df['label'])
    val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=42, stratify=temp_df['label'])

    logger.info(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForSequenceClassification.from_pretrained(args.base_model, num_labels=2)
    model.to(device)

    train_dataset = NewsDataset(train_df['text'].values, train_df['label'].values, tokenizer, args.max_len)
    val_dataset   = NewsDataset(val_df['text'].values,   val_df['label'].values,   tokenizer, args.max_len)
    test_dataset  = NewsDataset(test_df['text'].values,  test_df['label'].values,  tokenizer, args.max_len)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size)
    test_loader  = DataLoader(test_dataset,  batch_size=args.batch_size)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    total_steps = len(train_loader) * args.epochs
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1.0, end_factor=0.0, total_iters=total_steps
    )

    best_f1 = 0
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
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
        val_metrics = evaluate(model, val_loader, device)
        logger.info(f"Epoch {epoch+1} | Loss: {avg_loss:.4f} | Val: {val_metrics}")

        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            os.makedirs(SAVED_MODEL_PATH, exist_ok=True)
            model.save_pretrained(SAVED_MODEL_PATH)
            tokenizer.save_pretrained(SAVED_MODEL_PATH)
            logger.info(f"Best model saved (F1={best_f1})")

    logger.info("=== Test Set Evaluation ===")
    test_metrics = evaluate(model, test_loader, device)
    logger.info(f"Test Metrics: {test_metrics}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path',   default='data/WELFAKE_dataset.csv')
    parser.add_argument('--base_model',  default='roberta-base')
    parser.add_argument('--epochs',      type=int,   default=3)
    parser.add_argument('--batch_size',  type=int,   default=8)
    parser.add_argument('--lr',          type=float, default=2e-5)
    parser.add_argument('--max_len',     type=int,   default=256)
    train(parser.parse_args())
