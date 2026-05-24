import torch
import logging
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('prediction.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

SAVED_MODEL_PATH = os.path.join(os.path.dirname(__file__), 'saved_model', 'fine_tuned')
FALLBACK_MODEL = 'roberta-base'


class NewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
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


def load_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    model_path = SAVED_MODEL_PATH if os.path.exists(SAVED_MODEL_PATH) else FALLBACK_MODEL
    logger.info(f"Loading model from: {model_path}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path, num_labels=2
        )
        model.to(device)
        model.eval()
        logger.info("Model loaded successfully")
        return tokenizer, model, device
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise RuntimeError(f"Could not load model from {model_path}: {e}")


def predict_text(text, tokenizer, model, device, max_len=512):
    logger.info(f"Running prediction on: {text[:60]}...")
    encoding = tokenizer(
        text,
        add_special_tokens=True,
        max_length=max_len,
        return_token_type_ids=False,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )

    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)
        prediction = torch.argmax(probs, dim=1).item()
        confidence = probs[0][prediction].item()

    label = 'Real' if prediction == 1 else 'Fake'
    logger.info(f"Result => Label: {label}, Confidence: {confidence:.4f}")
    return label, confidence


def fine_tune_model(model, tokenizer, device, dataset_path='data/news.csv',
                    epochs=3, batch_size=8, learning_rate=2e-5):
    logger.info("Starting fine-tuning on custom dataset")

    df = pd.read_csv(dataset_path).dropna(subset=['text', 'label'])
    label_mapping = {'fake': 0, 'real': 1, '0': 0, '1': 1,
                     'FAKE': 0, 'REAL': 1, 'kaggle_fake': 0, 'kaggle_real': 1}
    df['label'] = df['label'].map(label_mapping)
    df = df.dropna(subset=['label'])

    texts = df['text'].values
    labels = df['label'].astype(int).values

    dataset = NewsDataset(texts, labels, tokenizer)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1.0, end_factor=0.1, total_iters=len(loader) * epochs
    )

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in tqdm(loader, desc=f"Epoch {epoch + 1}/{epochs}"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels_batch = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels_batch)
            loss = outputs.loss
            total_loss += loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        avg_loss = total_loss / len(loader)
        logger.info(f"Epoch {epoch + 1} — Avg Loss: {avg_loss:.4f}")

    model.eval()
    os.makedirs(SAVED_MODEL_PATH, exist_ok=True)
    model.save_pretrained(SAVED_MODEL_PATH)
    tokenizer.save_pretrained(SAVED_MODEL_PATH)
    logger.info(f"Fine-tuned model saved to {SAVED_MODEL_PATH}")
    return model, tokenizer


if __name__ == '__main__':
    tokenizer, model, device = load_model()
    sample = "Scientists discover a new planet that could support life outside the solar system."
    label, conf = predict_text(sample, tokenizer, model, device)
    print(f"Label: {label} | Confidence: {conf:.4f}")
