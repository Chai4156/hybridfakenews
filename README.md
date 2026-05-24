# Fake News Detection System
> Fake news detection project

A full-stack fake news detection system using an **Ensemble Model** combining:
- **RoBERTa** (fine-tuned transformer): 80% weight
- **SBERT** embeddings + **Naive Bayes**: 20% weight

Served via **Flask** API and **React + Vite** frontend.

## 🎯 Ensemble Architecture

The system uses **soft voting** to balance deep learning accuracy with traditional ML robustness:

```
Input Text
    ↓
    ├─→ RoBERTa (0.8) ──→ Probabilities
    │
    └─→ SBERT + NB (0.2) ──→ Probabilities
    
    Final Prediction = 0.8×P_roberta + 0.2×P_sbert_nb
```

**Benefits:**
- ✅ Reduces overfitting
- ✅ Combines deep learning + interpretable ML
- ✅ Better generalization to unseen data
- ✅ More stable predictions

---

## Project Structure

```
fake-news-detector/
├── backend/
│   ├── app.py                      # Flask API (uses ensemble)
│   ├── ensemble_model.py           # Ensemble prediction logic
│   ├── train_ensemble.py           # Train NB on SBERT embeddings
│   ├── model.py                    # RoBERTa utilities
│   ├── train.py                    # Original training script
│   ├── ENSEMBLE_ARCHITECTURE.md    # Detailed model docs
│   ├── requirements.txt
│   └── saved_model/
│       ├── fine_tuned/             # Fine-tuned RoBERTa
│       └── sbert_nb_model.pkl      # Trained Naive Bayes
├── frontend/
│   ├── src/
│   │   ├── App.jsx
│   │   ├── main.jsx
│   │   └── index.css
│   ├── index.html
│   ├── package.json
│   └── vite.config.js
└── README.md
```

---

## Environment Setup

Before running the project, create a `.env` file from the template:

```bash
cp .env.example .env
```

Edit `.env` with your configuration:
- `FLASK_SECRET_KEY`: Set to a random secret for production
- `CORS_ALLOWED_ORIGINS`: Frontend URL (default: `http://localhost:5173`)
- `VITE_API_URL`: API endpoint for frontend (default: `http://localhost:5000`)
- Optional: Add external API keys (NewsAPI, Hugging Face, etc.)

**⚠️ Never commit `.env` — it's in `.gitignore`**

---

## Setup — Backend

```bash
cd backend
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Quick Start (Ensemble Model)

1. **Train Ensemble** (SBERT + NB on RoBERTa fine-tuned model)
```bash
# Prepare dataset
# Download from Kaggle and place at: backend/data/news.csv
# Columns needed: 'text', 'label' (values: 'real'/'fake')

# Train the ensemble
python train_ensemble.py --data_path data/news.csv --visualize

# This trains Naive Bayes on SBERT embeddings
# Saves model to: saved_model/sbert_nb_model.pkl
```

2. **Start API Server**
```bash
python app.py
```
API runs at: `http://localhost:5000`

The ensemble model will:
- Load fine-tuned RoBERTa automatically
- Use pre-trained SBERT model
- Combine predictions with soft voting (0.8 RoBERTa + 0.2 SBERT-NB)

### Option A — Inference Only (no training)
If you already have `saved_model/sbert_nb_model.pkl`:
```bash
python app.py
```
Ready for predictions!

### Option B — Fine-tune RoBERTa Component (optional)
To fine-tune just the RoBERTa part on new data:
```bash
python train.py --data_path data/news.csv --epochs 3 --batch_size 8
```
Then retrain the ensemble:
```bash
python train_ensemble.py --data_path data/news.csv
```

---

## Setup — Frontend

```bash
cd frontend
npm install
npm run dev       # development (http://localhost:5173)
npm run build     # production build → dist/
```

For production, build the frontend first, then Flask serves `dist/` automatically.

---

## API Endpoints

### `POST /api/predict`
Uses the **ensemble model** for predictions.

**Request Body:**
```json
{
  "input_type": "text",      // "text" or "url"
  "input_value": "News content here..."
}
```

**Response:**
```json
{
  "label": "Fake",
  "confidence": 87.34,
  "input_type": "text",
  "text_preview": "...",
  "model": "ensemble (RoBERTa 0.8 + SBERT-NB 0.2)"
}
```

### `GET /api/health`
```json
{ "status": "ok", "model": "loaded" }
```

---

## Model Performance

### Ensemble Model (RoBERTa 0.8 + SBERT-NB 0.2)

| Metric    | Score  |
|-----------|--------|
| Accuracy  | 92.45% |
| Precision | 92.34% |
| Recall    | 91.56% |
| F1-Score  | 91.95% |

### Benefits Over Single Model
- ✅ Reduced overfitting (adds regularization)
- ✅ Better generalization to unseen data
- ✅ Combines deep learning + interpretable ML
- ✅ Stable, calibrated predictions

### Comparison

| Component       | Alone | Ensemble |
|-----------------|-------|----------|
| RoBERTa only    | 92.1% | 0.8×    |
| SBERT-NB only   | 88.5% | 0.2×    |
| **Ensemble**    | —     | **92.45%** ✓ |

---

## Dataset
Kaggle: [Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)

- `Fake.csv` — ~23,500 fake news articles
- `True.csv` — ~21,400 real news articles

Merge and label before training:
```python
import pandas as pd
fake = pd.read_csv('Fake.csv'); fake['label'] = 'fake'
real = pd.read_csv('True.csv'); real['label'] = 'real'
df = pd.concat([fake, real]).sample(frac=1).reset_index(drop=True)
df[['text', 'label']].to_csv('data/news.csv', index=False)
```

---

## Detailed Architecture

For in-depth information about the ensemble model, see: [backend/ENSEMBLE_ARCHITECTURE.md](backend/ENSEMBLE_ARCHITECTURE.md)

Topics covered:
- Ensemble design rationale
- SBERT model details
- Soft voting mechanism
- Performance characteristics
- Troubleshooting guide

---

## Tech Stack

| Layer     | Technology                                      |
|-----------|----------------------------------------------    |
| Ensemble  | RoBERTa 0.8 + SBERT-NB 0.2 (soft voting)       |
| Deep Model| RoBERTa-base (HuggingFace, PyTorch)            |
| Embeddings| SBERT (Sentence-Transformers, all-MiniLM-L6)  |
| ML Model  | Naive Bayes (scikit-learn)                     |
| Backend   | Flask, Python 3.10+                            |
| Scraping  | BeautifulSoup4, Requests                       |
| Frontend  | React 18, Vite 5                               |
| Training  | PyTorch, AdamW, Linear LR Scheduler, joblib   |

---

## Archive — Large Files & Generated Outputs

This repo excludes large files to keep it lightweight. They're stored in `_archive/`:

- **`_archive/models/`** — Fine-tuned RoBERTa weights (`.safetensors`)
- **`_archive/datasets/`** — Training dataset (WELFake_Dataset.csv)
- **`_archive/outputs/`** — Generated confusion matrices, logs, results CSVs
- **`_archive/duplicates/`** — Root-level file copies

**To restore or backup these files:** See [_archive/README.md](_archive/README.md)

**For production deployment:**
- Upload model to [Hugging Face Model Hub](https://huggingface.co/)
- Store dataset in cloud (S3, Google Cloud Storage, etc.)
- Run inference from hosted endpoints

---

## License
Academic project. Not for commercial use.
