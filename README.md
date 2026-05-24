# Fake News Detection System
> Final-year BE project — Fake news detection using a hybrid deep learning + ML ensemble

A full-stack fake news detection system using an **Ensemble Model** combining:
- **RoBERTa** (fine-tuned transformer): 80% weight
- **SBERT** embeddings + **Naive Bayes**: 20% weight

Served via **Flask** API and **React + Vite** frontend.

---

## 🎯 Ensemble Architecture

The system uses **soft voting** to balance deep learning accuracy with traditional ML robustness:

```
Input Text
    ↓
    ├─→ RoBERTa (fine-tuned, 0.8) ──→ Probabilities
    │
    └─→ SBERT + Naive Bayes (0.2) ──→ Probabilities

    Final Prediction = 0.8 × P_roberta + 0.2 × P_sbert_nb
```

**Why ensemble?**
- Reduces overfitting from RoBERTa's high-capacity fine-tuning
- Combines deep learning precision with interpretable ML
- Better generalization to unseen news content
- More stable, calibrated confidence scores

---

## Model Performance

### Ensemble (RoBERTa 0.8 + SBERT-NB 0.2)

| Metric    | Score   |
|-----------|---------|
| Accuracy  | 92.45%  |
| Precision | 92.34%  |
| Recall    | 91.56%  |
| F1-Score  | 91.95%  |

### Component Comparison

| Component     | Standalone | Role in Ensemble |
|---------------|------------|-----------------|
| RoBERTa only  | 92.1%      | 80% weight      |
| SBERT-NB only | 88.5%      | 20% weight      |
| **Ensemble**  | —          | **92.45%** ✓    |

---

## Dataset

**WELFake** — a merged dataset from 4 sources (Kaggle, McIntire, Reuters, BuzzFeed Political News)  
Kaggle: [WELFake: News Dataset for Fake News Classification](https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification)

- `WELFake_Dataset.csv` — columns: `title`, `text`, `label`
- Label encoding: `0 = real`, `1 = fake`

**Prepare the training file:**
```python
import pandas as pd

df = pd.read_csv('WELFake_Dataset.csv')
df['label'] = df['label'].map({0: 'real', 1: 'fake'})   # note: WELFake uses 0=real, 1=fake
df['text'] = (df['title'].fillna('') + ' ' + df['text'].fillna('')).str.strip()
df[['text', 'label']].to_csv('data/news.csv', index=False)
```

> ⚠️ Run this preprocessing step before any training script.

---

## Project Structure

```
fake-news-detector/
├── backend/
│   ├── app.py                      # Flask API (uses ensemble)
│   ├── ensemble_model.py           # Ensemble prediction logic
│   ├── train_ensemble.py           # Train NB on SBERT embeddings
│   ├── model.py                    # RoBERTa utilities
│   ├── train.py                    # RoBERTa fine-tuning script
│   ├── ENSEMBLE_ARCHITECTURE.md    # Detailed model docs
│   ├── requirements.txt
│   └── saved_model/
│       ├── fine_tuned/             # Fine-tuned RoBERTa weights
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

```bash
cp .env.example .env
```

Edit `.env`:
- `FLASK_SECRET_KEY` — random secret for production
- `CORS_ALLOWED_ORIGINS` — frontend URL (default: `http://localhost:5173`)
- `VITE_API_URL` — API endpoint (default: `http://localhost:5000`)

> ⚠️ Never commit `.env` — it's in `.gitignore`

---

## Setup — Backend

```bash
cd backend
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Step 1 — Fine-tune RoBERTa

RoBERTa was fine-tuned on the WELFake dataset using Google Colab (T4 GPU):

- **Base model:** `roberta-base` (HuggingFace)
- **Epochs:** 3
- **Batch size:** 16
- **Learning rate:** 2e-5 (AdamW + linear decay scheduler)
- **Max token length:** 256
- **Data split:** 70% train / 15% val / 15% test
- **Checkpoint:** Best model saved by validation F1

```bash
# Run on Colab (upload train.py and WELFake_Dataset.csv to Google Drive)
python train.py --data_path data/news.csv --epochs 3 --batch_size 16

# Output saved to: saved_model/fine_tuned/
```

### Step 2 — Train Ensemble (SBERT + Naive Bayes)

```bash
python train_ensemble.py --data_path data/news.csv --visualize

# Saves to: saved_model/sbert_nb_model.pkl
```

### Step 3 — Start API Server

```bash
python app.py
```

API runs at: `http://localhost:5000`

> If you already have both saved models, you can skip Steps 1–2 and go straight to Step 3.

---

## Setup — Frontend

```bash
cd frontend
npm install
npm run dev       # development — http://localhost:5173
npm run build     # production build → dist/
```

For production, build the frontend first — Flask serves `dist/` automatically.

---

## API Endpoints

### `POST /api/predict`

**Request:**
```json
{
  "input_type": "text",
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

## Tech Stack

| Layer      | Technology                                     |
|------------|------------------------------------------------|
| Ensemble   | RoBERTa 0.8 + SBERT-NB 0.2 (soft voting)      |
| Deep Model | RoBERTa-base (HuggingFace, PyTorch)            |
| Embeddings | SBERT (Sentence-Transformers, all-MiniLM-L6)   |
| ML Model   | Multinomial Naive Bayes (scikit-learn)         |
| Backend    | Flask, Python 3.10+                            |
| Scraping   | BeautifulSoup4, Requests                       |
| Frontend   | React 18, Vite 5                               |
| Training   | PyTorch, AdamW, Linear LR Scheduler, joblib    |

---

## Detailed Architecture

See [backend/ENSEMBLE_ARCHITECTURE.md](backend/ENSEMBLE_ARCHITECTURE.md) for:
- Ensemble design rationale
- SBERT model details
- Soft voting mechanism
- Performance characteristics
- Troubleshooting guide

---

## License

Academic project. Not for commercial use.
