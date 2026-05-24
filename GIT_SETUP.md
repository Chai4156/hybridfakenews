# Git Setup & Push Guide

## ✅ Pre-Push Checklist — COMPLETED

Your repository is now cleaned up and ready for GitHub! Here's what was done:

### 1. ✅ Created `.gitignore`
- Excludes `venv/`, `node_modules/`, `__pycache__/`, `.env`, etc.
- Prevents accidental commits of large files, secrets, and generated outputs

### 2. ✅ Created `.env.example`
- Template for configuration variables
- Users copy this to `.env` and fill in their values
- `.env` itself is ignored and never committed

### 3. ✅ Cleaned up duplicates
Moved to `_archive/duplicates/`:
- `App.jsx`, `app.py`, `clean_news.py`, `model.py`, `train.py`

### 4. ✅ Archived large files
**`_archive/datasets/`** → `WELFake_Dataset.csv`
**`_archive/models/`** → `roberta_fine_tuned/`, `*.zip` files
**`_archive/outputs/`** → confusion matrices, logs, CSVs
**`_archive/`** → `.venv/`, `__pycache__/`

### 5. ✅ Updated README
- Added Environment Setup section
- Added Archive section with recovery instructions

---

## 🚀 Next Steps — To Push to GitHub

### Step 1: Initialize Git (if not already done)
```bash
cd "c:\Users\lucif\OneDrive\Desktop\files"
git init
git add .
git commit -m "Initial commit: Fake news detection system

- Full-stack ML application with RoBERTa + Flask + React
- Fine-tuned transformer for fake news classification
- Hybrid ensemble model (RoBERTa + NB + LR + RF)
- Clean architecture with separated frontend/backend"
```

### Step 2: Create GitHub Repository
1. Go to [github.com/new](https://github.com/new)
2. Name: `fake-news-detector`
3. Description: `Full-stack fake news detection system using RoBERTa and ensemble methods`
4. Public/Private: Your choice
5. Don't initialize with README (you have one)
6. Click "Create repository"

### Step 3: Link & Push
```bash
git remote add origin https://github.com/YOUR_USERNAME/fake-news-detector.git
git branch -M main
git push -u origin main
```

### Step 4: Add Topics (on GitHub)
- `machine-learning`
- `fake-news-detection`
- `nlp`
- `roberta`
- `flask`
- `react`
- `ensemble-learning`

---

## 📦 Backup Archive — Safe Storage

Before pushing to GitHub, backup your large files:

### Option A: Store locally + upload later
```bash
# Create a backup zip of the archive
Compress-Archive -Path _archive -DestinationPath "C:\Users\lucif\OneDrive\Desktop\fake-news-detector-backup.zip"
```

### Option B: Upload model to Hugging Face Hub (recommended for sharing)
```bash
pip install huggingface-hub
huggingface-cli login
huggingface-cli upload YOUR_USERNAME/fake-news-detector _archive/models/roberta_fine_tuned --repo-type model
```

Then add to README:
```markdown
## Pre-trained Model

Download from: [huggingface.co/YOUR_USERNAME/fake-news-detector](https://huggingface.co/YOUR_USERNAME/fake-news-detector)

```python
from huggingface_hub import snapshot_download
model_path = snapshot_download("YOUR_USERNAME/fake-news-detector")
```
```

---

## 📋 Repository Structure After Push

```
fake-news-detector/
├── .gitignore                    ✅ Excludes large/secret files
├── .env.example                  ✅ Template for environment variables
├── README.md                     ✅ Setup & API documentation
├── backend/
│   ├── app.py                    (Flask server)
│   ├── model.py                  (Model loading & inference)
│   ├── train.py                  (Fine-tuning script)
│   ├── requirements.txt           (Python dependencies)
│   └── saved_model/              (Will download on first run)
├── frontend/
│   ├── package.json              (Node dependencies)
│   ├── vite.config.js            (Build config)
│   └── src/
│       ├── App.jsx               (React component)
│       └── main.jsx
├── special/
│   ├── QUICK_START.md            (Colab instructions)
│   └── requirements.txt           (For Colab environment)
└── _archive/
    ├── README.md                 (Instructions for archived files)
    ├── datasets/                 (Training data)
    ├── models/                   (Fine-tuned weights)
    └── outputs/                  (Generated results)
```

**What's in Git:** ✅ Source code + config  
**What's NOT in Git:** ❌ venv, node_modules, .env, large files (in `_archive/` for safekeeping)

---

## 🔑 API Keys & Secrets

### Current Implementation
The app doesn't require API keys for basic usage:
- ✅ Uses `roberta-base` (no key needed)
- ✅ URL scraping with BeautifulSoup (no key needed)
- ✅ Model fine-tuning on local Kaggle dataset

### Optional Future Keys (add to `.env`)
```bash
# NewsAPI (for fetching live articles)
NEWSAPI_KEY=your_api_key_here

# Hugging Face (for model uploads)
HUGGINGFACE_TOKEN=your_hf_token_here

# AWS S3 (for cloud storage)
AWS_ACCESS_KEY_ID=xxx
AWS_SECRET_ACCESS_KEY=xxx
```

### Docker for Production (optional)
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY backend/requirements.txt .
RUN pip install -r requirements.txt
COPY backend/ .
EXPOSE 5000
CMD ["python", "app.py"]
```

---

## ✨ Final Checklist Before Pushing

- [ ] Run `git status` — should show only source files
- [ ] Check `.gitignore` is working:
  ```bash
  git check-ignore -v _archive/models/roberta_fine_tuned
  git check-ignore -v .env
  ```
- [ ] Test backend setup:
  ```bash
  cd backend
  python -m venv venv
  source venv/bin/activate
  pip install -r requirements.txt
  python app.py  # Should start without errors
  ```
- [ ] Test frontend setup:
  ```bash
  cd frontend
  npm install
  npm run dev  # Should start dev server
  ```
- [ ] Create `.env` locally:
  ```bash
  cp .env.example .env
  # Edit if needed
  ```
- [ ] First push:
  ```bash
  git log --oneline  # Verify commits
  git push -u origin main
  ```

---

**Ready to push! 🎉**

For questions, check:
- [README.md](README.md) — Project overview
- [_archive/README.md](_archive/README.md) — File restoration guide
- [.env.example](.env.example) — Configuration template
