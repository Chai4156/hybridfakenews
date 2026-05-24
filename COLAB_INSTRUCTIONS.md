# RoBERTa Fine-tuning for Fake News Detection - Google Colab Guide

This directory contains code to fine-tune RoBERTa on Google Colab with GPU support for the fake news detection task.

## Files Included

1. **RoBERTa_Colab_Training.ipynb** - Ready-to-use Jupyter notebook (Recommended)
2. **roberta_colab_training.py** - Python script version

## Prerequisites

- Google Colab account (free with Google account)
- WELFake_Dataset.csv (download from Kaggle: https://www.kaggle.com/datasets/shoaib98/fake-news-dataset)
- Google Drive account with enough storage (~2-3 GB for model)

## Quick Start (Using Jupyter Notebook - Recommended)

### Step 1: Upload Dataset to Google Drive
1. Download `WELFake_Dataset.csv` from Kaggle
2. Upload it to your Google Drive (recommended: root or a dedicated folder)

### Step 2: Open the Notebook in Colab
1. Go to https://colab.research.google.com/
2. Click **File → Open notebook**
3. Select **GitHub** tab and paste: `roberta_colab_training.py`
   - OR upload the `.ipynb` file directly

### Step 3: Configure Parameters
In **Step 6 cell**, modify:
```python
DATA_PATH = '/content/drive/MyDrive/WELFake_Dataset.csv'
OUTPUT_DIR = '/content/drive/MyDrive/roberta_fine_tuned'
BATCH_SIZE = 16  # Increase to 32 if you have more GPU memory
EPOCHS = 3  # Increase for better performance
```

### Step 4: Run All Cells
Click **Runtime → Run all** (Ctrl+F9)

### Step 5: Monitor Progress
- The notebook will show training progress with metrics
- Best model is saved to Google Drive automatically
- Training takes approximately:
  - **~2 hours** with batch_size=16
  - **~4+ hours** with batch_size=8 for higher accuracy

## Training Configuration

### Recommended Settings
```python
EPOCHS = 5           # More epochs = better performance (but longer)
BATCH_SIZE = 16      # Good balance of speed and memory
LEARNING_RATE = 2e-5 # Standard for fine-tuning
MAX_LEN = 256        # Token sequence length
```

### If Out of Memory
Reduce `BATCH_SIZE` to 8 or 4

### If Too Slow
Reduce `EPOCHS` to 2, increase `BATCH_SIZE` to 32 (if GPU allows)

## Expected Performance

After fine-tuning for 3-5 epochs:
- **Accuracy**: 90-95%
- **F1 Score**: 90-95%
- **Training Time**: 2-4 hours

## Output Files

In your Google Drive (`roberta_fine_tuned/`):
```
roberta_fine_tuned/
├── config.json           # Model configuration
├── pytorch_model.bin     # Model weights (~500 MB)
├── tokenizer.json        # Tokenizer configuration
├── vocab.json            # RoBERTa vocabulary
├── merges.txt            # BPE merges
└── special_tokens_map.json
```

## Using the Fine-tuned Model

After training, load and use the model locally:

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load from Drive
model_path = "/path/to/roberta_fine_tuned"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Make predictions
text = "Your news article here..."
inputs = tokenizer(text, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    prediction = torch.argmax(logits, dim=1).item()

label = "Real" if prediction == 1 else "Fake"
print(f"Prediction: {label}")
```

## GPU Specification

Google Colab provides:
- **T4 GPU** (default) - 16GB VRAM - Supports batch_size up to 32
- **V100 GPU** (premium) - 32GB VRAM - Supports batch_size up to 64
- **P100 GPU** (premium) - 16GB VRAM - Supports batch_size up to 32

Check your GPU:
```python
!nvidia-smi
```

## Troubleshooting

### "No module named 'transformers'"
Solution: The notebook automatically installs it in Step 1. Make sure you ran that cell.

### CUDA Out of Memory
Solution: Reduce `BATCH_SIZE` to 8 or 4

### Data file not found
Solution: Check the path matches your Google Drive structure. Use:
```python
!ls /content/drive/MyDrive/  # List files in Drive
```

### Slow Training
Solution: 
- Use T4/V100 GPU (check runtime type)
- Reduce `MAX_LEN` to 128
- Increase `BATCH_SIZE` if memory allows

### Model not saving
Solution: Ensure you:
1. Properly mounted Google Drive (Step 2)
2. Have write permissions to the directory
3. Have enough space (at least 1 GB)

## Advanced Tips

### Training on Multiple GPUs
Colab supports multiple GPUs. Modify the notebook to use `torch.nn.DataParallel`:
```python
if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)
```

### Remove Validation Steps
If you want only training without validation to save time:
```python
# Comment out/remove validation section
```

### Save Checkpoint Every N Epochs
Modify the training loop to save checkpoints:
```python
if (epoch + 1) % 2 == 0:  # Save every 2 epochs
    model.save_pretrained(f"{OUTPUT_DIR}/checkpoint_{epoch+1}")
```

## Dataset Information

**WELFake Dataset Statistics:**
- Total samples: ~72,000
- Fake news: ~36,000
- Real news: ~36,000
- Split: 70% train, 15% val, 15% test

## References

- Transformers Library: https://huggingface.co/transformers/
- RoBERTa Paper: https://arxiv.org/abs/1907.11692
- WELFake Dataset: https://www.kaggle.com/datasets/shoaib98/fake-news-dataset

## Support

For issues:
1. Check the troubleshooting section above
2. Verify your data path is correct
3. Check GPU memory with `!nvidia-smi`
4. Review training logs for specific errors

---

**Happy Training! 🚀**
