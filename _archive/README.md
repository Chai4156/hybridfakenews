# Archive

This folder stores large files and generated outputs that are not kept in the main Git history.

## Layout
- `models/` - Fine-tuned RoBERTa weights and model artifacts
- `datasets/` - Training datasets such as `WELFake_Dataset.csv`
- `outputs/` - Confusion matrices, logs, and results CSVs
- `duplicates/` - Root-level file copies kept for backup or recovery

## Restore
Place the archived files back into the matching subfolders if you need to rebuild the full training environment.
