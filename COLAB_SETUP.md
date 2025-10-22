# Google Colab Setup Instructions

If you want to run the `agnews_train_eval.ipynb` notebook on Google Colab, follow these instructions:

## Method 1: Direct Upload

1. Go to [Google Colab](https://colab.research.google.com/)
2. Upload `notebooks/agnews_train_eval.ipynb`
3. Add a new cell at the beginning with this code:

```python
# Install dependencies
!pip install -q transformers datasets torch accelerate scikit-learn matplotlib seaborn tqdm

# Clone repository to get data
!git clone https://github.com/gsaco/News_Classification-lab.git
%cd News_Classification-lab

# Setup directories
!mkdir -p outputs/roberta outputs/deberta outputs/modernbert
```

4. Modify file paths in the notebook:
   - Change `../data/` to `data/`
   - Change `../outputs/` to `outputs/`

5. Enable GPU:
   - Runtime → Change runtime type → Hardware accelerator → GPU

## Method 2: From GitHub

1. Go to Google Colab
2. File → Open notebook → GitHub
3. Enter repository URL: `https://github.com/gsaco/News_Classification-lab`
4. Select `notebooks/agnews_train_eval.ipynb`
5. Follow steps 3-5 from Method 1

## Colab-Specific Modifications

Add this cell at the start of the notebook:

```python
# ====== GOOGLE COLAB SETUP (Run this first!) ======

import os
import sys

# Check if running on Colab
IN_COLAB = 'google.colab' in sys.modules

if IN_COLAB:
    print("Running on Google Colab")
    
    # Install packages
    print("Installing dependencies...")
    !pip install -q transformers datasets accelerate scikit-learn matplotlib seaborn tqdm
    
    # Clone repository
    if not os.path.exists('News_Classification-lab'):
        print("Cloning repository...")
        !git clone https://github.com/gsaco/News_Classification-lab.git
    
    # Change to repository directory
    os.chdir('News_Classification-lab')
    
    # Create output directories
    !mkdir -p outputs/roberta outputs/deberta outputs/modernbert
    
    # Check GPU
    import torch
    if torch.cuda.is_available():
        print(f"✓ GPU available: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️  No GPU available. Training will be slower.")
    
    print("\n✓ Setup complete! You can now run the rest of the notebook.")
else:
    print("Not running on Colab - skipping setup")
```

## Colab Tips

1. **Save Your Work**:
   - File → Save a copy in Drive (to save your progress)
   - Models will be lost when runtime disconnects unless saved to Drive

2. **Mount Google Drive (Optional)**:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   
   # Save outputs to Drive
   output_dir = '/content/drive/MyDrive/news_classification_outputs'
   !mkdir -p {output_dir}
   ```

3. **GPU Runtime**:
   - Free tier provides T4 GPU (16GB)
   - Sufficient for all three models
   - ~1-2 hours for complete training

4. **Memory Management**:
   - If you get OOM errors, reduce batch_size:
     ```python
     batch_size = 8  # Instead of 16
     ```

5. **Session Timeout**:
   - Colab disconnects after ~12 hours or 90 minutes idle
   - Save checkpoints regularly
   - Models auto-save during training

## Path Adjustments for Colab

If you're manually adjusting paths, change:

**Original (local):**
```python
output_dir='../outputs/roberta'
data_path = '../data/rpp_news_50.csv'
```

**For Colab (in repo directory):**
```python
output_dir='outputs/roberta'
data_path = 'data/rpp_news_50.csv'
```

## Download Results from Colab

After training, download results:

```python
# Zip outputs
!zip -r outputs.zip outputs/

# Download (will appear in Colab file browser)
from google.colab import files
files.download('outputs.zip')
```

## Troubleshooting Colab

### Issue: Disk Space
```python
# Clear space
!rm -rf /root/.cache/huggingface/transformers
!rm -rf outputs/*/checkpoint-*  # Keep only best models
```

### Issue: Runtime Disconnect
```python
# Add periodic saves to Drive
import time
def save_checkpoint():
    !cp -r outputs /content/drive/MyDrive/news_classification_outputs
    print(f"Checkpoint saved at {time.strftime('%H:%M:%S')}")
```

### Issue: Slow Download
```python
# Use Kaggle mirror for datasets (if available)
from datasets import load_dataset
dataset = load_dataset("ag_news", cache_dir="/content/cache")
```

## Expected Runtime on Colab

With T4 GPU:
- Data loading and preprocessing: ~5 minutes
- RoBERTa training (3 epochs): ~30-45 minutes
- DeBERTa training (3 epochs): ~35-50 minutes
- ModernBERT training (3 epochs): ~30-45 minutes
- Total: ~2-2.5 hours

**Total notebook runtime: ~2.5-3 hours**

## Alternative: Kaggle Notebooks

The notebook also works on Kaggle with minimal changes:
1. Create new Kaggle notebook
2. Upload `agnews_train_eval.ipynb`
3. Enable GPU accelerator
4. Similar setup as Colab

Kaggle advantages:
- More generous GPU quota
- Better uptime
- Direct dataset integration

## Summary

For the best Colab experience:
1. ✅ Enable GPU
2. ✅ Add Colab setup cell at the start
3. ✅ Save copy to Drive
4. ✅ Monitor execution time
5. ✅ Download results regularly

Happy training! 🚀
