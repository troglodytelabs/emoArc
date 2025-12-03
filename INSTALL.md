# Quick Start Guide

## Installation

Before running the notebooks, install the required dependencies:

### Option 1: Using pip (Recommended)

```bash
pip install -r requirements.txt
```

### Option 2: Install individually

```bash
pip install torch transformers numpy pandas matplotlib seaborn requests jupyter
```

### Option 3: Using conda (if you prefer conda)

```bash
conda install pytorch transformers numpy pandas matplotlib seaborn requests jupyter -c pytorch -c conda-forge
```

## Verify Installation

Open Python and test:

```python
import torch
import transformers
print("✓ PyTorch version:", torch.__version__)
print("✓ Transformers version:", transformers.__version__)
```

## Running the Notebooks

### 1. Data Validation Notebook

Validates Project Gutenberg data structure:

```bash
jupyter notebook notebooks/gutenberg_validation.ipynb
```

### 2. Single Book Analysis Notebook

Run complete emotional analysis on one book:

```bash
jupyter notebook notebooks/single_book_analysis.ipynb
```

**Before running:**
- Update `MODEL_PATH` in cell 3 to point to your trained model
- Default: `/Users/devindyson/Desktop/troglodytelabs/emoWork/emoBERT/models/best_model.pt`

## Quick Test

Test with just downloading and chapter detection (no model needed):

```bash
# Run just cells 1-12 in single_book_analysis.ipynb
# This tests download → clean → chapter detection
```

## Troubleshooting

### "ModuleNotFoundError: No module named 'torch'"

Your Jupyter notebook is using a different Python environment.

**Fix 1: Install in Jupyter's Python**
```bash
# Find which Python Jupyter uses
jupyter --paths

# Install packages there
python3 -m pip install -r requirements.txt
```

**Fix 2: Use the right kernel**
- In Jupyter, go to Kernel → Change Kernel
- Select the environment where you installed the packages

**Fix 3: Install in current notebook**

Run this in the first cell:
```python
import sys
!{sys.executable} -m pip install torch transformers numpy pandas matplotlib seaborn requests
```

### "Model not found"

Update `MODEL_PATH` in configuration cell:
```python
MODEL_PATH = "/Users/devindyson/Desktop/troglodytelabs/emoWork/emoBERT/models/best_model.pt"
```

Check the file exists:
```bash
ls -lh /Users/devindyson/Desktop/troglodytelabs/emoWork/emoBERT/models/best_model.pt
```

### Download failures

Project Gutenberg may be slow or blocked. The script tries multiple URLs automatically. If all fail, try:
- Wait a few minutes and retry
- Check your internet connection
- Try a different book ID

## Next Steps

Once installed:
1. Run `single_book_analysis.ipynb` to test Moby Dick
2. Try different books by changing `BOOK_ID`
3. Process full books by setting `MAX_CHAPTERS = None`
4. Export results are saved to `output/` directory
