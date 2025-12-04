# emoArc
discovering narrative patterns through distributed affective analysis

## Scripts

### 1. Romeo and Juliet Emotional Arc Analysis (`test_roberta_romeo_juliet.py`)

Test RoBERTa emotion model on Romeo and Juliet from Project Gutenberg.

**Usage:**
```bash
python test_roberta_romeo_juliet.py
```

**What it does:**
1. Loads your trained RoBERTa model
2. Fetches Romeo and Juliet from HuggingFace Project Gutenberg dataset
3. Splits the play into Acts and Scenes
4. Analyzes emotions for each chapter
5. Generates emotional arc visualization

**Output:**
- `romeo_juliet_emotions.json` - Emotion scores per chapter
- `romeo_juliet_emotional_arc.png` - Visualization

### 2. Sample Analysis (`analyze_gutenberg_sample.py`) **RECOMMENDED**

Efficiently analyzes emotional arcs across 100 Project Gutenberg books.

**Usage:**
```bash
python analyze_gutenberg_sample.py
```

**What it does:**
1. Streams first 100 books from Project Gutenberg dataset
2. Splits each book into text chunks
3. Analyzes emotions for each chunk (8 emotions: anger, anticipation, disgust, fear, joy, sadness, surprise, trust)
4. Calculates average emotions per book
5. Generates comparative visualizations

**Output:**
- `gutenberg_sample_analysis.json` - Full results for all books
- `emotion_distributions.png` - Distribution of emotions across all books
- `top_joyful_books.png` - Top 10 books by joy score
- `top_sad_books.png` - Top 10 books by sadness score

## Setup

```bash
pip install -r requirements.txt
```

### Model Path

Update the `MODEL_PATH` variable in the script if your model is located elsewhere:
```python
MODEL_PATH = "/Users/devindyson/Desktop/troglodytelabs/emoWork/emoBERT/models/best_model.pt"
```

### Customization

- Adjust `EMOTION_LABELS` if your model uses different emotion categories
- Modify `max_length` parameter for different text chunk sizes
- Update the model architecture in `RoBERTaEmotionClassifier` if your model has a different structure
