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

Efficiently analyzes emotional arcs across 100 Project Gutenberg books using **normalized segmentation** for comparative analysis.

**Usage:**
```bash
python analyze_gutenberg_sample.py
```

**What it does:**
1. Streams first 100 books from Project Gutenberg dataset
2. **Divides each book into 10 equal segments (10%, 20%, ..., 100%)** - enabling direct comparison across books of different lengths
3. Analyzes emotions for each segment (8 emotions: anger, anticipation, disgust, fear, joy, sadness, surprise, trust)
4. Calculates average emotions per book
5. Generates comparative visualizations showing emotional arcs on normalized timeline

**Key Feature - Normalized Segmentation:**
- All books are divided by percentage of total length (not fixed word counts)
- Each book has exactly 10 segments representing 0-10%, 10-20%, etc.
- Enables overlaying multiple books' emotional arcs on the same chart
- Allows comparison of "beginning", "middle", "end" patterns across different stories

**Output:**
- `gutenberg_sample_analysis.json` - Full results for all books
- `emotion_distributions.png` - Distribution of emotions across all books
- `average_emotional_arc.png` - Average emotional progression from beginning to end across all 100 books
- `comparative_joy_arcs.png` - 20 joyful books' arcs overlaid on normalized timeline
- `comparative_sadness_arcs.png` - 20 sad books' arcs overlaid on normalized timeline
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
