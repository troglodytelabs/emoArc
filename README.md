# emoArc
discovering narrative patterns through distributed affective analysis

## Romeo and Juliet Emotional Arc Analysis

Test RoBERTa emotion model on Romeo and Juliet from Project Gutenberg.

### Setup

```bash
pip install -r requirements.txt
```

### Usage

```bash
python test_roberta_romeo_juliet.py
```

### What it does

1. Loads your trained RoBERTa model from the specified path
2. Fetches Romeo and Juliet from Project Gutenberg
3. Splits the play into Acts and Scenes
4. Analyzes emotions for each chapter (8 emotions: anger, anticipation, disgust, fear, joy, sadness, surprise, trust)
5. Generates visualizations and saves results

### Output

- `romeo_juliet_emotions.json` - JSON file with emotion scores for each chapter
- `romeo_juliet_emotional_arc.png` - Visualization showing emotional arc across the story

### Model Path

Update the `MODEL_PATH` variable in the script if your model is located elsewhere:
```python
MODEL_PATH = "/Users/devindyson/Desktop/troglodytelabs/emoWork/emoBERT/models/best_model.pt"
```

### Customization

- Adjust `EMOTION_LABELS` if your model uses different emotion categories
- Modify `max_length` parameter for different text chunk sizes
- Update the model architecture in `RoBERTaEmotionClassifier` if your model has a different structure
