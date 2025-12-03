# Testing Guide

## Quick Local Test

Test the emotional analysis pipeline on a single book from Project Gutenberg.

### Prerequisites

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Ensure you have the trained model:**
   - Path: `/Users/devindyson/Desktop/troglodytelabs/emoWork/emoBERT/models/best_model.pt`

### Run Quick Test (5 segments)

```bash
./test_local.sh
```

This will:
- Download Pride and Prejudice from Project Gutenberg
- Detect chapters
- Run emotion predictions on first 5 chapters
- Save results to `output/pride_prejudice_test.json`

### Run Full Book Test

To process all chapters:

```bash
python3 test_single_book.py \
  --book-id 1342 \
  --title "Pride and Prejudice" \
  --author "Jane Austen" \
  --model /Users/devindyson/Desktop/troglodytelabs/emoWork/emoBERT/models/best_model.pt \
  --segment-unit chapter \
  --output output/pride_prejudice_full.json
```

### Test Different Books

**Frankenstein:**
```bash
python3 test_single_book.py \
  --book-id 84 \
  --title "Frankenstein" \
  --author "Mary Shelley" \
  --model /Users/devindyson/Desktop/troglodytelabs/emoWork/emoBERT/models/best_model.pt \
  --max-segments 10 \
  --output output/frankenstein_test.json
```

**Alice in Wonderland:**
```bash
python3 test_single_book.py \
  --book-id 11 \
  --title "Alice in Wonderland" \
  --author "Lewis Carroll" \
  --model /Users/devindyson/Desktop/troglodytelabs/emoWork/emoBERT/models/best_model.pt \
  --max-segments 10 \
  --output output/alice_test.json
```

### Using Fixed-Size Chunks

If chapter detection fails, use fixed-size chunks:

```bash
python3 test_single_book.py \
  --book-id 1952 \
  --title "The Yellow Wallpaper" \
  --author "Charlotte Perkins Gilman" \
  --model /Users/devindyson/Desktop/troglodytelabs/emoWork/emoBERT/models/best_model.pt \
  --segment-unit fixed_chunk \
  --chunk-size 500 \
  --max-segments 10 \
  --output output/yellow_wallpaper_test.json
```

## Expected Output

The script will output:
```json
{
  "book_metadata": {
    "book_id": "gutenberg:1342",
    "title": "Pride and Prejudice",
    "author": "Jane Austen",
    "total_segments": 61,
    "segment_method": "chapters",
    "processed_segments": 5
  },
  "emotional_trajectory": [
    {
      "segment": 1,
      "segment_name": "CHAPTER I",
      "word_count": 452,
      "emotions": {
        "joy": 0.45,
        "anticipation": 0.38,
        "surprise": 0.22,
        "trust": 0.31,
        "sadness": 0.12,
        "fear": 0.08,
        "anger": 0.05,
        "disgust": 0.04
      },
      "dominant_emotion": "joy",
      "intensity": 0.45
    }
  ],
  "narrative_features": {
    "climax_chapter": 45,
    "climax_intensity": 0.89,
    "emotional_range": 0.76,
    "pacing_score": 0.62
  }
}
```

## Validation Notebook

For detailed data validation, run the Jupyter notebook:

```bash
jupyter notebook notebooks/gutenberg_validation.ipynb
```

This will:
- Download 5 sample books
- Test chapter detection patterns
- Analyze text quality and structure
- Validate segment compatibility with RoBERTa-128
- Generate visualizations and go/no-go decision

## Troubleshooting

### Model Not Found
If you get "Model not found" error, check the path:
```bash
ls -lh /Users/devindyson/Desktop/troglodytelabs/emoWork/emoBERT/models/best_model.pt
```

### Download Failures
Project Gutenberg sometimes has network issues. The script tries multiple URLs automatically.

### Memory Issues
If processing full books causes memory issues, use `--max-segments` to limit:
```bash
--max-segments 20
```

### Token Length Warnings
Segments over 128 tokens will be truncated automatically by the tokenizer.
