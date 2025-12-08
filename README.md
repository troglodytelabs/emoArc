# EmoArc (Scratch Build)

A fresh, minimal content-based book recommendation system. Start with TF-IDF
similarity over book text and metadata; later, blend in experimental emotion
features once the core recommender is stable.

## Project layout
```
emoArc/
├── data/                 # place `gutenberg_metadata.csv` + book texts here
├── src/
│   ├── data.py           # load metadata and raw book text
│   ├── preprocess.py     # normalize and merge text fields
│   ├── features.py       # TF-IDF feature extraction utilities
│   └── recommender.py    # BookRecommender API + emotion feature hook
└── main.py               # CLI entry point
```

## Setup (pip, no uv/streamlit)
```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Prepare data
1. Download the Gutenberg metadata CSV from Kaggle and save as
   `data/gutenberg_metadata.csv`.
2. Ensure the CSV has a **title** column and a **path-like** column. Common
   aliases such as `book_title`, `name`, `file`, `filename`, `text_path`, and
   `relative_path` are accepted automatically; `path` should point to each text
   file relative to `data/`.
3. Place the referenced text files under `data/` (e.g., `data/books/<file>.txt`).

## Usage
Build the recommender and ask for similar books by title:
```bash
python main.py --metadata data/gutenberg_metadata.csv --books-dir data \
    --seed-title "Frankenstein; Or, The Modern Prometheus" --top-n 5
```

Or query with an ad-hoc text snippet:
```bash
python main.py --metadata data/gutenberg_metadata.csv --books-dir data \
    --seed-text "an explorer meets a creature in the frozen north" --top-n 3
```

If you want a quick smoke test on a small subset, add `--limit 50`.

## Extending with emotion features (future)
The recommender exposes `BookRecommender.attach_additional_features`, which can
append sparse feature matrices (e.g., NRC VAD/Wordlevel/Intensity signals) to
blend emotion-aware vectors with the TF-IDF baseline.
