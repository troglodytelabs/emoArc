# EmoArc - Emotion Trajectory Analysis and Recommendation System

A Spark-based big data analytics project that analyzes emotion trajectories in Project Gutenberg books and provides book recommendations based on emotional story arcs.

## Overview

EmoArc processes 75,000+ books from Project Gutenberg to:

1. **Segment** texts into fixed-length chunks (10,000 characters)
2. **Preprocess** text (remove stopwords, stemming, lemmatization)
3. **Score** each chunk using NRC Emotion Lexicon (8 Plutchik emotions) and NRC VAD Lexicon (Valence, Arousal, Dominance)
4. **Analyze** emotion trajectories to identify peaks, dominant emotions, and patterns
5. **Recommend** books with similar emotion trajectories

## Project Structure

```
emoArc/
├── src/
│   ├── __init__.py
│   ├── lexicon_loader.py      # Load NRC Emotion and VAD lexicons
│   ├── text_preprocessor.py   # Text chunking and preprocessing
│   ├── emotion_scorer.py      # Score chunks with emotions and VAD
│   ├── trajectory_analyzer.py # Analyze emotion trajectories
│   └── recommender.py         # Recommendation system
├── data/
│   ├── books/                 # Project Gutenberg book files
│   ├── gutenberg_metadata.csv # Book metadata
│   ├── NRC-Emotion-Lexicon-Wordlevel-v0.92.txt  # Download from NRC website
│   └── NRC-VAD-Lexicon-v2.1.txt  # Download from NRC website
├── main.py                    # Main pipeline script
├── demo.py                    # Demo script for presentations
├── pyproject.toml            # Project dependencies
└── README.md
```

## Installation

### Prerequisites

- Python 3.13+
- Java 8 or 11 (required for Spark)
- [uv](https://github.com/astral-sh/uv) package manager

### Setup

1. Install dependencies using uv:

```bash
uv sync
```

2. Download NLTK data (done automatically on first run):

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
```

3. Download NRC Lexicons:

   Place the following files in the `data/` directory:

   - `NRC-Emotion-Lexicon-Wordlevel-v0.92.txt` - Download from [NRC Emotion Lexicon](http://saifmohammad.com/WebPages/NRC-Emotion-Lexicon.htm)
   - `NRC-VAD-Lexicon-v2.1.txt` - Download from [NRC VAD Lexicon](http://saifmohammad.com/WebPages/nrc-vad.html)

   These files are not included in the repository.

## How to Run the Project

### Step 1: Generate Trajectories (main.py)

First, run the main pipeline to process books and generate trajectories:

```bash
# Process all English books (takes several hours)
python main.py

# Process limited number of books (for testing)
python main.py --limit 100

# Custom chunk size
python main.py --chunk-size 5000

# Custom output directory
python main.py --output results/
```

This generates:

- `output/chunk_scores/` - Emotion and VAD scores per chunk
- `output/trajectories/` - Aggregated trajectory statistics per book

### Step 2: Analyze and Get Recommendations (demo.py)

Once you have the output from main.py, use demo.py to analyze books or text files:

#### Analyze a Book

```bash
# Analyze a book (uses main.py output if available, otherwise processes)
python demo.py --book-id 11 --analyze
```

This will:

- Analyze the book's emotion trajectory
- Generate visualization plots
- Display emotion statistics

#### Analyze a Text File

```bash
# Analyze any text file
python demo.py --text-file my_story.txt --analyze
```

#### Get Recommendations

```bash
# Get recommendations for a book (requires main.py output)
python demo.py --book-id 11 --recommend

# Get recommendations for a text file (requires main.py output)
python demo.py --text-file my_story.txt --recommend

# Limit number of books to compare against
python demo.py --book-id 11 --recommend --limit 100
```

This will:

- Process your input (book or text file)
- Compare against trajectories from main.py output
- Find books with similar emotion trajectories
- Display top 10 recommendations with similarity scores
- Save results to CSV

### Command Line Options

**main.py options:**

- `--books-dir`: Directory containing book files (default: `data/books`)
- `--metadata`: Path to metadata CSV (default: `data/gutenberg_metadata.csv`)
- `--emotion-lexicon`: Path to NRC Emotion Lexicon
- `--vad-lexicon`: Path to NRC VAD Lexicon
- `--chunk-size`: Chunk size in characters (default: 10000)
- `--limit`: Limit number of books to process (for testing)
- `--output`: Output directory for results (default: `output`)
- `--language`: Filter books by language (default: `en`)

**demo.py options:**

- `--book-id`: Gutenberg book ID to analyze
- `--text-file`: Path to text file to analyze
- `--analyze`: Analyze input and create visualizations
- `--recommend`: Get recommendations based on input
- `--limit`: Limit number of books to consider for recommendations (optional)
- `--output-dir`: Directory with output from main.py (default: `output`)

## How It Works

### 1. Text Segmentation

- Books are split into fixed-length chunks (default: 10,000 characters)
- Each chunk is assigned a sequential index

### 2. Preprocessing

- Convert to lowercase
- Remove special characters
- Tokenize into words
- Remove stopwords (English)
- Apply Porter stemming

### 3. Emotion Scoring

- Map each word to NRC Emotion Lexicon (8 emotions: anger, anticipation, disgust, fear, joy, sadness, surprise, trust)
- Map each word to NRC VAD Lexicon (Valence, Arousal, Dominance)
- Aggregate scores per chunk (counts for emotions, averages for VAD)

### 4. Trajectory Analysis

- Compute statistics per book:
  - Maximum emotion peaks
  - Average emotions across chunks
  - VAD statistics (mean, stddev)
  - Emotion trajectory arrays

### 5. Recommendation

- Compute similarity between books using:
  - Feature-based similarity (Euclidean distance on normalized features)
  - Trajectory similarity (cosine similarity on emotion sequences)
- Return top N most similar books

## Output

The pipeline generates:

1. **chunk_scores/**: CSV files with emotion and VAD scores per chunk
2. **trajectories/**: CSV files with aggregated trajectory statistics per book
3. **demo_output/**: Visualization plots and recommendation CSVs (from demo.py)

## Example Output

### Emotion Statistics

```
Book: "Alice's Adventures in Wonderland"
Average Joy: 0.0234
Average Sadness: 0.0156
Average Fear: 0.0123
Average Anger: 0.0089
Average Valence: 0.234
Average Arousal: 0.156
```

### Recommendations

```
Top 10 Recommendations for "Alice's Adventures in Wonderland":
1. "Through the Looking-Glass" - Similarity: 0.8923
2. "The Wonderful Wizard of Oz" - Similarity: 0.8456
...
```

## Technical Details

### Spark Configuration

- Adaptive query execution enabled
- Automatic partition coalescing
- Optimized for large-scale text processing

### Lexicons

- **NRC Emotion Lexicon**: Word-level emotion associations (0/1)
- **NRC VAD Lexicon**: Valence-Arousal-Dominance scores (-1 to 1)

### Performance Considerations

- Processing 75,000 books may take several hours
- Use `--limit` for testing and demos
- Results are saved to output directory for reuse

## Future Enhancements

- [ ] Real-time recommendation API
- [ ] Interactive visualization dashboard
- [ ] Multi-language support
- [ ] Advanced trajectory pattern recognition
- [ ] User preference learning
