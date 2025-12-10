# EmoArc - Emotion Trajectory Analysis System

A big data analytics pipeline for analyzing emotional arcs in literature using Apache Spark, NRC Lexicons, and LDA topic modeling.

**Key Features:**
- **Normalized emotion density scoring** (per 1000 words) for fair cross-book comparison
- **Percentage-based chunking** (20 chunks per book) for comparable narrative arc analysis
- **Plutchik's emotional dyads** (complex emotions like love, guilt, despair)
- **LDA topic modeling** for thematic content discovery
- **Professional Django web interface** with search, filtering, and visualizations

---

## Overview

EmoArc processes books from Project Gutenberg to extract emotional and thematic features:

1. **Segment** texts into percentage-based chunks (20 chunks per book, regardless of length)
2. **Normalize** text (lowercase, remove stopwords, expand contractions)
3. **Score** each chunk using:
   - NRC Emotion Lexicon (Plutchik's 8 basic emotions)
   - NRC VAD Lexicon (Valence-Arousal-Dominance)
   - **Normalized density**: emotion words per 1000 words
4. **Model topics** using LDA (Latent Dirichlet Allocation)
5. **Analyze** emotional arcs to detect narrative structure (climax position, volatility)
6. **Visualize** through Django web app with interactive Plotly charts

---

## Project Structure

```
emoArc/
├── src/
│   ├── lexicon_loader.py       # load NRC emotion and VAD lexicons
│   ├── text_preprocessor.py    # percentage-based chunking, tokenization, word count tracking
│   ├── emotion_scorer.py       # normalize emotion scores by word count (density scoring)
│   ├── trajectory_analyzer.py  # narrative arc detection, climax position
│   ├── topic_modeling.py       # LDA topic extraction and interpretation
│   └── embeddings.py           # word2vec embeddings (optional)
├── web/                        # Django web application
│   ├── books/                  # main app
│   │   ├── models.py           # database models for books, emotions, dyads, topics
│   │   ├── views.py            # view logic with percentile calculations and similarity
│   │   ├── templates/          # HTML templates with Plotly visualizations
│   │   └── management/commands/load_books.py  # CSV import with dyad calculations
│   └── manage.py               # Django management
├── data/
│   ├── books/                  # Project Gutenberg book files (*.txt)
│   ├── gutenberg_metadata.csv  # book metadata (title, author, language)
│   ├── NRC-Emotion-Lexicon-Wordlevel-v0.92.txt
│   └── NRC-VAD-Lexicon-v2.1.txt
├── main.py                     # main Spark pipeline (preprocessing → scoring → LDA → save)
└── README.md
```

---

## Installation

### Prerequisites

- **Python 3.13+**
- **Java 8 or 11** (required for Apache Spark)
- **uv** package manager: [github.com/astral-sh/uv](https://github.com/astral-sh/uv)

### Setup

```bash
# 1. Install dependencies
uv sync

# 2. Download NRC Lexicons (place in data/ directory):
#    - NRC-Emotion-Lexicon-Wordlevel-v0.92.txt
#      from: http://saifmohammad.com/WebPages/NRC-Emotion-Lexicon.htm
#    - NRC-VAD-Lexicon-v2.1.txt
#      from: http://saifmohammad.com/WebPages/nrc-vad.html
```

---

## Quick Start

### Step 1: Generate Emotion Scores

```bash
# process first 100 books (for testing)
python main.py --limit 100

# process all english books (takes hours)
python main.py

# options:
#   --limit N          process only N books
#   --chunk-size N     override default (ignored if --num-chunks set)
#   --num-chunks N     use N percentage-based chunks per book (default: 20)
#   --num-topics K     number of LDA topics to extract (default: 10)
#   --skip-topics      skip LDA topic modeling
#   --skip-embeddings  skip word2vec embeddings
#   --output DIR       output directory (default: output/)
```

**Output:**
- `output/chunk_scores/` - emotion scores per chunk
- `output/trajectories/` - aggregated book-level statistics (CSV)

### Step 2: Load Data into Django

```bash
cd web

# run migrations (first time only)
python manage.py migrate

# load books from CSV
python manage.py load_books ../output/trajectories

# start development server
python manage.py runserver
```

Visit: **http://127.0.0.1:8000/**

---

## How It Works

### 1. Percentage-Based Chunking

Unlike fixed word-count chunking, we divide each book into **exactly 20 chunks** based on character percentages:

```
chunk_i boundaries:
  start = ⌊(i × T) / N⌋
  end   = ⌊((i+1) × T) / N⌋
where:
  T = total text length (characters)
  N = 20 (number of chunks)
  i ∈ [0, 19]
```

**Why?** This allows fair comparison of narrative arcs across books of different lengths. A 100k-word novel and a 20k-word novella both have climaxes at ~70%, for example.

### 2. Text Normalization

For each chunk:
1. Convert to lowercase
2. Remove special characters (keep apostrophes)
3. Expand contractions (`isn't → is not`)
4. Tokenize into words
5. Remove stopwords (NLTK English stopwords)
6. **Track word count** before exploding to word-per-row format

### 3. Normalized Density Scoring ⭐ CRITICAL

**Problem:** Raw emotion word counts are meaningless for comparison. "The Complete Works of William Shakespeare" (500k words) would have 10× higher counts than a 50k-word novel.

**Solution:** Normalize by word count to get **emotion word density per 1000 words**:

```
For chunk cᵢ with emotion e:
  count(cᵢ, e) = Σ(w∈cᵢ) 𝟙[Lexicon(w,e) = 1]
  density(cᵢ, e) = (count(cᵢ, e) / |cᵢ|) × 1000

Book-level score:
  emotion_score(e) = (1/N) Σ density(cᵢ, e)
```

Where `|cᵢ|` is the word count of chunk `i` after preprocessing.

**Result:** All books use the same scale (10-30 typical range), enabling:
- Fair percentile rankings
- Meaningful similarity calculations
- Direct score comparisons

### 4. Emotion Lexicons

**NRC Emotion Lexicon:**
- 14,182 words mapped to Plutchik's 8 basic emotions
- Emotions: `anger, anticipation, disgust, fear, joy, sadness, surprise, trust`
- Binary associations (word either has emotion or doesn't)

**NRC VAD Lexicon:**
- 20,007 words with continuous scores [0, 1]:
  - **Valence:** pleasure/displeasure (0 = very negative, 1 = very positive)
  - **Arousal:** activation (0 = very calm, 1 = very excited)
  - **Dominance:** control (0 = very submissive, 1 = very dominant)

**Plutchik's Dyads (Complex Emotions):**
Calculated from basic emotions:

*Primary dyads (adjacent on wheel):*
- Love = (joy + trust) / 2
- Submission = (trust + fear) / 2
- Alarm = (fear + surprise) / 2
- Disappointment = (surprise + sadness) / 2
- Remorse = (sadness + disgust) / 2
- Contempt = (disgust + anger) / 2
- Aggressiveness = (anger + anticipation) / 2
- Optimism = (anticipation + joy) / 2

*Secondary dyads (skip 1):*
- Guilt = (joy + fear) / 2
- Curiosity = (trust + surprise) / 2
- Despair = (fear + sadness) / 2
- Pride = (anger + joy) / 2
- Hope = (anticipation + trust) / 2
- Anxiety = (anticipation + fear) / 2
- ...and 4 more

### 5. Per-Book LDA Topic Modeling

Uses **per-book LDA** (each book trains its own model) for maximum topic interpretability:

1. **Stop word filtering:** NLTK English stopwords (179 words) + custom narrative stopwords (50+ words: "said", "told", "asked", "replied", "exclaimed", etc.)
2. **Feature extraction:** CountVectorizer per book with 1000-word vocabulary, min DF = 1
3. **LDA training:** 5 topics per book, 20 iterations, seed = 42
4. **Topic extraction:** Top 5 words per topic, stored as word arrays
5. **Probability assignment:** Exponential decay (decay factor = 0.6) for topic ranking
6. **Output:** Top 3 topics with words and normalized probabilities saved to CSV

**Why per-book instead of corpus-wide?**
- ✅ Topics are 100% specific to each book's content
- ✅ No cross-contamination (e.g., "tarzan" won't appear in Tale of Two Cities)
- ✅ More interpretable themes for individual book analysis
- ❌ Topics aren't directly comparable across books (trade-off accepted)

**Generative process (per book):**
```
For book b with 20 chunks:
  For each topic k: φₖ ~ Dir(β)
  For θ_b ~ Dir(α):
    For each word w in chunk:
      - Choose topic: z ~ Multinomial(θ_b)
      - Choose word: w ~ Multinomial(φ_z)
```

### 6. Narrative Arc Detection

Analyzes emotional intensity across the book:

```
Intensity(cᵢ) = Σ(e ∈ {joy, fear, sadness, anger, surprise}) score(cᵢ, e)

climax_index = argmax(Intensity(cᵢ))
climax_position = (climax_index / N) × 100%
```

**Arc classification:**
- `climax_position < 30%`: In Medias Res (starts at peak)
- `climax_position > 70%`: Slow Burn (builds to climactic ending)
- `30% ≤ climax_position ≤ 70%`: Classic Arc (rising → climax → falling)

### 7. Emotional Similarity

Recommends books using Euclidean distance in 8-dimensional emotion space:

```
d(A, B) = √(Σᵢ₌₁⁸ (eₐ,ᵢ - eᵦ,ᵢ)²)
```

Normalized to 0-100% similarity using exponential decay:
```
similarity = 100 × e^(-distance/scale)
where scale = median_distance / ln(2)
```

This ensures median-distance books ≈ 50% similar.

---

## Django Web Interface

### Features

**Home (Data Browser):**
- Searchable table of all analyzed books
- Sort by title, author, emotions, valence, etc.
- Shows normalized emotion scores (per 1000 words)

**Book Detail Pages:**
- **Emotional Tone:** VAD interpretation in prose
- **Narrative Arc:** 2-panel chart with climax marker
- **Emotion Heatmap:** All 8 emotions across 20 chunks
- **Percentile Rankings:** How this book compares to all others (e.g., "96th percentile - exceptionally uplifting")
- **Complex Emotions (Dyads):** Love, guilt, despair, hope, etc.
- **Topic Modeling:** Top 3 LDA topics with probabilities
- **Similar Books:** Top 10 emotionally similar recommendations

**Methodology Page:**
- Complete mathematical documentation
- Formulas with MathJax rendering
- Algorithm pseudocode
- Academic references

### Running the Web App

```bash
cd web
python manage.py runserver

# or bind to all interfaces:
python manage.py runserver 0.0.0.0:8000
```

---

## Data Flow

```
[Project Gutenberg Books]
         ↓
    main.py (Spark pipeline)
         ├─→ Load texts + metadata
         ├─→ Percentage-based chunking (20 chunks/book)
         ├─→ Tokenization + word count tracking
         ├─→ Emotion scoring (normalized density)
         ├─→ VAD scoring
         ├─→ LDA topic modeling
         └─→ Save to CSV: output/trajectories/
         ↓
    Django: python manage.py load_books
         ├─→ Import CSV data
         ├─→ Calculate dyads from basic emotions
         ├─→ Load topic probabilities
         └─→ Store in SQLite database
         ↓
    Django Web Interface
         ├─→ Search & filter books
         ├─→ Calculate percentiles dynamically
         ├─→ Generate Plotly visualizations
         └─→ Compute similarity recommendations
```

---

## Command Reference

### main.py Options

```bash
python main.py [OPTIONS]

Required (defaults provided):
  --books-dir PATH       Directory with book files (default: data/books)
  --metadata PATH        Metadata CSV (default: data/gutenberg_metadata.csv)
  --emotion-lexicon PATH NRC Emotion Lexicon (default: data/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt)
  --vad-lexicon PATH     NRC VAD Lexicon (default: data/NRC-VAD-Lexicon-v2.1.txt)

Optional:
  --limit N              Process only first N books (for testing)
  --num-chunks N         Number of chunks per book (default: 20)
  --num-topics K         LDA topics to extract (default: 10)
  --output DIR           Output directory (default: output/)
  --language LANG        Filter by language (default: en)
  --skip-topics          Skip LDA topic modeling
  --skip-embeddings      Skip word2vec embeddings
```

### Django Management Commands

```bash
# database setup (first time)
python manage.py migrate

# load book data from trajectories CSV
python manage.py load_books <path_to_csv_or_directory>
# example: python manage.py load_books ../output/trajectories

# create admin user (optional)
python manage.py createsuperuser

# run development server
python manage.py runserver [HOST:PORT]
```

---

## Performance Notes

### Spark Configuration

The pipeline uses:
- **Adaptive query execution** for dynamic optimization
- **Partition coalescing** to reduce small files
- **4 partitions** for output (repartition(4))
- **Memory settings:**
  - Driver: 4g
  - Executor: 4g

### Processing Time

Approximate times (varies by hardware):
- **100 books:** 2-5 minutes
- **1000 books:** 15-30 minutes
- **All books (~75k):** Several hours

### Memory Issues

If you encounter OutOfMemoryError:
1. Reduce `--limit` (test with smaller batches)
2. Increase Spark memory in `main.py`:
   ```python
   .config("spark.driver.memory", "8g")
   .config("spark.executor.memory", "8g")
   ```
3. Ensure chunk_text is dropped after tokenization (already implemented in text_preprocessor.py:286)

---

## Example Output

### Trajectory CSV Columns

```
book_id,title,author,
avg_joy,avg_sadness,avg_fear,avg_anger,avg_anticipation,avg_disgust,avg_surprise,avg_trust,
avg_valence,avg_arousal,avg_dominance,
avg_love,avg_submission,avg_alarm,avg_disappointment,... (all 18 dyads),
num_chunks,climax_position,emotional_volatility,
top_topic_1,top_topic_1_prob,top_topic_2,top_topic_2_prob,top_topic_3,top_topic_3_prob
```

### Sample Scores (Normalized Density)

```
"Pride and Prejudice" by Jane Austen:
  avg_joy: 15.234        (15.2 joy words per 1000 words)
  avg_trust: 12.456
  avg_anticipation: 11.234
  avg_love: 13.845       (dyad: (joy+trust)/2)
  avg_valence: 0.623     (moderately positive)
  climax_position: 72.5  (slow burn - builds to ending)

  top_topic_1: 3, prob: 0.341  (34.1% romance/social themes)
  top_topic_2: 7, prob: 0.227  (22.7% family/domestic themes)
```

---

## Technical Details

### Why Normalized Density?

**Before normalization:**
- "The Complete Works of Shakespeare": joy = 1484.4
- "Alice in Wonderland": joy = 342.6
- **Problem:** Scores not comparable!

**After normalization (per 1000 words):**
- "The Complete Works of Shakespeare": joy = 14.8
- "Alice in Wonderland": joy = 17.1
- **Now comparable:** Alice is actually more joyful per word!

### Percentile Rankings

For interpretability, raw scores are converted to percentiles:

```python
def calculate_percentile(book_score, all_scores):
    sorted_scores = sorted(all_scores)
    rank = count(s for s in sorted_scores if s ≤ book_score)
    return (rank / len(sorted_scores)) × 100
```

**Interpretation:**
- 90th percentile = more than 90% of all books
- 50th percentile = exactly median
- 10th percentile = less than 90% of books

### Limitations

1. **Word-level analysis:** Doesn't consider sentence context, negation, or sarcasm
2. **Lexicon coverage:** ~14k words; rare words and neologisms are ignored
3. **English-only:** Lexicons are English-language
4. **Topic interpretability:** LDA topics are abstract and require domain knowledge

---

## Academic References

- Mohammad, S. M., & Turney, P. D. (2013). *Crowdsourcing a word-emotion association lexicon.* Computational Intelligence, 29(3), 436-465.
- Mohammad, S. M. (2018). *Obtaining reliable human ratings of valence, arousal, and dominance for 20,000 English words.* ACL 2018.
- Plutchik, R. (1980). *Emotion: A psychoevolutionary synthesis.* Harper & Row.
- Blei, D. M., Ng, A. Y., & Jordan, M. I. (2003). *Latent Dirichlet allocation.* JMLR, 3, 993-1022.

---

## License

This project uses the NRC Emotion and VAD Lexicons, which are available for research purposes. Please cite the original papers if using this work academically.

---

## Troubleshooting

### "Java not found" error
Install Java 8 or 11 and ensure `JAVA_HOME` is set:
```bash
export JAVA_HOME=/path/to/java
export PATH=$JAVA_HOME/bin:$PATH
```

### "Lexicon file not found"
Download the NRC lexicons and place them in `data/`:
- NRC-Emotion-Lexicon-Wordlevel-v0.92.txt
- NRC-VAD-Lexicon-v2.1.txt

### "No books in database"
Run the data loading steps:
```bash
python main.py --limit 100
cd web
python manage.py load_books ../output/trajectories
```

### Plotly charts not showing
Ensure internet connection (Plotly loads from CDN) or use local Plotly installation.

---

## Contributing

When adding features:
1. Use lowercase comments in code explaining computations
2. Update the methodology page HTML with mathematical formulas
3. Test with `--limit 10` before processing full dataset
4. Run Django migrations if changing models

---

**Built with:** Apache Spark • Django • Plotly • NRC Lexicons • scikit-learn
