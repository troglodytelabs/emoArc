# EmoArc - Emotion Trajectory Analysis and Recommendation System

A Spark-based big data analytics project that analyzes emotion trajectories in Project Gutenberg books and provides book recommendations based on emotional story arcs.

## Overview

[Project Gutenberg](https://www.gutenberg.org/) is the oldest digital library archive with over 70,000 free eBooks -- primarily public domain books whose copyrights have expired -- that help make literature accessible to all.

**EmoArc** processes 75,000+ books from Project Gutenberg to:

1. **Segment** texts into percentage-based chunks (default: 20 chunks per book, 5% each)
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
├── app.py                     # Streamlit web application
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

2. Download Data (NRC Lexicons & Gutenberg Books):

   Place the following files in the `data/` directory:

   - `NRC-Emotion-Lexicon-Wordlevel-v0.92.txt` - Download from [NRC Emotion Lexicon](http://saifmohammad.com/WebPages/NRC-Emotion-Lexicon.htm)
   - `NRC-VAD-Lexicon-v2.1.txt` - Download from [NRC VAD Lexicon](http://saifmohammad.com/WebPages/nrc-vad.html)
   - `books` directory - Download from [Kaggle](https://www.kaggle.com/datasets/lokeshparab/gutenberg-books-and-metadata-2025?select=books)
   - `gutenberg_metadata.csv` - Download from [Kaggle](https://www.kaggle.com/datasets/lokeshparab/gutenberg-books-and-metadata-2025?select=books)

   These files are not included in the repository.

## How to Run the Project

### Step 1: Generate Trajectories (main.py)

First, run the main pipeline to process books and generate trajectories:

```bash
# Process all English books (takes several hours)
python main.py

# Process limited number of books (for testing)
python main.py --limit 100

# Custom number of chunks (percentage-based)
python main.py --num-chunks 100  # Creates 100 chunks per book

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

### Step 3: Interactive Web Application (app.py)

For a user-friendly web interface, use the Streamlit app:

```bash
streamlit run app.py
```

This will open a web browser at `http://localhost:8501` with an interactive interface.

#### Features:

1. **Book Analysis & Recommendations**:

   - Search books by title (partial match supported)
   - Enter book ID directly
   - Upload text files for analysis
   - View interactive emotion trajectory plots
   - See emotion statistics and trajectory summaries
   - Automatically get book recommendations based on emotion similarity

2. **Explore Books**:

   - Discover top books by emotion characteristics (Joy, Sadness, Fear, etc.)
   - Browse books ranked by specific emotions

3. **Interactive Visualizations**:
   - Plotly-based interactive charts
   - Zoom, pan, and hover for detailed exploration
   - Toggle emotions on/off in the legend
   - Download plots as PNG

#### Usage:

1. **Search and Analyze**:

   - Select input method (Search by Title, Enter Book ID, or Upload Text File)
   - If trajectories are available, adjust the number of recommendations (5-20)
   - Click "Analyze Book & Get Recommendations"
   - View analysis results and recommendations in one place

2. **Explore by Emotion**:
   - Select an emotion from the dropdown
   - Choose number of books to display (10-50)
   - Click "Show Top Books" to see rankings

**Note**: The app requires trajectories from `main.py` for recommendations. Run `python main.py` first to generate trajectories.

### Command Line Options

**main.py options:**

- `--books-dir`: Directory containing book files (default: `data/books`)
- `--metadata`: Path to metadata CSV (default: `data/gutenberg_metadata.csv`)
- `--emotion-lexicon`: Path to NRC Emotion Lexicon
- `--vad-lexicon`: Path to NRC VAD Lexicon
- `--num-chunks`: Number of chunks per book for percentage-based chunking (default: 20)
- `--limit`: Limit number of books to process (for testing)
- `--output`: Output directory for results (default: `output`)
- `--language`: Filter books by language (default: `en`)
- `--mode`: Run mode - `local` or `cluster` (default: `local`)
- `--driver-memory`: Driver memory (default: `8g`)
- `--executor-memory`: Executor memory (default: `8g`)
- `--skip-embeddings`: Skip Word2Vec embeddings computation
- `--skip-topics`: Skip LDA topic modeling

**demo.py options:**

- `--book-id`: Gutenberg book ID to analyze
- `--text-file`: Path to text file to analyze
- `--analyze`: Analyze input and create visualizations
- `--recommend`: Get recommendations based on input
- `--limit`: Limit number of books to consider for recommendations (optional)
- `--output-dir`: Directory with output from main.py (default: `output`)

**app.py (Streamlit app):**

- No command-line options needed - all configuration is done through the web interface
- Automatically uses `output/` directory for trajectories
- Supports all input methods through the UI

## Running on AWS EMR

This section describes how to run EmoArc on Amazon EMR for large-scale processing.

### Prerequisites

- AWS account with EMR permissions
- AWS CLI configured (`aws configure`)
- S3 bucket for data and code

### Step 1: Prepare S3 Bucket

Upload your data, code, and create a bootstrap script for dependencies:

```bash
# Create S3 bucket (if needed)
aws s3 mb s3://your-bucket-name

# Upload data files
aws s3 sync data/ s3://your-bucket-name/data/

# Package source code as zip (required for PySpark to find modules)
cd src && zip -r ../src.zip . && cd ..

# Upload main.py and src.zip
aws s3 cp main.py s3://your-bucket-name/
aws s3 cp src.zip s3://your-bucket-name/

# Upload bootstrap script
aws s3 cp bootstrap.sh s3://your-bucket-name/
```

### Step 2: Create EMR Cluster

Create an EMR cluster with Spark and the bootstrap action to install dependencies:

```bash
aws emr create-cluster \
  --name "EmoArc Cluster" \
  --release-label emr-7.12.0 \
  --applications Name=Spark \
  --instance-type m5.xlarge \
  --instance-count 3 \
  --use-default-roles \
  --ec2-attributes KeyName=your-key-pair \
  --log-uri s3://your-bucket-name/logs/ \
  --bootstrap-actions Path=s3://your-bucket-name/bootstrap.sh,Name="Install Dependencies"
```

### Step 3: Submit Spark Job

Submit via AWS CLI (note the `--py-files` to include source modules):

```bash
aws emr add-steps \
  --cluster-id j-XXXXXXXXXXXXX \
  --steps 'Type=Spark,Name=EmoArc Pipeline,ActionOnFailure=CONTINUE,Args=[--deploy-mode,cluster,--py-files,s3://your-bucket-name/src.zip,s3://your-bucket-name/main.py,--books-dir,s3://your-bucket-name/data/books,--metadata,s3://your-bucket-name/data/gutenberg_metadata.csv,--emotion-lexicon,s3://your-bucket-name/data/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt,--vad-lexicon,s3://your-bucket-name/data/NRC-VAD-Lexicon-v2.1.txt,--output,s3://your-bucket-name/output,--mode,cluster]'
```

### Step 4: Monitor Job Progress

```bash
# Check step status
aws emr describe-step --cluster-id j-XXXXXXXXXXXXX --step-id s-XXXXXXXXXXXXX

# View logs
aws s3 ls s3://your-bucket-name/logs/j-XXXXXXXXXXXXX/steps/

# Or use EMR console: https://console.aws.amazon.com/emr
```

### Step 5: Download Results

```bash
# Download output from S3
aws s3 sync s3://your-bucket-name/output/ ./output/
```

### Troubleshooting

**Out of Memory errors:**

- Use larger instance types (r5.2xlarge or r5.4xlarge)
- Or override memory: `--driver-memory 16g --executor-memory 16g`
- Reduce parallelism: `--conf spark.sql.shuffle.partitions=200`

**Missing Python packages:**

- Ensure bootstrap script ran successfully (check bootstrap logs in S3)
- SSH to a worker node and verify: `pip3 list | grep numpy`

**Python version mismatch:**

- Set PYSPARK_PYTHON: `--conf spark.pyspark.python=/usr/bin/python3`
- Ensure consistent Python versions across cluster

**S3 access issues:**

- Verify IAM roles have S3 read/write permissions
- Check bucket policy allows EMR access

## How It Works

The system processes books through a pipeline that extracts emotional features and compares them to find similar books.

### 1. Text Segmentation

- Books are split into percentage-based chunks (default: 20 chunks per book, 5% each)
- Each chunk is assigned a sequential index

### 2. Preprocessing

- Convert to lowercase
- Remove special characters
- Tokenize into words
- Remove stopwords (English)
- Apply Porter stemming

### 3. Emotion Scoring

- Map each word to NRC Emotion Lexicon using **Plutchik's 8 basic emotions**: anger, anticipation, disgust, fear, joy, sadness, surprise, trust
- Note: The NRC lexicon also includes "negative" and "positive" (sentiment labels), but we focus on the 8 core emotions for better accuracy
- Map each word to NRC VAD Lexicon (Valence, Arousal, Dominance)
- Aggregate scores per chunk (counts for emotions, averages for VAD)

### 4. Trajectory Analysis

For each book, we compute aggregated statistics that capture the emotional trajectory:

**Emotion Statistics (per book):**

- **Average emotions**: Mean value for each emotion across all chunks
  - `avg_anger`, `avg_anticipation`, `avg_disgust`, `avg_fear`, `avg_joy`, `avg_sadness`, `avg_surprise`, `avg_trust`
- **Emotion ratios**: Proportion of each emotion relative to total (for normalized comparison)
  - `ratio_anger`, `ratio_anticipation`, etc.
- **VAD statistics**:
  - Mean: `avg_valence`, `avg_arousal`, `avg_dominance`
- **Trajectory features**:
  - `num_chunks`: Total number of chunks in the book (always 20)
  - `emotion_trajectory`: Array of emotion scores per chunk for trajectory comparison
- **Enhanced features** (optional):
  - `book_embedding`: Word2Vec-based semantic embedding (100 dimensions)
  - `book_topics`: LDA topic distribution (10 topics by default)

**Why these features?**

- **Averages** capture the overall emotional tone
- **Ratios** enable cross-book comparison regardless of book length
- **Trajectories** track how emotions evolve through the narrative
- **Embeddings and topics** provide semantic similarity beyond just emotions

### 5. Recommendation System

The recommendation system finds books with similar emotional trajectories using **feature-based similarity**:

**Similarity Calculation:**

1. **Feature Extraction**: For each book, extract 11 normalized features:

   - 8 Plutchik emotion averages (anger, anticipation, disgust, fear, joy, sadness, surprise, trust)
   - 3 VAD scores (valence, arousal, dominance)

2. **Normalization**:

   - Each feature is normalized to 0-1 range using min-max normalization
   - Normalization is based on the range across all books (excluding the query book)
   - This ensures all features contribute equally despite different scales

3. **Distance Calculation**:

   - Compute Euclidean distance in the 11-dimensional normalized feature space
   - Formula: `distance = sqrt(Σ(feature_i - query_i)²)` for all 11 features

4. **Similarity Score**:

   - Convert distance to similarity: `similarity = 1 / (1 + distance)`
   - Range: 0 (completely different) to 1 (identical)
   - Typical range: 0.65-0.90 for similar books

5. **Ranking**:
   - Sort all books by similarity score (descending)
   - Return top N recommendations

**Note**: Trajectory similarity (cosine similarity on emotion sequences) is available but not currently used, as feature-based similarity is faster and provides good results.

## Output

The pipeline generates:

1. **trajectories/**: Parquet files with aggregated trajectory statistics per book (emotion scores, VAD, embeddings, topics)
2. **demo_output/**: Visualization plots and recommendation CSVs (from demo.py)
3. **Streamlit app**: Interactive web interface for analysis and recommendations (from app.py)

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
