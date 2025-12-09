"""
Core shared functionality for EmoArc CLI and web app.
Contains common functions for trajectory analysis, data loading, and processing.
"""

import os
import re
import glob
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col,
    trim,
    collect_list,
    sqrt,
    pow,
    lit,
    coalesce,
    min as spark_min,
    max as spark_max,
)
from pyspark.sql.types import StructType, StructField, StringType

# Import processing modules
from lexicon_loader import load_emotion_lexicon, load_vad_lexicon
from text_preprocessor import create_chunks_df
from emotion_scorer import (
    score_chunks_with_emotions,
    score_chunks_with_vad,
    combine_emotion_vad_scores,
)
from trajectory_analyzer import analyze_trajectory
from topic_modeling import (
    prepare_topic_features,
    train_lda,
    get_chunk_topics,
    compute_book_topics,
)


def create_spark_session(app_name: str = "EmoArc"):
    """Create Spark session with appropriate configuration."""
    spark = (
        SparkSession.builder.appName(app_name)
        .config("spark.sql.adaptive.enabled", "true")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark


def load_trajectories_with_types(spark, trajectories_path):
    """Load trajectories CSV and cast columns to proper types."""
    df = spark.read.option("header", "true").csv(trajectories_path)

    numeric_cols = [
        "max_anger",
        "max_anticipation",
        "max_disgust",
        "max_fear",
        "max_joy",
        "max_sadness",
        "max_surprise",
        "max_trust",
        "avg_anger",
        "avg_anticipation",
        "avg_disgust",
        "avg_fear",
        "avg_joy",
        "avg_sadness",
        "avg_surprise",
        "avg_trust",
        "avg_valence",
        "avg_arousal",
        "avg_dominance",
        "valence_std",
        "arousal_std",
        "num_chunks",
    ]

    for col_name in numeric_cols:
        if col_name in df.columns:
            df = df.withColumn(col_name, col(col_name).cast("double"))

    return df


def load_chunk_scores_with_types(spark, chunk_scores_path):
    """Load chunk scores CSV and cast columns to proper types."""
    df = spark.read.option("header", "true").csv(chunk_scores_path)

    if "chunk_index" in df.columns:
        df = df.withColumn("chunk_index", col("chunk_index").cast("int"))

    emotion_cols = [
        "anger",
        "anticipation",
        "disgust",
        "fear",
        "joy",
        "negative",
        "positive",
        "sadness",
        "surprise",
        "trust",
    ]

    for col_name in emotion_cols:
        if col_name in df.columns:
            df = df.withColumn(col_name, col(col_name).cast("double"))

    vad_cols = ["avg_valence", "avg_arousal", "avg_dominance", "vad_word_count"]
    for col_name in vad_cols:
        if col_name in df.columns:
            df = df.withColumn(col_name, col(col_name).cast("double"))

    return df


def load_metadata(spark, metadata_path="data/gutenberg_metadata.csv"):
    """Load metadata (not cached due to Spark DataFrame serialization issues)."""
    metadata_df = spark.read.option("header", "true").csv(metadata_path)
    # Filter English books only
    metadata_df = metadata_df.filter(col("Language") == "en")
    return metadata_df


def read_book_text(book_id: str, books_dir: str = "data/books") -> str:
    """Read book text from file."""
    try:
        book_path = f"{books_dir}/{book_id}"
        with open(book_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
            text = re.sub(r"\*\*\* START.*?\*\*\*", "", text, flags=re.DOTALL)
            text = re.sub(r"\*\*\* END.*?\*\*\*", "", text, flags=re.DOTALL)
            return text
    except Exception as e:
        raise RuntimeError(f"Could not read book file {book_id}: {e}")


def get_input_trajectory(
    spark,
    book_id=None,
    text_file=None,
    output_dir="output",
    books_dir="data/books",
    metadata_path="data/gutenberg_metadata.csv",
    emotion_lexicon="data/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt",
    vad_lexicon="data/NRC-VAD-Lexicon-v2.1.txt",
    compute_topics=False,
    num_topics=10,
):
    """
    Get trajectory for input (book ID or text file).

    Args:
        spark: SparkSession
        book_id: Gutenberg book ID
        text_file: Path to text file
        output_dir: Output directory for main.py results
        books_dir: Directory containing book files
        metadata_path: Path to metadata CSV
        emotion_lexicon: Path to emotion lexicon
        vad_lexicon: Path to VAD lexicon
        compute_topics: Whether to compute topic distributions (default: False)
        num_topics: Number of topics for LDA (default: 10)

    Returns:
        tuple: (trajectory_df, chunk_scores_df, title, author, book_topics_df)
        book_topics_df is None if compute_topics=False
    """
    trajectory = None
    chunk_scores = None
    title = None
    author = None
    book_topics = None

    # Load lexicons
    emotion_df = load_emotion_lexicon(spark, emotion_lexicon)
    vad_df = load_vad_lexicon(spark, vad_lexicon)

    # Case 1: Text file input
    if text_file:
        try:
            with open(text_file, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
            title = os.path.basename(text_file).replace(".txt", "").replace("_", " ")
            author = "File Input"

            schema = StructType(
                [
                    StructField("book_id", StringType(), True),
                    StructField("title", StringType(), True),
                    StructField("author", StringType(), True),
                    StructField("text", StringType(), True),
                ]
            )
            books_df = spark.createDataFrame(
                [("text_file", title, author, text)], schema
            )

            text_len = len(text)
            # Use percentage-based chunking: default 20 chunks
            # For very short texts, use fewer chunks
            num_chunks = 20
            if text_len < 10000:
                # For texts < 10k chars, use 10 chunks max
                num_chunks = max(5, min(10, text_len // 1000))

            # Use max_chunk_size to prevent memory issues with large texts
            chunks_df = create_chunks_df(
                spark, books_df, num_chunks=num_chunks, max_chunk_size=20000
            )
            emotion_scores = score_chunks_with_emotions(spark, chunks_df, emotion_df)
            vad_scores = score_chunks_with_vad(spark, chunks_df, vad_df)
            chunk_scores = combine_emotion_vad_scores(emotion_scores, vad_scores)
            trajectory = analyze_trajectory(spark, chunk_scores)

            # Compute topics if requested
            if compute_topics:
                feature_df, _ = prepare_topic_features(
                    spark, chunks_df, vocab_size=5000, min_df=2
                )
                lda_model = train_lda(
                    spark, feature_df, num_topics=num_topics, max_iter=50
                )
                chunk_topics = get_chunk_topics(spark, feature_df, lda_model)
                book_topics = compute_book_topics(spark, chunk_topics)

        except Exception as e:
            raise RuntimeError(f"Error processing text file: {e}")

    # Case 2: Book ID input
    elif book_id:
        # Try to load from main.py output first
        chunk_scores_path = f"{output_dir}/chunk_scores"
        trajectories_path = f"{output_dir}/trajectories"

        chunk_csv_files = glob.glob(f"{chunk_scores_path}/*.csv")
        traj_csv_files = glob.glob(f"{trajectories_path}/*.csv")

        if (chunk_csv_files or os.path.exists(chunk_scores_path)) and (
            traj_csv_files or os.path.exists(trajectories_path)
        ):
            try:
                output_chunks = load_chunk_scores_with_types(spark, chunk_scores_path)
                output_chunks = output_chunks.filter(col("book_id") == book_id)

                output_trajectories = load_trajectories_with_types(
                    spark, trajectories_path
                )
                output_trajectories = output_trajectories.filter(
                    col("book_id") == book_id
                )

                if output_chunks.count() > 0 and output_trajectories.count() > 0:
                    chunk_scores = output_chunks
                    trajectory = output_trajectories
                    book_info = trajectory.select("title", "author").first()
                    title = book_info["title"]
                    author = book_info["author"]
            except Exception as e:
                # Fall through to processing from Gutenberg data
                chunk_scores = None
                trajectory = None

        # Process from Gutenberg data if not in main.py output
        if chunk_scores is None or trajectory is None:
            metadata_df = load_metadata(spark, metadata_path)
            # Trim whitespace and compare as strings to handle any type mismatches
            metadata_df = metadata_df.filter(
                (col("Language") == "en")
                & (trim(col("Etext Number")) == str(book_id).strip())
            )

            if metadata_df.count() == 0:
                # Try without language filter in case language column has issues
                metadata_df_retry = load_metadata(spark, metadata_path)
                metadata_df_retry = metadata_df_retry.filter(
                    trim(col("Etext Number")) == str(book_id).strip()
                )
                if metadata_df_retry.count() == 0:
                    raise ValueError(f"Book {book_id} not found in Gutenberg metadata!")
                else:
                    metadata_df = metadata_df_retry

            book_info = metadata_df.select(
                col("Etext Number").alias("book_id"),
                col("Title").alias("title"),
                col("Authors").alias("author"),
            ).first()

            title = book_info["title"]
            author = book_info["author"]

            book_text = read_book_text(book_id, books_dir)
            if not book_text:
                raise ValueError(f"Could not read book file for {book_id}!")

            schema = StructType(
                [
                    StructField("book_id", StringType(), True),
                    StructField("title", StringType(), True),
                    StructField("author", StringType(), True),
                    StructField("text", StringType(), True),
                ]
            )
            books_df = spark.createDataFrame(
                [(book_id, title, author, book_text)], schema
            )

            text_len = len(book_text)
            # Use percentage-based chunking: default 20 chunks
            # For very short texts, use fewer chunks
            num_chunks = 20
            if text_len < 10000:
                # For texts < 10k chars, use 10 chunks max
                num_chunks = max(5, min(10, text_len // 1000))

            # Use max_chunk_size to prevent memory issues with large texts
            chunks_df = create_chunks_df(
                spark, books_df, num_chunks=num_chunks, max_chunk_size=20000
            )
            emotion_scores = score_chunks_with_emotions(spark, chunks_df, emotion_df)
            vad_scores = score_chunks_with_vad(spark, chunks_df, vad_df)
            chunk_scores = combine_emotion_vad_scores(emotion_scores, vad_scores)
            trajectory = analyze_trajectory(spark, chunk_scores)

            # Compute topics if requested
            if compute_topics:
                feature_df, _ = prepare_topic_features(
                    spark, chunks_df, vocab_size=5000, min_df=2
                )
                lda_model = train_lda(
                    spark, feature_df, num_topics=num_topics, max_iter=50
                )
                chunk_topics = get_chunk_topics(spark, feature_df, lda_model)
                book_topics = compute_book_topics(spark, chunk_topics)

    else:
        raise ValueError("No input specified! Please provide book_id or text_file.")

    return trajectory, chunk_scores, title, author, book_topics


def find_books_by_emotion_preferences(
    spark, trajectories, emotion_preferences, top_n=20
):
    """
    Find books that match user's emotion preferences.

    Args:
        spark: SparkSession
        trajectories: DataFrame with book trajectories
        emotion_preferences: Dict with emotion names as keys and desired levels (0-1) as values
        top_n: Number of books to return

    Returns:
        DataFrame with matching books and match scores
    """
    # Get emotion columns
    emotion_cols = [
        "avg_anger",
        "avg_anticipation",
        "avg_disgust",
        "avg_fear",
        "avg_joy",
        "avg_sadness",
        "avg_surprise",
        "avg_trust",
    ]

    # Normalize emotion preferences to match column names
    pref_dict = {}
    for emotion, value in emotion_preferences.items():
        if emotion.lower() in [
            "anger",
            "anticipation",
            "disgust",
            "fear",
            "joy",
            "sadness",
            "surprise",
            "trust",
        ]:
            col_name = f"avg_{emotion.lower()}"
            pref_dict[col_name] = value

    # Compute min/max for normalization
    stats = trajectories.agg(
        *[
            spark_min(col(col_name)).alias(f"min_{col_name}")
            for col_name in emotion_cols
            if col_name in trajectories.columns
        ],
        *[
            spark_max(col(col_name)).alias(f"max_{col_name}")
            for col_name in emotion_cols
            if col_name in trajectories.columns
        ],
    ).first()

    # Convert stats to dict for easier access
    stats_dict = stats.asDict() if stats else {}

    # Normalize preferences and compute distance
    distance_expr = lit(0.0)
    has_any_preference = False

    for col_name, pref_value in pref_dict.items():
        if col_name in trajectories.columns and pref_value > 0:
            has_any_preference = True
            # Get min/max for this emotion
            min_key = f"min_{col_name}"
            max_key = f"max_{col_name}"
            min_val = stats_dict.get(min_key, 0.0) or 0.0
            max_val = stats_dict.get(max_key, 1.0) or 1.0

            # Normalize preference to 0-1 range based on actual data range
            if max_val > min_val:
                norm_pref = (pref_value - min_val) / (max_val - min_val)
            else:
                norm_pref = 0.5

            # Normalize book emotion value
            book_val = coalesce(col(col_name), lit(0.0))
            if max_val > min_val:
                norm_book_val = (book_val - min_val) / (max_val - min_val)
            else:
                norm_book_val = lit(0.5)

            # Add squared difference to distance
            distance_expr = distance_expr + pow(norm_book_val - norm_pref, 2)

    if not has_any_preference:
        # If no preferences set, return empty
        return spark.createDataFrame([], trajectories.schema)

    # Compute match score: higher is better (inverse of distance)
    # Use 1 / (1 + distance) to convert distance to similarity
    match_score = 1.0 / (1.0 + sqrt(distance_expr))

    # Add match score and select relevant columns
    result = (
        trajectories.withColumn("match_score", match_score)
        .select(
            "book_id",
            "title",
            "author",
            "match_score",
            "avg_anger",
            "avg_anticipation",
            "avg_disgust",
            "avg_fear",
            "avg_joy",
            "avg_sadness",
            "avg_surprise",
            "avg_trust",
            "avg_valence",
            "avg_arousal",
        )
        .orderBy(col("match_score").desc())
        .limit(top_n)
    )

    return result
