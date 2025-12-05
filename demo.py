"""
Demo script for class presentation - analyzes text/book and provides recommendations.

Workflow:
1. Accepts either a text file or book ID
2. Gets trajectory for the input (from main.py output or by processing)
3. Compares against main.py output for recommendations
4. Generates visualizations
"""

import sys
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from lexicon_loader import load_emotion_lexicon, load_vad_lexicon
from text_preprocessor import load_books, create_chunks_df
from emotion_scorer import (
    score_chunks_with_emotions,
    score_chunks_with_vad,
    combine_emotion_vad_scores,
)
from trajectory_analyzer import analyze_trajectory
from recommender import recommend_by_features


def load_trajectories_with_types(spark, trajectories_path):
    """Load trajectories CSV and cast columns to proper types."""
    df = spark.read.option("header", "true").csv(trajectories_path)

    # Cast numeric columns
    numeric_cols = [
        "max_anger",
        "max_joy",
        "max_fear",
        "max_sadness",
        "max_surprise",
        "avg_anger",
        "avg_joy",
        "avg_fear",
        "avg_sadness",
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

    # Cast chunk_index to int
    if "chunk_index" in df.columns:
        df = df.withColumn("chunk_index", col("chunk_index").cast("int"))

    # Cast emotion columns to double
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

    # Cast VAD columns to double
    vad_cols = ["avg_valence", "avg_arousal", "avg_dominance", "vad_word_count"]
    for col_name in vad_cols:
        if col_name in df.columns:
            df = df.withColumn(col_name, col(col_name).cast("double"))

    return df


def create_spark_session():
    """Create Spark session."""
    spark = (
        SparkSession.builder.appName("EmoArc Demo")
        .config("spark.sql.adaptive.enabled", "true")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark


def plot_emotion_trajectory(chunk_scores_pd, book_title, output_path):
    """Plot emotion trajectory for a book."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # Plot 1: Emotion scores over chunks
    emotions = ["joy", "sadness", "fear", "anger"]
    colors = ["green", "blue", "red", "orange"]

    for emotion, color in zip(emotions, colors):
        if emotion in chunk_scores_pd.columns:
            axes[0].plot(
                chunk_scores_pd["chunk_index"],
                chunk_scores_pd[emotion],
                label=emotion.capitalize(),
                color=color,
                linewidth=2,
            )

    axes[0].set_xlabel("Chunk Index", fontsize=12)
    axes[0].set_ylabel("Emotion Score", fontsize=12)
    axes[0].set_title(
        f"Emotion Trajectory: {book_title}", fontsize=14, fontweight="bold"
    )
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot 2: VAD scores
    if "avg_valence" in chunk_scores_pd.columns:
        axes[1].plot(
            chunk_scores_pd["chunk_index"],
            chunk_scores_pd["avg_valence"],
            label="Valence",
            color="purple",
            linewidth=2,
        )
    if "avg_arousal" in chunk_scores_pd.columns:
        axes[1].plot(
            chunk_scores_pd["chunk_index"],
            chunk_scores_pd["avg_arousal"],
            label="Arousal",
            color="brown",
            linewidth=2,
        )

    axes[1].set_xlabel("Chunk Index", fontsize=12)
    axes[1].set_ylabel("VAD Score", fontsize=12)
    axes[1].set_title(
        "Valence-Arousal-Dominance Trajectory", fontsize=14, fontweight="bold"
    )
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved plot to {output_path}")


def get_input_trajectory(
    spark,
    book_id=None,
    text_file=None,
    output_dir="output",
    books_dir="data/books",
    metadata_path="data/gutenberg_metadata.csv",
    emotion_lexicon="data/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt",
    vad_lexicon="data/NRC-VAD-Lexicon-v2.1.txt",
):
    """
    Get trajectory for input (book ID or text file).

    Returns:
        tuple: (trajectory_df, chunk_scores_df, title, author)
    """
    trajectory = None
    chunk_scores = None
    title = None
    author = None

    # Case 1: Text file input
    if text_file:
        print(f"  Processing text file: {text_file}")
        try:
            with open(text_file, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
            title = os.path.basename(text_file).replace(".txt", "").replace("_", " ")
            author = "File Input"

            # Process the text
            print("    Loading lexicons...")
            emotion_df = load_emotion_lexicon(spark, emotion_lexicon)
            vad_df = load_vad_lexicon(spark, vad_lexicon)

            # Create DataFrame with the text
            from pyspark.sql.types import StructType, StructField, StringType

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

            print("    Creating chunks...")
            chunks_df = create_chunks_df(spark, books_df, chunk_size=10000)

            print("    Scoring chunks...")
            emotion_scores = score_chunks_with_emotions(spark, chunks_df, emotion_df)
            vad_scores = score_chunks_with_vad(spark, chunks_df, vad_df)
            chunk_scores = combine_emotion_vad_scores(emotion_scores, vad_scores)

            print("    Analyzing trajectory...")
            trajectory = analyze_trajectory(spark, chunk_scores)

        except Exception as e:
            print(f"  ❌ Error processing text file: {e}")
            return None, None, None, None

    # Case 2: Book ID input
    elif book_id:
        print(f"  Processing book ID: {book_id}")

        # Try to load from main.py output first
        chunk_scores_path = f"{output_dir}/chunk_scores"
        trajectories_path = f"{output_dir}/trajectories"

        import glob

        chunk_csv_files = glob.glob(f"{chunk_scores_path}/*.csv")
        traj_csv_files = glob.glob(f"{trajectories_path}/*.csv")

        # Check if both chunk_scores and trajectories exist in output
        if (chunk_csv_files or os.path.exists(chunk_scores_path)) and (
            traj_csv_files or os.path.exists(trajectories_path)
        ):
            try:
                print("    Checking main.py output...")
                output_chunks = load_chunk_scores_with_types(spark, chunk_scores_path)
                output_chunks = output_chunks.filter(col("book_id") == book_id)

                output_trajectories = load_trajectories_with_types(
                    spark, trajectories_path
                )
                output_trajectories = output_trajectories.filter(
                    col("book_id") == book_id
                )

                if output_chunks.count() > 0 and output_trajectories.count() > 0:
                    print("    ✓ Using results from main.py output")
                    chunk_scores = output_chunks
                    trajectory = output_trajectories
                    book_info = trajectory.select("title", "author").first()
                    title = book_info["title"]
                    author = book_info["author"]
                else:
                    print(
                        "    Book not in main.py output, processing from Gutenberg data..."
                    )
                    chunk_scores = None
                    trajectory = None
            except Exception as e:
                print(f"    Could not load from main.py output: {e}")
                chunk_scores = None
                trajectory = None

        # Process from Gutenberg data if not in main.py output
        if chunk_scores is None or trajectory is None:
            print("    Loading from Gutenberg data...")
            emotion_df = load_emotion_lexicon(spark, emotion_lexicon)
            vad_df = load_vad_lexicon(spark, vad_lexicon)

            books_df = load_books(
                spark, books_dir, metadata_path, language="en", limit=None
            )
            books_df = books_df.filter(col("book_id") == book_id)

            if books_df.count() == 0:
                print(f"  ❌ Book {book_id} not found in Gutenberg data!")
                return None, None, None, None

            book_info = books_df.select("title", "author").first()
            title = book_info["title"]
            author = book_info["author"]
            print(f"    Title: {title}")
            print(f"    Author: {author}")

            print("    Creating chunks...")
            chunks_df = create_chunks_df(spark, books_df, chunk_size=10000)

            print("    Scoring chunks...")
            emotion_scores = score_chunks_with_emotions(spark, chunks_df, emotion_df)
            vad_scores = score_chunks_with_vad(spark, chunks_df, vad_df)
            chunk_scores = combine_emotion_vad_scores(emotion_scores, vad_scores)

            print("    Analyzing trajectory...")
            trajectory = analyze_trajectory(spark, chunk_scores)

    else:
        print("  ❌ Error: No input specified!")
        return None, None, None, None

    return trajectory, chunk_scores, title, author


def demo_analysis(
    spark,
    book_id=None,
    text_file=None,
    output_dir="output",
    books_dir="data/books",
    metadata_path="data/gutenberg_metadata.csv",
    emotion_lexicon="data/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt",
    vad_lexicon="data/NRC-VAD-Lexicon-v2.1.txt",
):
    """Analyze input and create visualizations."""
    print(f"\n{'=' * 80}")
    print("Emotion Trajectory Analysis")
    print(f"{'=' * 80}")

    # Get trajectory and chunk scores
    trajectory, chunk_scores, title, author = get_input_trajectory(
        spark,
        book_id,
        text_file,
        output_dir,
        books_dir,
        metadata_path,
        emotion_lexicon,
        vad_lexicon,
    )

    if trajectory is None or chunk_scores is None:
        return

    print(f"\n  Title: {title}")
    print(f"  Author: {author}")

    # Convert to pandas for plotting
    chunk_scores_pd = chunk_scores.orderBy("chunk_index").toPandas()

    # Create plot
    plot_id = (
        book_id
        if book_id
        else os.path.basename(text_file).replace(".txt", "")
        if text_file
        else "custom"
    )
    plot_id = str(plot_id).replace("/", "_").replace("\\", "_")
    output_path = f"demo_output/{plot_id}_trajectory.png"
    os.makedirs("demo_output", exist_ok=True)
    plot_emotion_trajectory(chunk_scores_pd, title, output_path)

    # Print statistics
    print("\n  Emotion Statistics:")
    print(f"    Average Joy: {chunk_scores_pd['joy'].mean():.4f}")
    print(f"    Average Sadness: {chunk_scores_pd['sadness'].mean():.4f}")
    print(f"    Average Fear: {chunk_scores_pd['fear'].mean():.4f}")
    print(f"    Average Anger: {chunk_scores_pd['anger'].mean():.4f}")
    print(f"    Average Valence: {chunk_scores_pd['avg_valence'].mean():.4f}")
    print(f"    Average Arousal: {chunk_scores_pd['avg_arousal'].mean():.4f}")


def demo_recommendations(
    spark,
    book_id=None,
    text_file=None,
    output_dir="output",
    limit=None,
    books_dir="data/books",
    metadata_path="data/gutenberg_metadata.csv",
    emotion_lexicon="data/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt",
    vad_lexicon="data/NRC-VAD-Lexicon-v2.1.txt",
):
    """Get recommendations based on input."""
    print(f"\n{'=' * 80}")
    print("Recommendation System")
    print(f"{'=' * 80}")

    # Get trajectory for input
    liked_trajectory, _, title, author = get_input_trajectory(
        spark,
        book_id,
        text_file,
        output_dir,
        books_dir,
        metadata_path,
        emotion_lexicon,
        vad_lexicon,
    )

    if liked_trajectory is None:
        return

    print(f"  Input: {title} by {author}")

    # Load trajectories from main.py output for comparison (required)
    trajectories_path = f"{output_dir}/trajectories"
    import glob

    csv_files = glob.glob(f"{trajectories_path}/*.csv")

    if not csv_files and not os.path.exists(trajectories_path):
        print(f"  ❌ Error: No trajectories found in {trajectories_path}")
        print(
            "  Please run 'python main.py' first to generate trajectories for comparison."
        )
        print(f"  Example: python main.py --limit 100 --output {output_dir}")
        return

    try:
        print("  Loading trajectories from main.py output for comparison...")
        trajectories = load_trajectories_with_types(spark, trajectories_path)

        # Filter to limit if specified
        total_count = trajectories.count()
        if limit and total_count > limit:
            print(f"  Limiting to {limit} books for comparison...")
            trajectories = trajectories.limit(limit)

        print(f"  ✓ Loaded {trajectories.count()} trajectories for comparison")
    except Exception as e:
        print(f"  ❌ Error loading trajectories from main.py output: {e}")
        print("  Please run 'python main.py' first to generate trajectories.")
        return

    # Get recommendations
    print("  Computing recommendations...")
    liked_id = liked_trajectory.select("book_id").first()["book_id"]

    # Combine liked trajectory with trajectories from main.py output
    all_trajectories = trajectories.union(liked_trajectory)

    recommendations = recommend_by_features(spark, all_trajectories, liked_id, top_n=10)

    # Display recommendations
    print("\n  Top 10 Recommendations:")
    print(f"  {'-' * 80}")
    recs_pd = recommendations.toPandas()
    for idx, row in recs_pd.iterrows():
        print(f"  {idx + 1}. {row['title']}")
        print(f"     Author: {row['author']}")
        print(f"     Similarity: {row['similarity']:.4f}")
        print(
            f"     Emotions - Joy: {row['avg_joy']:.3f}, Sadness: {row['avg_sadness']:.3f}"
        )
        print()

    # Save to CSV
    os.makedirs("demo_output", exist_ok=True)
    rec_id = str(liked_id).replace("/", "_").replace("\\", "_")
    recs_pd.to_csv(f"demo_output/recommendations_for_{rec_id}.csv", index=False)
    print(f"  ✓ Saved recommendations to demo_output/recommendations_for_{rec_id}.csv")


def main():
    """Main demo function."""
    import argparse

    parser = argparse.ArgumentParser(
        description="EmoArc Demo - Analyzes text/book and provides recommendations",
        epilog="For recommendations, main.py should be run first to generate trajectories.",
    )
    parser.add_argument("--book-id", type=str, help="Gutenberg book ID to analyze")
    parser.add_argument("--text-file", type=str, help="Path to text file to analyze")
    parser.add_argument(
        "--analyze", action="store_true", help="Analyze input and create visualizations"
    )
    parser.add_argument(
        "--recommend", action="store_true", help="Get recommendations based on input"
    )
    parser.add_argument(
        "--output-dir",
        default="output",
        help="Directory with output from main.py (default: output)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of books to consider for recommendations (optional)",
    )
    parser.add_argument(
        "--books-dir",
        default="data/books",
        help="Directory containing Gutenberg book files",
    )
    parser.add_argument(
        "--metadata", default="data/gutenberg_metadata.csv", help="Path to metadata CSV"
    )
    parser.add_argument(
        "--emotion-lexicon",
        default="data/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt",
        help="Path to NRC Emotion Lexicon",
    )
    parser.add_argument(
        "--vad-lexicon",
        default="data/NRC-VAD-Lexicon-v2.1.txt",
        help="Path to NRC VAD Lexicon",
    )

    args = parser.parse_args()

    # Validate input
    if not args.book_id and not args.text_file:
        print("❌ Error: Please specify either --book-id or --text-file")
        print("\nExample usage:")
        print("  python demo.py --book-id 11 --analyze")
        print("  python demo.py --text-file story.txt --analyze")
        print("  python demo.py --book-id 11 --recommend")
        print("  python demo.py --text-file story.txt --recommend")
        return

    if not args.analyze and not args.recommend:
        print("❌ Error: Please specify --analyze or --recommend (or both)")
        return

    spark = create_spark_session()

    try:
        if args.analyze:
            demo_analysis(
                spark,
                book_id=args.book_id,
                text_file=args.text_file,
                output_dir=args.output_dir,
                books_dir=args.books_dir,
                metadata_path=args.metadata,
                emotion_lexicon=args.emotion_lexicon,
                vad_lexicon=args.vad_lexicon,
            )

        if args.recommend:
            demo_recommendations(
                spark,
                book_id=args.book_id,
                text_file=args.text_file,
                output_dir=args.output_dir,
                limit=args.limit,
                books_dir=args.books_dir,
                metadata_path=args.metadata,
                emotion_lexicon=args.emotion_lexicon,
                vad_lexicon=args.vad_lexicon,
            )

    finally:
        spark.stop()


if __name__ == "__main__":
    main()
