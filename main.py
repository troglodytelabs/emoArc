"""
Main pipeline for emotion trajectory analysis.
"""

import sys
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col

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


def create_spark_session(app_name: str = "EmoArc"):
    """Create Spark session with appropriate configuration."""
    spark = (
        SparkSession.builder.appName(app_name)
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("WARN")
    return spark


def main():
    """Main pipeline execution."""
    import argparse

    parser = argparse.ArgumentParser(description="Emotion Trajectory Analysis Pipeline")
    parser.add_argument(
        "--books-dir", default="data/books", help="Directory containing book files"
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
    parser.add_argument(
        "--chunk-size", type=int, default=10000, help="Chunk size in characters"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of books to process (for testing)",
    )
    parser.add_argument(
        "--output", default="output", help="Output directory for results"
    )
    parser.add_argument("--language", default="en", help="Filter books by language")

    args = parser.parse_args()

    print("=" * 80)
    print("EmoArc - Emotion Trajectory Analysis Pipeline")
    print("=" * 80)

    # Create Spark session
    spark = create_spark_session()

    try:
        # Step 1: Load lexicons
        print("\n[Step 1/6] Loading lexicons...")
        emotion_df = load_emotion_lexicon(spark, args.emotion_lexicon)
        vad_df = load_vad_lexicon(spark, args.vad_lexicon)
        print(f"  ✓ Loaded {emotion_df.count()} emotion word-emotion pairs")
        print(f"  ✓ Loaded {vad_df.count()} VAD terms")

        # Step 2: Load books
        print("\n[Step 2/6] Loading books...")
        books_df = load_books(
            spark,
            args.books_dir,
            args.metadata,
            language=args.language,
            limit=args.limit,
        )
        print(f"  ✓ Loaded {books_df.count()} books")

        # Step 3: Create chunks
        print("\n[Step 3/6] Creating text chunks...")
        chunks_df = create_chunks_df(spark, books_df, chunk_size=args.chunk_size)
        print(f"  ✓ Created chunks (total rows: {chunks_df.count()})")

        # Step 4: Score chunks with emotions
        print("\n[Step 4/6] Scoring chunks with emotions...")
        emotion_scores = score_chunks_with_emotions(spark, chunks_df, emotion_df)
        print(f"  ✓ Scored {emotion_scores.count()} chunks with emotions")

        # Step 5: Score chunks with VAD
        print("\n[Step 5/6] Scoring chunks with VAD...")
        vad_scores = score_chunks_with_vad(spark, chunks_df, vad_df)
        print(f"  ✓ Scored {vad_scores.count()} chunks with VAD")

        # Combine scores
        chunk_scores = combine_emotion_vad_scores(emotion_scores, vad_scores)

        # Step 6: Analyze trajectories
        print("\n[Step 6/6] Analyzing emotion trajectories...")
        trajectories = analyze_trajectory(spark, chunk_scores)
        
        # Filter out books with no chunks (shouldn't happen, but safety check)
        trajectories = trajectories.filter(col("num_chunks") > 0)
        trajectory_count = trajectories.count()
        print(f"  ✓ Analyzed {trajectory_count} book trajectories")
        
        if args.limit and trajectory_count < args.limit:
            print(f"  ⚠ Warning: Expected {args.limit} books but got {trajectory_count}")
            print(f"  Some books may have had no chunks or processing errors.")

        # Save results
        print(f"\n[Saving] Writing results to {args.output}/...")
        os.makedirs(args.output, exist_ok=True)

        chunk_scores.coalesce(1).write.mode("overwrite").option("header", "true").csv(
            f"{args.output}/chunk_scores"
        )

        # Remove emotion_trajectory column (array type not supported by CSV)
        # It can be recomputed if needed, and we have all the stats we need
        trajectories_for_csv = trajectories.drop("emotion_trajectory")
        trajectories_for_csv.coalesce(1).write.mode("overwrite").option(
            "header", "true"
        ).csv(f"{args.output}/trajectories")

        print("  ✓ Results saved successfully!")

        # Show sample results
        print("\n" + "=" * 80)
        print("Sample Results:")
        print("=" * 80)
        print("\nTop 10 books by average joy:")
        trajectories.orderBy(col("avg_joy").desc()).select(
            "book_id", "title", "author", "avg_joy", "avg_sadness", "avg_valence"
        ).show(10, truncate=False)

        print("\n" + "=" * 80)
        print("Pipeline completed successfully!")
        print("=" * 80)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    finally:
        spark.stop()


if __name__ == "__main__":
    main()
