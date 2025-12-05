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
from word_embeddings import (
    train_word2vec,
    compute_chunk_embeddings,
    compute_book_embedding,
)
from topic_modeling import (
    prepare_topic_features,
    train_lda,
    get_chunk_topics,
    compute_book_topics,
)


def create_spark_session(app_name: str = "EmoArc", mode: str = "local"):
    """
    Create Spark session with appropriate configuration.

    Args:
        app_name: Application name
        mode: Execution mode - "local" or "cluster"
            - "local": Higher memory settings, more partitions, Arrow disabled
            - "cluster": Uses cluster defaults, adaptive partitions

    Returns:
        Configured SparkSession
    """
    import os

    mode = mode.lower()
    if mode not in ["local", "cluster"]:
        raise ValueError(f"Mode must be 'local' or 'cluster', got '{mode}'")

    is_local = mode == "local"

    # Get memory settings from environment or use adaptive defaults
    if is_local:
        # Local mode: Need explicit memory settings (Spark defaults are too low)
        driver_memory = os.environ.get("SPARK_DRIVER_MEMORY", "4g")
        executor_memory = os.environ.get("SPARK_EXECUTOR_MEMORY", "4g")
        max_result_size = os.environ.get("SPARK_MAX_RESULT_SIZE", "2g")
        shuffle_partitions = int(os.environ.get("SPARK_SHUFFLE_PARTITIONS", "200"))
        # Disable Arrow in local mode to save memory
        arrow_enabled = os.environ.get("SPARK_ARROW_ENABLED", "false").lower() == "true"
    else:
        # Cluster mode: Let Spark/EMR use defaults or cluster settings
        # Only override if explicitly set
        driver_memory = os.environ.get("SPARK_DRIVER_MEMORY", None)
        executor_memory = os.environ.get("SPARK_EXECUTOR_MEMORY", None)
        max_result_size = os.environ.get("SPARK_MAX_RESULT_SIZE", None)
        # For clusters, partitions should be based on cluster size (1-2x vCores)
        # Let Spark adaptive execution handle it, or set based on cluster
        shuffle_partitions = int(os.environ.get("SPARK_SHUFFLE_PARTITIONS", "200"))
        # Arrow can be enabled on clusters (better performance, more memory available)
        arrow_enabled = os.environ.get("SPARK_ARROW_ENABLED", "true").lower() == "true"

    # Build Spark session
    builder = SparkSession.builder.appName(app_name)

    # Always enable adaptive execution (works in both local and cluster)
    builder = builder.config("spark.sql.adaptive.enabled", "true")
    builder = builder.config("spark.sql.adaptive.coalescePartitions.enabled", "true")

    # Memory settings (only set if provided or in local mode)
    if driver_memory:
        builder = builder.config("spark.driver.memory", driver_memory)
    if executor_memory:
        builder = builder.config("spark.executor.memory", executor_memory)
    if max_result_size:
        builder = builder.config("spark.driver.maxResultSize", max_result_size)

    # Arrow: Enable on clusters, disable on local for memory savings
    builder = builder.config(
        "spark.sql.execution.arrow.pyspark.enabled", str(arrow_enabled).lower()
    )

    # Shuffle partitions: Set explicitly (adaptive execution will optimize)
    # For clusters, this should ideally be 1-2x number of vCores
    builder = builder.config("spark.sql.shuffle.partitions", str(shuffle_partitions))

    # Memory fractions: Only set in local mode (clusters usually have better defaults)
    # These are Spark defaults: fraction=0.6, storageFraction=0.5
    # We adjust for local mode to use more memory
    if is_local:
        builder = builder.config("spark.memory.fraction", "0.8")  # Use 80% of heap
        builder = builder.config(
            "spark.memory.storageFraction", "0.3"
        )  # 30% storage, 70% execution

    # Create session
    spark = builder.getOrCreate()

    # Log environment info
    actual_master = spark.sparkContext.master
    if is_local:
        print(f"  Running in LOCAL mode (master: {actual_master})")
        print(f"  Memory settings: driver={driver_memory}, executor={executor_memory}")
    else:
        print(f"  Running in CLUSTER mode (master: {actual_master})")
        print("  Using cluster defaults (override via environment variables if needed)")

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
    parser.add_argument(
        "--vector-size",
        type=int,
        default=100,
        help="Word2Vec vector size (default: 100)",
    )
    parser.add_argument(
        "--num-topics",
        type=int,
        default=10,
        help="Number of LDA topics (default: 10)",
    )
    parser.add_argument(
        "--driver-memory",
        type=str,
        default=None,
        help="Spark driver memory (default: auto-detect based on environment)",
    )
    parser.add_argument(
        "--executor-memory",
        type=str,
        default=None,
        help="Spark executor memory (default: auto-detect based on environment)",
    )
    parser.add_argument(
        "--skip-embeddings",
        action="store_true",
        help="Skip word embeddings computation",
    )
    parser.add_argument(
        "--skip-topics",
        action="store_true",
        help="Skip topic modeling",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["local", "cluster"],
        default="local",
        help="Spark execution mode: 'local' (default) or 'cluster' (for EMR/YARN)",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("EmoArc - Emotion Trajectory Analysis Pipeline")
    print("=" * 80)

    # Set Spark memory from arguments (only if provided)
    import os

    if args.driver_memory:
        os.environ["SPARK_DRIVER_MEMORY"] = args.driver_memory
    if args.executor_memory:
        os.environ["SPARK_EXECUTOR_MEMORY"] = args.executor_memory

    # Create Spark session with specified mode
    spark = create_spark_session(mode=args.mode)

    try:
        # Step 1: Load lexicons
        print("\n[Step 1/8] Loading lexicons...")
        emotion_df = load_emotion_lexicon(spark, args.emotion_lexicon)
        vad_df = load_vad_lexicon(spark, args.vad_lexicon)
        print(f"  ✓ Loaded {emotion_df.count()} emotion word-emotion pairs")
        print(f"  ✓ Loaded {vad_df.count()} VAD terms")

        # Step 2: Load books
        print("\n[Step 2/8] Loading books...")
        books_df = load_books(
            spark,
            args.books_dir,
            args.metadata,
            language=args.language,
            limit=args.limit,
        )
        print(f"  ✓ Loaded {books_df.count()} books")

        # Step 3: Create chunks
        print("\n[Step 3/8] Creating text chunks...")
        chunks_df = create_chunks_df(spark, books_df, chunk_size=args.chunk_size)
        print(f"  ✓ Created chunks (total rows: {chunks_df.count()})")

        # Step 4: Score chunks with emotions
        print("\n[Step 4/8] Scoring chunks with emotions...")
        emotion_scores = score_chunks_with_emotions(spark, chunks_df, emotion_df)
        print(f"  ✓ Scored {emotion_scores.count()} chunks with emotions")

        # Step 5: Score chunks with VAD
        print("\n[Step 5/8] Scoring chunks with VAD...")
        vad_scores = score_chunks_with_vad(spark, chunks_df, vad_df)
        print(f"  ✓ Scored {vad_scores.count()} chunks with VAD")

        # Combine scores
        chunk_scores = combine_emotion_vad_scores(emotion_scores, vad_scores)

        # Step 6: Analyze trajectories
        print("\n[Step 6/8] Analyzing emotion trajectories...")
        trajectories = analyze_trajectory(spark, chunk_scores)

        # Filter out books with no chunks (shouldn't happen, but safety check)
        trajectories = trajectories.filter(col("num_chunks") > 0)
        trajectory_count = trajectories.count()
        print(f"  ✓ Analyzed {trajectory_count} book trajectories")

        # Step 7: Compute word embeddings
        book_embeddings = None
        if not args.skip_embeddings:
            print("\n[Step 7/8] Computing word embeddings...")
            print("  Training Word2Vec model...")
            word2vec_model = train_word2vec(
                spark, chunks_df, vector_size=args.vector_size, min_count=5
            )
            print("  ✓ Word2Vec model trained")

            print("  Computing chunk embeddings...")
            chunk_embeddings = compute_chunk_embeddings(
                spark, chunks_df, word2vec_model
            )
            # Cache to avoid recomputation
            chunk_embeddings.cache()
            chunk_count = chunk_embeddings.count()
            print(f"  ✓ Computed embeddings for {chunk_count} chunks")

            print("  Computing book-level embeddings...")
            book_embeddings = compute_book_embedding(spark, chunk_embeddings)
            book_count = book_embeddings.count()
            print(f"  ✓ Computed embeddings for {book_count} books")

            # Unpersist cached data
            chunk_embeddings.unpersist()
        else:
            print("\n[Step 7/8] Skipping word embeddings (--skip-embeddings)")

        # Step 8: Compute topic distributions
        book_topics = None
        if not args.skip_topics:
            print("\n[Step 8/8] Computing topic distributions...")
            print("  Preparing topic features...")
            feature_df, cv_model = prepare_topic_features(
                spark, chunks_df, vocab_size=5000, min_df=2
            )
            # Cache feature_df to avoid recomputation
            feature_df.cache()
            print("  ✓ Features prepared")

            print(f"  Training LDA model with {args.num_topics} topics...")
            lda_model = train_lda(
                spark, feature_df, num_topics=args.num_topics, max_iter=50
            )
            print("  ✓ LDA model trained")

            print("  Computing chunk topics...")
            chunk_topics = get_chunk_topics(spark, feature_df, lda_model)
            chunk_topics.cache()
            chunk_topic_count = chunk_topics.count()
            print(f"  ✓ Computed topics for {chunk_topic_count} chunks")

            print("  Computing book-level topics...")
            book_topics = compute_book_topics(spark, chunk_topics)
            topic_count = book_topics.count()
            print(f"  ✓ Computed topics for {topic_count} books")

            # Unpersist cached data
            feature_df.unpersist()
            chunk_topics.unpersist()
        else:
            print("\n[Step 8/8] Skipping topic modeling (--skip-topics)")

        # Join embeddings and topics with trajectories
        if book_embeddings is not None:
            trajectories = trajectories.join(book_embeddings, on="book_id", how="left")
            print("  ✓ Joined embeddings with trajectories")

        if book_topics is not None:
            trajectories = trajectories.join(book_topics, on="book_id", how="left")
            print("  ✓ Joined topics with trajectories")

        if args.limit and trajectory_count < args.limit:
            print(
                f"  ⚠ Warning: Expected {args.limit} books but got {trajectory_count}"
            )
            print("  Some books may have had no chunks or processing errors.")

        # Save results
        print(f"\n[Saving] Writing results to {args.output}/...")
        os.makedirs(args.output, exist_ok=True)

        chunk_scores.coalesce(1).write.mode("overwrite").option("header", "true").csv(
            f"{args.output}/chunk_scores"
        )

        # Remove array columns (not supported by CSV)
        # These can be recomputed if needed
        columns_to_drop = ["emotion_trajectory"]
        if "book_embedding" in trajectories.columns:
            columns_to_drop.append("book_embedding")
        if "book_topics" in trajectories.columns:
            columns_to_drop.append("book_topics")

        trajectories_for_csv = trajectories.drop(*columns_to_drop)
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
