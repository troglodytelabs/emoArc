"""
Main pipeline for emotion trajectory analysis.
"""

import sys
import os
import time
import json
from datetime import datetime
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark import StorageLevel

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from lexicon_loader import load_emotion_lexicon, load_vad_lexicon
from text_preprocessor import load_and_chunk_books
from emotion_scorer import score_chunks
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
from model_persistence import save_models, load_models


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
        driver_memory = os.environ.get("SPARK_DRIVER_MEMORY", "8g")
        executor_memory = os.environ.get("SPARK_EXECUTOR_MEMORY", "8g")
        max_result_size = os.environ.get("SPARK_MAX_RESULT_SIZE", "4g")
        shuffle_partitions = int(os.environ.get("SPARK_SHUFFLE_PARTITIONS", "200"))
        # Limit parallelism in local mode to prevent OOM when loading large files
        # Each task can load a multi-MB book file, so fewer parallel tasks = less memory
        local_parallelism = int(os.environ.get("SPARK_LOCAL_PARALLELISM", "4"))
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
        local_parallelism = None  # Not applicable for cluster mode
        # Arrow can be enabled on clusters (better performance, more memory available)
        arrow_enabled = os.environ.get("SPARK_ARROW_ENABLED", "true").lower() == "true"

    # Build Spark session
    builder = SparkSession.builder.appName(app_name)

    # Set master based on mode
    if is_local and local_parallelism:
        # Local mode: limit parallelism to prevent OOM when loading large book files
        builder = builder.master(f"local[{local_parallelism}]")
    elif not is_local:
        # Cluster mode: use YARN (EMR default)
        # If running via spark-submit, this may be overridden by spark-submit args
        builder = builder.master("yarn")

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


def process_batch(
    spark, batch_ids, batch_index, args, emotion_df, vad_df, models, timings
):
    """
    Process a single batch of books.

    Args:
        spark: SparkSession
        batch_ids: List of book IDs to process
        batch_index: Index of the current batch (0-based)
        args: Command line arguments
        emotion_df: Loaded emotion lexicon
        vad_df: Loaded VAD lexicon
        models: Dictionary of existing models (word2vec, lda, cv_model) or None
        timings: Dictionary to update with timing info

    Returns:
        Updated models dictionary
    """
    batch_start = time.time()

    # Step 1: Load and chunk books
    print(f"\n[Batch {batch_index + 1}] Loading and chunking {len(batch_ids)} books...")
    stage_start = time.time()
    chunks_df = load_and_chunk_books(
        spark,
        args.books_dir,
        args.metadata,
        num_chunks=args.num_chunks,
        language=args.language,
        book_ids=batch_ids,
    )
    chunks_df.persist(StorageLevel.MEMORY_AND_DISK)
    chunk_count = chunks_df.count()

    if chunk_count == 0:
        print(f"  ⚠ Batch {batch_index + 1} has no chunks. Skipping.")
        chunks_df.unpersist()
        return models

    timings[f"batch_{batch_index}_load"] = time.time() - stage_start
    print(f"  ✓ Created {chunk_count} chunks")

    # Step 2: Score chunks
    print(f"\n[Batch {batch_index + 1}] Scoring chunks...")
    stage_start = time.time()
    chunk_scores = score_chunks(spark, chunks_df, emotion_df, vad_df)

    # Step 3: Analyze trajectories
    print(f"\n[Batch {batch_index + 1}] Analyzing trajectories...")
    trajectories = analyze_trajectory(spark, chunk_scores)
    trajectories = trajectories.filter(col("num_chunks") > 0)
    timings[f"batch_{batch_index}_score"] = time.time() - stage_start

    # Step 4: Word Embeddings
    book_embeddings = None
    if not args.skip_embeddings:
        print(f"\n[Batch {batch_index + 1}] Processing embeddings...")
        stage_start = time.time()

        if models.get("word2vec") is None:
            print("  Training Word2Vec model (on this batch)...")
            models["word2vec"] = train_word2vec(
                spark, chunks_df, vector_size=args.vector_size, min_count=5
            )
            models["_word2vec_trained_this_run"] = True

        chunk_embeddings = compute_chunk_embeddings(
            spark, chunks_df, models["word2vec"]
        )
        book_embeddings = compute_book_embedding(spark, chunk_embeddings)
        timings[f"batch_{batch_index}_embeddings"] = time.time() - stage_start

    # Step 5: Topic Modeling
    book_topics = None
    if not args.skip_topics:
        print(f"\n[Batch {batch_index + 1}] Processing topics...")
        stage_start = time.time()

        if models.get("lda") is None:
            print("  Preparing topic features and training LDA (on this batch)...")
            feature_df, cv_model = prepare_topic_features(
                spark, chunks_df, vocab_size=5000, min_df=2
            )
            models["cv_model"] = cv_model

            lda_model = train_lda(
                spark, feature_df, num_topics=args.num_topics, max_iter=50
            )
            models["lda"] = lda_model
            models["_lda_trained_this_run"] = True
        else:
            # Transform using existing CV model
            word_sequences = chunks_df.select("book_id", "chunk_index", col("words"))
            feature_df = models["cv_model"].transform(word_sequences)

        chunk_topics = get_chunk_topics(spark, feature_df, models["lda"])
        book_topics = compute_book_topics(spark, chunk_topics)
        timings[f"batch_{batch_index}_topics"] = time.time() - stage_start

    # Cleanup chunks
    chunks_df.unpersist()
    # chunk_scores is not persisted, so no need to unpersist

    # Join results
    if book_embeddings is not None:
        trajectories = trajectories.join(book_embeddings, on="book_id", how="left")
    if book_topics is not None:
        trajectories = trajectories.join(book_topics, on="book_id", how="left")

    # Save results
    print(f"\n[Batch {batch_index + 1}] Saving results...")
    stage_start = time.time()
    mode = "overwrite" if batch_index == 0 else "append"
    trajectories.write.mode(mode).parquet(f"{args.output}/trajectories")
    timings[f"batch_{batch_index}_save"] = time.time() - stage_start

    print(f"  ✓ Batch {batch_index + 1} complete in {time.time() - batch_start:.2f}s")
    return models


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
        "--num-chunks",
        type=int,
        default=20,
        help="Fixed number of chunks per book (5%% each with default 20)",
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
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Process books in batches of this size to avoid OOM",
    )
    parser.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="Start fresh, overwriting existing data (default: resume if data exists)",
    )
    parser.set_defaults(resume=True)

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

    # Track timing for each stage
    timings = {}
    pipeline_start = time.time()

    try:
        # Step 1: Load lexicons (small, safe to cache)
        print("\n[Step 1/6] Loading lexicons...")
        stage_start = time.time()
        emotion_df = load_emotion_lexicon(spark, args.emotion_lexicon)
        vad_df = load_vad_lexicon(spark, args.vad_lexicon)
        # Cache lexicons - they're small and reused for every chunk
        emotion_df.cache()
        vad_df.cache()
        emotion_count = emotion_df.count()
        vad_count = vad_df.count()
        timings["1_lexicon_loading"] = time.time() - stage_start
        print(f"  ✓ Loaded {emotion_count} emotion word-emotion pairs")
        print(f"  ✓ Loaded {vad_count} VAD terms")
        print(f"  ⏱ Time: {timings['1_lexicon_loading']:.2f}s")

        # Determine books to process
        print("\n[Step 2/6] Determining books to process...")
        metadata_df = spark.read.option("header", "true").csv(args.metadata)
        if args.language:
            metadata_df = metadata_df.filter(col("Language") == args.language)

        metadata_df = metadata_df.filter(
            col("Etext Number").isNotNull() & (col("Etext Number") != "")
        )

        if args.limit:
            metadata_df = metadata_df.limit(args.limit)

        all_book_ids = [
            row["Etext Number"] for row in metadata_df.select("Etext Number").collect()
        ]

        # Shuffle book IDs to ensure the first batch (used for training) is representative
        import random

        random.seed(42)  # For reproducibility
        random.shuffle(all_book_ids)

        print(f"  ✓ Found {len(all_book_ids)} books to process (shuffled)")

        # Check for resume capability
        batch_size = args.batch_size if args.batch_size else len(all_book_ids)
        start_batch = 0

        if args.resume:
            # Check which batches already exist
            trajectories_path = f"{args.output}/trajectories"
            if os.path.exists(trajectories_path):
                try:
                    existing_df = spark.read.parquet(trajectories_path)
                    existing_ids = set(
                        row["book_id"]
                        for row in existing_df.select("book_id").collect()
                    )

                    # Find which batches are complete
                    for i in range(0, len(all_book_ids), batch_size):
                        batch_ids = set(all_book_ids[i : i + batch_size])
                        if batch_ids.issubset(existing_ids):
                            start_batch = (i // batch_size) + 1
                        else:
                            break

                    if start_batch > 0:
                        print(
                            f"  ✓ Resuming from batch {start_batch + 1} ({start_batch * batch_size} books already processed)"
                        )
                except Exception as e:
                    print(f"  ⚠ Could not check existing data: {e}. Starting fresh.")
                    start_batch = 0

        # Process in batches
        models = {}

        # Load existing models if resuming
        if args.resume and start_batch > 0:
            print("\n[Resume] Loading previously trained models...")
            models = load_models(args.output)
            if not models:
                print(
                    "  ⚠ No saved models found. Models will be retrained on first batch."
                )

        for i in range(start_batch * batch_size, len(all_book_ids), batch_size):
            batch_ids = all_book_ids[i : i + batch_size]
            batch_index = i // batch_size

            models = process_batch(
                spark, batch_ids, batch_index, args, emotion_df, vad_df, models, timings
            )

            # Save models after first batch (when they're trained)
            if models.get("_word2vec_trained_this_run") or models.get(
                "_lda_trained_this_run"
            ):
                print("\n[Models] Saving trained models for resume capability...")
                save_models(models, args.output)
                # Clear the flags so we don't save again
                models.pop("_word2vec_trained_this_run", None)
                models.pop("_lda_trained_this_run", None)

            # Save timings after each batch (in case of crash)
            timings["last_completed_batch"] = batch_index
            timings["elapsed"] = time.time() - pipeline_start
            timings_file = os.path.join(args.output, "timings.json")
            with open(timings_file, "w") as f:
                json.dump(
                    {
                        "timestamp": datetime.now().isoformat(),
                        "stages": {
                            k: round(v, 2) if isinstance(v, float) else v
                            for k, v in timings.items()
                        },
                    },
                    f,
                    indent=2,
                )

        # Calculate total time
        timings["total"] = time.time() - pipeline_start

        # # Show sample results (read from disk since we processed in batches)
        # print("\n" + "=" * 80)
        # print("Sample Results:")
        # print("=" * 80)
        # try:
        #     trajectories = spark.read.parquet(f"{args.output}/trajectories")
        #     print("\nTop 10 books by average joy:")
        #     trajectories.orderBy(col("avg_joy").desc()).select(
        #         "book_id", "title", "author", "avg_joy", "avg_sadness", "avg_valence"
        #     ).show(10, truncate=False)
        # except Exception as e:
        #     print(f"Could not read results for sample display: {e}")

        # Print timing summary
        print("\n" + "=" * 80)
        print("TIMING SUMMARY")
        print("=" * 80)
        for stage, t in sorted(timings.items()):
            if stage != "total":
                pct = t / timings["total"] * 100
                print(f"  {stage:<35} {t:>8.2f}s  ({pct:>5.1f}%)")
        print("  " + "-" * 50)
        print(f"  {'TOTAL':<35} {timings['total']:>8.2f}s")

        # Save timings to JSON
        timings_data = {
            "timestamp": datetime.now().isoformat(),
            "stages": {k: round(v, 2) for k, v in timings.items()},
            "percentages": {
                k: round(v / timings["total"] * 100, 1)
                for k, v in timings.items()
                if k != "total"
            },
        }
        timings_file = os.path.join(args.output, "timings.json")
        with open(timings_file, "w") as f:
            json.dump(timings_data, f, indent=2)
        print(f"\n  ✓ Timings saved to {timings_file}")

        print("\n" + "=" * 80)
        print("Pipeline completed successfully!")
        print("Run 'python evaluate.py' for evaluation metrics.")
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
