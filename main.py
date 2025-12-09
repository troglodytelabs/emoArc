"""
Main pipeline for emotion trajectory analysis.
"""

import os
import sys

from pyspark.sql import SparkSession
from pyspark.sql.functions import col

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from emotion_scorer import (
    combine_emotion_vad_scores,
    score_chunks_with_emotions,
    score_chunks_with_vad,
)
from lexicon_loader import load_emotion_lexicon, load_vad_lexicon
from text_preprocessor import create_chunks_df, load_books
from topic_modeling import (
    compute_book_topics,
    get_chunk_topics,
    prepare_topic_features,
    train_lda,
    train_per_book_lda,
)
from trajectory_analyzer import analyze_trajectory
from word_embeddings import (
    compute_book_embedding,
    compute_chunk_embeddings,
    train_word2vec,
)


def create_spark_session(app_name: str = "EmoArc"):
    """Create Spark session with appropriate configuration."""
    spark = (
        SparkSession.builder.appName(app_name)
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
        .config("spark.driver.memory", "4g")
        .config("spark.executor.memory", "4g")
        .config("spark.driver.maxResultSize", "2g")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("WARN")
    return spark

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
        "--chunk-size",
        type=int,
        default=10000,
        help="Chunk size in characters (ignored if --num-chunks is set)",
    )
    parser.add_argument(
        "--num-chunks",
        type=int,
        default=20,
        help="Number of chunks per book for percentage-based chunking (default: 20). Set to 0 to use --chunk-size instead",
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
    spark = create_spark_session()

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
        # use percentage-based chunking if num_chunks > 0, otherwise use fixed size
        if args.num_chunks > 0:
            print(
                f"  Using percentage-based chunking: {args.num_chunks} chunks per book"
            )
            chunks_df = create_chunks_df(spark, books_df, num_chunks=args.num_chunks)
        else:
            print(
                f"  Using fixed-size chunking: {args.chunk_size} characters per chunk"
            )
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

        # Unpersist chunks_df now that we have scores (saves memory)
        chunks_df.unpersist()

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
            # Reload chunks_df for embeddings (we unpersisted it earlier)
            print("  Recreating chunks for embeddings...")
            chunks_df_emb = create_chunks_df(
                spark,
                books_df,
                num_chunks=args.num_chunks,
                max_chunk_size=args.max_chunk_size,
            )
            print("  Training Word2Vec model...")
            word2vec_model = train_word2vec(
                spark, chunks_df_emb, vector_size=args.vector_size, min_count=5
            )
            print("  ✓ Word2Vec model trained")

            print("  Computing chunk embeddings...")
            chunk_embeddings = compute_chunk_embeddings(
                spark, chunks_df_emb, word2vec_model
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
            # Unpersist chunks_df_emb
            chunks_df_emb.unpersist()
        else:
            print("\n[Step 7/8] Skipping word embeddings (--skip-embeddings)")

        # Step 8: Compute per-book topic models
        book_topics = None
        if not args.skip_topics:
            print("\n[Step 8/8] Computing per-book topic models...")
            print(
                f"  Training separate LDA model for each book ({args.num_topics} topics per book)..."
            )

            # train per-book LDA (each book gets its own topic model)
            book_topics = train_per_book_lda(
                spark,
                chunks_df,
                num_topics=min(args.num_topics, 5),  # cap at 5 topics per book
                vocab_size=1000,
                min_df=1,
                max_iter=20,
            )

            topic_count = book_topics.count()
            print(f"  ✓ Computed per-book topics for {topic_count} books")
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

        # calculate emotional volatility from trajectory before dropping it
        # volatility = standard deviation of total emotional intensity across chunks
        if "emotion_trajectory" in trajectories.columns:
            print("  Calculating emotional volatility...")
            from pyspark.sql.functions import explode, udf
            from pyspark.sql.types import (
                ArrayType,
                FloatType,
                StringType,
                StructField,
                StructType,
            )

            # udf to calculate volatility from emotion trajectory array
            def calculate_volatility(trajectory):
                """
                calculate emotional volatility as std dev of total intensity across chunks
                trajectory format: [[chunk_idx, anger, anticipation, disgust, fear, joy, sadness, surprise, trust], ...]
                """
                if not trajectory or len(trajectory) == 0:
                    return 0.0
                try:
                    # sum all emotions for each chunk (indices 1-8)
                    intensities = [sum(chunk[1:9]) for chunk in trajectory]
                    if len(intensities) < 2:
                        return 0.0
                    # calculate standard deviation
                    mean_intensity = sum(intensities) / len(intensities)
                    variance = sum(
                        (x - mean_intensity) ** 2 for x in intensities
                    ) / len(intensities)
                    return float(variance**0.5)
                except Exception:
                    return 0.0

            volatility_udf = udf(calculate_volatility, FloatType())
            trajectories = trajectories.withColumn(
                "emotional_volatility", volatility_udf(col("emotion_trajectory"))
            )
            print("  ✓ Emotional volatility calculated")

            # convert emotion_trajectory to JSON string for CSV storage
            # this preserves the data so django can load and display arc charts
            print("  Converting emotion trajectories to JSON...")

            def trajectory_to_json(trajectory):
                """
                convert trajectory array to json string, then base64 encode to avoid csv issues
                format: base64([{"anger": 12.3, "joy": 15.4, ...}, {"anger": 11.2, ...}, ...])
                """
                if not trajectory or len(trajectory) == 0:
                    return ""
                try:
                    import base64
                    import json

                    emotion_names = [
                        "anger",
                        "anticipation",
                        "disgust",
                        "fear",
                        "joy",
                        "sadness",
                        "surprise",
                        "trust",
                    ]
                    chunks = []
                    for chunk_data in trajectory:
                        # chunk_data = [chunk_idx, anger, anticipation, ..., trust]
                        chunk_dict = {}
                        for i, emotion in enumerate(emotion_names):
                            chunk_dict[emotion] = float(
                                chunk_data[i + 1]
                            )  # +1 to skip chunk_idx
                        chunks.append(chunk_dict)
                    json_str = json.dumps(chunks)
                    # base64 encode to avoid CSV quoting/escaping issues
                    return base64.b64encode(json_str.encode("utf-8")).decode("utf-8")
                except Exception:
                    return ""

            trajectory_json_udf = udf(trajectory_to_json, StringType())
            trajectories = trajectories.withColumn(
                "emotion_trajectory_json",
                trajectory_json_udf(col("emotion_trajectory")),
            )
            print("  ✓ Emotion trajectories converted to JSON")

        # flatten per-book topics to CSV-compatible columns
        if book_topics is not None and "book_topics" in trajectories.columns:
            print("  Flattening per-book topics for CSV export...")
            from pyspark.sql.functions import udf
            from pyspark.sql.types import FloatType, IntegerType, StringType

            # extract top 3 topics from per-book topic arrays
            # book_topics format: [[words for topic 0], [words for topic 1], ...]
            def get_topic_words(topics, rank=0):
                """get comma-separated words for nth topic"""
                if not topics or len(topics) == 0:
                    return ""
                if rank >= len(topics):
                    return ""
                words = topics[rank]
                return ",".join(words[:5]) if words else ""  # top 5 words per topic

            # for per-book LDA, we assign equal probability (1/num_topics) to each topic
            # or we can use a simple heuristic based on topic position (earlier = more important)
            def get_topic_prob(topics, rank=0):
                """assign probability based on topic rank (exponential decay)"""
                if not topics or len(topics) == 0:
                    return 0.0
                if rank >= len(topics):
                    return 0.0
                # exponential decay: first topic gets highest probability
                # normalize so all probs sum to ~1.0
                num_topics = len(topics)
                decay_factor = 0.6  # each subsequent topic gets 60% of previous
                weights = [decay_factor**i for i in range(num_topics)]
                total_weight = sum(weights)
                normalized_prob = (
                    weights[rank] / total_weight
                    if total_weight > 0
                    else 1.0 / num_topics
                )
                return float(normalized_prob)

            # register udfs for top 3 topics
            topic_1_words_udf = udf(lambda t: get_topic_words(t, 0), StringType())
            topic_1_prob_udf = udf(lambda t: get_topic_prob(t, 0), FloatType())

            topic_2_words_udf = udf(lambda t: get_topic_words(t, 1), StringType())
            topic_2_prob_udf = udf(lambda t: get_topic_prob(t, 1), FloatType())

            topic_3_words_udf = udf(lambda t: get_topic_words(t, 2), StringType())
            topic_3_prob_udf = udf(lambda t: get_topic_prob(t, 2), FloatType())

            # add flattened topic columns
            # topic_id is just the rank (0, 1, 2) since these are per-book topics
            trajectories = (
                trajectories.withColumn(
                    "top_topic_1",
                    udf(lambda t: 0 if t and len(t) > 0 else -1, IntegerType())(
                        col("book_topics")
                    ),
                )
                .withColumn("top_topic_1_prob", topic_1_prob_udf(col("book_topics")))
                .withColumn("top_topic_1_words", topic_1_words_udf(col("book_topics")))
                .withColumn(
                    "top_topic_2",
                    udf(lambda t: 1 if t and len(t) > 1 else -1, IntegerType())(
                        col("book_topics")
                    ),
                )
                .withColumn("top_topic_2_prob", topic_2_prob_udf(col("book_topics")))
                .withColumn("top_topic_2_words", topic_2_words_udf(col("book_topics")))
                .withColumn(
                    "top_topic_3",
                    udf(lambda t: 2 if t and len(t) > 2 else -1, IntegerType())(
                        col("book_topics")
                    ),
                )
                .withColumn("top_topic_3_prob", topic_3_prob_udf(col("book_topics")))
                .withColumn("top_topic_3_words", topic_3_words_udf(col("book_topics")))
            )
            print("  ✓ Per-book topics flattened with word labels")

        # Save results
        print(f"\n[Saving] Writing results to {args.output}/...")
        os.makedirs(args.output, exist_ok=True)

        # Save chunk scores (text columns already dropped during preprocessing)
        chunk_scores.repartition(4).write.mode("overwrite").option(
            "header", "true"
        ).csv(f"{args.output}/chunk_scores")
        print("  ✓ Chunk scores saved")

        # Remove array columns and text from trajectories (not supported by CSV)
        columns_to_drop = ["emotion_trajectory"]
        if "book_embedding" in trajectories.columns:
            columns_to_drop.append("book_embedding")
        if "book_topics" in trajectories.columns:
            columns_to_drop.append(
                "book_topics"
            )  # drop original array, keep flattened columns
        if "text" in trajectories.columns:
            columns_to_drop.append("text")

        trajectories_for_csv = trajectories.drop(*columns_to_drop)
        trajectories_for_csv.coalesce(1).write.mode("overwrite").option(
            "header", "true"
        ).csv(f"{args.output}/trajectories")
        print("  ✓ Trajectories saved")

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
