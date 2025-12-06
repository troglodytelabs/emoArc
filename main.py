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
        "--chunk-size", type=int, default=10000, help="Chunk size in characters (ignored if --num-chunks is set)"
    )
    parser.add_argument(
        "--num-chunks",
        type=int,
        default=20,
        help="Number of chunks per book for percentage-based chunking (default: 20). Set to 0 to use --chunk-size instead"
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
        "--skip-embeddings",
        action="store_true",
        help="Skip word embeddings computation",
    )
    parser.add_argument(
        "--skip-topics",
        action="store_true",
        help="Skip topic modeling",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("EmoArc - Emotion Trajectory Analysis Pipeline")
    print("=" * 80)

    # Create Spark session
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
            print(f"  Using percentage-based chunking: {args.num_chunks} chunks per book")
            chunks_df = create_chunks_df(spark, books_df, num_chunks=args.num_chunks)
        else:
            print(f"  Using fixed-size chunking: {args.chunk_size} characters per chunk")
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

            print("  Computing book-level embeddings (memory-efficient aggregation)...")
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
            lda_model = train_lda(spark, feature_df, num_topics=args.num_topics, max_iter=50)
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

        # flatten topic distributions to CSV-compatible columns
        if book_topics is not None and "book_topics" in trajectories.columns:
            print("  Flattening topic distributions for CSV export...")
            from pyspark.sql.functions import udf
            from pyspark.sql.types import FloatType, IntegerType

            # extract top 3 topics and their probabilities
            # topic distributions are arrays like [0.05, 0.32, 0.11, 0.08, 0.24, ...]
            def get_top_topic_idx(topics, rank=0):
                """get index of nth highest topic probability"""
                if not topics or len(topics) == 0:
                    return -1
                # sort indices by probability descending
                sorted_indices = sorted(range(len(topics)), key=lambda i: topics[i], reverse=True)
                return int(sorted_indices[rank]) if rank < len(sorted_indices) else -1

            def get_top_topic_prob(topics, rank=0):
                """get nth highest topic probability"""
                if not topics or len(topics) == 0:
                    return 0.0
                sorted_probs = sorted(topics, reverse=True)
                return float(sorted_probs[rank]) if rank < len(sorted_probs) else 0.0

            # register udfs
            top_idx_udf = udf(lambda t: get_top_topic_idx(t, 0), IntegerType())
            top_prob_udf = udf(lambda t: get_top_topic_prob(t, 0), FloatType())
            second_idx_udf = udf(lambda t: get_top_topic_idx(t, 1), IntegerType())
            second_prob_udf = udf(lambda t: get_top_topic_prob(t, 1), FloatType())
            third_idx_udf = udf(lambda t: get_top_topic_idx(t, 2), IntegerType())
            third_prob_udf = udf(lambda t: get_top_topic_prob(t, 2), FloatType())

            # add flattened topic columns
            trajectories = trajectories.withColumn("top_topic_1", top_idx_udf(col("book_topics"))) \
                                     .withColumn("top_topic_1_prob", top_prob_udf(col("book_topics"))) \
                                     .withColumn("top_topic_2", second_idx_udf(col("book_topics"))) \
                                     .withColumn("top_topic_2_prob", second_prob_udf(col("book_topics"))) \
                                     .withColumn("top_topic_3", third_idx_udf(col("book_topics"))) \
                                     .withColumn("top_topic_3_prob", third_prob_udf(col("book_topics")))
            print("  ✓ Topic distributions flattened")

        # Save results
        print(f"\n[Saving] Writing results to {args.output}/...")
        os.makedirs(args.output, exist_ok=True)

        # Save chunk scores (text columns already dropped during preprocessing)
        chunk_scores.repartition(4).write.mode("overwrite").option("header", "true").csv(
            f"{args.output}/chunk_scores"
        )
        print("  ✓ Chunk scores saved")

        # Remove array columns and text from trajectories (not supported by CSV)
        columns_to_drop = ["emotion_trajectory"]
        if "book_embedding" in trajectories.columns:
            columns_to_drop.append("book_embedding")
        if "book_topics" in trajectories.columns:
            columns_to_drop.append("book_topics")  # drop original array, keep flattened columns
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
