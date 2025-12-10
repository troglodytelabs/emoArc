"""
Analyze emotion trajectories: peaks, dominant emotions, patterns.

Computes per-book emotion ratios for comparable similarity metrics.
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col,
    avg,
    array,
    collect_list,
    sort_array,
    count,
    when,
)


def analyze_trajectory(spark: SparkSession, chunk_scores_df):
    """
    Analyze emotion trajectory for each book.

    Computes:
    - avg_* (8 emotions): Average emotion scores per book
    - avg_valence/arousal/dominance: VAD averages
    - ratio_* (8 emotions): Emotion proportions (sum to 1) for cross-book comparison
    - emotion_trajectory: Array of chunk-level emotions for trajectory similarity
    - num_chunks: Number of chunks (informational)

    Args:
        spark: SparkSession
        chunk_scores_df: DataFrame with emotion and VAD scores per chunk

    Returns:
        DataFrame with trajectory analysis per book
    """
    emotions = [
        "anger",
        "anticipation",
        "disgust",
        "fear",
        "joy",
        "sadness",
        "surprise",
        "trust",
    ]

    # Calculate trajectory statistics per book
    book_trajectories = chunk_scores_df.groupBy("book_id", "title", "author").agg(
        # Average emotions (used in fallback similarity and output display)
        avg("anger").alias("avg_anger"),
        avg("anticipation").alias("avg_anticipation"),
        avg("disgust").alias("avg_disgust"),
        avg("fear").alias("avg_fear"),
        avg("joy").alias("avg_joy"),
        avg("sadness").alias("avg_sadness"),
        avg("surprise").alias("avg_surprise"),
        avg("trust").alias("avg_trust"),
        # VAD statistics (used in feature similarity)
        avg("avg_valence").alias("avg_valence"),
        avg("avg_arousal").alias("avg_arousal"),
        avg("avg_dominance").alias("avg_dominance"),
        # Chunk count (informational)
        count("chunk_index").alias("num_chunks"),
        # Collect emotion trajectory as array (used in trajectory similarity)
        sort_array(
            collect_list(
                array(
                    col("chunk_index").cast("double"),
                    col("anger").cast("double"),
                    col("anticipation").cast("double"),
                    col("disgust").cast("double"),
                    col("fear").cast("double"),
                    col("joy").cast("double"),
                    col("sadness").cast("double"),
                    col("surprise").cast("double"),
                    col("trust").cast("double"),
                )
            )
        ).alias("emotion_trajectory"),
    )

    # Calculate emotion ratios (proportion of total emotion)
    # This is the PRIMARY feature used for similarity comparison
    book_trajectories = book_trajectories.withColumn(
        "total_emotion",
        col("avg_anger")
        + col("avg_anticipation")
        + col("avg_disgust")
        + col("avg_fear")
        + col("avg_joy")
        + col("avg_sadness")
        + col("avg_surprise")
        + col("avg_trust"),
    )

    for emotion in emotions:
        book_trajectories = book_trajectories.withColumn(
            f"ratio_{emotion}",
            when(
                col("total_emotion") > 0, col(f"avg_{emotion}") / col("total_emotion")
            ).otherwise(0.125),  # Default to equal distribution (1/8)
        )

    # Drop intermediate column
    book_trajectories = book_trajectories.drop("total_emotion")

    return book_trajectories
