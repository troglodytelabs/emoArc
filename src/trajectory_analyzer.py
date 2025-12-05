"""
Analyze emotion trajectories: peaks, dominant emotions, patterns.

Includes per-book normalization to make emotion scores comparable across books.
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col,
    max as spark_max,
    avg,
    stddev,
    array,
    collect_list,
    sort_array,
    count,
    lit,
    when,
    sqrt as spark_sqrt,
)


def analyze_trajectory(spark: SparkSession, chunk_scores_df):
    """
    Analyze emotion trajectory for each book.

    Args:
        spark: SparkSession
        chunk_scores_df: DataFrame with emotion and VAD scores per chunk

    Returns:
        DataFrame with trajectory analysis per book
    """
    # Calculate trajectory statistics per book
    # Use all 8 Plutchik emotions: anger, anticipation, disgust, fear, joy, sadness, surprise, trust
    book_trajectories = chunk_scores_df.groupBy("book_id", "title", "author").agg(
        # Emotion peaks (all 8 Plutchik emotions)
        spark_max("anger").alias("max_anger"),
        spark_max("anticipation").alias("max_anticipation"),
        spark_max("disgust").alias("max_disgust"),
        spark_max("fear").alias("max_fear"),
        spark_max("joy").alias("max_joy"),
        spark_max("sadness").alias("max_sadness"),
        spark_max("surprise").alias("max_surprise"),
        spark_max("trust").alias("max_trust"),
        # Average emotions (all 8 Plutchik emotions)
        avg("anger").alias("avg_anger"),
        avg("anticipation").alias("avg_anticipation"),
        avg("disgust").alias("avg_disgust"),
        avg("fear").alias("avg_fear"),
        avg("joy").alias("avg_joy"),
        avg("sadness").alias("avg_sadness"),
        avg("surprise").alias("avg_surprise"),
        avg("trust").alias("avg_trust"),
        # VAD statistics
        avg("avg_valence").alias("avg_valence"),
        avg("avg_arousal").alias("avg_arousal"),
        avg("avg_dominance").alias("avg_dominance"),
        stddev("avg_valence").alias("valence_std"),
        stddev("avg_arousal").alias("arousal_std"),
        # Trajectory features
        count("chunk_index").alias("num_chunks"),
        # Collect emotion trajectory as array (all 8 Plutchik emotions)
        sort_array(
            collect_list(
                array(
                    col("chunk_index"),
                    col("anger"),
                    col("anticipation"),
                    col("disgust"),
                    col("fear"),
                    col("joy"),
                    col("sadness"),
                    col("surprise"),
                    col("trust"),
                )
            )
        ).alias("emotion_trajectory"),
    )

    # Add per-book normalized emotion scores (z-scores)
    # This makes emotions comparable across books of different lengths/intensities
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

    # Calculate mean and std of all emotions for each book
    book_trajectories = book_trajectories.withColumn(
        "emotion_mean",
        (
            col("avg_anger")
            + col("avg_anticipation")
            + col("avg_disgust")
            + col("avg_fear")
            + col("avg_joy")
            + col("avg_sadness")
            + col("avg_surprise")
            + col("avg_trust")
        )
        / 8.0,
    )

    # Calculate variance for standard deviation
    book_trajectories = book_trajectories.withColumn(
        "emotion_var",
        (
            (col("avg_anger") - col("emotion_mean"))
            * (col("avg_anger") - col("emotion_mean"))
            + (col("avg_anticipation") - col("emotion_mean"))
            * (col("avg_anticipation") - col("emotion_mean"))
            + (col("avg_disgust") - col("emotion_mean"))
            * (col("avg_disgust") - col("emotion_mean"))
            + (col("avg_fear") - col("emotion_mean"))
            * (col("avg_fear") - col("emotion_mean"))
            + (col("avg_joy") - col("emotion_mean"))
            * (col("avg_joy") - col("emotion_mean"))
            + (col("avg_sadness") - col("emotion_mean"))
            * (col("avg_sadness") - col("emotion_mean"))
            + (col("avg_surprise") - col("emotion_mean"))
            * (col("avg_surprise") - col("emotion_mean"))
            + (col("avg_trust") - col("emotion_mean"))
            * (col("avg_trust") - col("emotion_mean"))
        )
        / 8.0,
    )

    book_trajectories = book_trajectories.withColumn(
        "emotion_std", spark_sqrt(col("emotion_var"))
    )

    # Add normalized emotion scores (z-scores)
    # Use when() to avoid division by zero
    for emotion in emotions:
        book_trajectories = book_trajectories.withColumn(
            f"norm_{emotion}",
            when(
                col("emotion_std") > 0,
                (col(f"avg_{emotion}") - col("emotion_mean")) / col("emotion_std"),
            ).otherwise(0.0),
        )

    # Calculate emotion ratios (proportion of total emotion)
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
            ).otherwise(0.125),  # Default to equal distribution
        )

    return book_trajectories
