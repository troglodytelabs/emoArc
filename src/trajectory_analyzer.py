"""
Analyze emotion trajectories: peaks, dominant emotions, patterns.
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
)


def get_dominant_emotion(chunk_row):
    """
    Get the dominant emotion for a chunk.

    Args:
        chunk_row: Row with emotion scores

    Returns:
        Name of dominant emotion
    """
    emotion_cols = [
        "anger",
        "anticipation",
        "disgust",
        "fear",
        "joy",
        "sadness",
        "surprise",
        "trust",
    ]

    max_emotion = None
    max_value = -1

    for emotion in emotion_cols:
        if emotion in chunk_row.asDict():
            value = chunk_row[emotion] or 0
            if value > max_value:
                max_value = value
                max_emotion = emotion

    return max_emotion if max_emotion else "neutral"


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
    book_trajectories = chunk_scores_df.groupBy("book_id", "title", "author").agg(
        # Emotion peaks
        spark_max("anger").alias("max_anger"),
        spark_max("joy").alias("max_joy"),
        spark_max("fear").alias("max_fear"),
        spark_max("sadness").alias("max_sadness"),
        spark_max("surprise").alias("max_surprise"),
        # Average emotions
        avg("anger").alias("avg_anger"),
        avg("joy").alias("avg_joy"),
        avg("fear").alias("avg_fear"),
        avg("sadness").alias("avg_sadness"),
        # VAD statistics
        avg("avg_valence").alias("avg_valence"),
        avg("avg_arousal").alias("avg_arousal"),
        avg("avg_dominance").alias("avg_dominance"),
        stddev("avg_valence").alias("valence_std"),
        stddev("avg_arousal").alias("arousal_std"),
        # Trajectory features
        count("chunk_index").alias("num_chunks"),
        # Collect emotion trajectory as array
        sort_array(
            collect_list(
                array(
                    col("chunk_index"),
                    col("joy"),
                    col("sadness"),
                    col("fear"),
                    col("anger"),
                )
            )
        ).alias("emotion_trajectory"),
    )

    return book_trajectories


def find_emotion_peaks(
    spark: SparkSession, chunk_scores_df, emotion: str, threshold: float = 0.1
):
    """
    Find peaks for a specific emotion.

    Args:
        spark: SparkSession
        chunk_scores_df: DataFrame with emotion scores
        emotion: Emotion name to analyze
        threshold: Minimum value to consider a peak

    Returns:
        DataFrame with peak locations
    """
    from pyspark.sql.functions import lag, lead
    from pyspark.sql import Window

    window = Window.partitionBy("book_id").orderBy("chunk_index")

    # Add previous and next values
    chunk_scores_df = chunk_scores_df.withColumn(
        "prev_value", lag(col(emotion), 1).over(window)
    ).withColumn("next_value", lead(col(emotion), 1).over(window))

    # Find local maxima (peaks)
    peaks = chunk_scores_df.filter(
        (col(emotion) >= threshold)
        & ((col("prev_value").isNull()) | (col(emotion) >= col("prev_value")))
        & ((col("next_value").isNull()) | (col(emotion) >= col("next_value")))
    ).select("book_id", "title", "chunk_index", emotion.alias("peak_value"))

    return peaks


def get_emotion_arc_type(trajectory_df):
    """
    Classify emotion arc type (e.g., "rise-fall", "steady", "climax").

    Args:
        trajectory_df: DataFrame with trajectory data

    Returns:
        DataFrame with arc_type column
    """
    # Simple classification based on valence trajectory
    # This is a simplified version - can be enhanced

    def classify_arc(row):
        valence_std = row["valence_std"] or 0
        avg_valence = row["avg_valence"] or 0

        if valence_std < 0.1:
            return "steady"
        elif avg_valence > 0.3:
            return "positive"
        elif avg_valence < -0.3:
            return "negative"
        else:
            return "mixed"

    # This would need to be implemented as a UDF
    # For now, return the trajectory_df as-is
    return trajectory_df
