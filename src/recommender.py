"""
Recommendation system based on emotion trajectory similarity.
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, pow, udf
from pyspark.sql.types import DoubleType
import math


def compute_trajectory_similarity(traj1, traj2):
    """
    Compute similarity between two emotion trajectories.
    Uses cosine similarity or Euclidean distance.

    Args:
        traj1: First trajectory (array of emotion scores)
        traj2: Second trajectory (array of emotion scores)

    Returns:
        Similarity score (higher = more similar)
    """
    if not traj1 or not traj2:
        return 0.0

    # Flatten trajectories if needed
    if isinstance(traj1[0], list):
        traj1 = [item for sublist in traj1 for item in sublist]
    if isinstance(traj2[0], list):
        traj2 = [item for sublist in traj2 for item in sublist]

    # Pad to same length
    max_len = max(len(traj1), len(traj2))
    traj1 = traj1 + [0.0] * (max_len - len(traj1))
    traj2 = traj2 + [0.0] * (max_len - len(traj2))

    # Compute cosine similarity
    dot_product = sum(a * b for a, b in zip(traj1, traj2))
    norm1 = math.sqrt(sum(a * a for a in traj1))
    norm2 = math.sqrt(sum(b * b for b in traj2))

    if norm1 == 0 or norm2 == 0:
        return 0.0

    similarity = dot_product / (norm1 * norm2)
    return float(similarity)


def compute_feature_similarity(features1, features2):
    """
    Compute similarity based on aggregated features.

    Args:
        features1: Dict with emotion statistics
        features2: Dict with emotion statistics

    Returns:
        Similarity score
    """
    # Extract key features
    feature_names = [
        "avg_anger",
        "avg_joy",
        "avg_fear",
        "avg_sadness",
        "avg_valence",
        "avg_arousal",
        "avg_dominance",
        "valence_std",
        "arousal_std",
    ]

    similarity = 0.0
    count = 0

    for feat in feature_names:
        if feat in features1 and feat in features2:
            val1 = features1[feat] or 0.0
            val2 = features2[feat] or 0.0
            # Use 1 - normalized difference as similarity
            diff = abs(val1 - val2)
            max_val = max(abs(val1), abs(val2), 1.0)
            similarity += 1.0 - (diff / max_val)
            count += 1

    return similarity / count if count > 0 else 0.0


def recommend_books(
    spark: SparkSession, trajectory_df, liked_book_id: str, top_n: int = 10
):
    """
    Recommend books with similar emotion trajectories.

    Args:
        spark: SparkSession
        trajectory_df: DataFrame with book trajectories
        liked_book_id: Book ID that user likes
        top_n: Number of recommendations

    Returns:
        DataFrame with recommended books and similarity scores
    """
    # Get liked book trajectory
    liked_book = trajectory_df.filter(col("book_id") == liked_book_id).collect()

    if not liked_book:
        return spark.createDataFrame([], trajectory_df.schema)

    liked_row = liked_book[0]

    # Compute similarity for all other books
    similarity_udf = udf(
        lambda traj: compute_trajectory_similarity(
            liked_row["emotion_trajectory"], traj
        ),
        DoubleType(),
    )

    # Add similarity scores
    recommendations = (
        trajectory_df.filter(col("book_id") != liked_book_id)
        .withColumn("similarity", similarity_udf(col("emotion_trajectory")))
        .orderBy(col("similarity").desc())
        .limit(top_n)
    )

    return recommendations.select(
        "book_id",
        "title",
        "author",
        "similarity",
        "avg_joy",
        "avg_sadness",
        "avg_fear",
        "avg_anger",
        "avg_valence",
        "avg_arousal",
    )


def recommend_by_features(
    spark: SparkSession, trajectory_df, liked_book_id: str, top_n: int = 10
):
    """
    Recommend books based on aggregated feature similarity.
    This is faster than trajectory similarity.

    Uses Euclidean distance on normalized features for similarity.

    Args:
        spark: SparkSession
        trajectory_df: DataFrame with book trajectories
        liked_book_id: Book ID that user likes
        top_n: Number of recommendations

    Returns:
        DataFrame with recommended books
    """
    # Get liked book features
    liked_book = trajectory_df.filter(col("book_id") == liked_book_id).collect()

    if not liked_book:
        return spark.createDataFrame([], trajectory_df.schema)

    liked_row = liked_book[0]

    # Extract feature values (handle None)
    def get_val(val):
        return val if val is not None else 0.0

    liked_anger = get_val(liked_row["avg_anger"])
    liked_joy = get_val(liked_row["avg_joy"])
    liked_fear = get_val(liked_row["avg_fear"])
    liked_sadness = get_val(liked_row["avg_sadness"])
    liked_valence = get_val(liked_row["avg_valence"])
    liked_arousal = get_val(liked_row["avg_arousal"])
    liked_dominance = get_val(liked_row["avg_dominance"])

    # Compute similarity using Euclidean distance (inverted as similarity)
    # Similarity = 1 / (1 + distance)
    from pyspark.sql.functions import sqrt

    recommendations = (
        trajectory_df.filter(col("book_id") != liked_book_id)
        .withColumn(
            "similarity",
            1.0
            / (
                1.0
                + sqrt(
                    pow((col("avg_anger") - liked_anger), 2)
                    + pow((col("avg_joy") - liked_joy), 2)
                    + pow((col("avg_fear") - liked_fear), 2)
                    + pow((col("avg_sadness") - liked_sadness), 2)
                    + pow((col("avg_valence") - liked_valence), 2)
                    + pow((col("avg_arousal") - liked_arousal), 2)
                    + pow((col("avg_dominance") - liked_dominance), 2)
                )
            ),
        )
        .orderBy(col("similarity").desc())
        .limit(top_n)
    )

    return recommendations.select(
        "book_id",
        "title",
        "author",
        "similarity",
        "avg_joy",
        "avg_sadness",
        "avg_fear",
        "avg_anger",
        "avg_valence",
        "avg_arousal",
    )
