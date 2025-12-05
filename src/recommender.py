"""
Recommendation system based on emotion trajectory similarity.
- Uses per-book normalized emotion scores (z-scores and ratios) for better comparison
- Emotion ratios make books of different lengths/intensities comparable
- Z-scores highlight what makes each book emotionally distinctive
"""

from pyspark.sql import SparkSession, Window
from pyspark.sql.functions import (
    col,
    pow,
    udf,
    lit,
    sqrt,
    min as spark_min,
    max as spark_max,
    row_number,
    coalesce,
)
from pyspark.sql.types import DoubleType
import math


def compute_trajectory_similarity(traj1, traj2):
    """
    Compute similarity between two emotion trajectories.
    Uses cosine similarity.

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


def compute_feature_similarity(
    spark: SparkSession,
    trajectory_df,
    liked_book_id: str,
):
    """
    Compute feature-based similarity for all books compared to the liked book.

    Uses emotion RATIOS when available (ratio_*), which represent
    the proportion of each emotion relative to total emotion. This makes
    books of different lengths/intensities comparable.

    Falls back to normalized averages if ratios not available.

    Features used:
    - 8 emotion ratios (or averages): anger, anticipation, disgust, fear, joy, sadness, surprise, trust
    - 3 VAD scores (valence, arousal, dominance) - already normalized 0-1

    Args:
        spark: SparkSession
        trajectory_df: DataFrame with book trajectories
        liked_book_id: Book ID that user likes

    Returns:
        DataFrame with feature_similarity column
    """
    # Get liked book features
    liked_book = trajectory_df.filter(col("book_id") == liked_book_id).collect()
    if not liked_book:
        return spark.createDataFrame([], trajectory_df.schema)

    liked_row = liked_book[0]

    # Check if ratio columns are available (from improved trajectory_analyzer)
    use_ratios = "ratio_anger" in trajectory_df.columns

    # Extract feature values (handle None)
    def get_val(row, col_name, default=0.0):
        try:
            val = row[col_name]
            return val if val is not None else default
        except (KeyError, TypeError):
            return default

    # Compute min/max for normalization (exclude the liked book to avoid bias)
    other_books = trajectory_df.filter(col("book_id") != liked_book_id)

    if use_ratios:
        # Use emotion ratios - they're already normalized (sum to 1)
        # Apply DISCRIMINATIVE WEIGHTING based on genre differentiation analysis:
        # - Fear, Joy have highest std (2.13%, 1.92%) → weight = 3.0
        # - Trust has high std but is uniform across genres → weight = 0.5
        # - Anticipation has high std → weight = 2.0
        # - Others (anger, sadness, disgust, surprise) → weight = 1.5
        # - VAD scores show big differences across genres → weight = 4.0

        # Get liked book ratios
        liked_anger = get_val(liked_row, "ratio_anger", 0.125)
        liked_anticipation = get_val(liked_row, "ratio_anticipation", 0.125)
        liked_disgust = get_val(liked_row, "ratio_disgust", 0.125)
        liked_fear = get_val(liked_row, "ratio_fear", 0.125)
        liked_joy = get_val(liked_row, "ratio_joy", 0.125)
        liked_sadness = get_val(liked_row, "ratio_sadness", 0.125)
        liked_surprise = get_val(liked_row, "ratio_surprise", 0.125)
        liked_trust = get_val(liked_row, "ratio_trust", 0.125)

        # VAD scores - multiply by 10 to scale from [-1,1] to comparable range
        liked_valence = get_val(liked_row, "avg_valence", 0.5)
        liked_arousal = get_val(liked_row, "avg_arousal", 0.5)
        liked_dominance = get_val(liked_row, "avg_dominance", 0.5)

        # Use coalesce to handle potentially missing columns
        # Apply discriminative weights to emphasize genre-distinguishing features
        feature_sim_df = other_books.withColumn(
            "feature_similarity",
            1.0
            / (
                1.0
                + sqrt(
                    # HIGH discriminative emotions (fear, joy) - weight 3.0
                    pow((coalesce(col("ratio_fear"), lit(0.125)) - liked_fear) * 3.0, 2)
                    + pow((coalesce(col("ratio_joy"), lit(0.125)) - liked_joy) * 3.0, 2)
                    # MEDIUM discriminative emotions - weight 2.0
                    + pow(
                        (
                            coalesce(col("ratio_anticipation"), lit(0.125))
                            - liked_anticipation
                        )
                        * 2.0,
                        2,
                    )
                    + pow(
                        (coalesce(col("ratio_sadness"), lit(0.125)) - liked_sadness)
                        * 2.0,
                        2,
                    )
                    + pow(
                        (coalesce(col("ratio_anger"), lit(0.125)) - liked_anger) * 1.5,
                        2,
                    )
                    + pow(
                        (coalesce(col("ratio_disgust"), lit(0.125)) - liked_disgust)
                        * 1.5,
                        2,
                    )
                    + pow(
                        (coalesce(col("ratio_surprise"), lit(0.125)) - liked_surprise)
                        * 1.5,
                        2,
                    )
                    # LOW discriminative (trust is uniform) - weight 0.5
                    + pow(
                        (coalesce(col("ratio_trust"), lit(0.125)) - liked_trust) * 0.5,
                        2,
                    )
                    # VAD features - amplified (weight 4.0) as they show larger genre differences
                    + pow(
                        (coalesce(col("avg_valence"), lit(0.5)) - liked_valence) * 4.0,
                        2,
                    )
                    + pow(
                        (coalesce(col("avg_arousal"), lit(0.5)) - liked_arousal) * 4.0,
                        2,
                    )
                    + pow(
                        (coalesce(col("avg_dominance"), lit(0.5)) - liked_dominance)
                        * 4.0,
                        2,
                    )
                )
            ),
        ).select(
            "book_id",
            "title",
            "author",
            "feature_similarity",
            "avg_anger",
            "avg_anticipation",
            "avg_disgust",
            "avg_fear",
            "avg_joy",
            "avg_sadness",
            "avg_surprise",
            "avg_trust",
            "avg_valence",
            "avg_arousal",
        )

        return feature_sim_df

    # Fallback: Original approach using min-max normalization
    # (when ratio columns not available)

    # Get ranges for normalization (all 8 Plutchik emotions + VAD)
    stats = other_books.agg(
        # All 8 Plutchik emotions
        spark_min("avg_anger").alias("min_anger"),
        spark_max("avg_anger").alias("max_anger"),
        spark_min("avg_anticipation").alias("min_anticipation"),
        spark_max("avg_anticipation").alias("max_anticipation"),
        spark_min("avg_disgust").alias("min_disgust"),
        spark_max("avg_disgust").alias("max_disgust"),
        spark_min("avg_fear").alias("min_fear"),
        spark_max("avg_fear").alias("max_fear"),
        spark_min("avg_joy").alias("min_joy"),
        spark_max("avg_joy").alias("max_joy"),
        spark_min("avg_sadness").alias("min_sadness"),
        spark_max("avg_sadness").alias("max_sadness"),
        spark_min("avg_surprise").alias("min_surprise"),
        spark_max("avg_surprise").alias("max_surprise"),
        spark_min("avg_trust").alias("min_trust"),
        spark_max("avg_trust").alias("max_trust"),
        # VAD scores
        spark_min("avg_valence").alias("min_valence"),
        spark_max("avg_valence").alias("max_valence"),
        spark_min("avg_arousal").alias("min_arousal"),
        spark_max("avg_arousal").alias("max_arousal"),
        spark_min("avg_dominance").alias("min_dominance"),
        spark_max("avg_dominance").alias("max_dominance"),
    ).first()

    # Normalize liked book features
    def normalize(val, min_val, max_val):
        if min_val is None or max_val is None:
            return 0.5
        if max_val == min_val:
            return 0.5  # If all values are same, use middle
        return (val - min_val) / (max_val - min_val)

    # Get min/max values with defaults (all 8 Plutchik emotions + VAD)
    min_anger = stats["min_anger"] if stats["min_anger"] is not None else 0.0
    max_anger = stats["max_anger"] if stats["max_anger"] is not None else 1.0
    min_anticipation = (
        stats["min_anticipation"] if stats["min_anticipation"] is not None else 0.0
    )
    max_anticipation = (
        stats["max_anticipation"] if stats["max_anticipation"] is not None else 1.0
    )
    min_disgust = stats["min_disgust"] if stats["min_disgust"] is not None else 0.0
    max_disgust = stats["max_disgust"] if stats["max_disgust"] is not None else 1.0
    min_fear = stats["min_fear"] if stats["min_fear"] is not None else 0.0
    max_fear = stats["max_fear"] if stats["max_fear"] is not None else 1.0
    min_joy = stats["min_joy"] if stats["min_joy"] is not None else 0.0
    max_joy = stats["max_joy"] if stats["max_joy"] is not None else 1.0
    min_sadness = stats["min_sadness"] if stats["min_sadness"] is not None else 0.0
    max_sadness = stats["max_sadness"] if stats["max_sadness"] is not None else 1.0
    min_surprise = stats["min_surprise"] if stats["min_surprise"] is not None else 0.0
    max_surprise = stats["max_surprise"] if stats["max_surprise"] is not None else 1.0
    min_trust = stats["min_trust"] if stats["min_trust"] is not None else 0.0
    max_trust = stats["max_trust"] if stats["max_trust"] is not None else 1.0
    min_valence = stats["min_valence"] if stats["min_valence"] is not None else -1.0
    max_valence = stats["max_valence"] if stats["max_valence"] is not None else 1.0
    min_arousal = stats["min_arousal"] if stats["min_arousal"] is not None else -1.0
    max_arousal = stats["max_arousal"] if stats["max_arousal"] is not None else 1.0
    min_dominance = (
        stats["min_dominance"] if stats["min_dominance"] is not None else -1.0
    )
    max_dominance = (
        stats["max_dominance"] if stats["max_dominance"] is not None else 1.0
    )

    # Normalize liked book features (all 8 Plutchik emotions + VAD)
    liked_anger_norm = normalize(get_val(liked_row["avg_anger"]), min_anger, max_anger)
    liked_anticipation_norm = normalize(
        get_val(liked_row["avg_anticipation"]), min_anticipation, max_anticipation
    )
    liked_disgust_norm = normalize(
        get_val(liked_row["avg_disgust"]), min_disgust, max_disgust
    )
    liked_fear_norm = normalize(get_val(liked_row["avg_fear"]), min_fear, max_fear)
    liked_joy_norm = normalize(get_val(liked_row["avg_joy"]), min_joy, max_joy)
    liked_sadness_norm = normalize(
        get_val(liked_row["avg_sadness"]), min_sadness, max_sadness
    )
    liked_surprise_norm = normalize(
        get_val(liked_row["avg_surprise"]), min_surprise, max_surprise
    )
    liked_trust_norm = normalize(get_val(liked_row["avg_trust"]), min_trust, max_trust)
    liked_valence_norm = normalize(
        get_val(liked_row["avg_valence"]), min_valence, max_valence
    )
    liked_arousal_norm = normalize(
        get_val(liked_row["avg_arousal"]), min_arousal, max_arousal
    )
    liked_dominance_norm = normalize(
        get_val(liked_row["avg_dominance"]), min_dominance, max_dominance
    )

    # Normalize all features in the DataFrame and compute similarity
    def norm_col(col_name, min_val, max_val):
        if min_val == max_val:
            return lit(0.5)
        return (col(col_name) - min_val) / (max_val - min_val)

    feature_sim_df = (
        other_books.withColumn(
            "anger_norm", norm_col("avg_anger", min_anger, max_anger)
        )
        .withColumn(
            "anticipation_norm",
            norm_col("avg_anticipation", min_anticipation, max_anticipation),
        )
        .withColumn("disgust_norm", norm_col("avg_disgust", min_disgust, max_disgust))
        .withColumn("fear_norm", norm_col("avg_fear", min_fear, max_fear))
        .withColumn("joy_norm", norm_col("avg_joy", min_joy, max_joy))
        .withColumn("sadness_norm", norm_col("avg_sadness", min_sadness, max_sadness))
        .withColumn(
            "surprise_norm", norm_col("avg_surprise", min_surprise, max_surprise)
        )
        .withColumn("trust_norm", norm_col("avg_trust", min_trust, max_trust))
        .withColumn("valence_norm", norm_col("avg_valence", min_valence, max_valence))
        .withColumn("arousal_norm", norm_col("avg_arousal", min_arousal, max_arousal))
        .withColumn(
            "dominance_norm", norm_col("avg_dominance", min_dominance, max_dominance)
        )
        .withColumn(
            "feature_similarity",
            1.0
            / (
                1.0
                + sqrt(
                    pow((col("anger_norm") - liked_anger_norm), 2)
                    + pow((col("anticipation_norm") - liked_anticipation_norm), 2)
                    + pow((col("disgust_norm") - liked_disgust_norm), 2)
                    + pow((col("fear_norm") - liked_fear_norm), 2)
                    + pow((col("joy_norm") - liked_joy_norm), 2)
                    + pow((col("sadness_norm") - liked_sadness_norm), 2)
                    + pow((col("surprise_norm") - liked_surprise_norm), 2)
                    + pow((col("trust_norm") - liked_trust_norm), 2)
                    + pow((col("valence_norm") - liked_valence_norm), 2)
                    + pow((col("arousal_norm") - liked_arousal_norm), 2)
                    + pow((col("dominance_norm") - liked_dominance_norm), 2)
                )
            ),
        )
        .select(
            "book_id",
            "title",
            "author",
            "feature_similarity",
            "avg_anger",
            "avg_anticipation",
            "avg_disgust",
            "avg_fear",
            "avg_joy",
            "avg_sadness",
            "avg_surprise",
            "avg_trust",
            "avg_valence",
            "avg_arousal",
        )
    )

    return feature_sim_df


def recommend(
    spark: SparkSession,
    trajectory_df,
    liked_book_id: str,
    top_n: int = 10,
    feature_weight: float = 0.6,
    trajectory_weight: float = 0.4,
):
    """
    Recommend books with similar emotion trajectories.

    Combines feature-based similarity (11 aggregated features) and trajectory similarity
    (emotion sequences) when available. Falls back to feature-based only when trajectory
    arrays are not available.

    Args:
        spark: SparkSession
        trajectory_df: DataFrame with book trajectories
        liked_book_id: Book ID that user likes
        top_n: Number of recommendations
        feature_weight: Weight for feature-based similarity (default: 0.6)
        trajectory_weight: Weight for trajectory similarity (default: 0.4)
        Note: weights are normalized to sum to 1.0

    Returns:
        DataFrame with recommended books and similarity scores
    """
    # Remove duplicates (same book_id) - keep first occurrence
    window = Window.partitionBy("book_id").orderBy("book_id")
    trajectory_df = (
        trajectory_df.withColumn("rn", row_number().over(window))
        .filter(col("rn") == 1)
        .drop("rn")
    )

    # Normalize weights
    total_weight = feature_weight + trajectory_weight
    if total_weight > 0:
        feature_weight = feature_weight / total_weight
        trajectory_weight = trajectory_weight / total_weight

    # Get liked book
    liked_book = trajectory_df.filter(col("book_id") == liked_book_id).collect()
    if not liked_book:
        return spark.createDataFrame([], trajectory_df.schema)

    liked_row = liked_book[0]
    has_trajectory = False
    if "emotion_trajectory" in trajectory_df.columns:
        try:
            traj_value = liked_row["emotion_trajectory"]
            has_trajectory = traj_value is not None
        except (KeyError, AttributeError):
            has_trajectory = False

    # Compute feature-based similarity (always available)
    feature_sim_df = compute_feature_similarity(spark, trajectory_df, liked_book_id)

    # Compute trajectory similarity if available
    if has_trajectory and "emotion_trajectory" in trajectory_df.columns:
        # Check if other books also have trajectories
        other_books_with_traj = trajectory_df.filter(
            (col("book_id") != liked_book_id) & (col("emotion_trajectory").isNotNull())
        )

        if other_books_with_traj.count() > 0:
            # Compute trajectory similarity for all books
            similarity_udf = udf(
                lambda traj: compute_trajectory_similarity(
                    liked_row["emotion_trajectory"], traj
                )
                if traj is not None
                else 0.0,
                DoubleType(),
            )

            trajectory_sim_df = (
                trajectory_df.filter(col("book_id") != liked_book_id)
                .withColumn(
                    "trajectory_similarity", similarity_udf(col("emotion_trajectory"))
                )
                .select("book_id", "trajectory_similarity")
            )

            # Join feature and trajectory similarities
            combined_df = feature_sim_df.join(
                trajectory_sim_df, on="book_id", how="outer"
            )

            # Fill missing trajectory similarity with 0 (when trajectory arrays not available)
            combined_df = combined_df.fillna(0.0, subset=["trajectory_similarity"])

            # Compute combined similarity: weighted average
            combined_df = combined_df.withColumn(
                "similarity",
                (feature_weight * col("feature_similarity"))
                + (trajectory_weight * col("trajectory_similarity")),
            )
        else:
            # No other books have trajectories, use only feature-based
            combined_df = feature_sim_df.withColumn(
                "similarity", col("feature_similarity")
            )
    else:
        # No trajectory data available, use only feature-based
        combined_df = feature_sim_df.withColumn("similarity", col("feature_similarity"))

    # Select final columns and order by similarity
    result = (
        combined_df.select(
            "book_id",
            "title",
            "author",
            "similarity",
            "avg_anger",
            "avg_anticipation",
            "avg_disgust",
            "avg_fear",
            "avg_joy",
            "avg_sadness",
            "avg_surprise",
            "avg_trust",
            "avg_valence",
            "avg_arousal",
        )
        .orderBy(col("similarity").desc())
        .limit(top_n)
    )

    return result
