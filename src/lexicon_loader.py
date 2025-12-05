"""
Load NRC Emotion and VAD lexicons into Spark DataFrames.
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, split, trim


def load_emotion_lexicon(spark: SparkSession, lexicon_path: str):
    """
    Load NRC Emotion Lexicon (word-level) into Spark DataFrame.

    Format: word\temotion\tvalue (0 or 1)
    Emotions: anger, anticipation, disgust, fear, joy, negative, positive, sadness, surprise, trust

    Returns:
        DataFrame with columns: word, emotion, value
    """
    # Read as text and split by tab
    df = spark.read.text(lexicon_path)
    df = df.select(
        split(col("value"), "\t")[0].alias("word"),
        split(col("value"), "\t")[1].alias("emotion"),
        split(col("value"), "\t")[2].alias("value"),
    )

    # Filter out invalid rows and convert value to int
    df = df.filter(
        (col("word").isNotNull())
        & (col("emotion").isNotNull())
        & (col("value").isNotNull())
    )
    df = df.withColumn("value", col("value").cast("int"))

    # Filter only emotions (exclude positive/negative if needed, or keep them)
    # Keep all emotions including positive/negative
    df = df.filter(col("value") == 1)  # Only words that have the emotion

    return df


def load_vad_lexicon(spark: SparkSession, lexicon_path: str):
    """
    Load NRC VAD Lexicon into Spark DataFrame.

    Format: term\tvalence\tarousal\tdominance

    Returns:
        DataFrame with columns: term, valence, arousal, dominance
    """
    # Read as text, skip header
    df = spark.read.text(lexicon_path)

    # Split by tab and extract columns
    df = df.select(
        split(col("value"), "\t")[0].alias("term"),
        split(col("value"), "\t")[1].alias("valence"),
        split(col("value"), "\t")[2].alias("arousal"),
        split(col("value"), "\t")[3].alias("dominance"),
    )

    # Filter out header row and invalid rows
    df = df.filter(
        (col("term") != "term")  # Skip header
        & (col("term").isNotNull())
        & (col("valence").isNotNull())
    )

    # Convert to proper types
    df = df.withColumn("valence", col("valence").cast("double"))
    df = df.withColumn("arousal", col("arousal").cast("double"))
    df = df.withColumn("dominance", col("dominance").cast("double"))

    # Trim whitespace from term
    df = df.withColumn("term", trim(col("term")))

    return df


def create_emotion_dict_broadcast(spark: SparkSession, emotion_df):
    """
    Create a broadcast dictionary mapping (word, emotion) -> value for fast lookups.

    Returns:
        Broadcast variable containing dict: {(word, emotion): value}
    """
    # Collect to driver and create dictionary
    emotion_dict = {}
    for row in emotion_df.collect():
        word = row["word"].lower()
        emotion = row["emotion"]
        emotion_dict[(word, emotion)] = 1

    return spark.sparkContext.broadcast(emotion_dict)


def create_vad_dict_broadcast(spark: SparkSession, vad_df):
    """
    Create a broadcast dictionary mapping term -> (valence, arousal, dominance).

    Returns:
        Broadcast variable containing dict: {term: (valence, arousal, dominance)}
    """
    vad_dict = {}
    for row in vad_df.collect():
        term = row["term"].lower()
        vad_dict[term] = (row["valence"], row["arousal"], row["dominance"])

    return spark.sparkContext.broadcast(vad_dict)
