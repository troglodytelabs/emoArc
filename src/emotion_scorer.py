"""
Score chunks with NRC Emotion and VAD lexicons.
Uses UDF-based scoring with words as arrays (memory efficient).
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf, broadcast
from pyspark.sql.types import (
    StructType,
    StructField,
    IntegerType,
    DoubleType,
)


def score_chunks(spark: SparkSession, chunks_df, emotion_df, vad_df):
    """
    Score each chunk with both NRC Emotion and VAD lexicons using UDF-based approach.

    This approach keeps words as arrays and uses broadcast variables for lexicons,
    which is much more memory efficient than exploding words into rows.

    Args:
        spark: SparkSession
        chunks_df: DataFrame with columns: book_id, title, author, chunk_index, words (array)
        emotion_df: DataFrame with columns: word, emotion, value
        vad_df: DataFrame with columns: term, valence, arousal, dominance

    Returns:
        DataFrame with combined emotion and VAD scores per chunk
    """
    # Convert lexicons to dictionaries and broadcast them
    # Emotion lexicon: word -> {emotion: count}
    emotion_rows = (
        emotion_df.filter(col("value") == 1).select("word", "emotion").collect()
    )
    emotion_dict = {}
    for row in emotion_rows:
        word = row["word"]
        emotion = row["emotion"]
        if word not in emotion_dict:
            emotion_dict[word] = set()
        emotion_dict[word].add(emotion)

    # VAD lexicon: word -> (valence, arousal, dominance)
    vad_rows = vad_df.select("term", "valence", "arousal", "dominance").collect()
    vad_dict = {
        row["term"]: (row["valence"], row["arousal"], row["dominance"])
        for row in vad_rows
    }

    # Broadcast the dictionaries
    emotion_bc = spark.sparkContext.broadcast(emotion_dict)
    vad_bc = spark.sparkContext.broadcast(vad_dict)

    # Define the emotions we care about (Plutchik's 8 basic emotions)
    emotions_list = [
        "anger",
        "anticipation",
        "disgust",
        "fear",
        "joy",
        "sadness",
        "surprise",
        "trust",
    ]

    # Define output schema for the scoring UDF
    score_schema = StructType(
        [
            StructField("anger", IntegerType(), True),
            StructField("anticipation", IntegerType(), True),
            StructField("disgust", IntegerType(), True),
            StructField("fear", IntegerType(), True),
            StructField("joy", IntegerType(), True),
            StructField("sadness", IntegerType(), True),
            StructField("surprise", IntegerType(), True),
            StructField("trust", IntegerType(), True),
            StructField("avg_valence", DoubleType(), True),
            StructField("avg_arousal", DoubleType(), True),
            StructField("avg_dominance", DoubleType(), True),
            StructField("vad_word_count", IntegerType(), True),
        ]
    )

    def score_words(words):
        """Score a list of words against emotion and VAD lexicons."""
        if not words:
            return (0, 0, 0, 0, 0, 0, 0, 0, 0.0, 0.0, 0.0, 0)

        emotion_lookup = emotion_bc.value
        vad_lookup = vad_bc.value

        # Count emotions
        emotion_counts = {e: 0 for e in emotions_list}

        # Collect VAD values
        valence_sum = 0.0
        arousal_sum = 0.0
        dominance_sum = 0.0
        vad_count = 0

        for word in words:
            # Emotion scoring
            if word in emotion_lookup:
                for emotion in emotion_lookup[word]:
                    if emotion in emotion_counts:
                        emotion_counts[emotion] += 1

            # VAD scoring
            if word in vad_lookup:
                v, a, d = vad_lookup[word]
                valence_sum += v
                arousal_sum += a
                dominance_sum += d
                vad_count += 1

        # Compute VAD averages
        avg_v = valence_sum / vad_count if vad_count > 0 else 0.0
        avg_a = arousal_sum / vad_count if vad_count > 0 else 0.0
        avg_d = dominance_sum / vad_count if vad_count > 0 else 0.0

        return (
            emotion_counts["anger"],
            emotion_counts["anticipation"],
            emotion_counts["disgust"],
            emotion_counts["fear"],
            emotion_counts["joy"],
            emotion_counts["sadness"],
            emotion_counts["surprise"],
            emotion_counts["trust"],
            avg_v,
            avg_a,
            avg_d,
            vad_count,
        )

    score_udf = udf(score_words, score_schema)

    # Apply scoring UDF and extract fields
    scored = chunks_df.withColumn("scores", score_udf(col("words")))

    # Extract individual score columns
    result = scored.select(
        col("book_id"),
        col("title"),
        col("author"),
        col("chunk_index"),
        col("scores.anger").alias("anger"),
        col("scores.anticipation").alias("anticipation"),
        col("scores.disgust").alias("disgust"),
        col("scores.fear").alias("fear"),
        col("scores.joy").alias("joy"),
        col("scores.sadness").alias("sadness"),
        col("scores.surprise").alias("surprise"),
        col("scores.trust").alias("trust"),
        col("scores.avg_valence").alias("avg_valence"),
        col("scores.avg_arousal").alias("avg_arousal"),
        col("scores.avg_dominance").alias("avg_dominance"),
        col("scores.vad_word_count").alias("vad_word_count"),
    )

    return result
