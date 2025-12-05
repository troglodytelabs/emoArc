"""
Score chunks with NRC Emotion and VAD lexicons.
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, sum as spark_sum, avg, count, when


def score_chunks_with_emotions(spark: SparkSession, chunks_df, emotion_df):
    """
    Score each chunk with NRC Emotion lexicon.

    Args:
        spark: SparkSession
        chunks_df: DataFrame with columns: book_id, title, author, chunk_index, word
        emotion_df: DataFrame with columns: word, emotion, value

    Returns:
        DataFrame with emotion scores per chunk
    """
    # Join chunks with emotion lexicon
    emotion_scores = chunks_df.join(
        emotion_df, chunks_df.word == emotion_df.word, "left"
    )

    # Group by book_id, chunk_index and emotion, count occurrences
    chunk_emotions = emotion_scores.groupBy(
        "book_id", "title", "author", "chunk_index", "emotion"
    ).agg(count("emotion").alias("emotion_count"))

    # Pivot to get one row per chunk with all emotions
    chunk_emotions_pivot = (
        chunk_emotions.groupBy("book_id", "title", "author", "chunk_index")
        .pivot(
            "emotion",
            values=[
                "anger",
                "anticipation",
                "disgust",
                "fear",
                "joy",
                "negative",
                "positive",
                "sadness",
                "surprise",
                "trust",
            ],
        )
        .agg(spark_sum("emotion_count").alias("count"))
        .fillna(0)
    )

    return chunk_emotions_pivot


def score_chunks_with_vad(spark: SparkSession, chunks_df, vad_df):
    """
    Score each chunk with NRC VAD lexicon.

    Args:
        spark: SparkSession
        chunks_df: DataFrame with columns: book_id, title, author, chunk_index, word
        vad_df: DataFrame with columns: term, valence, arousal, dominance

    Returns:
        DataFrame with VAD scores per chunk
    """
    # Join chunks with VAD lexicon
    vad_scores = chunks_df.join(vad_df, chunks_df.word == vad_df.term, "left")

    # Group by book_id, chunk_index and compute averages
    chunk_vad = vad_scores.groupBy("book_id", "title", "author", "chunk_index").agg(
        avg("valence").alias("avg_valence"),
        avg("arousal").alias("avg_arousal"),
        avg("dominance").alias("avg_dominance"),
        count("valence").alias("vad_word_count"),
    )

    # Fill NaN values with 0
    chunk_vad = chunk_vad.fillna(
        0, subset=["avg_valence", "avg_arousal", "avg_dominance"]
    )

    return chunk_vad


def combine_emotion_vad_scores(emotion_scores_df, vad_scores_df):
    """
    Combine emotion and VAD scores into a single DataFrame.

    Args:
        emotion_scores_df: DataFrame with emotion scores per chunk
        vad_scores_df: DataFrame with VAD scores per chunk

    Returns:
        Combined DataFrame with all scores
    """
    combined = emotion_scores_df.join(
        vad_scores_df, on=["book_id", "title", "author", "chunk_index"], how="outer"
    )

    # Fill missing values with 0
    emotion_cols = [
        "anger",
        "anticipation",
        "disgust",
        "fear",
        "joy",
        "negative",
        "positive",
        "sadness",
        "surprise",
        "trust",
    ]
    for col_name in emotion_cols:
        if col_name in combined.columns:
            combined = combined.fillna(0, subset=[col_name])

    return combined


def normalize_emotion_scores(chunks_df):
    """
    Normalize emotion scores by total word count in chunk.
    This gives us emotion density rather than raw counts.

    Args:
        chunks_df: DataFrame with emotion scores

    Returns:
        DataFrame with normalized scores
    """
    # Calculate total emotion words per chunk
    emotion_cols = [
        "anger",
        "anticipation",
        "disgust",
        "fear",
        "joy",
        "negative",
        "positive",
        "sadness",
        "surprise",
        "trust",
    ]

    # Sum all emotions to get total
    chunks_df = chunks_df.withColumn(
        "total_emotion_words",
        sum([col(c) for c in emotion_cols if c in chunks_df.columns]),
    )

    # Normalize each emotion by total (avoid division by zero)
    for col_name in emotion_cols:
        if col_name in chunks_df.columns:
            chunks_df = chunks_df.withColumn(
                f"{col_name}_normalized",
                when(
                    col("total_emotion_words") > 0,
                    col(col_name) / col("total_emotion_words"),
                ).otherwise(0.0),
            )

    return chunks_df
